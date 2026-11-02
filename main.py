"""在 AstrBot 消息管线进入 LLM 会话锁之前收集连续私聊图文。"""

from __future__ import annotations

import asyncio
import base64
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import astrbot
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At, Image, Plain, Reply
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star

from .adaptive_timer import AdaptiveSettings, adaptive_wait
from .message_buffer import BufferedMessage, MessageBuffer, SessionState
from .quote_parser import QuotedMessage, parse_quote
from .typing_detector import is_notice, parse_notice

SessionKey = tuple[str, str]


def _message_event_hook(handler):
    """在类加载前检查消息 API，避免缺少装饰器时插件直接导入失败。"""
    register = getattr(filter, "event_message_type", None)
    all_messages = getattr(getattr(filter, "EventMessageType", None), "ALL", None)
    if callable(register) and all_messages is not None:
        return register(all_messages, priority=100)(handler)
    return handler


def _legacy_llm_hook(handler):
    """缺少旧版请求钩子的定制宿主继续使用早期消息入口。"""
    register = getattr(filter, "on_llm_request", None)
    return register(priority=100)(handler) if callable(register) else handler


class DebouncePlugin(Star):
    """按平台实例、机器人账号和统一会话来源隔离防抖状态。"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.settings = AdaptiveSettings.from_config(config)
        self.sessions: dict[SessionKey, SessionState] = {}
        self._finishing: dict[int, SessionState] = {}
        self._closed = False
        self._detect_version_support()

    def _detect_version_support(self) -> None:
        """将运行版本与消息 API 分开检测，按版本选择基础或增强模式。"""
        register = getattr(filter, "event_message_type", None)
        all_messages = getattr(getattr(filter, "EventMessageType", None), "ALL", None)
        self.supports_message_events = callable(register) and all_messages is not None
        self.astrbot_version = str(getattr(astrbot, "__version__", "unknown"))
        match = re.match(r"^v?(\d+)\.(\d+)(?:\.(\d+))?", self.astrbot_version)
        version = tuple(int(part or 0) for part in match.groups()) if match else None
        prerelease = bool(
            match and re.search(r"(?:a|b|rc|dev)\d*", self.astrbot_version[match.end() :])
        )
        self.supports_enhanced_events = bool(
            self.supports_message_events
            and version is not None
            and version >= (4, 29, 0)
            and not prerelease
        )
        self.legacy_mode = not self.supports_enhanced_events
        if not self.supports_message_events:
            # 官方受支持版本均提供该 API。定制宿主缺失时直通，避免锁内等待。
            logger.warning("[消息防抖] 宿主缺少 event_message_type API，已停用防抖并放行消息")
        elif self.legacy_mode:
            logger.info(
                f"[消息防抖] AstrBot {self.astrbot_version} 使用基础防抖，输入感知和撤回过滤停用"
            )

    @staticmethod
    def _now() -> float:
        """用单调时钟，避免系统校时改变消息等待时长。"""
        return asyncio.get_running_loop().time()

    @staticmethod
    def _key(event: AstrMessageEvent) -> SessionKey:
        """优先使用统一来源；旧事件用平台实例、消息类型与会话重建来源。"""
        message = getattr(event, "message_obj", None)
        get_self_id = getattr(event, "get_self_id", None)
        account = get_self_id() if callable(get_self_id) else getattr(message, "self_id", "")
        origin = getattr(event, "unified_msg_origin", None)
        if origin:
            return str(account), str(origin)
        get_platform_id = getattr(event, "get_platform_id", None)
        platform_meta = getattr(event, "platform_meta", None) or getattr(event, "platform", None)
        platform_id = get_platform_id() if callable(get_platform_id) else None
        platform_id = platform_id or getattr(platform_meta, "id", None)
        if not platform_id:
            get_platform_name = getattr(event, "get_platform_name", None)
            platform_id = get_platform_name() if callable(get_platform_name) else "unknown-platform"
        message_type = getattr(message, "type", None)
        message_type = getattr(message_type, "value", message_type) or "FriendMessage"
        session_id = getattr(event, "session_id", None) or getattr(message, "session_id", None)
        if not session_id:
            get_sender_id = getattr(event, "get_sender_id", None)
            session_id = (
                get_sender_id()
                if callable(get_sender_id)
                else getattr(getattr(message, "sender", None), "user_id", "")
            )
        return str(account), f"{platform_id}:{message_type}:{session_id}"

    def _is_command(self, text: str) -> bool:
        """识别有效的非空前缀，兼容消息前导空白。"""
        prefixes = self.config.get("command_prefixes", ["/", "#"])
        if not isinstance(prefixes, (list, tuple)):
            prefixes = ["/", "#"]
        return any(
            isinstance(prefix, str) and prefix and text.lstrip().startswith(prefix)
            for prefix in prefixes
        )

    @staticmethod
    def _silence(event: AstrMessageEvent) -> None:
        """停止插件传播和默认 LLM 路径，保留框架停止标记。"""
        # 显式关闭默认 LLM 路径，并保留 STOP 结果给后续管线检查。
        should_call_llm = getattr(event, "should_call_llm", None)
        if callable(should_call_llm):
            should_call_llm(True)
        else:
            event.call_llm = True
        event._debounce_suppressed = True
        event.stop_event()

    @staticmethod
    def _sources(event: AstrMessageEvent) -> tuple[str, ...]:
        """只收集原图来源，优先 URL，按首次出现顺序去重。"""
        return tuple(
            dict.fromkeys(
                str(component.url or component.file)
                for component in event.get_messages()
                if isinstance(component, Image) and (component.url or component.file)
            )
        )

    @staticmethod
    async def _preserve_sources(sources: tuple[str, ...]) -> tuple[str, ...]:
        """临时文件在事件结束时可能被清理，读入内存；远程图片保持原 URL。"""
        preserved = []
        for source in sources:
            if source.startswith(("http://", "https://", "base64://", "data:image/")):
                preserved.append(source)
                continue
            if source.startswith("file://"):
                parsed = urlparse(source)
                filename = unquote(parsed.path)
                if parsed.netloc:
                    filename = f"//{parsed.netloc}{filename}"
                elif len(filename) > 2 and filename[0] == "/" and filename[2] == ":":
                    filename = filename[1:]
            else:
                filename = source
            try:
                data = await asyncio.to_thread(Path(filename).read_bytes)
            except (OSError, ValueError):
                logger.warning("[消息防抖] 图片来源不可读，保留原始来源")
                preserved.append(source)
            else:
                preserved.append("base64://" + base64.b64encode(data).decode("ascii"))
        return tuple(dict.fromkeys(preserved))

    async def _prepare_item(
        self, state: SessionState, item: BufferedMessage, event: AstrMessageEvent
    ) -> None:
        """入队后再查询引用，保证慢查询不会改变消息顺序。"""
        try:
            item.image_urls = await self._preserve_sources(item.image_urls)
            quote = await parse_quote(event)
            if quote:
                images = await self._preserve_sources(quote.image_urls)
                item.quote = QuotedMessage(quote.content, quote.sender, images)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error("[消息防抖] 消息内容准备失败，保留已收集内容", exc_info=True)

    def _cancel_timer(self, state: SessionState) -> None:
        """替换计时器时取消旧任务，避免取消当前调用栈。"""
        if state.timer_task and state.timer_task is not asyncio.current_task():
            state.timer_task.cancel()
        state.timer_task = None

    def _schedule(self, key: SessionKey, state: SessionState) -> None:
        """输入状态暂停普通倒计时，两个硬截止时间始终生效。"""
        self._cancel_timer(state)
        if state.flush_event.is_set():
            return
        now = self._now()
        deadline = state.buffer.first_message_time + self.settings.max_total_wait
        if state.is_typing and state.typing_start_time is not None:
            deadline = min(deadline, state.typing_start_time + self.settings.max_typing_wait)
            delay = max(0.0, deadline - now)
        else:
            delay = adaptive_wait(state.buffer, self.settings, now)
        if delay <= 0:
            state.flush_event.set()
        else:
            state.timer_task = asyncio.create_task(self._countdown(key, state, delay))

    async def _countdown(self, key: SessionKey, state: SessionState, delay: float) -> None:
        """到期只唤醒首事件，消息重构继续在其原管线执行。"""
        try:
            await asyncio.sleep(delay)
            # 旧轮次的任务即使晚醒，也不会触发同一用户的新一轮收集。
            if self.sessions.get(key) is state:
                state.flush_event.set()
        except asyncio.CancelledError:
            pass

    async def _flush_before_passthrough(self, key: SessionKey) -> None:
        """指令及其他媒体到达时，先让已有图文完成重构。"""
        previous = [state for state in self._finishing.values() if self._key(state.event) == key]
        state = self.sessions.get(key)
        if state:
            self._cancel_timer(state)
            state.flush_event.set()
            previous.append(state)
        for state in previous:
            await state.done_event.wait()

    def _handle_notice(self, event: AstrMessageEvent) -> None:
        """将受支持的私聊通知更新到同一账号与会话的缓存。"""
        if not self.supports_enhanced_events:
            return
        notice = parse_notice(event)
        if notice is None:
            return
        # UMO 使用适配器实例 ID；额外核对账号和用户，隔离多个 QQ 连接。
        key = self._key(event)
        if notice.self_id and notice.self_id != key[0]:
            return
        if notice.recalled_message_id is not None and self.config.get("enable_recall_filter", True):
            candidates = [
                state for state in self._finishing.values() if self._key(state.event) == key
            ]
            collecting = self.sessions.get(key)
            if collecting:
                candidates.append(collecting)
            # 等待引用查询或上一轮期间，仍未交给下游的消息也可撤回。
            for state in candidates:
                if str(state.event.get_sender_id()) != notice.user_id:
                    continue
                removed = [
                    item
                    for item in state.buffer.items
                    if item.message_id == notice.recalled_message_id
                ]
                if state.buffer.remove_by_id(notice.recalled_message_id):
                    for item in removed:
                        if item.prepare_task:
                            item.prepare_task.cancel()
                    if not state.buffer.items:
                        self._cancel_timer(state)
                        state.flush_event.set()
            return
        state = self.sessions.get(key)
        if state is None or str(state.event.get_sender_id()) != notice.user_id:
            return
        if notice.typing is None or not self.config.get("enable_typing_detection", True):
            return
        if notice.typing and not state.is_typing:
            state.is_typing = True
            state.typing_start_time = self._now()
            self._schedule(key, state)
        elif not notice.typing and state.is_typing:
            state.is_typing = False
            state.typing_start_time = None
            self._schedule(key, state)

    async def _finish(self, state: SessionState) -> None:
        """等待已启动的内容准备，然后在首事件上重构合并结果。"""
        if self._closed or not state.buffer.items:
            for task in state.prepare_tasks:
                task.cancel()
            self._silence(state.event)
            return
        tasks = set(state.prepare_tasks)
        if tasks:
            remaining = state.buffer.first_message_time + self.settings.max_total_wait - self._now()
            if state.is_typing and state.typing_start_time is not None:
                remaining = min(
                    remaining, state.typing_start_time + self.settings.max_typing_wait - self._now()
                )
            _, pending = await asyncio.wait(tasks, timeout=max(0.0, min(3.0, remaining)))
            for task in pending:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._closed or not state.buffer.items:
            self._silence(state.event)
            return
        # 硬截止可能中止本地图片读取；失效路径用明确占位替代。
        # 避免后续事件清理文件后，合并结果仍把临时路径交给模型。
        for item in state.buffer.items:
            stable = ("http://", "https://", "base64://", "data:image/")
            retained = tuple(url for url in item.image_urls if url.startswith(stable))
            missing_images = len(retained) != len(item.image_urls)
            item.image_urls = retained
            if item.quote:
                quote_images = tuple(url for url in item.quote.image_urls if url.startswith(stable))
                missing_images |= len(quote_images) != len(item.quote.image_urls)
                item.quote = QuotedMessage(item.quote.content, item.quote.sender, quote_images)
            if missing_images:
                item.text += ("\n" if item.text else "") + "[图片读取失败或超时，请重新发送图片]"
                logger.warning("[消息防抖] 图片读取失败或超过本轮等待上限，已使用文字占位")
        text = state.buffer.merged_text(str(self.config.get("merge_separator", "\n")))
        images = state.buffer.image_urls
        components = ([Plain(text)] if text else []) + [Image(file=url) for url in images]
        state.event.message_str = text
        message = state.event.message_obj
        message.message_str = text
        message.message = components
        message.message_id = state.buffer.items[-1].message_id
        state.event._debounce_merged = True
        # 更新 OneBot 原始链，避免读取 raw_message 的下游插件拿到旧正文。
        if (
            isinstance(message.raw_message, dict)
            and message.raw_message.get("post_type") == "message"
        ):
            raw = dict(message.raw_message)
            raw["message_id"] = message.message_id
            raw["message"] = ([{"type": "text", "data": {"text": text}}] if text else []) + [
                {"type": "image", "data": {"file": url}} for url in images
            ]
            raw["raw_message"] = text
            message.raw_message = raw
        logger.debug(f"[消息防抖] 已合并 {len(state.buffer.items)} 条消息、{len(images)} 张图片")

    @_message_event_hook
    async def handle_message(self, event: AstrMessageEvent) -> None:
        """通知只更新状态；普通私聊在早期消息钩子收集，不占用 LLM 会话锁。"""
        if self._closed or not self.config.get("enabled", True) or not self.supports_message_events:
            return
        if getattr(event, "_debounce_collected", False):
            return
        if is_notice(event):
            if not self.supports_enhanced_events:
                return
            if event.is_private_chat() and parse_notice(event) is not None:
                self._handle_notice(event)
                self._silence(event)
            return
        if not event.is_private_chat():
            return
        if not getattr(event, "is_at_or_wake_command", True):
            return
        key = self._key(event)
        text = event.message_str or ""
        # 含语音、文件等未实现的组件整条放行，交给对应插件处理。
        original_text = "".join(
            component.text for component in event.get_messages() if isinstance(component, Plain)
        )
        if (
            self._is_command(text)
            or self._is_command(original_text)
            or any(
                not isinstance(component, (Plain, Image, Reply, At))
                for component in event.get_messages()
            )
        ):
            await self._flush_before_passthrough(key)
            return
        sources = self._sources(event)
        if not text.strip() and not sources:
            return

        now = self._now()
        state = self.sessions.get(key)
        owner = state is None
        if owner:
            state = SessionState(MessageBuffer(now), event)
            self.sessions[key] = state
        item = BufferedMessage(str(event.message_obj.message_id), text, sources, now)
        event._debounce_collected = True
        # 即使查询超时，也保留有引用这一事实。
        reply = next(
            (component for component in event.get_messages() if isinstance(component, Reply)), None
        )
        if reply:
            item.quote = QuotedMessage(f"[引用消息 ID: {reply.id}]", "未知用户")
        state.buffer.items.append(item)
        preparation = asyncio.create_task(self._prepare_item(state, item, event))
        item.prepare_task = preparation
        state.prepare_tasks.add(preparation)
        self._schedule(key, state)
        if not owner:
            self._silence(event)
            # 后续事件存活到本地图片读入内存，避免临时文件先被框架清理。
            try:
                await asyncio.shield(preparation)
            except asyncio.CancelledError:
                # 内部截止/撤回会取消准备任务，正常结束已停止的后续事件。
                # 外部取消通过 shield 与准备任务分离，继续向框架传播。
                if not preparation.cancelled():
                    raise
            return

        try:
            await state.flush_event.wait()
            if self.sessions.get(key) is state:
                self.sessions.pop(key)
            previous = [
                older for older in self._finishing.values() if self._key(older.event) == key
            ]
            self._finishing[id(state)] = state
            self._cancel_timer(state)
            for older in previous:
                await older.done_event.wait()
            await self._finish(state)
        except asyncio.CancelledError:
            self._silence(event)
            raise
        finally:
            if self.sessions.get(key) is state:
                self.sessions.pop(key)
            self._cancel_timer(state)
            for task in state.prepare_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*state.prepare_tasks, return_exceptions=True)
            state.done_event.set()
            self._finishing.pop(id(state), None)

    @_legacy_llm_hook
    async def on_llm_request_legacy(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        """旧版请求阶段兜底阻止被收拢消息；此处立即返回，不重复收集。"""
        if self._closed or not self.config.get("enabled", True) or not self.legacy_mode:
            return
        # 4.26–4.28 的请求钩子在会话锁内执行。防抖等待已在早期消息钩子完成，
        # 这里仅保留后续消息的停止状态；核心添加的提示前缀和图片加工保持原样。
        if getattr(event, "_debounce_suppressed", False):
            self._silence(event)

    async def terminate(self) -> None:
        """卸载时唤醒等待者并回收任务，丢弃本轮内存缓存。"""
        self._closed = True
        states = list(
            {
                id(state): state for state in (*self.sessions.values(), *self._finishing.values())
            }.values()
        )
        self.sessions.clear()
        tasks = []
        for state in states:
            self._silence(state.event)
            if state.timer_task:
                state.timer_task.cancel()
                tasks.append(state.timer_task)
            for task in state.prepare_tasks:
                task.cancel()
                tasks.append(task)
            state.flush_event.set()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if states:
            await asyncio.gather(*(state.done_event.wait() for state in states))
