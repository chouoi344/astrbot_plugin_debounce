"""解析消息引用，使用组件已有内容并为 OneBot v11 补充缺失内容。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from html import escape
from typing import TYPE_CHECKING, Any

from astrbot.api import logger

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent


@dataclass(frozen=True)
class QuotedMessage:
    """被引用消息的纯文本、发送者及有序图片来源。"""

    content: str
    sender: str
    image_urls: tuple[str, ...] = ()

    def as_xml(self) -> str:
        """转义外部文本，防止原消息中的标签破坏引用边界。"""
        return (
            f'<quoted_message sender="{escape(self.sender, quote=True)}">\n'
            f"{escape(self.content, quote=False)}\n</quoted_message>"
        )


def _field(value: Any, name: str, default: Any = None) -> Any:
    """兼容 AstrBot 消息组件与 OneBot 原始消息段。"""
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _kind(segment: Any) -> str:
    """归一化 AstrBot ComponentType 和 OneBot 小写类型名。"""
    kind = _field(segment, "type", type(segment).__name__)
    return str(getattr(kind, "value", kind)).lower()


def _content(chain: Any) -> tuple[str, tuple[str, ...]]:
    """读取一层引用内容；嵌套引用仅加标记，避免递归查询。"""
    if isinstance(chain, str):
        return chain.strip(), ()
    if not isinstance(chain, (list, tuple)):
        return "", ()

    parts: list[str] = []
    images: list[str] = []
    placeholders = {
        "record": "[语音]",
        "video": "[视频]",
        "file": "[文件]",
        "face": "[表情]",
        "reply": "[引用消息]",
        "forward": "[合并转发]",
        "json": "[卡片]",
    }
    for segment in chain:
        kind = _kind(segment)
        data = _field(segment, "data", segment)
        if kind in ("plain", "text"):
            text = _field(data, "text", "")
            if text:
                parts.append(str(text))
        elif kind == "image":
            parts.append("[图片]")
            source = _field(data, "url") or _field(data, "file")
            if source and str(source) not in images:
                images.append(str(source))
        elif kind == "at":
            target = _field(data, "name") or _field(data, "qq", "")
            parts.append(f"@{target}")
        elif kind in placeholders:
            parts.append(placeholders[kind])
    return "".join(parts).strip(), tuple(images)


def _sender_name(event: AstrMessageEvent, sender_id: Any, nickname: Any) -> str:
    """将引用 Bot 自己的消息标注为 Bot，并兼容数值与字符串 ID。"""
    get_self_id = getattr(event, "get_self_id", None)
    bot_id = get_self_id() if callable(get_self_id) else None
    if bot_id is None:
        bot_id = getattr(getattr(event, "message_obj", None), "self_id", None)
    if sender_id not in (None, "", 0, "0") and bot_id is not None and str(sender_id) == str(bot_id):
        return "Bot"
    return str(nickname or sender_id or "未知用户")


async def parse_quote(event: AstrMessageEvent) -> QuotedMessage | None:
    """解析首个 Reply；内容缺失时最多等待 3 秒调用 OneBot get_msg。

    其他平台通过 Reply.chain/message_str 使用引用功能。平台缺少 get_msg
    或查询失败时保留引用 ID 标记，当前用户消息仍然继续参与防抖。
    """
    components = getattr(getattr(event, "message_obj", None), "message", ()) or ()
    reply = next((item for item in components if _kind(item) == "reply"), None)
    if reply is None:
        return None

    reply_data = _field(reply, "data", reply)
    reply_id = _field(reply_data, "id", "未知")
    sender_id = _field(reply_data, "sender_id")
    nickname = _field(reply_data, "sender_nickname")
    content, images = _content(_field(reply_data, "chain"))
    if not content:
        content = str(_field(reply_data, "message_str") or _field(reply_data, "text") or "").strip()

    if not content:
        api = getattr(getattr(event, "bot", None), "api", None)
        call_action = getattr(api, "call_action", None)
        if callable(call_action):
            try:
                original = await asyncio.wait_for(
                    call_action("get_msg", message_id=reply_id), timeout=3.0
                )
                if isinstance(original, dict):
                    if isinstance(original.get("data"), dict):
                        original = original["data"]
                    raw_sender = original.get("sender") or {}
                    if isinstance(raw_sender, dict):
                        sender_id = raw_sender.get("user_id") or sender_id
                        nickname = raw_sender.get("card") or raw_sender.get("nickname") or nickname
                    content, images = _content(original.get("message"))
                    if not content:
                        content = str(original.get("raw_message") or "").strip()
            except Exception:
                logger.debug("[消息防抖] 引用内容查询失败，保留引用 ID", exc_info=True)

    if not content:
        content = f"[引用消息 ID: {reply_id}]"
    return QuotedMessage(content, _sender_name(event, sender_id, nickname), images)
