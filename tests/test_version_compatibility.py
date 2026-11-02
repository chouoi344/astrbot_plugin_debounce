"""AstrBot 4.26—4.29 与未知版本的行为兼容回归。"""

from __future__ import annotations

import asyncio
import copy
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from astrbot_stub import FakeEvent, Image, Plain, ProviderRequest, install

install()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from astrbot_plugin_debounce.main import DebouncePlugin


def enhanced_notice(*, recall: str | None = None, typing: bool | None = None) -> FakeEvent:
    raw = {"post_type": "notice", "user_id": "10", "self_id": "99"}
    if recall is not None:
        raw.update(notice_type="friend_recall", message_id=recall)
    else:
        raw.update(notice_type="notify", sub_type="input_status", is_typing=typing)
    return FakeEvent(raw=raw)


def legacy_event(*, account="99", platform="qq-a", user="10", session="chat"):
    """移除可选便捷 API，保留核心消息对象和旧式平台元信息。"""
    event = FakeEvent("你好", account=account, platform=platform, user=user)
    del event.unified_msg_origin
    event.get_self_id = None
    event.get_sender_id = None
    event.get_platform_id = None
    event.get_platform_name = None
    event.session_id = session
    event.message_obj.session_id = session
    return event


class VersionCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.plugins = []
        self.tasks = []

    async def asyncTearDown(self):
        for plugin in self.plugins:
            await plugin.terminate()
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    def plugin(self, version="4.29.0", config=None):
        settings = {
            "adaptive_min_wait": 0.06,
            "adaptive_max_wait": 0.06,
            "adaptive_max_total_wait": 0.4,
            "max_typing_wait": 0.3,
        }
        settings.update(config or {})
        with patch.object(sys.modules["astrbot"], "__version__", version, create=True):
            plugin = DebouncePlugin(object(), settings)
        self.plugins.append(plugin)
        return plugin

    async def start(self, plugin, event):
        task = asyncio.create_task(plugin.handle_message(event))
        self.tasks.append(task)
        await asyncio.sleep(0)
        return task

    async def test_version_flags_cover_supported_generations(self):
        for version, legacy, enhanced in (
            ("4.26.0", True, False),
            ("4.26.8", True, False),
            ("4.28.0", True, False),
            ("4.29.0", False, True),
            ("4.30.1", False, True),
        ):
            with self.subTest(version=version):
                plugin = self.plugin(version)
                self.assertTrue(plugin.supports_message_events)
                self.assertEqual(plugin.legacy_mode, legacy)
                self.assertEqual(plugin.supports_enhanced_events, enhanced)
                self.assertTrue(plugin.astrbot_version)

    async def test_unknown_versions_default_to_basic_mode(self):
        for version in ("", "unknown", "dev", None):
            with self.subTest(version=version):
                plugin = self.plugin(version)
                self.assertTrue(plugin.supports_message_events)
                self.assertTrue(plugin.legacy_mode)
                self.assertFalse(plugin.supports_enhanced_events)

    async def test_every_supported_generation_merges_text_and_image_once(self):
        for version in ("4.26.0", "4.28.0", "4.29.0", "unknown"):
            with self.subTest(version=version):
                plugin = self.plugin(version)
                first = FakeEvent("先到", message_id="1")
                owner = await self.start(plugin, first)
                follower = FakeEvent(
                    "后到",
                    message_id="2",
                    components=[Plain("后到"), Image(url="https://example.test/picture.png")],
                )
                await plugin.handle_message(follower)
                await asyncio.wait_for(owner, 0.3)
                self.assertEqual(first.message_str, "先到\n后到")
                self.assertFalse(first.stopped)
                self.assertTrue(follower.stopped)
                self.assertEqual(plugin.sessions, {})
                self.assertEqual(
                    [part.file for part in first.get_messages() if isinstance(part, Image)],
                    ["https://example.test/picture.png"],
                )

    async def test_basic_mode_does_not_apply_real_typing_or_recall_notices(self):
        for version in ("4.26.8", "4.28.0", "unknown"):
            with self.subTest(version=version):
                plugin = self.plugin(version)
                first = FakeEvent("应保留", message_id="1")
                owner = await self.start(plugin, first)
                await plugin.handle_message(enhanced_notice(typing=True))
                await plugin.handle_message(enhanced_notice(recall="1"))
                await asyncio.wait_for(owner, 0.2)
                self.assertFalse(first.stopped)
                self.assertEqual(first.message_str, "应保留")

    async def test_modern_mode_applies_recall_notice(self):
        plugin = self.plugin("4.29.0")
        first = FakeEvent("应撤回")
        owner = await self.start(plugin, first)
        await plugin.handle_message(enhanced_notice(recall="1"))
        await asyncio.wait_for(owner, 0.2)
        self.assertTrue(first.stopped)

    async def test_modern_mode_applies_typing_notice(self):
        plugin = self.plugin("4.29.0")
        owner = await self.start(plugin, FakeEvent("等待输入"))
        await plugin.handle_message(enhanced_notice(typing=True))
        await asyncio.sleep(0.1)
        self.assertFalse(owner.done())
        await plugin.handle_message(enhanced_notice(typing=False))
        await asyncio.wait_for(owner, 0.2)

    async def test_legacy_hook_does_not_rebuffer_or_change_merged_request(self):
        plugin = self.plugin("4.26.8")
        first = FakeEvent("先到")
        owner = await self.start(plugin, first)
        follower = FakeEvent("后到", message_id="2")
        await plugin.handle_message(follower)
        await asyncio.wait_for(owner, 0.3)
        request = ProviderRequest(
            prompt="其他插件生成的完整提示词",
            system_prompt="系统原文前缀",
            image_urls=["https://example.test/a.png"],
            contexts=[{"role": "user", "content": "既有上下文"}],
        )
        before = copy.deepcopy(request)
        await plugin.on_llm_request_legacy(first, request)
        self.assertEqual(request, before)
        self.assertEqual(plugin.sessions, {})
        self.assertFalse(first.stopped)

    async def test_legacy_hook_suppresses_only_already_collected_follower(self):
        plugin = self.plugin("4.28.0")
        first = FakeEvent("先到")
        owner = await self.start(plugin, first)
        follower = FakeEvent("后到", message_id="2")
        await plugin.handle_message(follower)
        self.assertTrue(getattr(follower, "_debounce_suppressed", False))
        follower.stopped = False  # 模拟旧管线继续到了 LLM 请求钩子。
        request = ProviderRequest(
            prompt="后到", system_prompt="保留系统", image_urls=["https://example.test/a"]
        )
        before = copy.deepcopy(request)
        await asyncio.wait_for(plugin.on_llm_request_legacy(follower, request), 0.1)
        self.assertTrue(follower.stopped)
        self.assertEqual(request, before)
        self.assertEqual(len(next(iter(plugin.sessions.values())).buffer.items), 2)
        await asyncio.wait_for(owner, 0.3)
        self.assertEqual(first.message_str, "先到\n后到")

    async def test_legacy_hook_leaves_unrelated_requests_untouched(self):
        plugin = self.plugin("4.26.0")
        event = FakeEvent("正常请求")
        request = ProviderRequest(
            prompt="既有正文", system_prompt="既有系统", image_urls=["https://x/a"]
        )
        before = copy.deepcopy(request)
        await plugin.on_llm_request_legacy(event, request)
        self.assertFalse(event.stopped)
        self.assertEqual(request, before)
        self.assertEqual(plugin.sessions, {})

    async def test_reentering_early_hook_does_not_duplicate_collected_follower(self):
        plugin = self.plugin("4.26.8")
        first = FakeEvent("先到")
        owner = await self.start(plugin, first)
        follower = FakeEvent("后到", message_id="2")
        await plugin.handle_message(follower)
        await asyncio.wait_for(plugin.handle_message(follower), 0.1)
        await asyncio.wait_for(owner, 0.3)
        self.assertTrue(follower.stopped)
        self.assertEqual(first.message_str, "先到\n后到")

    async def test_reentering_merged_event_does_not_start_another_batch(self):
        plugin = self.plugin("4.28.0")
        first = FakeEvent("合并后正文")
        owner = await self.start(plugin, first)
        await asyncio.wait_for(owner, 0.3)
        await asyncio.wait_for(plugin.handle_message(first), 0.1)
        self.assertFalse(first.stopped)
        self.assertEqual(first.message_str, "合并后正文")
        self.assertEqual(plugin.sessions, {})

    async def test_silence_handles_missing_should_call_llm_method(self):
        plugin = self.plugin("4.26.0")
        event = FakeEvent("后到")
        event.should_call_llm = None
        plugin._silence(event)
        self.assertTrue(event.stopped)
        self.assertTrue(getattr(event, "_debounce_suppressed", False))

    async def test_legacy_key_fallback_is_stable_and_separates_platforms_accounts_and_chats(self):
        plugin = self.plugin("4.26.8")
        first = legacy_event()
        keys = [
            plugin._key(first),
            plugin._key(legacy_event(platform="qq-b")),
            plugin._key(legacy_event(account="100")),
            plugin._key(legacy_event(session="other")),
        ]
        self.assertEqual(plugin._key(first), plugin._key(legacy_event()))
        self.assertEqual(len(set(keys)), 4)

    async def test_key_uses_platform_meta_when_platform_id_is_missing(self):
        plugin = self.plugin("4.26.8")
        first, second = legacy_event(platform="qq-a"), legacy_event(platform="qq-b")
        del first.platform_id
        del second.platform_id
        self.assertNotEqual(plugin._key(first), plugin._key(second))

    async def test_key_uses_sender_when_session_id_is_missing(self):
        plugin = self.plugin("4.26.8")
        first, second = legacy_event(user="10"), legacy_event(user="11")
        for event in (first, second):
            del event.session_id
            del event.message_obj.session_id
        self.assertNotEqual(plugin._key(first), plugin._key(second))


class MissingApiImportTests(unittest.TestCase):
    def test_missing_message_decorator_degrades_without_import_or_runtime_error(self):
        source_root = Path(__file__).resolve().parents[2]
        test_root = Path(__file__).resolve().parent
        code = r"""
import asyncio
import sys
from astrbot_stub import FakeEvent, install
install()
from astrbot.api.event import filter
del filter.event_message_type
del filter.EventMessageType
from astrbot_plugin_debounce.main import DebouncePlugin
plugin = DebouncePlugin(object(), {})
assert plugin.supports_message_events is False
async def run():
    event = FakeEvent("正常旁路")
    await plugin.handle_message(event)
    assert not event.stopped
    assert not plugin.sessions
    await plugin.terminate()
asyncio.run(run())
"""
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join((str(source_root), str(test_root)))
        result = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=10
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
