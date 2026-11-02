"""独立行为测试；运行 python -m unittest discover -s tests -v。"""

from __future__ import annotations

import asyncio
import base64
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from astrbot_stub import At, FakeEvent, Image, Plain, Reply, install

install()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from astrbot_plugin_debounce.adaptive_timer import AdaptiveSettings, adaptive_wait
from astrbot_plugin_debounce.main import DebouncePlugin
from astrbot_plugin_debounce.message_buffer import BufferedMessage, MessageBuffer
from astrbot_plugin_debounce.quote_parser import QuotedMessage, parse_quote
from astrbot_plugin_debounce.typing_detector import parse_notice


def image_sources(event: FakeEvent) -> list[str]:
    return [item.url or item.file for item in event.get_messages() if isinstance(item, Image)]


def notice(
    *,
    user: str = "10",
    account: str = "99",
    platform: str = "qq-a",
    typing: str | None = None,
    recall: str | int | None = None,
    private: bool = True,
    **extra,
) -> FakeEvent:
    raw = {"post_type": "notice", "user_id": user, "self_id": account}
    if recall is not None:
        raw.update(notice_type="friend_recall", message_id=recall)
    else:
        raw.update(notice_type="notify", sub_type="input_status", status_text=typing or "")
    raw.update(extra)
    return FakeEvent(user=user, account=account, platform=platform, private=private, raw=raw)


class AdaptiveTests(unittest.TestCase):
    def wait(self, texts, *, now=100.0, settings=None):
        buffer = MessageBuffer(100.0)
        buffer.items = [
            BufferedMessage(str(index), text, (), 100.0) for index, text in enumerate(texts)
        ]
        return adaptive_wait(buffer, settings or AdaptiveSettings(), now)

    def test_length_baselines_and_boundaries(self):
        for length, expected in ((1, 3.0), (10, 3.0), (11, 2.0), (50, 2.0), (51, 1.5)):
            with self.subTest(length=length):
                self.assertEqual(self.wait(["字" * length]), expected)

    def test_sentence_endings_and_pause_punctuation(self):
        for punctuation in "。！？.!?":
            with self.subTest(punctuation=punctuation):
                self.assertAlmostEqual(self.wait(["字" * 20 + punctuation]), 1.4)
        for punctuation in "，,、":
            with self.subTest(punctuation=punctuation):
                self.assertAlmostEqual(self.wait(["字" * 20 + punctuation]), 2.4)

    def test_second_short_message_extends_wait(self):
        self.assertEqual(self.wait(["如果", "明天"]), 4.5)
        self.assertEqual(self.wait(["如果", "明天", "天气好"]), 4.5)

    def test_image_only_messages_do_not_count_as_short_sentences(self):
        self.assertEqual(self.wait(["", "", "你好"]), 3.0)

    def test_single_wait_clamped(self):
        self.assertEqual(self.wait(["如果", "明天"], settings=AdaptiveSettings(max_wait=4)), 4)
        self.assertEqual(self.wait(["字" * 51 + "。"], settings=AdaptiveSettings(min_wait=2)), 2)

    def test_total_deadline_overrides_minimum(self):
        self.assertAlmostEqual(self.wait(["你好"], now=111.9), 0.1)
        self.assertEqual(self.wait(["你好"], now=112), 0)
        self.assertEqual(self.wait(["你好"], now=113), 0)

    def test_custom_short_threshold(self):
        self.assertEqual(
            self.wait(["字" * 11], settings=AdaptiveSettings(short_message_threshold=12)), 3
        )

    def test_invalid_config_uses_safe_finite_values(self):
        config = AdaptiveSettings.from_config(
            {
                "adaptive_min_wait": "nan",
                "adaptive_max_wait": "bad",
                "adaptive_max_total_wait": None,
                "adaptive_short_message_threshold": "inf",
                "max_typing_wait": -4,
            }
        )
        self.assertEqual(config.min_wait, 1)
        self.assertEqual(config.max_wait, 6)
        self.assertEqual(config.max_total_wait, 12)
        self.assertEqual(config.short_message_threshold, 10)
        self.assertGreater(config.max_typing_wait, 0)

    def test_reversed_wait_range_is_normalized(self):
        settings = AdaptiveSettings.from_config({"adaptive_min_wait": 5, "adaptive_max_wait": 2})
        self.assertGreaterEqual(settings.max_wait, settings.min_wait)


class BufferTests(unittest.TestCase):
    def test_quotes_stay_with_each_message_in_order(self):
        buffer = MessageBuffer(
            0,
            [
                BufferedMessage("1", "甲", (), 0, QuotedMessage("引用甲", "Bot")),
                BufferedMessage("2", "乙", (), 1),
                BufferedMessage("3", "丙", (), 2, QuotedMessage("引用丙", "用户")),
            ],
        )
        self.assertEqual(
            buffer.merged_text("\n"),
            '<quoted_message sender="Bot">\n引用甲\n</quoted_message>\n\n甲\n乙\n'
            '<quoted_message sender="用户">\n引用丙\n</quoted_message>\n\n丙',
        )

    def test_recall_removes_text_quote_and_unique_image_together(self):
        buffer = MessageBuffer(
            0,
            [
                BufferedMessage("1", "甲", ("a", "b"), 0),
                BufferedMessage("2", "乙", ("b", "c"), 1, QuotedMessage("引用", "Bot", ("q",))),
            ],
        )
        self.assertEqual(buffer.image_urls, ["a", "b", "q", "c"])
        self.assertTrue(buffer.remove_by_id(2))
        self.assertEqual(buffer.merged_text("\n"), "甲")
        self.assertEqual(buffer.image_urls, ["a", "b"])
        self.assertFalse(buffer.remove_by_id("missing"))
        self.assertEqual(buffer.first_message_time, 0)


class NoticeTests(unittest.TestCase):
    def test_stop_typing_takes_precedence_over_typing_substring(self):
        for text in ("stop typing", "not typing", "停止输入"):
            with self.subTest(text=text):
                self.assertFalse(parse_notice(notice(typing=text)).typing)

    def test_typing_and_explicit_boolean_state(self):
        self.assertTrue(parse_notice(notice(typing="正在输入")).typing)
        self.assertTrue(parse_notice(notice(is_typing=True)).typing)
        self.assertFalse(parse_notice(notice(is_typing=False)).typing)

    def test_undocumented_numeric_state_is_ignored(self):
        for event_type in (0, 1, 2, 11):
            with self.subTest(event_type=event_type):
                self.assertIsNone(parse_notice(notice(event_type=event_type)))

    def test_group_notice_and_unrelated_notice_are_ignored(self):
        self.assertIsNone(parse_notice(notice(typing="正在输入", group_id=123)))
        event = FakeEvent(raw={"post_type": "notice", "notice_type": "friend_add", "user_id": "10"})
        self.assertIsNone(parse_notice(event))


class QuoteTests(unittest.IsolatedAsyncioTestCase):
    async def test_embedded_chain_bot_sender_images_and_xml(self):
        event = FakeEvent(
            "那明天呢",
            components=[
                Reply(
                    "original",
                    sender_id=99,
                    sender_nickname="助手",
                    chain=[
                        Plain("天气 <晴> & 好"),
                        Image(url="https://x/a"),
                        Image(url="https://x/a"),
                    ],
                ),
                Plain("那明天呢"),
            ],
        )
        quote = await parse_quote(event)
        self.assertEqual(quote.sender, "Bot")
        self.assertEqual(quote.image_urls, ("https://x/a",))
        self.assertIn("天气 &lt;晴&gt; &amp; 好", quote.as_xml())
        self.assertIn("[图片][图片]", quote.content)

    async def test_onebot_lookup_data_wrapper(self):
        calls = []

        async def lookup(action, **kwargs):
            calls.append((action, kwargs))
            return {
                "data": {
                    "sender": {"user_id": "20", "nickname": 'A"B'},
                    "message": [{"type": "text", "data": {"text": "旧消息"}}],
                }
            }

        event = FakeEvent(
            "新", components=[Reply(7), Plain("新")], api=types.SimpleNamespace(call_action=lookup)
        )
        quote = await parse_quote(event)
        self.assertEqual(calls, [("get_msg", {"message_id": 7})])
        self.assertEqual(quote.content, "旧消息")
        self.assertIn('sender="A&quot;B"', quote.as_xml())

    async def test_failed_lookup_keeps_reference_id(self):
        async def lookup(*args, **kwargs):
            raise RuntimeError("offline")

        event = FakeEvent(
            "新",
            components=[Reply("7"), Plain("新")],
            api=types.SimpleNamespace(call_action=lookup),
        )
        self.assertEqual((await parse_quote(event)).content, "[引用消息 ID: 7]")

    async def test_no_reply_returns_none(self):
        self.assertIsNone(await parse_quote(FakeEvent("你好")))

    async def test_nested_reply_does_not_fetch_recursively(self):
        async def lookup(*args, **kwargs):
            self.fail("已有引用链时不应发起额外查询")

        event = FakeEvent(
            "新",
            components=[Reply("7", chain=[Reply("6"), Plain("引用正文")]), Plain("新")],
            api=types.SimpleNamespace(call_action=lookup),
        )
        self.assertEqual((await parse_quote(event)).content, "[引用消息]引用正文")


class PluginBehaviorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.config = {
            "adaptive_min_wait": 0.08,
            "adaptive_max_wait": 0.08,
            "adaptive_max_total_wait": 0.4,
            "max_typing_wait": 0.25,
        }
        self.plugin = DebouncePlugin(object(), self.config)
        self.tasks = []
        self.delivered = []

    async def asyncTearDown(self):
        await self.plugin.terminate()
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def start(self, event):
        async def pipeline():
            await self.plugin.handle_message(event)
            if not event.stopped:
                self.delivered.append(event)

        task = asyncio.create_task(pipeline())
        self.tasks.append(task)
        await asyncio.sleep(0)
        return task

    async def finish(self, task):
        await asyncio.wait_for(task, timeout=0.8)

    async def wait_until(self, predicate, timeout=0.4):
        async with asyncio.timeout(timeout):
            while not predicate():
                await asyncio.sleep(0.005)

    async def test_three_texts_release_only_first_event_once(self):
        first, second, third = [
            FakeEvent(text, message_id=str(i)) for i, text in enumerate(("今天", "天气", "怎么样"))
        ]
        owner = await self.start(first)
        await self.plugin.handle_message(second)
        await self.plugin.handle_message(third)
        await self.finish(owner)
        self.assertEqual(self.delivered, [first])
        self.assertEqual(first.message_str, "今天\n天气\n怎么样")
        self.assertEqual(first.message_obj.message_str, first.message_str)
        self.assertEqual(first.message_obj.message_id, "2")
        self.assertTrue(second.stopped)
        self.assertTrue(third.stopped)
        self.assertEqual(self.plugin.sessions, {})

    async def test_multiple_users_platforms_and_accounts_are_isolated(self):
        events = [
            FakeEvent("甲", user="10"),
            FakeEvent("乙", user="11"),
            FakeEvent("丙", user="10", platform="qq-b"),
            FakeEvent("丁", user="10", account="100"),
        ]
        tasks = [await self.start(event) for event in events]
        await asyncio.gather(*(self.finish(task) for task in tasks))
        self.assertEqual(len(self.delivered), 4)
        self.assertEqual([event.message_str for event in events], ["甲", "乙", "丙", "丁"])

    async def test_image_only_batch_keeps_order_and_deduplicates(self):
        first = FakeEvent(components=[Image(url="https://x/a"), Image(url="https://x/b")])
        owner = await self.start(first)
        second = FakeEvent(
            message_id="2", components=[Image(url="https://x/b"), Image(url="https://x/c")]
        )
        await self.plugin.handle_message(second)
        await self.finish(owner)
        self.assertEqual(image_sources(first), ["https://x/a", "https://x/b", "https://x/c"])
        self.assertFalse(first.stopped)

    async def test_second_event_local_image_survives_original_file_deletion(self):
        first = FakeEvent("看这张")
        owner = await self.start(first)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "temporary.png"
            content = b"fake-image-payload"
            path.write_bytes(content)
            second = FakeEvent(message_id="2", components=[Image(file=str(path))])
            await self.plugin.handle_message(second)
            path.unlink()
            await self.finish(owner)
        self.assertEqual(
            image_sources(first), ["base64://" + base64.b64encode(content).decode("ascii")]
        )

    async def test_quotes_on_first_middle_and_last_message_stay_attached(self):
        first = FakeEvent(
            "甲", components=[Reply("q1", chain=[Plain("引用甲")], sender_id="99"), Plain("甲")]
        )
        owner = await self.start(first)
        for index, text in ((2, "乙"), (3, "丙")):
            await self.plugin.handle_message(
                FakeEvent(
                    text,
                    message_id=str(index),
                    components=[
                        Reply(
                            f"q{index}",
                            chain=[Plain(f"引用{text}")],
                            sender_id="20",
                            sender_nickname="朋友",
                        ),
                        Plain(text),
                    ],
                )
            )
        await self.finish(owner)
        output = first.message_str
        self.assertTrue(output.startswith('<quoted_message sender="Bot">'))
        self.assertLess(output.index("引用甲"), output.index("引用乙"))
        self.assertLess(output.index("引用乙"), output.index("引用丙"))
        self.assertEqual(output.count("<quoted_message"), 3)

    async def test_slow_reference_lookup_does_not_reorder_messages(self):
        async def lookup(*args, **kwargs):
            await asyncio.sleep(0.04)
            return {
                "sender": {"nickname": "朋友"},
                "message": [{"type": "text", "data": {"text": "引用"}}],
            }

        first = FakeEvent(
            "先到",
            components=[Reply("q"), Plain("先到")],
            api=types.SimpleNamespace(call_action=lookup),
        )
        owner = await self.start(first)
        await self.plugin.handle_message(FakeEvent("后到", message_id="2"))
        await self.finish(owner)
        self.assertIn("</quoted_message>\n\n先到\n后到", first.message_str)

    async def test_recall_drops_text_image_and_quote(self):
        first = FakeEvent("保留", message_id="1")
        owner = await self.start(first)
        await self.plugin.handle_message(
            FakeEvent(
                "删除",
                message_id="2",
                components=[
                    Reply("q", chain=[Plain("删除的引用"), Image(url="https://x/q")]),
                    Plain("删除"),
                    Image(url="https://x/remove"),
                ],
            )
        )
        await self.plugin.handle_message(notice(recall=2))
        await self.finish(owner)
        self.assertEqual(first.message_str, "保留")
        self.assertEqual(image_sources(first), [])

    async def test_recall_of_first_message_releases_remaining_text(self):
        first = FakeEvent("删除", message_id="1")
        owner = await self.start(first)
        await self.plugin.handle_message(FakeEvent("保留", message_id="2"))
        await self.plugin.handle_message(notice(recall="1"))
        await self.finish(owner)
        self.assertEqual(first.message_str, "保留")
        self.assertEqual(first.message_obj.message_id, "2")

    async def test_all_recalled_messages_suppress_entire_batch(self):
        first = FakeEvent("删除", message_id="1")
        owner = await self.start(first)
        await self.plugin.handle_message(notice(recall="1"))
        await self.finish(owner)
        self.assertTrue(first.stopped)
        self.assertEqual(self.delivered, [])
        self.assertEqual(self.plugin.sessions, {})

    async def test_recall_wrong_account_does_not_remove_content(self):
        first = FakeEvent("保留")
        owner = await self.start(first)
        await self.plugin.handle_message(notice(recall="1", account="100"))
        await self.finish(owner)
        self.assertEqual(first.message_str, "保留")

    async def test_command_flushes_existing_batch_before_passing(self):
        first = FakeEvent("先处理")
        owner = await self.start(first)
        command = FakeEvent("/help", message_id="2")
        await self.plugin.handle_message(command)
        self.assertTrue(owner.done())
        self.assertEqual(first.message_str, "先处理")
        self.assertFalse(command.stopped)
        self.assertEqual(self.plugin.sessions, {})

    async def test_group_empty_command_and_unsupported_media_bypass(self):
        events = [
            FakeEvent("群消息", private=False),
            FakeEvent(),
            FakeEvent(" #help"),
            FakeEvent("语音消息", components=[types.SimpleNamespace(type="Record")]),
        ]
        for event in events:
            await self.plugin.handle_message(event)
            self.assertFalse(event.stopped)
        self.assertEqual(self.plugin.sessions, {})

    async def test_disabled_plugin_passes_through(self):
        self.config["enabled"] = False
        event = FakeEvent("你好")
        await self.plugin.handle_message(event)
        self.assertFalse(event.stopped)
        self.assertEqual(self.plugin.sessions, {})

    async def test_typing_pauses_normal_timeout_and_stop_resumes(self):
        first = FakeEvent("等一下")
        owner = await self.start(first)
        await self.plugin.handle_message(notice(typing="正在输入"))
        await asyncio.sleep(0.11)
        self.assertFalse(owner.done())
        await self.plugin.handle_message(notice(typing="stop typing"))
        await self.finish(owner)
        self.assertEqual(self.delivered, [first])

    async def test_repeated_typing_notifications_do_not_extend_typing_protection(self):
        self.plugin.settings = AdaptiveSettings(
            min_wait=0.05, max_wait=0.05, max_total_wait=0.9, max_typing_wait=0.3
        )
        first = FakeEvent("等一下")
        owner = await self.start(first)
        await self.plugin.handle_message(notice(typing="正在输入"))
        await asyncio.sleep(0.24)
        await self.plugin.handle_message(notice(typing="正在输入"))
        await asyncio.wait_for(owner, 0.2)

    async def test_typing_respects_total_hard_deadline(self):
        self.plugin.settings = AdaptiveSettings(
            min_wait=0.05, max_wait=0.05, max_total_wait=0.13, max_typing_wait=0.5
        )
        owner = await self.start(FakeEvent("等一下"))
        await self.plugin.handle_message(notice(typing="正在输入"))
        await asyncio.wait_for(owner, 0.3)

    async def test_continuous_messages_respect_total_hard_deadline(self):
        self.plugin.settings = AdaptiveSettings(min_wait=0.08, max_wait=0.08, max_total_wait=0.15)
        first = FakeEvent("一")
        owner = await self.start(first)
        for index in range(2, 5):
            await asyncio.sleep(0.035)
            await self.plugin.handle_message(FakeEvent(str(index), message_id=str(index)))
        await self.finish(owner)
        self.assertEqual(first.message_str, "一\n2\n3\n4")

    async def test_following_cycle_is_not_flushed_by_previous_timer(self):
        first_task = await self.start(FakeEvent("第一轮"))
        await asyncio.sleep(0.03)
        await self.plugin.handle_message(FakeEvent("/help"))
        await self.finish(first_task)
        second_task = await self.start(FakeEvent("第二轮"))
        await asyncio.sleep(0.055)
        self.assertFalse(second_task.done())
        await self.finish(second_task)

    async def test_terminate_suppresses_waiting_batch_and_reaps_tasks(self):
        first = FakeEvent("等待中")
        owner = await self.start(first)
        await self.plugin.terminate()
        await self.finish(owner)
        self.assertTrue(first.stopped)
        self.assertEqual(self.plugin.sessions, {})
        self.assertEqual(self.delivered, [])

    async def test_custom_separator_and_raw_onebot_message_are_updated(self):
        self.config["merge_separator"] = " | "
        first = FakeEvent("甲")
        owner = await self.start(first)
        await self.plugin.handle_message(FakeEvent("乙", message_id="2"))
        await self.finish(owner)
        self.assertEqual(first.message_str, "甲 | 乙")
        self.assertEqual(
            first.message_obj.raw_message["message"],
            [{"type": "text", "data": {"text": "甲 | 乙"}}],
        )
        self.assertEqual(first.message_obj.raw_message["message_id"], "2")

    async def test_command_prefix_in_original_plain_component_is_preserved(self):
        first = FakeEvent("先处理")
        owner = await self.start(first)
        command = FakeEvent("help", components=[Plain("/help")])
        await self.plugin.handle_message(command)
        self.assertTrue(owner.done())
        self.assertFalse(command.stopped)
        self.assertEqual(command.message_str, "help")

    async def test_group_and_unknown_notices_pass_through(self):
        events = [
            notice(typing="正在输入", private=False, group_id="123"),
            FakeEvent(raw={"post_type": "notice", "notice_type": "friend_add", "user_id": "10"}),
        ]
        for event in events:
            await self.plugin.handle_message(event)
            self.assertFalse(event.stopped)

    async def test_recall_switch_can_be_disabled(self):
        self.config["enable_recall_filter"] = False
        first = FakeEvent("保留")
        owner = await self.start(first)
        await self.plugin.handle_message(notice(recall="1"))
        await self.finish(owner)
        self.assertFalse(first.stopped)
        self.assertEqual(first.message_str, "保留")

    async def test_typing_switch_can_be_disabled(self):
        self.config["enable_typing_detection"] = False
        owner = await self.start(FakeEvent("你好"))
        await self.plugin.handle_message(notice(typing="正在输入"))
        await asyncio.wait_for(owner, 0.2)

    async def test_new_message_while_typing_still_waits_for_typing_stop(self):
        first = FakeEvent("甲")
        owner = await self.start(first)
        await self.plugin.handle_message(notice(typing="正在输入"))
        await asyncio.sleep(0.04)
        await self.plugin.handle_message(FakeEvent("乙", message_id="2"))
        await asyncio.sleep(0.11)
        self.assertFalse(owner.done())
        await self.plugin.handle_message(notice(typing="停止输入"))
        await self.finish(owner)
        self.assertEqual(first.message_str, "甲\n乙")

    async def test_duplicate_stop_notification_does_not_restart_timer(self):
        self.plugin.settings = AdaptiveSettings(
            min_wait=0.3, max_wait=0.3, max_total_wait=1.5, max_typing_wait=0.5
        )
        owner = await self.start(FakeEvent("你好"))
        await self.plugin.handle_message(notice(typing="正在输入"))
        await self.plugin.handle_message(notice(typing="停止输入"))
        await asyncio.sleep(0.2)
        await self.plugin.handle_message(notice(typing="停止输入"))
        await asyncio.wait_for(owner, 0.2)

    async def test_slow_quote_is_cancelled_at_total_deadline_with_fallback(self):
        self.plugin.settings = AdaptiveSettings(min_wait=0.05, max_wait=0.05, max_total_wait=0.13)
        lookup_started = asyncio.Event()
        lookup_cancelled = asyncio.Event()

        async def lookup(*args, **kwargs):
            lookup_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                lookup_cancelled.set()

        first = FakeEvent(
            "新消息",
            components=[Reply("q"), Plain("新消息")],
            api=types.SimpleNamespace(call_action=lookup),
        )
        owner = await self.start(first)
        await lookup_started.wait()
        await asyncio.wait_for(owner, 0.3)
        self.assertTrue(lookup_cancelled.is_set())
        self.assertIn("[引用消息 ID: q]", first.message_str)
        self.assertTrue(first.message_str.endswith("新消息"))

    async def test_full_recall_during_slow_quote_finishes_without_waiting(self):
        lookup_started = asyncio.Event()

        async def lookup(*args, **kwargs):
            lookup_started.set()
            await asyncio.Event().wait()

        first = FakeEvent(
            "删除",
            components=[Reply("q"), Plain("删除")],
            api=types.SimpleNamespace(call_action=lookup),
        )
        owner = await self.start(first)
        await lookup_started.wait()
        await self.plugin.handle_message(notice(recall="1"))
        await asyncio.wait_for(owner, 0.2)
        self.assertTrue(first.stopped)

    async def test_terminate_during_quote_preparation_reaps_detached_batch(self):
        lookup_started = asyncio.Event()

        async def lookup(*args, **kwargs):
            lookup_started.set()
            await asyncio.Event().wait()

        first = FakeEvent(
            "等待",
            components=[Reply("q"), Plain("等待")],
            api=types.SimpleNamespace(call_action=lookup),
        )
        owner = await self.start(first)
        await lookup_started.wait()
        await asyncio.sleep(0.1)
        await asyncio.wait_for(self.plugin.terminate(), 0.3)
        await self.finish(owner)
        self.assertTrue(first.stopped)
        self.assertEqual(self.delivered, [])

    async def test_command_waits_for_quote_preparation_after_quiet_timer(self):
        lookup_started = asyncio.Event()
        lookup_release = asyncio.Event()

        async def lookup(*args, **kwargs):
            lookup_started.set()
            await lookup_release.wait()
            return {"message": [{"type": "text", "data": {"text": "引用"}}]}

        first = FakeEvent(
            "先处理",
            components=[Reply("q"), Plain("先处理")],
            api=types.SimpleNamespace(call_action=lookup),
        )
        owner = await self.start(first)
        await lookup_started.wait()
        await asyncio.sleep(0.1)
        command = FakeEvent("/help")
        command_task = asyncio.create_task(self.plugin.handle_message(command))
        self.tasks.append(command_task)
        await asyncio.sleep(0.02)
        self.assertFalse(command_task.done(), "已有批次准备引用时，指令应等待该批次重构")
        lookup_release.set()
        await self.finish(owner)
        await self.finish(command_task)
        self.assertFalse(command.stopped)

    async def test_cancelled_owner_cleans_session_and_background_tasks(self):
        first = FakeEvent("等待")
        owner = await self.start(first)
        owner.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await owner
        self.assertTrue(first.stopped)
        self.assertEqual(self.plugin.sessions, {})

    async def test_later_batch_waits_for_earlier_slow_quote_batch(self):
        self.plugin.settings = AdaptiveSettings(min_wait=0.05, max_wait=0.05, max_total_wait=0.4)
        lookup_started = asyncio.Event()
        lookup_release = asyncio.Event()

        async def lookup(*args, **kwargs):
            lookup_started.set()
            await lookup_release.wait()
            return {"message": [{"type": "text", "data": {"text": "引用"}}]}

        first = FakeEvent(
            "第一轮",
            components=[Reply("q"), Plain("第一轮")],
            api=types.SimpleNamespace(call_action=lookup),
        )
        first_task = await self.start(first)
        await lookup_started.wait()
        await asyncio.sleep(0.07)
        second = FakeEvent("第二轮", message_id="2")
        second_task = await self.start(second)
        await asyncio.sleep(0.08)
        self.assertEqual(self.delivered, [], "后到批次应等待前一批次完成重构")
        self.assertFalse(second_task.done())
        lookup_release.set()
        await self.finish(first_task)
        await self.finish(second_task)
        self.assertEqual(self.delivered, [first, second])

    async def test_unawakened_private_message_passes_without_buffering(self):
        event = FakeEvent("未唤醒的普通消息", awake=False)
        await self.plugin.handle_message(event)
        self.assertFalse(event.stopped)
        self.assertEqual(self.plugin.sessions, {})

    async def test_unawakened_message_does_not_absorb_following_awakened_message(self):
        ignored = FakeEvent("未唤醒", awake=False)
        await self.plugin.handle_message(ignored)
        first = FakeEvent("唤醒后", message_id="2")
        owner = await self.start(first)
        await self.finish(owner)
        self.assertEqual(first.message_str, "唤醒后")
        self.assertFalse(ignored.stopped)

    async def test_qq_reply_with_at_component_participates_in_merge(self):
        first = FakeEvent(
            "那明天呢",
            components=[
                Reply("q", chain=[Plain("今天晴天")], sender_id="99"),
                At("99"),
                Plain("那明天呢"),
            ],
        )
        owner = await self.start(first)
        await self.plugin.handle_message(FakeEvent("还会晴吗", message_id="2"))
        await self.finish(owner)
        self.assertTrue(first.message_str.startswith('<quoted_message sender="Bot">'))
        self.assertIn("今天晴天", first.message_str)
        self.assertTrue(first.message_str.endswith("那明天呢\n还会晴吗"))
        self.assertEqual(self.delivered, [first])

    async def test_recall_removes_message_from_batch_already_preparing_to_release(self):
        lookup_release = asyncio.Event()

        async def lookup(*args, **kwargs):
            await lookup_release.wait()
            return {"message": [{"type": "text", "data": {"text": "已撤回的引用"}}]}

        first = FakeEvent(
            "应删除",
            message_id="1",
            components=[Reply("q"), Plain("应删除"), Image(url="https://x/remove")],
            api=types.SimpleNamespace(call_action=lookup),
        )
        owner = await self.start(first)
        await self.plugin.handle_message(FakeEvent("应保留", message_id="2"))
        await self.wait_until(lambda: bool(self.plugin._finishing))
        await self.plugin.handle_message(notice(recall="1"))
        lookup_release.set()
        await self.finish(owner)
        self.assertEqual(first.message_str, "应保留")
        self.assertEqual(image_sources(first), [])
        self.assertEqual(first.message_obj.message_id, "2")

    async def test_full_recall_of_finishing_batch_cancels_slow_query_immediately(self):
        self.plugin.settings = AdaptiveSettings(min_wait=0.05, max_wait=0.05, max_total_wait=0.8)
        lookup_cancelled = asyncio.Event()

        async def lookup(*args, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                lookup_cancelled.set()

        first = FakeEvent(
            "应删除",
            components=[Reply("q"), Plain("应删除")],
            api=types.SimpleNamespace(call_action=lookup),
        )
        owner = await self.start(first)
        await self.wait_until(lambda: bool(self.plugin._finishing))
        await self.plugin.handle_message(notice(recall="1"))
        await asyncio.wait_for(owner, 0.2)
        self.assertTrue(lookup_cancelled.is_set())
        self.assertTrue(first.stopped)
        self.assertEqual(self.delivered, [])

    async def test_recall_finds_older_finishing_batch_while_new_batch_exists(self):
        self.plugin.settings = AdaptiveSettings(min_wait=0.05, max_wait=0.05, max_total_wait=0.8)

        async def lookup(*args, **kwargs):
            await asyncio.Event().wait()

        first = FakeEvent(
            "应删除的旧轮",
            components=[Reply("q"), Plain("应删除的旧轮")],
            api=types.SimpleNamespace(call_action=lookup),
        )
        old_owner = await self.start(first)
        await self.wait_until(lambda: bool(self.plugin._finishing))
        second = FakeEvent("保留的新轮", message_id="2")
        new_owner = await self.start(second)
        self.assertTrue(self.plugin.sessions)
        await self.plugin.handle_message(notice(recall="1"))
        await asyncio.wait_for(old_owner, 0.2)
        await self.finish(new_owner)
        self.assertTrue(first.stopped)
        self.assertFalse(second.stopped)
        self.assertEqual(second.message_str, "保留的新轮")
        self.assertEqual(self.delivered, [second])

    async def test_slow_local_image_deadline_drops_temporary_path_without_cancelling_handler(self):
        self.plugin.settings = AdaptiveSettings(min_wait=0.05, max_wait=0.05, max_total_wait=0.13)
        original_preserve = self.plugin._preserve_sources
        image_read_started = asyncio.Event()
        temporary_path = "C:\\temporary\\deadline-image.png"

        async def stalled_image_read(sources):
            if temporary_path in sources:
                image_read_started.set()
                await asyncio.Event().wait()
            return await original_preserve(sources)

        self.plugin._preserve_sources = stalled_image_read
        from astrbot_plugin_debounce.main import logger

        with patch.object(logger, "warning") as warning, patch.object(logger, "error") as error:
            first = FakeEvent("请看图")
            owner = await self.start(first)
            second = FakeEvent(message_id="2", components=[Image(file=temporary_path)])
            follower = await self.start(second)
            await image_read_started.wait()
            results = await asyncio.wait_for(
                asyncio.gather(owner, follower, return_exceptions=True), 0.4
            )
        self.assertEqual(results, [None, None], "内部读取超时应由插件处理，后续事件应正常返回")
        self.assertEqual(image_sources(first), [], "已失效的临时文件路径应从最终消息移除")
        self.assertIn("请看图", first.message_str)
        self.assertIn("图片", first.message_str, "图片读取失败需提供明确占位")
        self.assertTrue(warning.called or error.called, "读取超时需要留日志以便定位")
        self.assertTrue(second.stopped)

    async def test_unreadable_local_image_is_replaced_with_clear_placeholder(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_path = str(Path(directory) / "missing.png")
            first = FakeEvent("请看图", components=[Plain("请看图"), Image(file=missing_path)])
            owner = await self.start(first)
            await self.finish(owner)
        self.assertEqual(image_sources(first), [])
        self.assertIn("请看图", first.message_str)
        self.assertIn("图片", first.message_str)


if __name__ == "__main__":
    unittest.main()
