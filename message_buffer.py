"""按消息保存图文和引用，撤回时一并删除关联内容。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .quote_parser import QuotedMessage

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent


@dataclass
class BufferedMessage:
    """单条消息的快照；图片与引用属于当前消息。"""

    message_id: str
    text: str
    image_urls: tuple[str, ...]
    received_at: float
    quote: QuotedMessage | None = None
    prepare_task: asyncio.Task[None] | None = field(default=None, repr=False, compare=False)


@dataclass
class MessageBuffer:
    """会话内有序消息，首条进入时间在本轮内保持固定。"""

    first_message_time: float
    items: list[BufferedMessage] = field(default_factory=list)

    @property
    def messages(self) -> list[str]:
        """返回用户正文，用于自适应长度计算。"""
        return [item.text for item in self.items if item.text]

    @property
    def last_update_time(self) -> float:
        """返回最后一条保留消息的到达时间。"""
        return self.items[-1].received_at if self.items else self.first_message_time

    @property
    def image_urls(self) -> list[str]:
        """按首次出现顺序去重，包括引用中的图片。"""
        sources: dict[str, None] = {}
        for item in self.items:
            for url in (*((item.quote.image_urls) if item.quote else ()), *item.image_urls):
                if url:
                    sources.setdefault(url, None)
        return list(sources)

    def remove_by_id(self, message_id: str) -> bool:
        """删除对应消息及关联图片、引用；兼容整数字符串 ID。"""
        previous_count = len(self.items)
        self.items[:] = [item for item in self.items if item.message_id != str(message_id)]
        return len(self.items) < previous_count

    def merged_text(self, separator: str) -> str:
        """保留每条引用与其正文的相对位置，第一条引用自然位于最前。"""
        parts = []
        for item in self.items:
            text = item.text
            if item.quote:
                text = item.quote.as_xml() + ("\n\n" + text if text else "")
            if text:
                parts.append(text)
        return separator.join(parts)


@dataclass
class SessionState:
    """一轮收集状态；由首事件等待 flush_event 后继续原消息管线。"""

    buffer: MessageBuffer
    event: AstrMessageEvent
    timer_task: asyncio.Task[None] | None = None
    flush_event: asyncio.Event = field(default_factory=asyncio.Event)
    done_event: asyncio.Event = field(default_factory=asyncio.Event)
    is_typing: bool = False
    typing_start_time: float | None = None
    prepare_tasks: set[asyncio.Task[None]] = field(default_factory=set)
