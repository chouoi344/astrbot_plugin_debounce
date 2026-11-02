"""最小 AstrBot 测试替身；仅模拟插件使用的公开 API，不执行真实平台操作。"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Plain:
    text: str
    type: str = field(default="Plain", init=False)


@dataclass
class Image:
    file: str | None = None
    url: str | None = None
    type: str = field(default="Image", init=False)

    @staticmethod
    def fromURL(url: str) -> Image:
        return Image(file=url, url=url)

    @staticmethod
    def fromFileSystem(path: str) -> Image:
        return Image(file=Path(path).absolute().as_uri())

    @staticmethod
    def fromBase64(data: str) -> Image:
        return Image(file="base64://" + data)


@dataclass
class At:
    qq: str | int
    name: str | None = None
    type: str = field(default="At", init=False)


@dataclass
class Reply:
    id: str | int
    chain: list[Any] | None = None
    sender_id: str | int | None = None
    sender_nickname: str | None = None
    message_str: str | None = None
    text: str | None = None
    type: str = field(default="Reply", init=False)


class FakeEvent:
    def __init__(
        self,
        text: str = "",
        *,
        message_id: str = "1",
        user: str = "10",
        account: str = "99",
        platform: str = "qq-a",
        private: bool = True,
        components: list[Any] | None = None,
        raw: dict | None = None,
        api: Any = None,
        awake: bool = True,
    ):
        self.message_str = text
        self.user = str(user)
        self.account = str(account)
        self.private = private
        self.is_at_or_wake_command = awake
        self.unified_msg_origin = (
            f"{platform}:{'FriendMessage' if private else 'GroupMessage'}:{user}"
        )
        self.stopped = False
        self.call_llm = False
        self.message_obj = types.SimpleNamespace(
            message_str=text,
            message_id=message_id,
            self_id=account,
            sender=types.SimpleNamespace(user_id=user, nickname="用户"),
            session_id=user,
            group_id="" if private else "123",
            type=types.SimpleNamespace(value="FriendMessage" if private else "GroupMessage"),
            message=components if components is not None else ([Plain(text)] if text else []),
            raw_message=raw
            if raw is not None
            else {
                "post_type": "message",
                "message_type": "private" if private else "group",
                "user_id": user,
                "self_id": account,
                "message_id": message_id,
                "message": [{"type": "text", "data": {"text": text}}],
                "raw_message": text,
            },
        )
        self.bot = types.SimpleNamespace(api=api)

    def is_private_chat(self) -> bool:
        return self.private

    def get_self_id(self) -> str:
        return self.account

    def get_sender_id(self) -> str:
        return self.user

    def get_messages(self) -> list[Any]:
        return self.message_obj.message

    def stop_event(self) -> None:
        self.stopped = True

    def is_stopped(self) -> bool:
        return self.stopped

    def should_call_llm(self, value: bool) -> None:
        self.call_llm = value


def install() -> None:
    """测试运行中注入公开 API；单元测试不安装 AstrBot 及其运行依赖。"""
    if "astrbot.api" in sys.modules:
        return
    modules = {
        name: types.ModuleType(name)
        for name in (
            "astrbot",
            "astrbot.api",
            "astrbot.api.event",
            "astrbot.api.star",
            "astrbot.api.message_components",
        )
    }
    logger = types.SimpleNamespace(
        **{level: lambda *args, **kwargs: None for level in ("debug", "info", "warning", "error")}
    )
    modules["astrbot.api"].AstrBotConfig = dict
    modules["astrbot.api"].logger = logger
    modules["astrbot.api.event"].AstrMessageEvent = FakeEvent
    modules["astrbot.api.event"].filter = types.SimpleNamespace(
        EventMessageType=types.SimpleNamespace(ALL="all"),
        event_message_type=lambda *args, **kwargs: lambda method: method,
    )

    class Star:
        def __init__(self, context: Any):
            self.context = context

    modules["astrbot.api.star"].Star = Star
    modules["astrbot.api.star"].Context = object
    for name, cls in (("Plain", Plain), ("Image", Image), ("Reply", Reply), ("At", At)):
        setattr(modules["astrbot.api.message_components"], name, cls)
    sys.modules.update(modules)
