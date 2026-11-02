"""归一化 OneBot/NapCat 通知，不依赖不存在的通用通知装饰器。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PrivateNotice:
    """通知所属账号、用户和可选操作。"""

    user_id: str
    self_id: str
    typing: bool | None = None
    recalled_message_id: str | None = None


def raw_notice(event: Any) -> Mapping[str, Any]:
    """兼容 aiocqhttp CQEvent（dict 子类）和适配器原始字典。"""
    raw = getattr(getattr(event, "message_obj", None), "raw_message", None)
    return raw if isinstance(raw, Mapping) else {}


def is_notice(event: Any) -> bool:
    """通知消息应在防抖入口静默结束，避免被当作聊天正文。"""
    raw = raw_notice(event)
    return raw.get("post_type") in {"notice", "meta_event", "request"} or bool(
        raw.get("notice_type")
    )


def parse_notice(event: Any) -> PrivateNotice | None:
    """识别私聊输入、停止输入与好友撤回；群聊通知直接跳过。"""
    raw = raw_notice(event)
    if raw.get("group_id"):
        return None
    user_id = str(raw.get("user_id", "") or "")
    if not user_id:
        return None
    self_id = str(raw.get("self_id", "") or "")
    notice_type = raw.get("notice_type", "")
    if notice_type == "friend_recall":
        message_id = raw.get("message_id")
        if message_id is not None:
            return PrivateNotice(user_id, self_id, recalled_message_id=str(message_id))
        return None

    if "input_status" not in {notice_type, raw.get("sub_type", ""), raw.get("event_type", "")}:
        return None
    text = str(raw.get("status_text", "") or "").strip().lower()
    # 先判断停止，避免 'stop typing' 被 'typing' 子串匹配。
    if any(word in text for word in ("停止", "结束", "stop", "not typing", "非输入", "未输入")):
        return PrivateNotice(user_id, self_id, typing=False)
    if any(word in text for word in ("输入", "typing")):
        return PrivateNotice(user_id, self_id, typing=True)
    status = raw.get("is_typing", raw.get("typing"))
    if isinstance(status, bool):
        return PrivateNotice(user_id, self_id, typing=status)
    # NapCat 透传底层数值 event_type，公开源码未定义稳定的二态映射。
    # 缺少明确文本/布尔标记时沿用普通防抖，让总上限兜底。
    return None
