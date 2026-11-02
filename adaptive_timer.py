"""纯时间策略：单次等待范围服从整轮等待的硬上限。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .message_buffer import MessageBuffer


def _number(config: Mapping[str, Any], key: str, default: float, lower: float) -> float:
    """对手工编辑的无效数值采用默认值，避免计时器失效。"""
    try:
        value = float(config.get(key, default))
    except (TypeError, ValueError, OverflowError):
        return default
    return max(lower, value) if math.isfinite(value) else default


@dataclass(frozen=True)
class AdaptiveSettings:
    """经过归一化的时间配置。"""

    min_wait: float = 1.0
    max_wait: float = 6.0
    max_total_wait: float = 12.0
    short_message_threshold: int = 10
    max_typing_wait: float = 60.0

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> AdaptiveSettings:
        """修正反向的 min/max，并保留零以上的保护时间。"""
        minimum = _number(config, "adaptive_min_wait", 1.0, 0.01)
        return cls(
            min_wait=minimum,
            max_wait=max(minimum, _number(config, "adaptive_max_wait", 6.0, 0.01)),
            max_total_wait=_number(config, "adaptive_max_total_wait", 12.0, 0.01),
            short_message_threshold=int(_number(config, "adaptive_short_message_threshold", 10, 1)),
            max_typing_wait=_number(config, "max_typing_wait", 60.0, 0.01),
        )


def adaptive_wait(buffer: MessageBuffer, settings: AdaptiveSettings, now: float) -> float:
    """依据最新正文计算等待秒数；总上限到达时返回零。"""
    latest_text = buffer.items[-1].text.strip() if buffer.items else ""
    length = len(latest_text)
    if length <= settings.short_message_threshold:
        wait = 3.0
    elif length <= 50:
        wait = 2.0
    else:
        wait = 1.5

    if latest_text.endswith(("。", "！", "？", ".", "!", "?")):
        wait *= 0.7
    elif latest_text.endswith(("，", ",", "、")):
        wait *= 1.2

    # 与需求示例一致：本条入队后达到两条短句，即采用 1.5 倍等待。
    short_count = sum(
        1 for text in buffer.messages if 0 < len(text.strip()) <= settings.short_message_threshold
    )
    if 0 < length <= settings.short_message_threshold and short_count >= 2:
        wait *= 1.5

    wait = min(settings.max_wait, max(settings.min_wait, wait))
    remaining = buffer.first_message_time + settings.max_total_wait - now
    return max(0.0, min(wait, remaining))
