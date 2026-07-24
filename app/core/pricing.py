"""模型单价表（被 cost_tracker 引用）。

所有价格为 USD。LLM 按 token 计价，图像/视频/TTS 按次计价。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMPrice:
    """LLM 按 token 计价。"""
    input_per_million: float   # 输入 token 单价（USD / 1M tokens）
    output_per_million: float  # 输出 token 单价（USD / 1M tokens）


@dataclass(frozen=True)
class UnitPrice:
    """按次计价。"""
    per_unit: float
    unit: str  # "image" / "second" / "1k_chars" / "call" / "1k_tokens"


# 单价表（与  文档 2.2 节 COST_MODEL_PRICING 对应）
# 注意：媒体生成（image/video/tts）的实际单价以 config.py 中的 settings 为准，
# 此处仅保留 LLM 和其他服务的静态价格。媒体价格通过 get_media_price() 动态读取。
MODEL_PRICING: dict[str, LLMPrice | UnitPrice] = {
    # --- LLM ---
    "deepseek--pro": LLMPrice(input_per_million=0.27, output_per_million=1.10),
    "qwen3.7-max": LLMPrice(input_per_million=0.40, output_per_million=1.20),
    "qwen-turbo": LLMPrice(input_per_million=0.05, output_per_million=0.20),
    "qwen-vl-max": LLMPrice(input_per_million=0.50, output_per_million=1.50),
    # --- 图像（静态参考值，实际以 config.IMAGE_COST_PER_UNIT 为准）---
    "flux-schnell": UnitPrice(per_unit=0.003, unit="image"),
    "seedream": UnitPrice(per_unit=0.02, unit="image"),
    # --- 视频（静态参考值，实际以 config.VIDEO_COST_PER_SECOND 为准）---
    "kling-video": UnitPrice(per_unit=0.30, unit="second"),
    # --- TTS（静态参考值，实际以 config.TTS_COST_PER_CHAR 为准）---
    "azure-tts": UnitPrice(per_unit=0.016, unit="1k_chars"),
    # --- Seed Audio 1.0（实际以 config.SEED_AUDIO_COST_PER_SECOND 为准）---
    "seed-audio-1.0": UnitPrice(per_unit=0.02, unit="second"),
    # --- Embedding ---
    "dashscope-embedding": UnitPrice(per_unit=0.00007, unit="1k_tokens"),
    # --- 内容安全 ---
    "aliyun-safety": UnitPrice(per_unit=0.01, unit="call"),
}


def get_media_price(media_type: str) -> float:
    """从 config 获取媒体生成单价（L5: 价格集中化管理）。

    Args:
        media_type: "image" / "video_second" / "tts_char"

    Returns:
        单价（USD），如果类型未知返回 0.0
    """
    from app.core.config import settings
    prices = {
        "image": settings.IMAGE_COST_PER_UNIT,
        "video_second": settings.VIDEO_COST_PER_SECOND,
        "tts_char": settings.TTS_COST_PER_CHAR,
    }
    return prices.get(media_type, 0.0)


def calculate_llm_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """计算单次 LLM 调用成本（USD）。"""
    price = MODEL_PRICING.get(model)
    if price is None or not isinstance(price, LLMPrice):
        return 0.0
    return (
        input_tokens / 1_000_000 * price.input_per_million
        + output_tokens / 1_000_000 * price.output_per_million
    )


def calculate_unit_cost(model: str, count: float) -> float:
    """计算按次计价成本（USD）。

    count 的单位由模型决定：image / second / 1k_chars / call / 1k_tokens。
    """
    price = MODEL_PRICING.get(model)
    if price is None or not isinstance(price, UnitPrice):
        return 0.0
    if price.unit in ("1k_chars", "1k_tokens"):
        return count / 1000 * price.per_unit
    return count * price.per_unit
