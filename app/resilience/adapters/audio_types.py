"""音频适配器共享类型（TTSResult、音色映射表）。

独立模块，避免 Seed Audio adapter 通过 tts_adapter 间接依赖
websockets/tts_protocol（legacy TTS 协议栈）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TTSResult:
    """TTS / Seed Audio 统一的音频合成结果。"""
    audio_url: str
    local_path: str = ''
    text: str = ''
    voice_id: str = ''
    duration_s: float = 0.0
    model: str = ''
    cost_usd: float = 0.0
    word_timestamps: list[dict] = field(default_factory=list)
    minio_path: Optional[str] = None  # MinIO 上传后的对象路径（fire-and-forget，可能为 None）


# voice_id → speaker 映射（Seed Audio 与 seed-tts-2.0 共用豆包音色 ID）
VOICE_SPEAKER_MAP = {
    # 中文女声
    'zh_female_qingxin': 'zh_female_gaolengyujie_uranus_bigtts',
    'zh_female_xiaoxiao': 'zh_female_gaolengyujie_uranus_bigtts',
    'zh_female_yunxi': 'zh_female_gaolengyujie_uranus_bigtts',
    # 中文男声 - 使用霸总音色更符合赛博朋克侦探
    'zh_male_qingxin': 'zh_male_ruyayichen_uranus_bigtts',
    'zh_male_yunxi': 'zh_male_ruyayichen_uranus_bigtts',
    # 英文音色 fallback → 用对应性别的中文音色
    'en-US-DavisNeural': 'zh_male_ruyayichen_uranus_bigtts',      # 男主 K → 儒雅逸辰
    'en-US-GuyNeural': 'zh_male_ruyayichen_uranus_bigtts',         # 林博士 → 儒雅逸辰
    'en-US-JennyNeural': 'zh_female_gaolengyujie_uranus_bigtts',
    'en-US-ChristopherNeural': 'zh_male_ruyayichen_uranus_bigtts',
    'en-US-AriaNeural': 'zh_female_gaolengyujie_uranus_bigtts',   # 镜 → 高冷御姐
    # 直接使用 voice_type
    'zh_male_ruyayichen_uranus_bigtts': 'zh_male_ruyayichen_uranus_bigtts',
    'zh_female_gaolengyujie_uranus_bigtts': 'zh_female_gaolengyujie_uranus_bigtts',
}

# 男声音色列表（用于 _build_voice_map 的 fallback）
MALE_VOICE_IDS = {
    'zh_male_qingxin', 'zh_male_yunxi',
    'en-US-DavisNeural', 'en-US-GuyNeural', 'en-US-ChristopherNeural',
    'zh_male_ruyayichen_uranus_bigtts',
}

DEFAULT_SPEAKER = 'zh_female_gaolengyujie_uranus_bigtts'
DEFAULT_MALE_SPEAKER = 'zh_male_ruyayichen_uranus_bigtts'
