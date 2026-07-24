"""Seed Audio 1.0 适配器（整合 TTS + BGM + 音效）。

调用火山方舟 Seed Audio 1.0 HTTP API（/api/v3/tts/create），一次生成包含
对白、背景音乐、环境音效的完整音频场景。相比 seed-tts-2.0：
- 支持自然语言 prompt 描述声音场景（BGM/音效/语气）
- enable_subtitle 返回字级时间戳（比 TTS 估算更精确，直接驱动字幕同步）
- 单次最长 120s，覆盖短剧单个 shot 的需求

接口与 TTSAdapter.synthesize 兼容，返回 TTSResult，下游 composer 无需改动。
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import os
import uuid
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# 音频输出目录（与 tts_adapter 共用，便于 compositor 统一扫描）
AUDIO_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "output", "audio",
)


def _import_tts_types():
    """导入音频共享类型（从 audio_types，不依赖 websockets）。"""
    from app.resilience.adapters.audio_types import (
        TTSResult,
        VOICE_SPEAKER_MAP,
        DEFAULT_SPEAKER,
    )
    return TTSResult, VOICE_SPEAKER_MAP, DEFAULT_SPEAKER


class SeedAudioAdapter:
    """Seed Audio 1.0 适配器。

    通过 HTTP POST 调用 /api/v3/tts/create，同步返回 base64 音频 + 字幕。
    支持 speaker 参数复用豆包语音合成模型 2.0 的音色 ID（与 seed-tts-2.0 一致）。
    """

    def __init__(self) -> None:
        from app.core.config import settings
        self._api_url: str = settings.SEED_AUDIO_API_URL
        self._api_key: str = settings.VOLCENGINE_TTS_KEY
        self._model: str = settings.ARK_AUDIO_MODEL
        self._cost_per_second: float = settings.SEED_AUDIO_COST_PER_SECOND
        self._http: Optional[httpx.AsyncClient] = None

    def _client(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=120.0)
        return self._http

    @property
    def MODEL(self) -> str:
        return self._model

    async def synthesize(
        self,
        text: str,
        voice_id: str = "zh_female_qingxin",
        speed: float = 1.0,
        pitch: float = 0.0,
    ) -> TTSResult:
        """生成音频（兼容 TTSAdapter 接口）。

        Args:
            text: 待合成文本（对白/旁白）。Seed Audio 会根据文本内容自动
                  匹配语气，并在文本包含场景描述时生成相应背景音。
            voice_id: 角色音色标识（与 seed-tts-2.0 共用 VOICE_SPEAKER_MAP）。
            speed:   语速倍率（1.0 = 正常）。映射到 speech_rate[-50,100]。
            pitch:   音调偏移（0 = 正常）。映射到 pitch_rate[-12,12]。

        Returns:
            TTSResult（含 local_path、duration_s、word_timestamps、cost_usd）
        """
        if not self._api_key:
            logger.warning("Seed Audio: VOLCENGINE_TTS_KEY 未配置，返回 placeholder")
            return self._placeholder(text, voice_id)
        if not text.strip():
            return self._placeholder(text, voice_id)

        TTSResult, VOICE_SPEAKER_MAP, DEFAULT_SPEAKER = _import_tts_types()

        # voice_id → speaker（与 seed-tts-2.0 共用映射表）
        speaker = VOICE_SPEAKER_MAP.get(voice_id, DEFAULT_SPEAKER)

        # speed/pitch → Seed Audio 参数范围
        speech_rate = self._speed_to_rate(speed)
        pitch_rate = self._pitch_to_rate(pitch)

        payload = {
            "model": self._model,
            "text_prompt": text,
            "speaker": speaker,
            "audio_config": {
                "format": "mp3",
                "sample_rate": 24000,
                "speech_rate": speech_rate,
                "pitch_rate": pitch_rate,
                "enable_subtitle": True,  # 字级时间戳（文档：属 audio_config）
            },
        }

        headers = {
            "X-Api-Key": self._api_key,
            "X-Api-Request-Id": str(uuid.uuid4()),
            "Content-Type": "application/json",
        }

        # 3 层重试（与 TTSAdapter 对齐：1s/2s/4s 指数退避）
        for attempt in range(3):
            try:
                client = self._client()
                resp = await client.post(
                    self._api_url, json=payload, headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()

                code = data.get("code", 0)
                if code != 0:
                    raise RuntimeError(
                        f"Seed Audio API error: code={code}, message={data.get('message', '')}"
                    )

                audio_b64 = data.get("audio", "")
                if not audio_b64:
                    raise RuntimeError("Seed Audio returned empty audio data")

                duration_s = float(data.get("duration", 0.0)) or float(
                    data.get("original_duration", 0.0)
                )
                # 计费以 original_duration 为准（文档明确）
                billable_duration = float(data.get("original_duration", duration_s))

                # 时长合理性校验：中文约 5 字/秒，偏差超过 3 倍视为异常
                # 触发重试（非最后一次）；最后一次接受结果但记录 warning
                expected_dur = max(len(text) / 5.0, 0.5)
                ratio = duration_s / expected_dur if expected_dur > 0 else 1.0
                is_anomaly = ratio < 0.33 or ratio > 3.0
                if is_anomaly and attempt < 2:
                    raise RuntimeError(
                        f"duration anomaly: {duration_s:.2f}s vs expected ~{expected_dur:.2f}s "
                        f"(text={len(text)} chars, ratio={ratio:.2f})"
                    )

                # 保存音频文件
                os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
                fname = f"seed_audio_{hashlib.md5(text.encode()).hexdigest()[:12]}.mp3"
                local_path = os.path.join(AUDIO_OUTPUT_DIR, fname)
                with open(local_path, "wb") as f:
                    f.write(base64.b64decode(audio_b64))

                # 解析字级时间戳（Seed Audio 原生返回，比 TTS 估算精确）
                subtitle_data = data.get("subtitle")
                word_timestamps = self._extract_word_timestamps(subtitle_data)
                if not word_timestamps:
                    # API 对某些文本（如长文本或多句）可能不返回 subtitle，
                    # fallback 到基于文本长度和时长的均匀估算
                    logger.info(
                        "Seed Audio: subtitle 为空或无 words，使用估算时间戳 "
                        "(text=%d chars, dur=%.2fs)",
                        len(text), duration_s,
                    )
                    word_timestamps = self._estimate_word_timestamps(text, duration_s)

                cost_usd = billable_duration * self._cost_per_second

                if is_anomaly:
                    # 最后一次重试仍异常：接受结果（比 placeholder 好），但记录 warning
                    logger.warning(
                        "Seed Audio: 时长异常但已达最大重试，接受结果 "
                        "(text=%d chars, dur=%.2fs, expected~%.2fs, ratio=%.2f)",
                        len(text), duration_s, expected_dur, ratio,
                    )
                logger.info(
                    "Seed Audio: %s (%.1fs, %d chars, cost=%.4f)",
                    os.path.basename(local_path), duration_s, len(text), cost_usd,
                )

                result = TTSResult(
                    audio_url="file://" + local_path,
                    local_path=local_path,
                    text=text,
                    voice_id=voice_id,
                    duration_s=duration_s,
                    model=self._model,
                    cost_usd=cost_usd,
                    word_timestamps=word_timestamps,
                )

                # fire-and-forget MinIO 上传（与 TTSAdapter 行为一致）
                try:
                    from app.services.minio_client import fire_and_forget_upload
                    fire_and_forget_upload(result, local_path, "audio")
                except Exception as e:
                    logger.debug("MinIO upload skipped: %s", e)

                return result

            except Exception as e:
                delay = 2 ** attempt
                logger.warning(
                    "Seed Audio attempt %d/3 failed: %s, retry in %ds",
                    attempt + 1, e, delay,
                )
                if attempt < 2:
                    await asyncio.sleep(delay)

        logger.error(
            "Seed Audio: all retries exhausted for text (%d chars), falling back to placeholder",
            len(text),
        )
        return self._placeholder(text, voice_id)

    @staticmethod
    def _extract_word_timestamps(subtitle: Optional[dict]) -> list[dict]:
        """从 Seed Audio 响应提取字级时间戳。

        响应结构：subtitle.{sentences:[{words:[{start_time,end_time,text}]}]}
        时间单位：毫秒（距音频起始）。转换为 TTSResult 兼容的秒级格式。
        """
        if not subtitle:
            return []
        timestamps: list[dict] = []
        for sentence in subtitle.get("sentences", []):
            for word in sentence.get("words", []):
                timestamps.append({
                    "char": word.get("text", ""),
                    "start": round(word.get("start_time", 0) / 1000.0, 3),
                    "end": round(word.get("end_time", 0) / 1000.0, 3),
                })
        return timestamps

    @staticmethod
    def _estimate_word_timestamps(text: str, duration_s: float) -> list[dict]:
        """估算字级时间戳（当 API 不返回 subtitle 时的 fallback）。

        基于文本长度和音频时长均匀分布，与 TTSAdapter 行为一致。
        """
        if not text or duration_s <= 0:
            return []
        chars_per_sec = max(len(text) / duration_s, 0.1)
        timestamps: list[dict] = []
        t = 0.0
        for ch in text:
            d = 1.0 / chars_per_sec
            timestamps.append({
                "char": ch,
                "start": round(t, 3),
                "end": round(min(t + d, duration_s), 3),
            })
            t += d
        return timestamps

    @staticmethod
    def _speed_to_rate(speed: float) -> int:
        """speed 倍率 → speech_rate[-50,100]。1.0=0, 2.0=100, 0.5=-50。"""
        rate = int((speed - 1.0) * 100)
        return max(-50, min(100, rate))

    @staticmethod
    def _pitch_to_rate(pitch: float) -> int:
        """pitch 半音 → pitch_rate[-12,12]。"""
        return max(-12, min(12, int(pitch)))

    @staticmethod
    def _placeholder(text: str, voice_id: str):
        """降级：返回估算时长的占位音频（与 TTSAdapter 行为一致）。"""
        TTSResult, _, _ = _import_tts_types()
        dur = len(text) / 5.0 if text else 1.0
        # 用简单的字符级时间戳估算（不依赖 VolcengineTTSAdapter）
        timestamps = []
        if text and dur > 0:
            chars_per_sec = len(text) / dur
            t = 0.0
            for ch in text:
                d = 1.0 / chars_per_sec
                timestamps.append({"char": ch, "start": round(t, 3), "end": round(t + d, 3)})
                t += d
        return TTSResult(
            word_timestamps=timestamps,
            audio_url="tts://placeholder/" + hashlib.md5(text.encode()).hexdigest()[:8],
            text=text,
            voice_id=voice_id,
            duration_s=dur,
            model="seed-audio-1.0",
            cost_usd=0.0,
        )


def get_seed_audio_adapter() -> SeedAudioAdapter:
    """工厂方法：返回 SeedAudioAdapter 单例。"""
    return SeedAudioAdapter()
