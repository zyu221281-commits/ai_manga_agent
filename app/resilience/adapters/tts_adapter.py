'''TTS adapter - Volcengine seed-tts-2.0 bidirectional WebSocket'''
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import traceback
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

# 共享类型从 audio_types 导入（与 Seed Audio adapter 解耦，避免 websockets 强依赖）
from app.resilience.adapters.audio_types import (
    TTSResult,
    VOICE_SPEAKER_MAP,
    MALE_VOICE_IDS,
    DEFAULT_SPEAKER,
    DEFAULT_MALE_SPEAKER,
)

# tts_protocol（依赖 websockets）延迟导入到 _syn_once 内部，
# 避免模块加载时强依赖 websockets。

logger = logging.getLogger(__name__)
TTS_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / 'output' / 'audio'


class TTSAdapter(ABC):
    @abstractmethod
    async def synthesize(self, text: str, voice_id: str = 'zh_female_qingxin',
                         speed: float = 1.0, pitch: float = 0.0) -> TTSResult: ...


class VolcengineTTSAdapter(TTSAdapter):
    '''seed-tts-2.0 bidirectional WebSocket adapter.

    Uses the official Volcengine binary protocol via tts_protocol.protocols.
    Mirrors the working standalone test structure exactly.
    '''

    @property
    def MODEL(self) -> str:
        """从 config 读取 TTS 模型标识（避免硬编码）。"""
        from app.core.config import settings
        return settings.ARK_TTS_MODEL

    WS_URL = 'wss://openspeech.bytedance.com/api/v3/tts/bidirection'
    RESOURCE_ID = 'seed-tts-2.0'  # X-Api-Resource-Id（协议固定值，非模型名）

    @property
    def COST_PER_CHAR(self) -> float:
        """从 config 读取 TTS 每字符单价（L5: 价格集中化）。"""
        from app.core.config import settings
        return settings.TTS_COST_PER_CHAR

    async def synthesize(self, text: str, voice_id: str = 'zh_female_qingxin',
                         speed: float = 1.0, pitch: float = 0.0) -> TTSResult:
        """TTS 合成，带 3 层防御。

        L0: WebSocket 连接/协议失败时重连（3次，指数退避 1s/2s/4s）
        L1: SessionFailed 时换默认 voice_id 重试
        L2: 全部失败时降级到 placeholder（静音 TTS）
        """
        from app.core.config import settings
        tts_key = settings.VOLCENGINE_TTS_KEY
        if not tts_key:
            logger.warning('TTS: VOLCENGINE_TTS_KEY not set, returning placeholder')
            return self._placeholder(text, voice_id)
        if not text.strip():
            return self._placeholder(text, voice_id)

        # L1 备选 voice_id
        fallback_voices = ['zh_female_qingxin', 'zh_male_ruyayichen_uranus_bigtts']
        tried_voices = set()

        for attempt_voice in [voice_id] + [v for v in fallback_voices if v != voice_id]:
            if attempt_voice in tried_voices:
                continue
            tried_voices.add(attempt_voice)

            # L0: WebSocket 重连重试
            for ws_attempt in range(3):
                try:
                    result = await self._syn_once(text, attempt_voice, speed, pitch, tts_key)
                    if result and result.audio_url and not result.audio_url.startswith('tts://placeholder'):
                        if attempt_voice != voice_id:
                            logger.info('TTS L1 success with fallback voice: %s', attempt_voice)
                        return result
                except Exception as e:
                    delay = 2 ** ws_attempt  # 1s, 2s, 4s
                    logger.warning('TTS L0 attempt %d/3 failed (voice=%s): %s, retry in %ds',
                                   ws_attempt + 1, attempt_voice, e, delay)
                    if ws_attempt < 2:
                        import asyncio as _aio
                        await _aio.sleep(delay)

        # L2: 全部失败，降级到 placeholder
        logger.error('TTS L2 fallback: all retries exhausted for text (%d chars)', len(text))
        return self._placeholder(text, voice_id)

    async def _syn_once(self, text: str, voice_id: str, speed: float, pitch: float,
                        tts_key: str) -> TTSResult:
        """单次 TTS 合成（无重试逻辑，供 synthesize 调用）。"""
        # 延迟导入：websockets + tts_protocol 仅 legacy TTS 需要
        import websockets
        from app.resilience.adapters.tts_protocol.protocols import (
            EventType,
            MsgType,
            start_connection,
            finish_connection,
            start_session,
            finish_session,
            task_request,
            receive_message,
            wait_for_event,
        )
        speaker = VOICE_SPEAKER_MAP.get(voice_id, DEFAULT_SPEAKER)
        ws = None

        try:
            headers = {
                'X-Api-Key': tts_key,
                'X-Api-Resource-Id': self.RESOURCE_ID,
                'X-Api-Connect-Id': str(uuid.uuid4()),
            }
            ws = await websockets.connect(
                self.WS_URL,
                additional_headers=headers, max_size=10 * 1024 * 1024
            )

            # --- Start connection ---
            await start_connection(ws)
            await wait_for_event(ws, MsgType.FullServerResponse, EventType.ConnectionStarted)
            logger.debug('TTS: ConnectionStarted')

            # --- Start session ---
            session_id = str(uuid.uuid4())
            base_req = {
                'req_params': {
                    'speaker': speaker,
                    'audio_params': {'format': 'mp3', 'sample_rate': 24000},
                }
            }
            await start_session(ws, json.dumps(base_req).encode(), session_id)
            await wait_for_event(ws, MsgType.FullServerResponse, EventType.SessionStarted)
            logger.debug('TTS: SessionStarted')

            # --- Send full text in one request ---
            req = copy.deepcopy(base_req)
            req['req_params']['text'] = text
            await task_request(ws, json.dumps(req).encode(), session_id)

            # --- Finish session to trigger audio synthesis ---
            await finish_session(ws, session_id)
            logger.debug('TTS: FinishSession sent, collecting audio...')

            # --- Receive audio chunks until SessionFinished ---
            audio_data = bytearray()
            while True:
                msg = await receive_message(ws)
                if msg.type == MsgType.AudioOnlyServer:
                    audio_data.extend(msg.payload)
                elif msg.type == MsgType.FullServerResponse:
                    if msg.event == EventType.SessionFinished:
                        logger.debug('TTS: SessionFinished, audio=%d bytes', len(audio_data))
                        break
                    elif msg.event == EventType.SessionFailed:
                        raise RuntimeError(
                            f'TTS session failed: {msg.payload.decode("utf-8", "ignore")}'
                        )
                elif msg.type == MsgType.Error:
                    raise RuntimeError(
                        f'TTS protocol error: code={msg.error_code}, '
                        f'payload={msg.payload.decode("utf-8", "ignore")}'
                    )

            # --- Finish connection ---
            try:
                await finish_connection(ws)
            except Exception:
                pass
            await ws.close()
            ws = None

            if audio_data:
                os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)
                fname = 'tts_{}.mp3'.format(hashlib.md5(text.encode()).hexdigest()[:12])
                local = str(TTS_OUTPUT_DIR / fname)
                with open(local, 'wb') as f:
                    f.write(audio_data)
                dur = self._probe_mp3_duration(local)
                if dur <= 0:
                    dur = len(text) / 5.0 if text else 1.0
                logger.info('TTS: %s (%d bytes, %.1fs)', local[:80], len(audio_data), dur)
                word_timestamps = VolcengineTTSAdapter._estimate_word_timestamps(text, dur)
                tts_result = TTSResult(
                    word_timestamps=word_timestamps,
                    audio_url='file://' + local, local_path=local,
                    text=text, voice_id=voice_id, duration_s=dur,
                    model=self.MODEL,
                    cost_usd=len(text) * self.COST_PER_CHAR,
                )
                # fire-and-forget MinIO 上传（失败不阻塞管线，minio_path 异步填入）
                from app.services.minio_client import fire_and_forget_upload
                fire_and_forget_upload(tts_result, local, "audio")
                return tts_result

            return self._placeholder(text, voice_id)

        except Exception as e:
            logger.error('TTS _syn_once failed: %s\n%s', e, traceback.format_exc())
            if ws is not None:
                try:
                    await ws.close()
                except Exception:
                    pass
            raise  # 重新抛出，由上层 synthesize 的 L0 重试逻辑处理


    @staticmethod
    def _estimate_word_timestamps(text, duration_s):
        if not text or duration_s <= 0: return []
        chars_per_sec = len(text) / duration_s
        timestamps = []
        current_time = 0.0
        for char in text:
            char_duration = 1.0 / chars_per_sec
            timestamps.append({'char': char, 'start': round(current_time, 3), 'end': round(current_time + char_duration, 3)})
            current_time += char_duration
        return timestamps

    @staticmethod
    def _probe_mp3_duration(path: str) -> float:
        """Probe actual mp3 duration via ffprobe. Returns 0.0 on failure.

        Prefer this over len(text)/5.0 rough estimation for accurate
        subtitle timing. Falls back to 0.0 if ffprobe unavailable or fails.
        """
        import re
        import subprocess as sp
        try:
            proc = sp.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, timeout=10,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return float(proc.stdout.strip())
        except Exception:
            pass
        # Fallback: parse Duration from ffprobe stderr (compat with older versions)
        try:
            proc = sp.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", path],
                capture_output=True, text=True, timeout=10,
            )
            m = re.search(r"duration=([\d.]+)", proc.stdout)
            if m:
                return float(m.group(1))
        except Exception:
            pass
        return 0.0

    def _placeholder(self, text: str, voice_id: str) -> TTSResult:
        dur = len(text) / 5.0 if text else 1.0
        word_timestamps = VolcengineTTSAdapter._estimate_word_timestamps(text, dur)
        return TTSResult(
            word_timestamps=word_timestamps,
            audio_url='tts://placeholder/' + hashlib.md5(text.encode()).hexdigest()[:8],
            text=text, voice_id=voice_id,
            duration_s=dur, model=self.MODEL, cost_usd=0.0,
        )


def get_tts_adapter(provider: str = 'azure') -> TTSAdapter:
    """工厂方法：根据 config.AUDIO_PROVIDER 选择音频适配器。

    - seed_audio（默认）: Seed Audio 1.0，整合 TTS + BGM，返回 TTSResult
    - tts: legacy seed-tts-2.0 WebSocket 适配器
    """
    from app.core.config import settings
    if settings.AUDIO_PROVIDER == 'seed_audio':
        from app.resilience.adapters.seed_audio_adapter import SeedAudioAdapter
        return SeedAudioAdapter()  # type: ignore[return-value]
    return VolcengineTTSAdapter()
