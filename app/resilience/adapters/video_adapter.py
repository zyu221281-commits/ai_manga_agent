"""视频生成适配器：火山方舟 Seedance (ARK SDK content_generation.tasks)

真实实现：通过 volcengine SDK 调用 seedance 图生视频。
- 提交任务 → 轮询状态 → 下载视频
- KEY_SCENE → Seedance 图生视频，NORMAL_SCENE → FFmpeg Ken Burns 静态图运镜
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
from volcenginesdkarkruntime import Ark

logger = logging.getLogger(__name__)

VIDEO_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "output" / "videos"
DEFAULT_VIDEO_MODEL = "doubao-seedance-1-0-pro-fast-251015"


@dataclass
class VideoResult:
    url: str
    local_path: str = ""
    duration_s: float = 5.0
    scene_type: str = "normal"
    model: str = ""
    cost_usd: float = 0.0
    metadata: dict = field(default_factory=dict)
    minio_path: Optional[str] = None  # MinIO 上传后的对象路径（fire-and-forget，可能为 None）


class VideoAdapter(ABC):
    @abstractmethod
    async def generate(
        self, image_path: str = "", image_url: str = "",
        scene_type: str = "normal", duration_s: float = 5.0,
        motion_type: str = "zoom", **kwargs,
    ) -> VideoResult: ...


class VolcengineVideoAdapter(VideoAdapter):
    """火山方舟 Seedance 图生视频（ARK SDK）。"""

    MODEL = DEFAULT_VIDEO_MODEL

    @property
    def COST_PER_SECOND(self) -> float:
        """从 config 读取视频每秒单价（L5: 价格集中化）。"""
        from app.core.config import settings
        return settings.VIDEO_COST_PER_SECOND

    def __init__(self, model: str = ""):
        if model:
            self._model = model
        else:
            from app.core.config import settings
            self._model = settings.ARK_VIDEO_MODEL or DEFAULT_VIDEO_MODEL
        self._client: Optional[Ark] = None

    def _get_client(self) -> Ark:
        if self._client is None:
            from app.core.config import settings
            self._client = Ark(
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                api_key=settings.ARK_API_KEY,
            )
        return self._client

    async def generate(
        self, image_path: str = "", image_url: str = "",
        scene_type: str = "key", duration_s: float = 5.0,
        motion_type: str = "zoom", prompt_text: str = "",
        **kwargs,
    ) -> VideoResult:
        src_url = image_url
        if not src_url and image_path:
            src_url = image_path  # use local path as fallback

        if not src_url:
            logger.warning("No image source for seedance video")
            return VideoResult(
                url="", scene_type=scene_type, model=self.MODEL, cost_usd=0.0,
                metadata={"error": "no image source"},
            )

        text = prompt_text or "gentle camera motion, anime style, smooth"
        dur = max(int(duration_s), 1)
        text = "{}  --duration {} --camerafixed false --watermark false".format(text, dur)
        if motion_type == "zoom":
            text = "slow zoom in, " + text
        elif motion_type == "pan":
            text = "gentle pan, " + text


        try:
            client = self._get_client()

            # Submit task with L0 retry for transient API failures
            max_retries = 3
            create_result = None
            for attempt in range(max_retries + 1):
                try:
                    create_result = client.content_generation.tasks.create(
                        model=self._model,
                        content=[
                            {"type": "image_url", "image_url": {"url": src_url}},
                            {"type": "text", "text": text},
                        ],
                    )
                    break
                except Exception as e:
                    if attempt < max_retries:
                        delay = 1.0 * (3 ** attempt)
                        logger.warning("Seedance submit retry %d/%d after %.0fs: %s", attempt+1, max_retries, delay, e)
                        await asyncio.sleep(delay)
                        continue
                    raise

            if create_result is None:
                logger.error("Seedance submit failed after all retries")
                # 失败时 local_path 必须为空，让 composer 检测失败后进入 L1/L2 fallback
                return VideoResult(url=src_url, local_path="", scene_type=scene_type, model=self.MODEL, cost_usd=0.0, metadata={"error": "submit retries exhausted"})

            task_id = create_result.id
            logger.info("Seedance task submitted: %s", task_id)

            # Poll for completion (adaptive interval)
            from app.core.config import settings
            max_wait = settings.VIDEO_POLL_MAX_WAIT_S
            poll_intervals = [5, 5, 10, 10, 15]  # 自适应轮询：先快后慢
            elapsed = 0
            interval_idx = 0
            while elapsed < max_wait:
                interval = poll_intervals[min(interval_idx, len(poll_intervals) - 1)]
                await asyncio.sleep(interval)
                elapsed += interval
                interval_idx += 1
                try:
                    get_result = client.content_generation.tasks.get(task_id=task_id)
                except Exception as e:
                    logger.warning("Seedance poll error (elapsed %ds): %s", elapsed, e)
                    continue
                status = get_result.status
                if status == "succeeded":
                    logger.info("Seedance task succeeded: %s (%ds)", task_id, elapsed)
                    video_url = get_result.content.video_url or ""
                    local = await self._download(video_url, task_id)
                    vid_result = VideoResult(
                        url=video_url, local_path=local,
                        duration_s=dur, scene_type=scene_type,
                        model=self.MODEL,
                        cost_usd=dur * self.COST_PER_SECOND,
                        metadata={"task_id": task_id, "elapsed_s": elapsed},
                    )
                    # fire-and-forget MinIO 上传（失败不阻塞管线，minio_path 异步填入）
                    from app.services.minio_client import fire_and_forget_upload
                    fire_and_forget_upload(vid_result, local, "video")
                    return vid_result
                elif status == "failed":
                    err = str(get_result.error) if get_result.error else "unknown"
                    logger.error("Seedance task failed: %s", err)
                    # 失败时 local_path 必须为空，让 composer 检测失败后进入 L1/L2 fallback
                    return VideoResult(
                        url=src_url, local_path="",
                        scene_type=scene_type, model=self.MODEL, cost_usd=0.0,
                        metadata={"error": err, "task_id": task_id},
                    )
                else:
                    logger.debug("Seedance status: %s (elapsed %ds)", status, elapsed)

            logger.warning("Seedance task timeout: %s", task_id)
            # 超时时 local_path 必须为空，让 composer 检测失败后进入 L1/L2 fallback
            return VideoResult(
                url=src_url, local_path="",
                scene_type=scene_type, model=self.MODEL, cost_usd=0.0,
                metadata={"error": "timeout", "task_id": task_id},
            )

        except Exception as e:
            logger.error("Seedance exception: %s", e)
            # 异常时 local_path 必须为空，让 composer 检测失败后进入 L1/L2 fallback
            return VideoResult(
                url=src_url, local_path="",
                scene_type=scene_type, model=self.MODEL, cost_usd=0.0,
                metadata={"error": str(e)},
            )

    async def _download(self, video_url: str, task_id: str) -> str:
        if not video_url:
            return ""
        os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
        safe_id = task_id.replace("/", "_").replace("\\", "_")
        nonce = int(time.time() * 1000) % 100000
        fname = "seedance_{}_{}.mp4".format(safe_id, nonce)
        local = str(VIDEO_OUTPUT_DIR / fname)
        try:
            async with httpx.AsyncClient(timeout=300) as c:
                r = await c.get(video_url)
                if r.status_code == 200:
                    content = r.content
                    if len(content) < 10240:
                        logger.error(
                            "Video download too small: %s (%d bytes) — likely error page or corrupted",
                            local, len(content),
                        )
                        return ""
                    with open(local, "wb") as f:
                        f.write(content)
                    logger.info("Video downloaded: %s (%d bytes)", local, len(content))
                    if not self._validate_video_file(local):
                        logger.error("Video file validation failed after download: %s", local)
                        try:
                            os.remove(local)
                        except Exception:
                            pass
                        return ""
                    return local
                else:
                    logger.error("Video download HTTP %d: %s", r.status_code, video_url[:80])
        except Exception as e:
            logger.warning("Video download failed: %s", e)
        return ""

    @staticmethod
    def _validate_video_file(path: str) -> bool:
        """验证视频文件有效：存在、非空、ffprobe 可解析。"""
        if not path or not os.path.isfile(path):
            return False
        if os.path.getsize(path) < 10240:
            logger.warning("Video file too small: %s (%d bytes)", path, os.path.getsize(path))
            return False
        try:
            import subprocess
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, timeout=15,
            )
            if r.returncode != 0:
                logger.warning("ffprobe failed for %s: %s", path, r.stderr[:200])
                return False
            dur = float(r.stdout.strip() or "0")
            if dur < 0.5:
                logger.warning("Video duration too short: %s (%.2fs)", path, dur)
                return False
            return True
        except Exception as e:
            logger.warning("Video validation exception for %s: %s", path, e)
            return False


class FFmpegKenBurnsAdapter(VideoAdapter):
    """FFmpeg Ken Burns 适配器：静态图 → 推拉摇移视频。

    用 zoompan filter 在静态图上模拟相机运动（Ken Burns 效果），
    零 API 成本，适合 NORMAL_SCENE 过渡/对话场景。
    """
    MODEL = "ffmpeg_kenburns"

    _WIDTH = 1080
    _HEIGHT = 1920

    async def generate(self, **kwargs) -> VideoResult:
        """Generate Ken Burns video from static image via ffmpeg zoompan."""
        image_path = kwargs.get("image_path", "")
        image_url = kwargs.get("image_url", "")
        duration_s = max(kwargs.get("duration_s", 4.0), 2.0)
        motion_type = kwargs.get("motion_type", "zoom-in")
        scene_type = kwargs.get("scene_type", "normal")

        src = image_path or image_url
        if not src or (image_path and not os.path.isfile(image_path)):
            logger.warning("Ken Burns: no valid image source")
            return VideoResult(
                url=src or "", scene_type=scene_type,
                model=self.MODEL, cost_usd=0.0,
                metadata={"error": "no image source"},
            )

        frames = int(duration_s * 30)
        motion_filters = {
            "zoom-in":  "zoompan=z='min(zoom+0.0018,1.5)':d={}:s={}x{}:fps=30",
            "zoom-out": "zoompan=z='max(zoom-0.0018,1.0)':d={}:s={}x{}:fps=30",
            "dolly-in":  "zoompan=z='min(zoom+0.0022,1.6)':d={}:s={}x{}:fps=30",
            "dolly-out": "zoompan=z='max(zoom-0.0022,1.0)':d={}:s={}x{}:fps=30",
            "pan-left":  "zoompan=z=1.15:x='max(iw-iw/zoom-1,max(0,iw/zoom*(1-1/zoom)-n*3))':d={}:s={}x{}:fps=30",
            "pan-right": "zoompan=z=1.15:x='min(iw/zoom+1,max(0,n*3))':d={}:s={}x{}:fps=30",
            "static":   "zoompan=z=1.0:d={}:s={}x{}:fps=30",
        }
        tmpl = motion_filters.get(motion_type, motion_filters["zoom-in"])
        vf = tmpl.format(frames, self._WIDTH, self._HEIGHT)

        os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
        out_path = str(VIDEO_OUTPUT_DIR / "kenburns_{}_{}.mp4".format(
            hashlib.md5(str(src).encode()).hexdigest()[:10],
            int(time.time() * 1000) % 100000,
        ))

        try:
            ffmpeg = self._ffmpeg_bin()
            # 使用 asyncio.create_subprocess_exec 替代 run_in_executor + subprocess.run
            # 避免线程池在 asyncio 事件循环中的兼容性问题（composer 全流程中 kenburns 生成空文件）
            proc = await asyncio.create_subprocess_exec(
                ffmpeg, "-y", "-loop", "1", "-i", src,
                "-vf", vf, "-t", str(duration_s), "-r", "30",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart",
                out_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                logger.error("Ken Burns ffmpeg timeout (120s) for src=%s", src)
                return VideoResult(
                    url=image_url or image_path or "",
                    local_path="",
                    duration_s=duration_s, scene_type=scene_type,
                    model=self.MODEL, cost_usd=0.0,
                    metadata={"error": "ffmpeg timeout"},
                )

            if proc.returncode == 0 and os.path.isfile(out_path) and os.path.getsize(out_path) > 1024:
                logger.info("Ken Burns: %s (%.1fs, %s, %d bytes)",
                            os.path.basename(out_path), duration_s, motion_type,
                            os.path.getsize(out_path))
                return VideoResult(
                    url=out_path, local_path=out_path,
                    duration_s=duration_s, scene_type=scene_type,
                    model=self.MODEL, cost_usd=0.0,
                    metadata={"motion": motion_type},
                )
            else:
                # 失败时记录详细错误并删除空文件（避免污染 videos/ 目录）
                err_tail = stderr.decode('utf-8', errors='replace')[-1500:] if stderr else ""
                logger.warning("Ken Burns ffmpeg failed: returncode=%s, file_exists=%s, size=%d, stderr=%s",
                               proc.returncode, os.path.isfile(out_path),
                               os.path.getsize(out_path) if os.path.isfile(out_path) else 0,
                               err_tail)
                # 删除空文件或损坏文件
                if os.path.isfile(out_path) and os.path.getsize(out_path) <= 1024:
                    try:
                        os.remove(out_path)
                        logger.info("Removed empty/invalid kenburns file: %s", out_path)
                    except Exception:
                        pass
        except Exception as e:
            logger.error("Ken Burns ffmpeg failed: %s", e)

        # 失败时 local_path 必须为空，让 composer 检测失败后进入 L1/L2 fallback
        # 不要返回 image_path 作为 local_path（会被 checkpoint 误判为视频成功）
        return VideoResult(
            url=image_url or image_path or "",
            local_path="",
            duration_s=duration_s, scene_type=scene_type,
            model=self.MODEL, cost_usd=0.0,
            metadata={"error": "ffmpeg failed, no video generated"},
        )

    @staticmethod
    def _ffmpeg_bin() -> str:
        """Locate ffmpeg binary, prefer imageio-ffmpeg bundled."""
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return "ffmpeg"


def get_video_adapter(scene_type: str) -> VideoAdapter:
    """所有场景统一使用 Seedance i2v（废弃 Ken Burns）。

    借鉴 QinKunming/ai_manju video_pipeline.py：
    所有视频（包括 NORMAL_SCENE）都走 API i2v，用图片作为 first_frame，
    不依赖本地 FFmpeg 生成视频内容（kenburns 在 asyncio 事件循环中
    生成空文件的问题不值得继续修复）。

    scene_type 参数保留仅为向后兼容，不再影响 adapter 选择。
    FFmpegKenBurnsAdapter 类保留但不启用，以备未来需要降级使用。
    """
    return VolcengineVideoAdapter()
