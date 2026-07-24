"""MinIO 客户端封装 — 异步上传媒体资产到 MinIO。

设计要点：
- MinIO SDK 是 sync 的，用 asyncio.to_thread() 包装避免阻塞 event loop
- 上传失败时返回 None（fire-and-forget 语义，由调用方决定是否记录）
- bucket 名复用 docker-compose minio-init 已创建的：images / videos / assets
- 公共读 URL 格式：http://{endpoint}/{bucket}/{object_name}
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


class MinIOClient:
    """异步 MinIO 客户端（基于 sync minio SDK + asyncio.to_thread）。

    Usage:
        client = MinIOClient()
        minio_path = await client.upload_image(local_path="/path/to/img.png")
        # minio_path 形如 "images/img_xxx.png" 或 None（上传失败）
    """

    BUCKET_IMAGES = "images"
    BUCKET_VIDEOS = "videos"
    BUCKET_AUDIO = "assets"  # 复用 assets bucket 存音频

    def __init__(self):
        self._client = None
        self._endpoint = settings.MINIO_ENDPOINT
        self._secure = settings.MINIO_SECURE
        self._public_url_base = self._build_public_url_base()
        # 已确认存在/已创建的 bucket 缓存，避免每次上传都探测
        self._bucket_ready: set[str] = set()

    def _build_public_url_base(self) -> str:
        """构建公共读 URL 前缀。

        docker-compose 中 images bucket 已设为 anonymous download，
        可通过 http://{endpoint}/images/{object_name} 直接访问。
        """
        scheme = "https" if self._secure else "http"
        return f"{scheme}://{self._endpoint}"

    def _get_client(self):
        """懒加载 Minio client（sync）。"""
        if self._client is None:
            from minio import Minio
            endpoint = self._endpoint
            if ":" not in endpoint:
                endpoint = f"{endpoint}:9000"
            self._client = Minio(
                endpoint,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=self._secure,
            )
        return self._client

    def _ensure_bucket(self, bucket: str) -> bool:
        """确保 bucket 存在，不存在则创建。

        幂等：已确认存在的 bucket 直接返回 True（缓存在 _bucket_ready）。
        创建失败时记录 debug 日志（静默，不 WARNING），返回 False。
        """
        if bucket in self._bucket_ready:
            return True
        try:
            client = self._get_client()
            if not client.bucket_exists(bucket):
                client.make_bucket(bucket)
                logger.info("MinIO: bucket '%s' created", bucket)
            self._bucket_ready.add(bucket)
            return True
        except Exception as e:
            # 静默处理：bucket 创建失败不影响本地文件使用，仅 debug 级别日志
            logger.debug("MinIO: ensure_bucket('%s') failed: %s", bucket, e)
            return False

    def _is_no_such_bucket(self, e: Exception) -> bool:
        """判断异常是否为 NoSuchBucket（兼容 minio SDK S3Error 和通用 Exception）。"""
        code = getattr(e, "code", None) or ""
        return "NoSuchBucket" in str(code) or "NoSuchBucket" in str(e)

    def _upload_sync(self, bucket: str, object_name: str, file_path: str, content_type: str) -> bool:
        """同步上传（在 to_thread 中执行）。

        NoSuchBucket 处理：首次上传触发时自动创建 bucket 并重试一次，
        后续命中 _bucket_ready 缓存直接上传，避免日志噪音。
        """
        try:
            client = self._get_client()
            if not os.path.exists(file_path):
                logger.warning("MinIO upload skipped: file not found %s", file_path)
                return False
            try:
                client.fput_object(
                    bucket_name=bucket,
                    object_name=object_name,
                    file_path=file_path,
                    content_type=content_type,
                )
                return True
            except Exception as e:
                if self._is_no_such_bucket(e):
                    # 静默尝试创建 bucket 后重试一次
                    if self._ensure_bucket(bucket):
                        client.fput_object(
                            bucket_name=bucket,
                            object_name=object_name,
                            file_path=file_path,
                            content_type=content_type,
                        )
                        return True
                    # 创建失败：降级为 debug，避免日志噪音
                    logger.debug(
                        "MinIO upload skipped (bucket '%s' not available): %s",
                        bucket, object_name,
                    )
                    return False
                raise  # 其他异常继续向外抛
        except Exception as e:
            logger.warning("MinIO upload failed (%s/%s): %s", bucket, object_name, e)
            return False

    async def _upload(self, bucket: str, object_name: str, file_path: str, content_type: str) -> Optional[str]:
        """异步上传，成功返回 object_name，失败返回 None。"""
        ok = await asyncio.to_thread(
            self._upload_sync, bucket, object_name, file_path, content_type
        )
        if ok:
            return f"{bucket}/{object_name}"
        return None

    # ================================================================
    # Public API
    # ================================================================

    async def upload_image(self, local_path: str, object_name: Optional[str] = None) -> Optional[str]:
        """上传图片到 images bucket。返回 minio_path 或 None。"""
        if not object_name:
            object_name = self._gen_object_name(local_path, "img")
        return await self._upload(
            self.BUCKET_IMAGES, object_name, local_path, "image/png"
        )

    async def upload_video(self, local_path: str, object_name: Optional[str] = None) -> Optional[str]:
        """上传视频到 videos bucket。返回 minio_path 或 None。"""
        if not object_name:
            object_name = self._gen_object_name(local_path, "vid")
        return await self._upload(
            self.BUCKET_VIDEOS, object_name, local_path, "video/mp4"
        )

    async def upload_audio(self, local_path: str, object_name: Optional[str] = None) -> Optional[str]:
        """上传音频到 assets bucket。返回 minio_path 或 None。"""
        if not object_name:
            object_name = self._gen_object_name(local_path, "aud")
        # 根据扩展名确定 content_type
        ext = Path(local_path).suffix.lower()
        ct = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".m4a": "audio/mp4",
            ".aac": "audio/aac",
        }.get(ext, "audio/mpeg")
        return await self._upload(
            self.BUCKET_AUDIO, object_name, local_path, ct
        )

    def public_url(self, minio_path: str) -> str:
        """根据 minio_path 生成公共读 URL。"""
        return f"{self._public_url_base}/{minio_path}"

    @staticmethod
    def _gen_object_name(local_path: str, prefix: str) -> str:
        """根据本地路径生成 MinIO object_name（去重 + 防覆盖）。"""
        import hashlib
        import time
        stem = Path(local_path).stem
        # 用路径 hash + 时间戳保证唯一性
        path_hash = hashlib.md5(local_path.encode()).hexdigest()[:8]
        ts = int(time.time())
        ext = Path(local_path).suffix or ".bin"
        return f"{prefix}_{stem}_{path_hash}_{ts}{ext}"


# 模块级单例
minio_client = MinIOClient()


# ================================================================
# Fire-and-forget 上传 helper（adapter 调用）
# ================================================================

# 保存后台 task 引用，避免被 GC 回收导致上传中断
_background_tasks: set = set()


def fire_and_forget_upload(result_obj, local_path: str, upload_type: str) -> None:
    """启动 fire-and-forget MinIO 上传任务（不阻塞调用方）。

    Args:
        result_obj: ImageResult / VideoResult / TTSResult 对象（需要有 minio_path 属性）
        local_path: 本地文件路径
        upload_type: "image" / "video" / "audio"

    设计要点：
    - 用 asyncio.create_task 启动后台 task，调用方不 await
    - task 内部有 try/except，失败只 log warning，不抛异常
    - 上传成功后自动设置 result_obj.minio_path
    - task 引用保存在 _background_tasks set 中，避免 GC 回收
    - MINIO_UPLOAD_ENABLED=False 时直接跳过
    """
    import asyncio
    from app.core.config import settings

    if not settings.MINIO_UPLOAD_ENABLED or not local_path:
        return

    async def _upload():
        try:
            if upload_type == "image":
                minio_path = await minio_client.upload_image(local_path)
            elif upload_type == "video":
                minio_path = await minio_client.upload_video(local_path)
            elif upload_type == "audio":
                minio_path = await minio_client.upload_audio(local_path)
            else:
                return
            if minio_path:
                result_obj.minio_path = minio_path
                logger.debug("MinIO upload success: %s → %s", local_path, minio_path)
        except Exception as e:
            logger.warning("MinIO %s upload failed (%s): %s", upload_type, local_path, e)

    coro = _upload()
    try:
        task = asyncio.create_task(coro)
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
    except RuntimeError:
        # No event loop running — 关闭未启动的 coroutine 避免 RuntimeWarning，然后跳过上传
        coro.close()
        logger.debug("MinIO upload skipped (no event loop): %s", local_path)
