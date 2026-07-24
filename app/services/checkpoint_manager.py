"""Checkpoint Manager — 断点续传基础设施。

每生成 1 个资源（图片/视频/TTS）成功后立即写入 checkpoint。
当管道崩溃或部分失败时，可从 checkpoint 恢复已生成的资源，避免重复生成。

持久化方式：JSON 文件（output/checkpoints/{episode_id}_{stage}.json）
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent / "output" / "checkpoints"


class CheckpointManager:
    """管理生成资源的断点续传。

    Usage:
        ckpt = CheckpointManager(episode_id="ep001")

        # 保存单个 shot 的生成结果
        ckpt.save("image", shot_id=3, result={"url": "http://...", "local_path": "/tmp/..."})

        # 检查某个 shot 是否已生成
        if ckpt.has("image", shot_id=3):
            img = ckpt.load("image", shot_id=3)

        # 加载某阶段所有已生成的结果
        results = ckpt.load_all("image")

        # 清理（整集完成后）
        ckpt.clear()
    """

    def __init__(self, episode_id: str):
        self.episode_id = episode_id
        self._dir = CHECKPOINT_DIR / episode_id
        os.makedirs(self._dir, exist_ok=True)

    def _path(self, stage: str) -> Path:
        return self._dir / f"{stage}.json"

    def _read(self, stage: str) -> dict[str, Any]:
        """读取某阶段的 checkpoint 文件。"""
        p = self._path(stage)
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Checkpoint read failed (%s/%s): %s", self.episode_id, stage, e)
            return {}

    def _write(self, stage: str, data: dict[str, Any]) -> None:
        """写入某阶段的 checkpoint 文件。"""
        p = self._path(stage)
        try:
            p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("Checkpoint write failed (%s/%s): %s", self.episode_id, stage, e)

    def save(self, stage: str, shot_id: int, result: dict[str, Any]) -> None:
        """保存单个 shot 的生成结果到 checkpoint。

        Args:
            stage: "image" / "video" / "tts"
            shot_id: 分镜 ID
            result: 生成结果（url, local_path, duration_s 等）
        """
        data = self._read(stage)
        data[str(shot_id)] = {
            **result,
            "_saved_at": datetime.now().isoformat(),
        }
        self._write(stage, data)
        logger.debug("Checkpoint saved: %s/%s/shot_%d", self.episode_id, stage, shot_id)

    def has(self, stage: str, shot_id: int) -> bool:
        """检查某个 shot 是否已有 checkpoint。"""
        return str(shot_id) in self._read(stage)

    def load(self, stage: str, shot_id: int) -> Optional[dict[str, Any]]:
        """加载某个 shot 的 checkpoint。"""
        return self._read(stage).get(str(shot_id))

    def load_all(self, stage: str) -> dict[str, dict[str, Any]]:
        """加载某阶段所有已生成的 shot 结果。

        Returns:
            {shot_id_str: result_dict, ...}
        """
        return self._read(stage)

    def get_completed_shot_ids(self, stage: str) -> set[int]:
        """获取某阶段已完成的 shot ID 集合。"""
        return {int(k) for k in self._read(stage).keys()}

    def remove(self, stage: str, shot_id: int) -> None:
        """删除某个 shot 的 checkpoint（重试时使用）。"""
        data = self._read(stage)
        data.pop(str(shot_id), None)
        self._write(stage, data)

    def clear(self) -> None:
        """清理该 episode 的所有 checkpoint（整集完成后调用）。"""
        import shutil
        if self._dir.exists():
            shutil.rmtree(self._dir, ignore_errors=True)
            logger.info("Checkpoint cleared for episode %s", self.episode_id)

    def clear_stage(self, stage: str) -> None:
        """清理某阶段的 checkpoint。"""
        p = self._path(stage)
        if p.exists():
            p.unlink()
            logger.debug("Checkpoint cleared: %s/%s", self.episode_id, stage)

    @staticmethod
    def cleanup_all():
        """清理所有 checkpoint（定期维护用）。"""
        if CHECKPOINT_DIR.exists():
            import shutil
            shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
            logger.info("All checkpoints cleaned up")
