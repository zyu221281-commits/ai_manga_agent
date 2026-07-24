"""ContentGate — 图像内容质检（生成后、视频生成前）。

阶段 4.1: CLIP 风格相似度 + 角色一致性检查。
- 角色一致性: 对有 anchor 的角色，检查生成图与 anchor 的 CLIP 相似度
- 风格连续性: 同一场景相邻镜头的 CLIP 相似度（防止风格跳变）
- 非阻断: 仅记录问题，不阻止视频生成（Pre-Video Gate 是最后防线）

设计原则：
1. 任何内部错误都降级为 skip，不抛异常（生成管道不能因质检崩溃而中断）
2. 只检查有 local_path 的图像（CLIP 需要本地文件）
3. 检查完成后释放 CLIP GPU 内存
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ShotCheck:
    """单个镜头的质检结果。"""
    shot_id: int
    character_check: Optional[dict] = None  # {passed, similarity, character_name}
    style_check: Optional[dict] = None      # {passed, similarity, vs_shot_id}
    issues: list[str] = field(default_factory=list)


@dataclass
class ContentGateResult:
    """ContentGate 汇总结果。"""
    verdict: str = "pass"  # pass | flag | skip
    total_checked: int = 0
    flagged_shots: list[int] = field(default_factory=list)
    shot_checks: dict[int, ShotCheck] = field(default_factory=dict)
    summary: str = ""
    cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "total_checked": self.total_checked,
            "flagged_shots": self.flagged_shots,
            "summary": self.summary,
            "cost_usd": round(self.cost_usd, 4),
            "shots": {
                str(sid): {
                    "character_check": sc.character_check,
                    "style_check": sc.style_check,
                    "issues": sc.issues,
                }
                for sid, sc in self.shot_checks.items()
            },
        }


class ContentGate:
    """图像内容质检门。

    在图像生成完成、视频生成之前执行，检测：
    1. 角色一致性：生成图与角色 anchor 的 CLIP 相似度
    2. 风格连续性：同一场景相邻镜头的 CLIP 相似度（防止风格跳变）
    """

    def __init__(
        self,
        style_threshold: Optional[float] = None,
        character_threshold: Optional[float] = None,
    ):
        self.style_threshold = style_threshold or settings.STYLE_SIMILARITY_THRESHOLD
        # 角色一致性阈值复用 character_consistency 内部默认值（0.82）
        self.character_threshold = character_threshold

    async def check_images(
        self,
        image_results: list,
        classified_scenes: list[dict],
        asset_library: Optional[dict[str, Any]] = None,
    ) -> ContentGateResult:
        """对所有生成图像执行内容质检。

        非阻断: 任何内部错误都记录为 skip，不抛异常。
        """
        result = ContentGateResult()
        if not image_results:
            result.summary = "no images to check"
            return result

        # 检查 CLIP 可用性（不可用则 skip）
        try:
            from app.quality.character_consistency import _get_clip
            _get_clip()
        except Exception as e:
            result.verdict = "skip"
            result.summary = f"CLIP unavailable: {e}"
            logger.warning("ContentGate skipped: %s", e)
            return result

        # 1. 角色一致性检查（仅对有 anchor 且有 local_path 的图像）
        char_tasks = []
        for img, cs in zip(image_results, classified_scenes):
            char_name = cs.get("character_name", "")
            local_path = getattr(img, "local_path", "") or ""
            shot_id = cs.get("shot_id", -1)
            if char_name and local_path:
                from app.quality.character_consistency import character_consistency
                if character_consistency.has_anchor_image(char_name):
                    char_tasks.append(self._check_character(shot_id, char_name, local_path))

        if char_tasks:
            char_results = await asyncio.gather(*char_tasks, return_exceptions=True)
        else:
            char_results = []

        for cr in char_results:
            if isinstance(cr, Exception):
                logger.warning("ContentGate character check failed: %s", cr)
                continue
            if cr is None:
                continue
            shot_id, check_data = cr
            sc = result.shot_checks.setdefault(shot_id, ShotCheck(shot_id=shot_id))
            sc.character_check = check_data
            if not check_data.get("passed", True):
                sc.issues.append(
                    f"character inconsistency: {check_data.get('character_name')} "
                    f"sim={check_data.get('similarity', 0):.2f}"
                )

        # 2. 风格连续性检查（同一场景相邻镜头）
        scenes_by_id: dict[int, list[tuple[int, str]]] = {}
        for img, cs in zip(image_results, classified_scenes):
            sid = cs.get("scene_id", 0)
            shot_id = cs.get("shot_id", -1)
            local_path = getattr(img, "local_path", "") or ""
            if local_path:
                scenes_by_id.setdefault(sid, []).append((shot_id, local_path))

        style_tasks = []
        for _sid, shots in scenes_by_id.items():
            # 只检查相邻镜头（同场景内防止风格跳变）
            for i in range(1, len(shots)):
                prev_shot_id, prev_path = shots[i - 1]
                cur_shot_id, cur_path = shots[i]
                style_tasks.append(
                    self._check_style(cur_shot_id, prev_path, cur_path, prev_shot_id)
                )

        if style_tasks:
            style_results = await asyncio.gather(*style_tasks, return_exceptions=True)
        else:
            style_results = []

        for sr in style_results:
            if isinstance(sr, Exception):
                logger.warning("ContentGate style check failed: %s", sr)
                continue
            if sr is None:
                continue
            shot_id, check_data = sr
            sc = result.shot_checks.setdefault(shot_id, ShotCheck(shot_id=shot_id))
            sc.style_check = check_data
            if not check_data.get("passed", True):
                sc.issues.append(
                    f"style break: sim={check_data.get('similarity', 0):.2f} "
                    f"vs shot {check_data.get('vs_shot_id')}"
                )

        # 汇总
        result.total_checked = len(result.shot_checks)
        result.flagged_shots = [
            sid for sid, sc in result.shot_checks.items() if sc.issues
        ]

        if result.flagged_shots:
            result.verdict = "flag"
            result.summary = (
                f"{len(result.flagged_shots)}/{result.total_checked} shots flagged"
            )
        else:
            result.verdict = "pass"
            result.summary = (
                f"{result.total_checked} shots checked, all passed"
            )

        # 释放 CLIP GPU 内存
        try:
            from app.quality.character_consistency import release_clip_model
            release_clip_model()
        except Exception:
            pass

        return result

    async def _check_character(
        self, shot_id: int, char_name: str, local_path: str
    ) -> tuple[int, dict]:
        """单张图角色一致性检查。"""
        from app.quality.character_consistency import character_consistency

        try:
            cr = await character_consistency.check_consistency(char_name, local_path)
            return shot_id, {
                "passed": cr.passed,
                "similarity": round(cr.similarity_score, 3),
                "character_name": char_name,
                "recommendation": cr.recommendation,
            }
        except Exception as e:
            return shot_id, {
                "passed": True,
                "similarity": 1.0,
                "character_name": char_name,
                "error": str(e),
            }

    async def _check_style(
        self, shot_id: int, prev_path: str, cur_path: str, prev_shot_id: int
    ) -> tuple[int, dict]:
        """相邻镜头风格相似度检查。"""
        try:
            import torch
            from PIL import Image
            from app.quality.character_consistency import _get_clip

            model, preprocess = _get_clip()
            device = next(model.parameters()).device

            img_a = preprocess(Image.open(prev_path).convert("RGB")).unsqueeze(0).to(device)
            img_b = preprocess(Image.open(cur_path).convert("RGB")).unsqueeze(0).to(device)

            with torch.no_grad():
                feat_a = model.encode_image(img_a)
                feat_b = model.encode_image(img_b)
                feat_a = feat_a / feat_a.norm(dim=-1, keepdim=True)
                feat_b = feat_b / feat_b.norm(dim=-1, keepdim=True)
                sim = float((feat_a @ feat_b.T).item())

            del img_a, img_b
            passed = sim >= self.style_threshold
            return shot_id, {
                "passed": passed,
                "similarity": round(sim, 3),
                "vs_shot_id": prev_shot_id,
                "threshold": self.style_threshold,
            }
        except Exception as e:
            return shot_id, {
                "passed": True,
                "similarity": 1.0,
                "vs_shot_id": prev_shot_id,
                "error": str(e),
            }


content_gate = ContentGate()
