"""VQAChecker — 轻量视觉问答质检（仅 KEY_SCENE）。

阶段 4.2: 使用 qwen-vl-max 检查关键镜头的物理异常。
- 仅检查 KEY_SCENE 图像（成本控制，单集最多 VQA_MAX_IMAGES 张）
- 检查项: 多指/缺指、肢体畸形、物理不可能、水印、面部扭曲
- 非阻断: LLM 失败 → 该 shot 记录 error，不阻止视频生成

设计原则：
1. 成本控制：只查 KEY_SCENE，限制单集检查数量
2. 非阻断：任何异常都降级为 skip 或记录 error
3. 支持本地文件（base64）和 URL 两种图像来源
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.services.video_strategy import KEY_SCENE

logger = logging.getLogger(__name__)


VQA_PROMPT = """请检查这张动漫图像是否存在以下物理异常（只看 KEY_SCENE 关键镜头）：

检查项：
1. extra_fingers: 多指/缺指/手指畸形
2. deformed_anatomy: 肢体畸形/身体结构错误
3. physical_impossibility: 物理不可能（如扭曲的关节、穿模）
4. watermark_text: 水印或乱码文字
5. face_distortion: 面部扭曲

输出严格 JSON:
{
  "has_issues": false,
  "issues": ["问题描述"],
  "severity": "none|minor|major",
  "recommendation": "pass|regenerate"
}

判断规则:
- 无异常 → has_issues=false, severity=none, recommendation=pass
- 轻微异常（不易察觉）→ severity=minor, recommendation=pass
- 严重异常（明显畸形）→ severity=major, recommendation=regenerate
"""


@dataclass
class VQAShotResult:
    """单张 KEY_SCENE 的 VQA 质检结果。"""
    shot_id: int
    has_issues: bool = False
    issues: list[str] = field(default_factory=list)
    severity: str = "none"
    recommendation: str = "pass"
    error: str = ""
    cost_usd: float = 0.0


@dataclass
class VQAResult:
    """VQA 质检汇总结果。"""
    verdict: str = "pass"  # pass | flag | skip
    total_checked: int = 0
    flagged_shots: list[int] = field(default_factory=list)
    shot_results: dict[int, VQAShotResult] = field(default_factory=dict)
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
                    "has_issues": sr.has_issues,
                    "issues": sr.issues,
                    "severity": sr.severity,
                    "recommendation": sr.recommendation,
                    "error": sr.error,
                }
                for sid, sr in self.shot_results.items()
            },
        }


class VQAChecker:
    """轻量 VQA 质检器（仅 KEY_SCENE）。

    使用 qwen-vl-max 多模态 LLM 检查关键镜头的物理异常。
    仅检查 KEY_SCENE 图像以控制成本。
    """

    def __init__(
        self,
        model: Optional[str] = None,
        max_images: Optional[int] = None,
    ):
        self.model = model or settings.VQA_MODEL
        self.max_images = max_images or settings.VQA_MAX_IMAGES

    async def check_key_scenes(
        self,
        image_results: list,
        classified_scenes: list[dict],
    ) -> VQAResult:
        """对 KEY_SCENE 图像执行 VQA 质检。

        非阻断: API key 未配置或无可用图像 → skip，不阻止流程。
        """
        result = VQAResult()

        # 前置检查: API key
        if not settings.DASHSCOPE_API_KEY:
            result.verdict = "skip"
            result.summary = "DASHSCOPE_API_KEY not configured"
            logger.info("VQA skipped: DASHSCOPE_API_KEY not set")
            return result

        # 收集 KEY_SCENE 图像
        key_shots: list[tuple[int, str]] = []  # (shot_id, image_source)
        for img, cs in zip(image_results, classified_scenes):
            if cs.get("type") != KEY_SCENE:
                continue
            shot_id = cs.get("shot_id", -1)
            local_path = getattr(img, "local_path", "") or ""
            url = getattr(img, "url", "") or ""
            source = ""
            if local_path and Path(local_path).is_file():
                try:
                    source = self._encode_local_image(local_path)
                except Exception as e:
                    logger.warning("VQA: failed to encode %s: %s", local_path, e)
            elif url:
                source = url
            if source:
                key_shots.append((shot_id, source))

        if not key_shots:
            result.verdict = "skip"
            result.summary = "no KEY_SCENE images with valid source"
            return result

        # 成本控制: 限制检查数量
        if len(key_shots) > self.max_images:
            logger.info(
                "VQA: %d KEY_SCENE images, checking first %d (cost cap)",
                len(key_shots), self.max_images,
            )
            key_shots = key_shots[: self.max_images]

        # 并行检查
        import asyncio
        tasks = [self._check_one(sid, src) for sid, src in key_shots]
        shot_results = await asyncio.gather(*tasks, return_exceptions=True)

        for sr in shot_results:
            if isinstance(sr, Exception):
                logger.warning("VQA check failed: %s", sr)
                continue
            if sr is None:
                continue
            result.shot_results[sr.shot_id] = sr
            result.cost_usd += sr.cost_usd

        result.total_checked = len(result.shot_results)
        result.flagged_shots = [
            sid
            for sid, sr in result.shot_results.items()
            if sr.has_issues and sr.recommendation == "regenerate"
        ]

        if result.flagged_shots:
            result.verdict = "flag"
            result.summary = (
                f"{len(result.flagged_shots)}/{result.total_checked} KEY scenes flagged"
            )
        else:
            result.verdict = "pass"
            result.summary = (
                f"{result.total_checked} KEY scenes checked, all passed"
            )

        return result

    async def _check_one(self, shot_id: int, image_source: str) -> VQAShotResult:
        """单张图 VQA 检查。"""
        from app.services.llm_client import llm_client

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_source}},
                    {"type": "text", "text": VQA_PROMPT},
                ],
            },
        ]

        try:
            resp = await llm_client.completions(
                messages=messages,
                model=self.model,
                temperature=0.1,
                max_tokens=512,
                enable_cache=False,
            )
        except Exception as e:
            logger.warning("VQA LLM call failed (shot %d): %s", shot_id, e)
            return VQAShotResult(shot_id=shot_id, error=str(e))

        content = resp.content or ""
        cost = resp.cost_usd

        # 尝试解析 JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # 提取 {...} 块
            first = content.find("{")
            last = content.rfind("}")
            if first != -1 and last != -1 and last > first:
                try:
                    data = json.loads(content[first : last + 1])
                except json.JSONDecodeError:
                    return VQAShotResult(
                        shot_id=shot_id,
                        error=f"non-JSON response: {content[:100]}",
                        cost_usd=cost,
                    )
            else:
                return VQAShotResult(
                    shot_id=shot_id,
                    error=f"non-JSON response: {content[:100]}",
                    cost_usd=cost,
                )

        return VQAShotResult(
            shot_id=shot_id,
            has_issues=bool(data.get("has_issues", False)),
            issues=data.get("issues", []),
            severity=data.get("severity", "none"),
            recommendation=data.get("recommendation", "pass"),
            cost_usd=cost,
        )

    @staticmethod
    def _encode_local_image(path: str) -> str:
        """本地图片转 base64 data URL。"""
        ext = Path(path).suffix.lower().lstrip(".")
        mime = {
            "jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp",
        }.get(ext, "png")
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"data:image/{mime};base64,{b64}"


vqa_checker = VQAChecker()
