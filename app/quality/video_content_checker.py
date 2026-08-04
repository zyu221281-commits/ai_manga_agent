"""VideoContentChecker — 视频内容质检（视频生成后、合成前）。

在每段视频生成完成后立即执行内容质检：
  1. ffmpeg 抽帧（前/中/后 3 帧）
  2. CLIP 相似度检查（视频帧 vs 源图，所有视频）
  3. VLM 人物完整性检查（仅 KEY_SCENE，复用 qwen-vl-max）
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

# 人物完整性检查 VLM prompt
VLM_CHARACTER_INTEGRITY_PROMPT = """请检查这些从动漫视频中截取的帧是否存在以下内容质量问题：

检查项：
1. character_completeness: 人物是否完整（有无缺胳膊少腿、身体被裁切）
2. face_distortion: 面部是否扭曲变形
3. body_deformation: 肢体是否畸形（多指/缺指/关节异常弯曲）
4. appearance_drift: 角色外观是否与源图差异过大（发色/服饰/体型突变）
5. visual_artifacts: 是否有严重伪影（闪烁/重影/马赛克/花屏）

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
- 严重异常（人物明显残缺/变形）→ severity=major, recommendation=regenerate
"""


@dataclass
class VideoContentResult:
    """视频内容质检结果。"""
    passed: bool = True
    clip_similarity: float = 1.0
    clip_passed: bool = True
    vlm_has_issues: bool = False
    vlm_issues: list[str] = field(default_factory=list)
    vlm_severity: str = "none"
    vlm_error: str = ""
    detail: str = ""
    cost_usd: float = 0.0


class VideoContentChecker:
    """视频内容质检器。

    质检策略：
    - 所有视频：CLIP 相似度检查（视频帧与源图，阈值从 config 读取）
    - KEY_SCENE 额外：VLM 人物完整性检查
    - 任一不通过 → 返回 passed=False
    - API/模型不可用时 → 降级为 passed=True
    """

    def __init__(
        self,
        clip_threshold: Optional[float] = None,
        vlm_model: Optional[str] = None,
    ):
        self.clip_threshold = clip_threshold or settings.VIDEO_CONTENT_CHECK_CLIP_THRESHOLD
        self.vlm_model = vlm_model or settings.VQA_MODEL
        self._enabled = settings.VIDEO_CONTENT_CHECK_ENABLED
        self._vlm_enabled = settings.VIDEO_CONTENT_CHECK_VLM_ENABLED

    async def check_video(
        self,
        video_path: str,
        source_image_path: str,
        scene_type: str = "normal",
        character_name: str = "",
    ) -> VideoContentResult:
        """对生成的视频执行内容质检。

        Args:
            video_path: 生成的视频文件路径
            source_image_path: 源图路径（用于 CLIP 相似度比较）
            scene_type: "key" (KEY_SCENE) 或 "normal" (NORMAL_SCENE)
            character_name: 角色名（用于 VLM prompt 上下文）

        Returns:
            VideoContentResult with passed=True/False
        """
        if not self._enabled:
            return VideoContentResult(passed=True, detail="content check disabled")

        result = VideoContentResult()

        # Step 1: 抽帧
        frames = self._extract_frames(video_path)
        if len(frames) < 2:
            result.passed = False
            result.detail = f"frame extraction failed: only {len(frames)} frames"
            return result

        try:
            # Step 2: CLIP 相似度检查（所有视频）
            if source_image_path and os.path.isfile(source_image_path):
                result.clip_similarity = await self._check_clip_similarity(frames, source_image_path)
                result.clip_passed = result.clip_similarity >= self.clip_threshold
                if not result.clip_passed:
                    result.passed = False
                    result.detail = (
                        f"CLIP similarity {result.clip_similarity:.3f} < threshold {self.clip_threshold}"
                    )
                    return result
            else:
                # 无源图时跳过 CLIP 检查
                result.clip_similarity = 1.0
                result.clip_passed = True

            # Step 3: VLM 人物完整性检查（仅 KEY_SCENE）
            is_key = scene_type in ("key", "KEY_SCENE")
            if is_key and self._vlm_enabled and settings.DASHSCOPE_API_KEY:
                vlm_ok = await self._check_character_integrity(frames, result)
                if not vlm_ok:
                    result.passed = False
                    result.detail = (
                        f"VLM integrity check failed: severity={result.vlm_severity}, "
                        f"issues={result.vlm_issues}"
                    )

        finally:
            # Step 4: 清理临时帧文件
            for fp in frames:
                try:
                    os.remove(fp)
                except Exception:
                    pass

        if result.passed:
            result.detail = f"CLIP={result.clip_similarity:.3f} passed"
        return result

    # ----------------------------------------------------------------
    # Frame extraction
    # ----------------------------------------------------------------

    @staticmethod
    def _extract_frames(video_path: str, num_frames: int = 3) -> list[str]:
        """从视频中抽取 N 帧（均匀分布：30%/50%/70% 位置）。

        Returns:
            临时帧文件路径列表（调用方负责清理）
        """
        dur = VideoContentChecker._probe_duration(video_path)
        if dur < 0.5:
            logger.warning("Video duration too short for frame extraction: %.2fs", dur)
            return []

        positions = [0.30, 0.50, 0.70][:num_frames]
        frames: list[str] = []

        for i, pos in enumerate(positions):
            t = dur * pos
            tmp = tempfile.mktemp(suffix=".jpg", prefix=f"vframe_{i}_")
            try:
                r = subprocess.run(
                    ["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", video_path,
                     "-frames:v", "1", "-q:v", "2", tmp],
                    capture_output=True, timeout=15,
                )
                if r.returncode == 0 and os.path.isfile(tmp) and os.path.getsize(tmp) > 512:
                    frames.append(tmp)
                else:
                    logger.debug("Frame %d extraction failed at %.2fs: %s", i, t, r.stderr[:200])
            except Exception as e:
                logger.debug("Frame %d extraction exception: %s", i, e)

        if len(frames) < 2:
            # 清理失败的帧
            for fp in frames:
                try:
                    os.remove(fp)
                except Exception:
                    pass
            return []

        logger.debug("Extracted %d frames from %s (dur=%.2fs)", len(frames), video_path, dur)
        return frames

    @staticmethod
    def _probe_duration(path: str) -> float:
        """探测视频时长（秒）。"""
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                return float(r.stdout.strip() or "0")
        except Exception:
            pass
        return 0.0

    # ----------------------------------------------------------------
    # CLIP similarity check
    # ----------------------------------------------------------------

    async def _check_clip_similarity(self, frame_paths: list[str], source_image_path: str) -> float:
        """计算视频帧与源图的平均 CLIP 余弦相似度。

        使用 CLIP 模型（ViT-B/32）。
        模型不可用时返回 1.0（降级通过）。
        """
        try:
            from app.quality.character_consistency import _get_clip
            import torch
            from PIL import Image

            model, preprocess = _get_clip()
            device = next(model.parameters()).device

            # 编码源图
            src_img = preprocess(Image.open(source_image_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                src_feat = model.encode_image(src_img)
                src_feat = src_feat / src_feat.norm(dim=-1, keepdim=True)

            # 编码每帧并计算相似度
            similarities: list[float] = []
            for fp in frame_paths:
                try:
                    frame_img = preprocess(Image.open(fp).convert("RGB")).unsqueeze(0).to(device)
                    with torch.no_grad():
                        frame_feat = model.encode_image(frame_img)
                        frame_feat = frame_feat / frame_feat.norm(dim=-1, keepdim=True)
                        sim = float((src_feat @ frame_feat.T).item())
                    similarities.append(sim)
                    del frame_img
                except Exception as e:
                    logger.debug("CLIP encode failed for frame %s: %s", fp, e)

            del src_img, src_feat
            if not similarities:
                return 1.0  # 降级通过
            avg = sum(similarities) / len(similarities)
            return round(avg, 4)
        except Exception as e:
            logger.warning("CLIP similarity check failed (degraded to pass): %s", e)
            return 1.0  # 模型不可用时降级通过

    # ----------------------------------------------------------------
    # VLM character integrity check
    # ----------------------------------------------------------------

    async def _check_character_integrity(
        self, frame_paths: list[str], result: VideoContentResult,
    ) -> bool:
        """VLM 人物完整性检查（仅 KEY_SCENE）。

        Returns:
            True 如果通过（无需重试），False 如果检测到严重问题需要重试。
        """
        from app.services.llm_client import llm_client

        # 构建多图消息：多帧 + prompt
        content_parts: list[dict] = []
        for fp in frame_paths:
            b64_url = self._encode_frame(fp)
            if b64_url:
                content_parts.append({"type": "image_url", "image_url": {"url": b64_url}})

        if not content_parts:
            result.vlm_error = "no frames to encode"
            return True  # 降级通过

        content_parts.append({"type": "text", "text": VLM_CHARACTER_INTEGRITY_PROMPT})

        messages = [{"role": "user", "content": content_parts}]

        try:
            resp = await llm_client.completions(
                messages=messages,
                model=self.vlm_model,
                temperature=0.1,
                max_tokens=512,
                enable_cache=False,
            )
        except Exception as e:
            logger.warning("VLM integrity check API failed (degraded to pass): %s", e)
            result.vlm_error = str(e)
            return True  # API 不可用时降级通过

        content = resp.content or ""
        result.cost_usd = resp.cost_usd

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            first = content.find("{")
            last = content.rfind("}")
            if first != -1 and last != -1 and last > first:
                try:
                    data = json.loads(content[first: last + 1])
                except json.JSONDecodeError:
                    result.vlm_error = f"non-JSON response: {content[:100]}"
                    return True  # 解析失败降级通过
            else:
                result.vlm_error = f"non-JSON response: {content[:100]}"
                return True

        result.vlm_has_issues = bool(data.get("has_issues", False))
        result.vlm_issues = data.get("issues", [])
        result.vlm_severity = data.get("severity", "none")
        recommendation = data.get("recommendation", "pass")

        if result.vlm_has_issues and recommendation == "regenerate":
            logger.info(
                "VLM integrity check: FAIL — severity=%s, issues=%s",
                result.vlm_severity, result.vlm_issues,
            )
            return False

        logger.info("VLM integrity check: PASS")
        return True

    @staticmethod
    def _encode_frame(path: str) -> str:
        """帧图片转 base64 data URL。"""
        try:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return f"data:image/jpeg;base64,{b64}"
        except Exception as e:
            logger.debug("Frame encode failed: %s", e)
            return ""


# 模块级单例
video_content_checker = VideoContentChecker()
