"""Composer Agent - image, video, TTS, subtitles, BGM, cover generation and FFmpeg composition.

Input: Writer output + AssetManager asset library.
Output: EpisodeAsset (final video + subtitles + BGM + cover + AI content label).
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Optional

from app.services.checkpoint_manager import CheckpointManager

from app.agents.base import BaseAgent, AgentResult
from app.resilience.adapters.image_adapter import get_image_adapter, ImageResult
from app.resilience.adapters.video_adapter import get_video_adapter, VideoResult
from app.resilience.adapters.tts_adapter import get_tts_adapter, TTSResult
from app.services.video_strategy import classify_scene, KEY_SCENE, NORMAL_SCENE
from app.services.episode_compositor import compose_episode

from app.quality.character_consistency import character_consistency
from app.quality.subtitle_generator import SubtitleGenerator
from app.quality.content_gate import content_gate
from app.quality.vqa_checker import vqa_checker
logger = logging.getLogger(__name__)



def _extract_main_character(scene, script):
    """Extract main character name from a scene."""
    for d in scene.get("dialogue", []):
        if isinstance(d, dict) and d.get("character"):
            return d["character"]
    chars = script.get("characters", [])
    if chars:
        return chars[0].get("name", "") if isinstance(chars[0], dict) else str(chars[0])
    return ""


# BGM_MOOD_MAP 已移除：Seed Audio 1.0 在生成对白时同步产出背景音，
# 不再需要独立的 BGM 选取与混合步骤。


class ComposerAgent(BaseAgent):
    """Composer Agent: turns script text into final video assets."""

    agent_name = "composer"

    async def execute(
        self,
        script: dict[str, Any],
        storyboard: list[dict],
        image_prompts: list[dict],
        asset_library: dict[str, Any],
        key_scene_ratio: Optional[float] = None,
    ) -> AgentResult:
        """Run the full production pipeline.

        Returns EpisodeAsset with images, videos, audio, subtitles,
        BGM, covers, final video path, and AI content label metadata.
        """
        from app.core.config import settings
        ratio = key_scene_ratio if key_scene_ratio is not None else settings.VIDEO_KEY_SCENE_RATIO
        episode_id = self.episode_id or "ep_001"

        total_cost = 0.0
        metadata: dict[str, Any] = {}

        # 1. Scene classification
        classified_scenes = self._classify_scenes(storyboard, image_prompts, script)
        metadata["key_scenes"] = sum(1 for s in classified_scenes if s["type"] == KEY_SCENE)
        metadata["normal_scenes"] = sum(1 for s in classified_scenes if s["type"] == NORMAL_SCENE)
        logger.info("Scene classification: %d key, %d normal", metadata["key_scenes"], metadata["normal_scenes"])

        # 2. Image + Audio in parallel (they share no dependencies)
        image_task = asyncio.create_task(
            self._generate_images(classified_scenes, asset_library)
        )
        audio_task = asyncio.create_task(
            self._generate_audio(script, asset_library, storyboard)
        )

        # Wait for images (video depends on them)
        image_results = await image_task
        total_cost += sum(img.cost_usd for img in image_results)
        metadata["images_generated"] = len(image_results)
        logger.info("Images: %d generated", len(image_results))

        # Wait for audio (TTS 完成后才能拿到真实时长，用于驱动视频生成时长)
        # TTS 通常比图像快，await 几乎不增加等待时间
        audio_segments, scene_subtitle_data = await audio_task
        total_cost += sum(a["result"].cost_usd for a in audio_segments if a["result"])
        metadata["audio_segments"] = len(audio_segments)
        logger.info("TTS: %d segments generated (shot-aligned, real duration available)", len(audio_segments))

        # 构建每个 shot 的真实 TTS 时长（音画对齐的关键：用真实时长驱动视频生成）
        shot_durations: dict[int, float] = {}
        for a in audio_segments:
            tts = a.get("result")
            if not tts or not tts.duration_s:
                continue
            sid = a.get("shot_id")
            if sid is not None:
                shot_durations[sid] = shot_durations.get(sid, 0.0) + tts.duration_s

        # Pre-Video Gate: abort if too many image failures
        if not self._pre_video_gate(image_results, classified_scenes):
            return AgentResult(
                success=False,
                error="Pre-video gate: image failure rate exceeds threshold",
            )

        # ContentGate + VQA: 内容质检（非阻断，仅记录问题，不阻止视频生成）
        # 阶段4.1: CLIP 风格相似度 + 角色一致性检查
        if settings.CONTENT_GATE_ENABLED:
            try:
                gate_result = await content_gate.check_images(
                    image_results, classified_scenes, asset_library
                )
                metadata["content_gate"] = gate_result.to_dict()
                if gate_result.flagged_shots:
                    logger.warning(
                        "ContentGate flagged %d shots: %s",
                        len(gate_result.flagged_shots), gate_result.flagged_shots,
                    )
            except Exception as e:
                logger.warning("ContentGate error (non-blocking): %s", e)
                metadata["content_gate"] = {"verdict": "error", "error": str(e)}

        # 阶段4.2: 轻量 VQA（仅 KEY_SCENE 物理异常检查）
        if settings.VQA_ENABLED:
            try:
                vqa_result = await vqa_checker.check_key_scenes(
                    image_results, classified_scenes
                )
                metadata["vqa_check"] = vqa_result.to_dict()
                if vqa_result.flagged_shots:
                    logger.warning(
                        "VQA flagged %d KEY scenes: %s",
                        len(vqa_result.flagged_shots), vqa_result.flagged_shots,
                    )
            except Exception as e:
                logger.warning("VQA error (non-blocking): %s", e)
                metadata["vqa_check"] = {"verdict": "error", "error": str(e)}

        # 3. Video generation (用 TTS 真实时长驱动视频时长，实现音画对齐)
        video_segments = await self._generate_videos(
            classified_scenes, image_results, ratio, shot_durations=shot_durations,
        )
        total_cost += sum(v.cost_usd for v in video_segments)
        metadata["videos_generated"] = len(video_segments)
        logger.info("Videos: %d generated", len(video_segments))

        # 5. Subtitles
        subtitles = scene_subtitle_data
        metadata["subtitle_entries"] = len(subtitles)

        # 6. BGM 已整合到 Seed Audio 输出中，无需独立选取（删除 _select_bgm）

        # 7. Cover selection
        covers = self._select_cover_candidates(image_results)
        metadata["covers"] = covers

       # 8. FFmpeg final composition（不再混合 BGM，Seed Audio 已含背景音）
        compose_result = await self._compose_final_video(
            video_segments, audio_segments, subtitles, "",
            classified_scenes, episode_id, bgm_path="",
       )
        metadata["composition"] = compose_result

        # Add previously lost auto-narration TTS cost (O10 fix)
        total_cost += compose_result.get("extra_tts_cost", 0.0)

        # 9. AI content label metadata
        ai_label = self._build_ai_label(script, metadata, total_cost)

        episode_asset = {
            "episode_id": episode_id,
            "script": script,
            "storyboard": storyboard,
            "image_prompts": image_prompts,
            "images": [self._image_to_dict(img) for img in image_results],
          "video_segments": [self._video_to_dict(v, i) for i, v in enumerate(video_segments)],
            "audio_segments": [self._tts_to_dict(a["result"]) for a in audio_segments if a["result"]],
            "subtitles": subtitles,
            "bgm_track": "",  # BGM 已整合到 Seed Audio 输出，不再独立选取
            "covers": covers,
            "final_video_path": compose_result.get("final_video_path", ""),
            "ai_label": ai_label,
            "cost_usd": round(total_cost, 4),
            "metadata": metadata,
        }

        logger.info(
            "Composer finished: %d images, %d videos, %d audio, cost=%.4f USD",
            metadata["images_generated"], metadata["videos_generated"],
            metadata["audio_segments"], total_cost,
        )

        return AgentResult(
            success=True,
            data=episode_asset,
            cost_usd=total_cost,
            metadata=metadata,
        )

    # -- 1. Scene classification --

    def _classify_scenes(
        self, storyboard: list[dict], image_prompts: list[dict], script: dict[str, Any]
    ) -> list[dict]:
        prompts_by_shot = {p.get("shot_id"): p for p in image_prompts}
        scenes = script.get("scenes", [])
        classified = []
        for shot in storyboard:
            shot_id = shot.get("shot_id")
            scene_id = shot.get("scene_id", 0)
            prompt_data = prompts_by_shot.get(shot_id, {})
            explicit_type = prompt_data.get("scene_type", "")
            scene_data = next((s for s in scenes if s.get("scene_id") == scene_id), {})
            if explicit_type in (KEY_SCENE, NORMAL_SCENE):
                scene_type = explicit_type
            else:
                scene_type = classify_scene(scene_data)
            classified.append({
                "shot_id": shot_id, "scene_id": scene_id, "type": scene_type,
                "shot": shot, "prompt_data": prompt_data, "scene_data": scene_data,
                "character_name": _extract_main_character(scene_data, script),
            })
        return classified

   # -- 2. Batch image generation --

    async def _generate_images(
        self, classified_scenes: list[dict], asset_library: dict[str, Any]
    ) -> list[ImageResult]:
        """Generate images with parallelism + 4-layer defense + checkpoint resume.

        Phase 1: Parallel first-shot of every scene (no ref dependency).
        Phase 2: Parallel remaining shots (using first shot as ref, Semaphore controlled).

        Defense layers:
          L0: adapter-level retry (handled in image_adapter)
          L1: retry without ref_image if first attempt fails
          L2: NORMAL_SCENE → reuse first scene image if all attempts fail
          L3: circuit breaker at >30% failure rate → graceful degradation (no raise)

        Checkpoint:
          每张图成功后立即写入 checkpoint，管道崩溃后重启可跳过已生成的 shot。
        """
        adapter = get_image_adapter()
        style_template = asset_library.get("style_template", {})
        global_suffix = style_template.get("global_prompt_suffix", "")

        # 初始化 checkpoint
        ckpt = CheckpointManager(self.episode_id or "unknown")
        completed_ids = ckpt.get_completed_shot_ids("image")
        if completed_ids:
            logger.info("Image checkpoint found: %d shots already completed", len(completed_ids))

        # Group shots by scene_id
        scenes_by_id: dict[int, list[dict]] = {}
        for cs in classified_scenes:
            sid = cs.get("scene_id", 0)
            scenes_by_id.setdefault(sid, []).append(cs)

        # Phase 1: Parallel first-shot generation for all scenes
        async def _gen_first_shot(scene_id: int, shots: list[dict]) -> tuple[int, ImageResult]:
            shot_id = shots[0].get("shot_id", 0)
            # Checkpoint 命中：跳过生成
            if shot_id in completed_ids:
                saved = ckpt.load("image", shot_id)
                if saved and saved.get("url"):
                    local_path = saved.get("local_path", "")
                    # 校验 local_path 有效性：如果 local_path 非空，文件必须存在
                    # 防止文件被删除后仍使用过期 checkpoint
                    if not local_path or os.path.isfile(local_path):
                        logger.info("Checkpoint hit: shot %d skipped", shot_id)
                        return scene_id, ImageResult(
                            url=saved["url"], local_path=local_path,
                            prompt=saved.get("prompt", ""), width=1080, height=1920,
                            model=adapter.MODEL, cost_usd=0.0,
                        )
                    else:
                        logger.warning(
                            "Image checkpoint invalid (shot %d): local_path=%s not found; removing",
                            shot_id, local_path,
                        )
                        ckpt.remove("image", shot_id)
            return scene_id, await self._gen_single_image(
                shots[0], adapter, asset_library, global_suffix, ref_url="",
            )

        first_tasks = [_gen_first_shot(sid, shots) for sid, shots in scenes_by_id.items()]
        first_raw = await asyncio.gather(*first_tasks, return_exceptions=True)

        scene_first: dict[int, ImageResult] = {}
        for result in first_raw:
            if isinstance(result, Exception):
                logger.error("First-shot generation failed: %s", result)
                continue
            sid, img = result
            scene_first[sid] = img
            # 写入 checkpoint
            shot_id = next((s[0].get("shot_id", 0) for s_id, s in scenes_by_id.items() if s_id == sid), 0)
            if img and img.url:
                ckpt.save("image", shot_id, {
                    "url": img.url, "local_path": img.local_path or "", "prompt": img.prompt,
                })

        # Phase 2: Parallel per-scene remaining shots (using first shot as ref)
        # 收集需要 Phase 2 生成的 shot（i > 0 且无 checkpoint 命中）
        phase2_specs: list[tuple[dict, int]] = []  # (cs, scene_id)
        for scene_id, shots in sorted(scenes_by_id.items()):
            for i, cs in enumerate(shots):
                if i == 0:
                    continue  # Phase 1 已处理
                shot_id = cs.get("shot_id", 0)
                # Checkpoint 命中：跳过生成（校验 local_path 有效性）
                if shot_id in completed_ids:
                    saved = ckpt.load("image", shot_id)
                    if saved and saved.get("url"):
                        local_path = saved.get("local_path", "")
                        if not local_path or os.path.isfile(local_path):
                            continue
                        # local_path 非空但文件不存在：删除 checkpoint，重新生成
                        logger.warning(
                            "Image checkpoint invalid (shot %d): local_path=%s not found; removing",
                            shot_id, local_path,
                        )
                        ckpt.remove("image", shot_id)
                phase2_specs.append((cs, scene_id))

        # 并行生成 Phase 2 shots（Semaphore 控制并发，避免 rate limit）
        sem = asyncio.Semaphore(8)

        async def _gen_phase2_shot(cs: dict, scene_id: int) -> ImageResult:
            ref = scene_first.get(scene_id)
            ref_url = ref.url if ref and ref.url else ""
            async with sem:
                return await self._gen_single_image(
                    cs, adapter, asset_library, global_suffix, ref_url=ref_url,
                )

        phase2_raw = await asyncio.gather(
            *[_gen_phase2_shot(cs, sid) for cs, sid in phase2_specs],
            return_exceptions=True,
        )

        # 构建 Phase 2 结果映射 + 写入 checkpoint
        phase2_results: dict[int, Optional[ImageResult]] = {}
        for (cs, _), result in zip(phase2_specs, phase2_raw):
            shot_id = cs.get("shot_id", 0)
            if isinstance(result, Exception):
                logger.error("Phase 2 shot %d generation failed: %s", shot_id, result)
                phase2_results[shot_id] = None
            elif result and result.url:
                phase2_results[shot_id] = result
                ckpt.save("image", shot_id, {
                    "url": result.url, "local_path": result.local_path or "", "prompt": result.prompt,
                })
            else:
                phase2_results[shot_id] = None

        # 按 classified_scenes 顺序组装最终结果（含 L2 fallback + L3 熔断）
        all_results: list[ImageResult] = []
        total_shots = len(classified_scenes)
        failures = 0
        failed_shots: list[int] = []

        for cs in classified_scenes:
            shot_id = cs.get("shot_id", 0)
            scene_id = cs.get("scene_id", 0)
            shots = scenes_by_id.get(scene_id, [])
            is_first = bool(shots) and shots[0].get("shot_id") == shot_id

            img: Optional[ImageResult] = None
            if is_first:
                # Phase 1 结果
                img = scene_first.get(scene_id)
            elif shot_id in completed_ids:
                # Checkpoint 命中（校验 local_path 有效性）
                saved = ckpt.load("image", shot_id)
                if saved and saved.get("url"):
                    local_path = saved.get("local_path", "")
                    if not local_path or os.path.isfile(local_path):
                        img = ImageResult(
                            url=saved["url"], local_path=local_path,
                            prompt=saved.get("prompt", ""), width=1080, height=1920,
                            model=adapter.MODEL, cost_usd=0.0,
                        )
                    else:
                        # local_path 非空但文件不存在：删除 checkpoint，img 留空触发 fallback
                        logger.warning(
                            "Image checkpoint invalid (shot %d): local_path=%s not found; removing",
                            shot_id, local_path,
                        )
                        ckpt.remove("image", shot_id)
            else:
                # Phase 2 结果
                img = phase2_results.get(shot_id)

            if img and img.url:
                all_results.append(img)
            else:
                failures += 1
                failed_shots.append(shot_id)
                # L2: NORMAL scenes fall back to first scene image
                first_img = scene_first.get(scene_id)
                if cs.get("type") == NORMAL_SCENE and first_img and first_img.url:
                    fallback = ImageResult(
                        url=first_img.url,
                        local_path=first_img.local_path or "",
                        prompt=first_img.prompt,
                        width=1080, height=1920,
                        model=adapter.MODEL, cost_usd=0.0,
                    )
                    all_results.append(fallback)
                    logger.warning(
                        "Shot %s image failed, using scene %d first image as fallback",
                        shot_id, scene_id,
                    )
                else:
                    all_results.append(ImageResult(
                        url="", prompt=cs.get("prompt_data", {}).get("prompt", ""),
                        width=1080, height=1920,
                        model=adapter.MODEL, cost_usd=0.0,
                    ))

        # L3: Circuit breaker — 优雅降级而非 raise
        if total_shots > 0 and failures / total_shots > 0.3:
            logger.error(
                "Image failure rate %d/%d (%.0f%%) exceeds 30%% threshold — "
                "graceful degradation: %d shots failed %s, continuing with %d valid",
                failures, total_shots, failures / total_shots * 100,
                failures, failed_shots, total_shots - failures,
            )
            # 不再 raise RuntimeError，而是继续流程
            # Pre-Video Gate 会检查图片有效性，KEY_SCENE 缺图会在那里被拦截

        return all_results

    async def _gen_single_image(
        self,
        cs: dict,
        adapter,
        asset_library: dict[str, Any],
        global_suffix: str,
        ref_url: str = "",
    ) -> ImageResult:
        """Generate a single image with L0 adapter retry + L1 no-ref retry.

        L0: adapter.generate() already has internal retries in FluxAdapter.
        L1: if first attempt with ref_image fails, retry without ref_image.
        """
        prompt_data = cs["prompt_data"]
        scene_data = cs.get("scene_data", {})
        shot_id = cs.get("shot_id", "?")
        prompt = prompt_data.get("prompt", "anime comic style, high quality")
        if global_suffix and global_suffix not in prompt:
            prompt += ", " + global_suffix

        # Inject scene-level visual context
        vc = scene_data.get("visual_context", {})
        if vc:
            ctx_parts = [p for p in [vc.get("environment", ""), vc.get("lighting", ""), vc.get("color_tone", "")] if p]
            if ctx_parts and ", ".join(ctx_parts) not in prompt:
                prompt = prompt.rstrip(", ") + ", " + ", ".join(ctx_parts)

        negative = prompt_data.get("negative_prompt", "")

        # P3: character anchor injection for ref_url if no scene ref provided
        if not ref_url:
            char_name = cs.get("character_name", "")
            if char_name and character_consistency.has_anchor_image(char_name):
                ref_url = character_consistency.get_anchor_ref_image(char_name) or ""

        # L0: first attempt (adapter handles its own retries internally)
        try:
            results = await adapter.generate(
                prompt=prompt, negative_prompt=negative,
                width=1080, height=1920, num_images=1,
                ref_image_url=ref_url,
            )
            if results and results[0].url:
                results[0].prompt = prompt
                return results[0]
        except Exception as e:
            logger.warning("Image gen L0 failed for shot %s: %s", shot_id, e)

        # L1: retry without ref_image
        if ref_url:
            try:
                logger.info("L1 retry: shot %s without ref_image", shot_id)
                results = await adapter.generate(
                    prompt=prompt, negative_prompt=negative,
                    width=1080, height=1920, num_images=1,
                    ref_image_url="",
                )
                if results and results[0].url:
                    results[0].prompt = prompt
                    return results[0]
            except Exception as e:
                logger.warning("Image gen L1 failed for shot %s: %s", shot_id, e)

        # All layers exhausted
        return ImageResult(
            url="", prompt=prompt, width=1080, height=1920,
            model=adapter.MODEL, cost_usd=0.0,
        )

    def _pre_video_gate(
        self, image_results: list[ImageResult], classified_scenes: list[dict],
    ) -> bool:
        """Validate images before video generation (Pre-Video Gate).

        Returns False if too many image failures, which should abort the pipeline
        rather than producing a video full of black frames.
        """
        if not image_results:
            logger.error("Pre-video gate: no image results at all")
            return False

        valid_count = 0
        for img, cs in zip(image_results, classified_scenes):
            has_url = bool(img.url)
            has_local = bool(img.local_path) and os.path.isfile(img.local_path or "")
            scene_type = cs["type"]

            if has_url or has_local:
                valid_count += 1
            elif scene_type == KEY_SCENE:
                # Key scenes MUST have valid images
                logger.error(
                    "Pre-video gate: KEY_SCENE shot %s has no valid image",
                    cs.get("shot_id"),
                )
                return False

        failure_rate = 1.0 - valid_count / max(len(image_results), 1)
        if failure_rate > 0.3:
            logger.error(
                "Pre-video gate: %.0f%% failure rate (%d/%d valid)",
                failure_rate * 100, valid_count, len(image_results),
            )
            return False

        logger.info(
            "Pre-video gate: %d/%d images valid (%.0f%%), proceeding",
            valid_count, len(image_results), (1 - failure_rate) * 100,
        )
        return True

    # -- 3. Video generation --

    async def _generate_videos(
        self, classified_scenes: list[dict], image_results: list[ImageResult], key_scene_ratio: float,
        shot_durations: dict[int, float] | None = None,
    ) -> list[VideoResult]:
        """Generate videos for all shots in parallel with 3-layer defense + checkpoint.

        Defense layers:
          L0: adapter-level retry (Seedance 已有内部重试)
          L1: KEY_SCENE 失败时换 motion_type 重试
          L2: KEY_SCENE 仍失败时再用 Seedance 重试一次（废弃 Ken Burns 后，
              get_video_adapter 统一返回 Seedance；失败则跳过该 shot）
          L3: 失败率 > 30% → 优雅降级（不 raise，返回部分结果）

        借鉴 QinKunming/ai_manju：失败的 shot 直接跳过，compose 阶段只收集
        valid segments，字幕时间轴自动对齐到成功的 shot。

        Checkpoint:
          每个视频成功后立即写入 checkpoint，崩溃后重启可跳过已生成的 shot。
        """
        image_map = {cs["shot_id"]: img for cs, img in zip(classified_scenes, image_results)}
        sem = asyncio.Semaphore(8)

        # 初始化 checkpoint
        ckpt = CheckpointManager(self.episode_id or "unknown")
        completed_ids = ckpt.get_completed_shot_ids("video")
        if completed_ids:
            logger.info("Video checkpoint found: %d shots already completed", len(completed_ids))

        # L1 备选 motion 类型（按优先级）
        fallback_motions = ["zoom-in", "pan-left", "dolly-in", "static"]

        async def _gen_one(cs: dict) -> VideoResult:
            scene_type = cs["type"]
            shot_id = cs["shot_id"]
            shot = cs["shot"]

            # Checkpoint 命中
            if shot_id in completed_ids:
                saved = ckpt.load("video", shot_id)
                if saved and saved.get("local_path"):
                    local_path = saved["local_path"]
                    # 校验 local_path 有效性：必须是 .mp4 文件且实际存在
                    # 防止错误的 checkpoint（如 image 路径误写入、文件被覆盖/删除）
                    if local_path.lower().endswith(".mp4") and os.path.isfile(local_path):
                        logger.info("Video checkpoint hit: shot %d skipped", shot_id)
                        return VideoResult(
                            url=saved.get("url", ""), local_path=local_path,
                            scene_type=scene_type, duration_s=saved.get("duration_s", 5.0),
                            model=saved.get("model", "checkpoint"), cost_usd=0.0,
                        )
                    else:
                        # 校验失败：从 checkpoint 中删除，让流程重新生成
                        logger.warning(
                            "Video checkpoint invalid (shot %d): local_path=%s, exists=%s; removing",
                            shot_id, local_path, os.path.isfile(local_path),
                        )
                        ckpt.remove("video", shot_id)

            img = image_map.get(shot_id)
            # 优先用 TTS 真实时长（音画对齐），fallback 到 storyboard 估算
            real_dur = shot_durations.get(shot_id) if shot_durations else None
            if real_dur and real_dur >= 1.0:
                duration = max(int(real_dur), 3)  # seedance 最小 3s，用 TTS 真实时长
            else:
                # 无 TTS 的 shot：用 storyboard 时长（fallback）
                if scene_type == KEY_SCENE:
                    duration = max(shot.get("duration_s", 5.0), 5.0)
                else:
                    duration = max(shot.get("duration_s", 3.0), 3.0)
            image_path = img.local_path if img else ""
            image_url = img.url if img and img.url else ""
            motion = shot.get("camera_motion", "static")
            prompt_text = self._build_video_prompt(cs)

            # 调试日志：记录传入视频生成的 image_path/image_url
            logger.info(
                "Video gen shot %d: scene_type=%s, image_path=%s (exists=%s), image_url=%s, duration=%.1f",
                shot_id, scene_type, image_path,
                os.path.isfile(image_path) if image_path else False,
                image_url[:60] if image_url else "", duration,
            )

            async with sem:
                # L0: 正常生成
                adapter = get_video_adapter(scene_type)
                try:
                    result = await adapter.generate(
                        image_path=image_path, image_url=image_url,
                        scene_type="key" if scene_type == KEY_SCENE else "normal",
                        duration_s=duration, motion_type=motion, prompt_text=prompt_text,
                    )
                    if result.local_path:
                        result.scene_type = scene_type
                        ckpt.save("video", shot_id, {
                            "local_path": result.local_path, "url": result.url,
                            "duration_s": result.duration_s, "model": result.model,
                        })
                        return result
                except Exception as e:
                    logger.warning("Video L0 failed (shot %d): %s", shot_id, e)

                # L1: KEY_SCENE 换 motion_type 重试
                if scene_type == KEY_SCENE:
                    for alt_motion in fallback_motions:
                        if alt_motion == motion:
                            continue
                        try:
                            logger.info("Video L1 retry: shot %d with motion=%s", shot_id, alt_motion)
                            result = await adapter.generate(
                                image_path=image_path, image_url=image_url,
                                scene_type="key", duration_s=duration,
                                motion_type=alt_motion, prompt_text=prompt_text,
                            )
                            if result.local_path:
                                result.scene_type = scene_type
                                ckpt.save("video", shot_id, {
                                    "local_path": result.local_path, "url": result.url,
                                    "duration_s": result.duration_s, "model": result.model,
                                })
                                logger.info("Video L1 success: shot %d motion=%s", shot_id, alt_motion)
                                return result
                        except Exception as e:
                            logger.warning("Video L1 failed (shot %d, motion=%s): %s", shot_id, alt_motion, e)

                # L2: 再用 Seedance 重试一次（废弃 Ken Burns 后，所有场景统一 Seedance i2v）
                # 借鉴 QinKunming/ai_manju：失败则跳过该 shot，compose 阶段只收集 valid segments
                if scene_type == KEY_SCENE:
                    logger.warning("Video L2 fallback: shot %d → Seedance retry", shot_id)
                    kb_adapter = get_video_adapter(NORMAL_SCENE)
                    try:
                        result = await kb_adapter.generate(
                            image_path=image_path, image_url=image_url,
                            scene_type="normal", duration_s=duration,
                            motion_type="zoom-in", prompt_text=prompt_text,
                        )
                        if result.local_path:
                            result.scene_type = scene_type
                            result.metadata = {**result.metadata, "fallback": "seedance_retry"}
                            ckpt.save("video", shot_id, {
                                "local_path": result.local_path, "url": result.url,
                                "duration_s": result.duration_s, "model": result.model,
                            })
                            return result
                    except Exception as e:
                        logger.error("Video L2 Seedance retry failed (shot %d): %s", shot_id, e)

                # 全部失败：返回空 VideoResult
                return VideoResult(
                    url="", local_path="", scene_type=scene_type,
                    model="failed", cost_usd=0.0, metadata={"error": "all retries exhausted"},
                )

        tasks = [_gen_one(cs) for cs in classified_scenes]
        results = await asyncio.gather(*tasks)

        # L3: 统计失败率（优雅降级，不 raise）
        failures = sum(1 for r in results if not r.local_path)
        total = len(results)
        if total > 0 and failures / total > 0.3:
            logger.error(
                "Video failure rate %d/%d (%.0f%%) exceeds 30%% — "
                "graceful degradation, %d videos will use fallback handling",
                failures, total, failures / total * 100, failures,
            )

        return list(results)

    def _build_video_prompt(self, cs: dict) -> str:
        shot = cs.get("shot", {})
        scene_data = cs.get("scene_data", {})
        parts = [
            shot.get("description", ""),
            "anime style",
            "cinematic lighting",
            scene_data.get("emotion", ""),
        ]
        return ", ".join(p for p in parts if p)

   # -- 4. TTS multi-character voice acting --

    async def _generate_audio(
        self, script: dict[str, Any], asset_library: dict[str, Any],
        storyboard: list[dict] | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """Generate TTS for both dialogue and narration, aligned to shots.

        New behavior (bugfix):
        - Generates TTS for narration too (not just dialogue)
        - Aligns each audio segment to a specific shot_id (not just scene_index)
        - If storyboard is provided, assigns narration/dialogue to shots in order
        """
        adapter = get_tts_adapter()
        scenes = script.get("scenes", [])
        char_voice_map = self._build_voice_map(asset_library)
        audio_segments: list[dict] = []
        subtitle_entries: list[dict] = []
        narrator_voice = "zh_male_ruyayichen_uranus_bigtts"
        shot_sub_time: dict[int, int] = {}

        # Audio checkpoint：支持断点恢复，避免重复调用 Seed Audio API
        ckpt = CheckpointManager(self.episode_id or "unknown")
        ckpt_loaded: dict[int, dict] = {}  # shot_id → saved entry (from checkpoint)

        def _try_load_ckpt(shot_id: int, key: str) -> dict | None:
            """从 checkpoint 加载音频段（文件存在性校验）。"""
            if shot_id is None:
                return None
            saved = ckpt.load("tts", shot_id)
            if not saved:
                return None
            local_path = saved.get("local_path", "")
            if local_path and not os.path.isfile(local_path):
                logger.info("TTS checkpoint invalid (shot %d): file missing %s", shot_id, local_path)
                ckpt.remove("tts", shot_id)
                return None
            return saved

        def _save_ckpt(shot_id: int, result, text: str, voice_id: str,
                       seg_type: str, character: str, scene_id: int, scene_idx: int):
            """保存音频段到 checkpoint。"""
            if shot_id is None or result is None:
                return
            ckpt.save("tts", shot_id, {
                "local_path": result.local_path,
                "audio_url": result.audio_url,
                "text": text,
                "voice_id": voice_id,
                "duration_s": result.duration_s,
                "cost_usd": result.cost_usd,
                "model": result.model,
                "type": seg_type,
                "character": character,
                "scene_id": scene_id,
                "scene_index": scene_idx,
                "word_timestamps": result.word_timestamps,
            })

        def _restore_from_ckpt(saved: dict, shot_id: int, scene_idx: int) -> dict:
            """从 checkpoint 数据重建 audio_segment + subtitle。"""
            result = TTSResult(
                audio_url=saved.get("audio_url", ""),
                local_path=saved.get("local_path", ""),
                text=saved.get("text", ""),
                voice_id=saved.get("voice_id", ""),
                duration_s=saved.get("duration_s", 0.0),
                model=saved.get("model", ""),
                cost_usd=0.0,  # checkpoint 恢复不重复计费
                word_timestamps=saved.get("word_timestamps", []),
            )
            return {
                "result": result,
                "scene_index": scene_idx,
                "scene_id": saved.get("scene_id", 0),
                "shot_id": shot_id,
                "type": saved.get("type", "dialogue"),
                "text": saved.get("text", ""),
                "character": saved.get("character", ""),
            }

        # Per-scene shot assignment: dialogue is mapped to shots WITHIN the same
        # scene, preventing cross-scene misalignment (e.g., scene 3 dialogue going
        # to scene 1's shots).
        # scene_id → list of shot_ids in that scene
        scene_shots: dict[int, list[int]] = {}
        if storyboard:
            for s in storyboard:
                sid = s.get("scene_id", 0)
                scene_shots.setdefault(sid, []).append(s.get("shot_id", 0))
        scene_shot_idx: dict[int, int] = {}  # scene_id → next shot index

        def _next_shot_in_scene(scene_id: int) -> int | None:
            """Return next available shot_id within the given scene."""
            shots = scene_shots.get(scene_id, [])
            idx = scene_shot_idx.get(scene_id, 0)
            if idx < len(shots):
                scene_shot_idx[scene_id] = idx + 1
                return shots[idx]
            return None  # no more shots in this scene

        def _assign_subtitle_timing(
            shot_id: int | None, text: str, speaker: str, scene_idx: int,
            subs: list[str], tts_dur_ms: int,
        ) -> None:
            """Assign shot-relative start_ms/end_ms to subtitle entries based on TTS duration.

            start_ms/end_ms are relative to the SHOT's start time (0 = shot start).
            The compositor will offset them by the shot's absolute start in the final video.
            Duration is distributed proportionally to each subtitle's character count.
            """
            cum_ms = shot_sub_time.get(shot_id, 0)
            total_chars = sum(len(s) for s in subs) or 1
            for entry_text in subs:
                entry_dur_ms = max(int(tts_dur_ms * len(entry_text) / total_chars), 500)
                subtitle_entries.append({
                    "text": entry_text, "speaker": speaker,
                    "scene_index": scene_idx,
                    "shot_id": shot_id,
                    "start_ms": cum_ms,
                    "end_ms": cum_ms + entry_dur_ms,
                })
                cum_ms += entry_dur_ms
            shot_sub_time[shot_id] = cum_ms

        for scene_idx, scene in enumerate(scenes):
            scene_id = scene.get("scene_id", scene_idx + 1)

            # NOTE: Scene-level narration TTS is intentionally SKIPPED here.
            # The writer already splits scene narration into per-shot narration
            # fields (shot.narration), which are handled in stage 3 below.
            # Generating scene-level narration would:
            #   1. Duplicate content already covered by shot narrations
            #   2. Assign a long audio (e.g. 19s) to a single short shot (e.g. 8s),
            #      causing severe audio-visual desync
            # Scene narration is only used as fallback for shots with empty
            # narration (handled in stage 3).

            # 1) Dialogue TTS — assigned to shots WITHIN the same scene
            for d in scene.get("dialogue", []):
                if isinstance(d, dict):
                    character = d.get("character", "")
                    line = d.get("line", "")
                elif isinstance(d, str):
                    character = ""
                    line = d
                else:
                    continue
                if not line:
                    continue
                # 优先用 voice_map；未匹配的角色默认用男声（适合侦探/旁白类）
                voice_id = char_voice_map.get(character, "zh_male_ruyayichen_uranus_bigtts")
                shot_id = _next_shot_in_scene(scene_id) if storyboard else None
                tts_result = None

                # Checkpoint 恢复：优先从 tts.json 加载，避免重复调用 API
                saved = _try_load_ckpt(shot_id, "dialogue")
                if saved and saved.get("type") == "dialogue":
                    seg = _restore_from_ckpt(saved, shot_id, scene_idx)
                    audio_segments.append(seg)
                    tts_result = seg["result"]
                    logger.info("TTS checkpoint hit: shot %d dialogue skipped", shot_id)
                else:
                    try:
                        tts_result = await adapter.synthesize(
                            text=line, voice_id=voice_id, speed=1.0, pitch=0.0,
                        )
                        audio_segments.append({
                            "result": tts_result,
                            "scene_index": scene_idx,
                            "scene_id": scene_id,
                            "shot_id": shot_id,
                            "type": "dialogue",
                            "character": character,
                            "text": line,
                        })
                        _save_ckpt(shot_id, tts_result, line, voice_id,
                                   "dialogue", character, scene_id, scene_idx)
                    except Exception as e:
                        logger.error("TTS failed for '%s': %s", line[:30], e)
                        audio_segments.append({
                            "result": None, "scene_index": scene_idx, "scene_id": scene_id,
                            "shot_id": None, "type": "dialogue",
                        })

                # Assign shot-relative timing to subtitles based on TTS duration
                tts_dur_ms = int((tts_result.duration_s if tts_result else 0) * 1000)
                _assign_subtitle_timing(
                    shot_id, line, character, scene_idx,
                    self._split_text_for_subtitles(line), tts_dur_ms,
                )

        # 3. Shot-level narration: 补全未分配 TTS 的 shot（保证 100% 音频覆盖）
        # Writer 为无 dialogue 的 shot 写了 narration 字段（语音叙事，非画面描述）
        if storyboard:
            assigned_shots = {a.get("shot_id") for a in audio_segments if a.get("shot_id") is not None}
            # Build scene_id → scene narration mapping for fallback (stage 3b)
            scene_narration_map: dict[int, str] = {}
            for scene in scenes:
                sid = scene.get("scene_id", 0)
                narr = (scene.get("narration") or "").strip()
                if narr:
                    scene_narration_map[sid] = narr

            # Pre-compute uncovered sentences per scene:
            # scene_id → list of sentences NOT already covered by any shot's narration
            scene_uncovered: dict[int, list[str]] = {}
            for scene in scenes:
                sid = scene.get("scene_id", 0)
                scene_narr = (scene.get("narration") or "").strip()
                if not scene_narr:
                    continue
                all_sentences = [s.strip() for s in scene_narr.replace("。", "。\n").split("\n") if s.strip()]
                # Collect narration text from shots that have non-empty narration in this scene
                used_text = ""
                for s in storyboard:
                    if s.get("scene_id") == sid:
                        used_text += (s.get("narration") or "").strip()
                # Filter: keep sentences NOT already covered by shot narrations
                uncovered = [sent for sent in all_sentences if sent not in used_text]
                scene_uncovered[sid] = uncovered

            for shot in storyboard:
                shot_id = shot.get("shot_id")
                if shot_id is None or shot_id in assigned_shots:
                    continue
                narration = (shot.get("narration") or "").strip()
                scene_id = shot.get("scene_id", 0)
                shot_idx = max(shot_id - 1, 0)

                # 3b. Fallback: shot with empty narration → use UNCOVERED scene narration
                if not narration:
                    uncovered = scene_uncovered.get(scene_id, [])
                    if not uncovered:
                        continue  # scene narration fully covered by other shots → silent
                    # Count empty-narration shots in this scene to distribute evenly
                    empty_shots_in_scene = [
                        s.get("shot_id") for s in storyboard
                        if s.get("scene_id") == scene_id
                        and s.get("shot_id") not in assigned_shots
                        and not (s.get("narration") or "").strip()
                    ]
                    my_idx = empty_shots_in_scene.index(shot_id) if shot_id in empty_shots_in_scene else 0
                    total = len(empty_shots_in_scene) or 1
                    # Distribute uncovered sentences: each empty shot gets ~1/total
                    per_shot = max(len(uncovered) // total, 1)
                    start = min(my_idx * per_shot, len(uncovered))
                    end = min(start + per_shot, len(uncovered))
                    narration = "".join(uncovered[start:end]) if uncovered[start:end] else ""
                    if not narration:
                        continue
                    logger.debug(
                        "Shot %d: empty narration, using scene %d uncovered fallback (portion %d/%d, %d sentences)",
                        shot_id, scene_id, my_idx + 1, total, end - start,
                    )

                tts_result = None

                # Checkpoint 恢复：优先从 tts.json 加载 narration
                saved = _try_load_ckpt(shot_id, "narration")
                if saved and saved.get("type") == "shot_narration":
                    seg = _restore_from_ckpt(saved, shot_id, shot_idx)
                    audio_segments.append(seg)
                    tts_result = seg["result"]
                    logger.info("TTS checkpoint hit: shot %d narration skipped", shot_id)
                else:
                    try:
                        # 将 audio_scene 组合到 text_prompt，让 Seed Audio 生成包含背景音的完整音频场景
                        # audio_scene 描述环境声/BGM/情绪氛围，narration 是要被"说出来"的文本
                        audio_scene = (shot.get("audio_scene") or "").strip()
                        text_prompt = narration
                        if audio_scene:
                            text_prompt = f"{narration}（背景音：{audio_scene}）"
                        tts_result = await adapter.synthesize(
                            text=text_prompt, voice_id=narrator_voice, speed=0.95, pitch=0.0,
                        )
                        audio_segments.append({
                            "result": tts_result,
                            "scene_index": shot_idx,
                            "scene_id": scene_id,
                            "shot_id": shot_id,
                            "type": "shot_narration",
                            "text": narration,  # 字幕用纯 narration（不含 audio_scene 描述）
                            "character": "narrator",
                        })
                        _save_ckpt(shot_id, tts_result, narration, narrator_voice,
                                   "shot_narration", "narrator", scene_id, shot_idx)
                        tts_dur_ms = int((tts_result.duration_s if tts_result else 0) * 1000)
                        _assign_subtitle_timing(
                            shot_id, narration, "narrator", shot_idx,
                            self._split_text_for_subtitles(narration), tts_dur_ms,
                        )
                        logger.debug("Shot narration audio for shot %d: %s", shot_id, narration[:30])
                    except Exception as e:
                        logger.error("Shot narration TTS failed (shot %d): %s", shot_id, e)
                        audio_segments.append({
                            "result": None, "scene_index": shot_idx, "scene_id": scene_id,
                            "shot_id": shot_id, "type": "shot_narration",
                        })

        return audio_segments, subtitle_entries

    def _build_voice_map(self, asset_library: dict) -> dict[str, str]:
        from app.resilience.adapters.tts_adapter import DEFAULT_MALE_SPEAKER
        voice_map: dict[str, str] = {}
        for c in asset_library.get("characters", []):
            name = c.get("name", "")
            vt = c.get("voice_traits", {})
            vid = vt.get("suggested_voice_id", "")
            if vid:
                # Safety: if voice_id starts with en-US and not in VOICE_SPEAKER_MAP,
                # fall back to gender-based Chinese voice instead
                if vid.startswith("en-"):
                    from app.resilience.adapters.tts_adapter import VOICE_SPEAKER_MAP
                    if vid not in VOICE_SPEAKER_MAP:
                        logger.warning(
                            "Character '%s' has English voice_id '%s' not in VOICE_SPEAKER_MAP, "
                            "falling back to gender-based Chinese voice", name, vid
                        )
                        vid = DEFAULT_MALE_SPEAKER if vt.get("gender") == "male" else "zh_female_qingxin"
                voice_map[name] = vid
            elif vt.get("gender") == "male":
                voice_map[name] = DEFAULT_MALE_SPEAKER
            else:
                voice_map[name] = "zh_female_qingxin"
        return voice_map

    @staticmethod
    def _split_text_for_subtitles(text: str, max_chars: int = 20) -> list[str]:
        """Split text at punctuation boundaries for subtitle display.

        Delegates to the canonical SubtitleGenerator._split_text implementation.
        """
        return SubtitleGenerator._split_text(text, max_chars)

    # -- 5. BGM selection 已移除 --
    # Seed Audio 1.0 在生成对白时同步产出背景音效与 BGM 元素，
    # 无需独立选取 BGM 曲目，也无需在 compositor 中混合。

    # -- 6. Cover selection --

    def _select_cover_candidates(
        self, image_results: list[ImageResult], count: int = 3,
    ) -> list[dict]:
        candidates = []
        for img in image_results:
            if img.local_path and os.path.isfile(img.local_path):
                candidates.append({
                    "local_path": img.local_path,
                    "prompt": img.prompt,
                    "width": img.width,
                    "height": img.height,
                })
        return candidates[:count]

    # -- 7. FFmpeg final composition --

    async def _compose_final_video(
        self,
        video_segments: list[VideoResult],
        audio_segments: list[dict],
        subtitles: list[dict],
        bgm_track: str,
        classified_scenes: list[dict],
        episode_id: str,
        bgm_path: str = "",
    ) -> dict:
        """Compose final video with SCENE-LEVEL composition.

        Groups all shot videos + TTS by scene_id, so a scene's narration
        plays across ALL its shots naturally (no slow-motion distortion).

        Seed Audio 1.0 生成的音频已包含背景音/BGM/环境音，TTS 音频直接按
        scene 分组拼接后对齐到 scene video，无需 audio_timeline 时间线编排。
        """
        # Build shot_id -> scene_id mapping from classified_scenes
        shot_to_scene: dict[int, int] = {}
        # shot_id -> duration_s from storyboard
        shot_dur_map: dict[int, float] = {}
        for cs in classified_scenes:
            shot = cs.get("shot", {})
            sid = shot.get("shot_id")
            scene_id = shot.get("scene_id", 1)
            if sid is not None:
                shot_to_scene[sid] = scene_id
                shot_dur_map[sid] = float(shot.get("duration_s", 5.0) or 5.0)

        # Group video segments by scene_id (preserve shot order within scene)
        # video_segments is ordered same as classified_scenes
        scenes_by_id: dict[int, list[dict]] = {}
        for i, v in enumerate(video_segments):
            if i < len(classified_scenes):
                shot = classified_scenes[i].get("shot", {})
                shot_id = shot.get("shot_id")
                scene_id = shot.get("scene_id", 1)
            else:
                shot_id = None
                scene_id = 1
            scenes_by_id.setdefault(scene_id, []).append({
                "local_path": v.local_path,
                "duration_s": v.duration_s,
                "shot_id": shot_id,
            })

        # Group audio segments by scene_id (collect all TTS for a scene, in shot order)
        audio_by_scene: dict[int, list[dict]] = {}
        for a in audio_segments:
            tts = a.get("result")
            if not tts or not tts.local_path or not os.path.isfile(tts.local_path):
                continue
            shot_id = a.get("shot_id")
            scene_id = shot_to_scene.get(shot_id, a.get("scene_id", 1))
            audio_by_scene.setdefault(scene_id, []).append({
                "local_path": tts.local_path,
                "duration_s": tts.duration_s,
                "shot_id": shot_id,
                "text": a.get("text", ""),
            })

        # Build ordered scene_groups (sorted by scene_id)
        scene_ids = sorted(scenes_by_id.keys())
        scene_groups: list[dict] = []
        for sid in scene_ids:
            shots_meta: list[dict] = []
            for v in scenes_by_id[sid]:
                vid = v.get("shot_id")
                shot_dur = shot_dur_map.get(vid, v.get("duration_s", 5.0)) if vid is not None else v.get("duration_s", 5.0)
                shots_meta.append({
                    "shot_id": vid,
                    "duration_s": shot_dur,
                })
            scene_groups.append({
                "scene_id": sid,
                "videos": scenes_by_id[sid],
                "audios": audio_by_scene.get(sid, []),
                "shots_meta": shots_meta,
            })

        logger.info(
            "Scene-level composition: %d scenes, %d videos, %d audios",
            len(scene_groups),
            sum(len(sg["videos"]) for sg in scene_groups),
            sum(len(sg["audios"]) for sg in scene_groups),
        )

        result = await compose_episode(
            scene_groups=scene_groups,
            subtitles=subtitles,
            episode_id=episode_id,
            bgm_path=bgm_path,
        )
        result["extra_tts_cost"] = 0.0
        result["auto_narration_segments"] = []
        if result["success"]:
            logger.info(
                "Final video composed (scene-level): %s (%.1f MB, %.0fs)",
                result["final_video_path"],
                result.get("size_mb", 0),
                result.get("duration_s", 0),
            )
        else:
            logger.error("Final composition failed: %s", result.get("error", "unknown"))
        return result

    # -- 8. AI content label --

    def _build_ai_label(
        self, script: dict[str, Any], metadata: dict[str, Any], total_cost_usd: float,
    ) -> dict[str, Any]:
        from app.core.config import settings
        # 模型标识统一从 config 读取（L5: 避免分散硬编码）
        audio_model = (
            settings.ARK_AUDIO_MODEL
            if settings.AUDIO_PROVIDER == "seed_audio"
            else settings.ARK_TTS_MODEL
        )
        audio_server = (
            settings.SEED_AUDIO_API_URL
            if settings.AUDIO_PROVIDER == "seed_audio"
            else "wss://openspeech.bytedance.com/api/v3/tts/bidirection"
        )
        return {
            "deep_synthesis": True,
            "generation_time": datetime.utcnow().isoformat() + "Z",
            "models_used": {
                "text": settings.ARK_LLM_MODEL,
                "image": settings.ARK_IMAGE_MODEL,
                "video": settings.ARK_VIDEO_MODEL,
                "audio": audio_model,
            },
            "servers": {
                "image": "https://ark.cn-beijing.volces.com/api/v3/images/generations",
                "video": "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks",
                "audio": audio_server,
            },
            "generation_parameters": {
                "scenes_count": len(script.get("scenes", [])),
                "images_generated": metadata.get("images_generated", 0),
                "videos_generated": metadata.get("videos_generated", 0),
                "audio_segments": metadata.get("audio_segments", 0),
            },
            "cost": {"total_usd": round(total_cost_usd, 4)},
            "venue": "ARK (Volcengine)",
        }

    # -- Utilities --

    @staticmethod
    def _image_to_dict(img: ImageResult) -> dict:
        return {
            "url": img.url, "local_path": img.local_path, "prompt": img.prompt,
            "width": img.width, "height": img.height, "model": img.model, "cost_usd": img.cost_usd,
        }

    @staticmethod
    def _video_to_dict(v: VideoResult, idx: int) -> dict:
        return {
            "index": idx, "url": v.url, "local_path": v.local_path, "duration_s": v.duration_s,
            "scene_type": v.scene_type, "model": v.model, "cost_usd": v.cost_usd, "metadata": v.metadata,
        }

    @staticmethod
    def _tts_to_dict(a: TTSResult) -> dict:
        return {
            "audio_url": a.audio_url, "local_path": a.local_path, "text": a.text,
            "voice_id": a.voice_id, "duration_s": a.duration_s, "model": a.model, "cost_usd": a.cost_usd,
       }
