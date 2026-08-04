"""Composer Agent - image, video, TTS,, BGM, cover generation and FFmpeg composition.

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
from app.services.video_strategy import classify_scene, _extract_main_character, KEY_SCENE, NORMAL_SCENE
from app.services.episode_compositor import compose_episode

from app.quality.character_consistency import character_consistency
from app.quality.content_gate import content_gate
from app.quality.vqa_checker import vqa_checker
from app.quality.video_content_checker import video_content_checker, VideoContentResult
logger = logging.getLogger(__name__)


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
        visual_specs: Optional[list[dict]] = None,
    ) -> AgentResult:
        """Run the full production pipeline.

        Returns EpisodeAsset with images, videos, audio,,
        BGM, covers, final video path, and AI content label metadata.
        """
        from app.agents.pipeline.context import PipelineContext
        from app.agents.pipeline.orchestrator import ComposerPipeline

        episode_id = self.episode_id or "ep_001"

        # 若未传 visual_specs，尝试从 storyboard 构建
        if visual_specs is None:
            try:
                from app.quality.visual_descriptor import visual_descriptor
                visual_specs = visual_descriptor.build_all(
                    storyboard=storyboard, script=script, asset_library=asset_library,
                )
            except Exception as e:
                logger.warning("visual_specs rebuild failed (degraded): %s", e)
                visual_specs = []

        context = PipelineContext(
            script=script,
            storyboard=storyboard,
            image_prompts=image_prompts,
            asset_library=asset_library,
            episode_id=episode_id,
            key_scene_ratio=key_scene_ratio,
            visual_specs=visual_specs,
        )

        pipeline = ComposerPipeline(context, self)
        error = await pipeline.run()

        if error:
            return AgentResult(
                success=False,
                error=error,
                metadata=context.metadata,
            )

        episode_asset = {
            "episode_id": episode_id,
            "script": script,
            "storyboard": storyboard,
            "image_prompts": image_prompts,
            "images": [self._image_to_dict(img) for img in context.image_results],
            "video_segments": [self._video_to_dict(v, i) for i, v in enumerate(context.video_segments)],
            "audio_segments": [self._tts_to_dict(a["result"]) for a in context.audio_segments if a["result"]],
            "bgm_track": "",
            "covers": context.covers,
            "final_video_path": context.compose_result.get("final_video_path", ""),
            "ai_label": context.ai_label,
            "cost_usd": round(context.total_cost_usd, 4),
            "metadata": context.metadata,
        }

        return AgentResult(
            success=True,
            data=episode_asset,
            cost_usd=context.total_cost_usd,
            metadata=context.metadata,
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

        # 按 shot_angle 选最佳视角 ref_image
        # 若 cs 中已带 visual_spec，则用 PromptTemplateEngine 重新渲染 prompt（确定性输出）
        visual_spec = cs.get("visual_spec")
        if visual_spec:
            try:
                from app.quality.visual_descriptor import prompt_template_engine
                rendered_prompt, rendered_negative = prompt_template_engine.render_both(visual_spec)
                if rendered_prompt:
                    prompt = rendered_prompt
                if rendered_negative:
                    negative = rendered_negative
            except Exception as e:
                logger.warning("PromptTemplateEngine render failed for shot %s: %s", shot_id, e)

        # P3: character anchor injection for ref_url if no scene ref provided
        # 优先使用 multi-view anchor 的最佳视角（按 camera_angle）
        if not ref_url:
            char_name = cs.get("character_name", "")
            shot_angle = (visual_spec or {}).get("composition", {}).get("camera_angle", "")
            if char_name and shot_angle and character_consistency.has_multi_view(char_name):
                # 多视角 anchor：按 shot_angle 选最佳视图
                ref_url = character_consistency.get_best_ref_view(char_name, shot_angle) or ""
            elif char_name and character_consistency.has_anchor_image(char_name):
                # 旧 anchor（仅 front / seed_image）
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

        硬约束（音画对齐）：shot_durations 必须传入，且每个 shot 必须有 TTS 真实时长。
        无 TTS 时长的 shot 直接跳过（返回空 VideoResult），不再使用 storyboard 估算 fallback，
        严防与 TTS 真实时长偏差导致的音画不同步。

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

        # 音画对齐硬约束：shot_durations 必传
        if shot_durations is None:
            logger.error(
                "shot_durations is None — video generation requires TTS durations "
                "for audio-visual sync; aborting to prevent desync"
            )
            return [VideoResult(
                url="", local_path="", scene_type=cs.get("type", NORMAL_SCENE),
                model="skipped_no_tts", cost_usd=0.0,
                metadata={"error": "shot_durations is None (no TTS duration)"},
            ) for cs in classified_scenes]

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
            # 音画对齐硬约束：必须有 TTS 真实时长，不再使用 storyboard 估算 fallback
            real_dur = shot_durations.get(shot_id) if shot_durations else None
            if not real_dur or real_dur < 1.0:
                # 无 TTS 时长 → 跳过该 shot（不生成视频，严防音画不同步）
                logger.warning(
                    "Video skip shot %d: no TTS duration (shot_durations=%s) — "
                    "audio-visual sync requires TTS, skipping to prevent desync",
                    shot_id, f"{real_dur:.2f}s" if real_dur else "None",
                )
                return VideoResult(
                    url="", local_path="", scene_type=scene_type,
                    model="skipped_no_tts", cost_usd=0.0,
                    metadata={"error": "no TTS duration (audio-visual sync required)"},
                )
            duration = max(int(real_dur), 3)  # seedance 最小 3s，用 TTS 真实时长

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
                        ok, reason = self._validate_video_result(result, shot_id)
                        if not ok:
                            logger.warning("Video L0 file invalid (shot %d): %s", shot_id, reason)
                            try:
                                os.remove(result.local_path)
                            except Exception:
                                pass
                        else:
                            # 文件校验通过，执行内容质检
                            content_ok, content_detail = await self._check_video_content(
                                video_path=result.local_path,
                                source_image_path=image_path,
                                scene_type=scene_type,
                                character_name=cs.get("character_name", ""),
                            )
                            if not content_ok:
                                logger.warning(
                                    "Video L0 content check failed (shot %d): %s",
                                    shot_id, content_detail,
                                )
                                try:
                                    os.remove(result.local_path)
                                except Exception:
                                    pass
                            else:
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
                                ok, reason = self._validate_video_result(result, shot_id)
                                if not ok:
                                    logger.warning("Video L1 file invalid (shot %d, motion=%s): %s", shot_id, alt_motion, reason)
                                    try:
                                        os.remove(result.local_path)
                                    except Exception:
                                        pass
                                    continue
                                # 文件校验通过，执行内容质检
                                content_ok, content_detail = await self._check_video_content(
                                    video_path=result.local_path,
                                    source_image_path=image_path,
                                    scene_type=scene_type,
                                    character_name=cs.get("character_name", ""),
                                )
                                if not content_ok:
                                    logger.warning(
                                        "Video L1 content check failed (shot %d, motion=%s): %s",
                                        shot_id, alt_motion, content_detail,
                                    )
                                    try:
                                        os.remove(result.local_path)
                                    except Exception:
                                        pass
                                    continue
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
                            ok, reason = self._validate_video_result(result, shot_id)
                            if not ok:
                                logger.warning("Video L2 file invalid (shot %d): %s", shot_id, reason)
                                try:
                                    os.remove(result.local_path)
                                except Exception:
                                    pass
                            else:
                                # 文件校验通过，执行内容质检
                                content_ok, content_detail = await self._check_video_content(
                                    video_path=result.local_path,
                                    source_image_path=image_path,
                                    scene_type=scene_type,
                                    character_name=cs.get("character_name", ""),
                                )
                                if not content_ok:
                                    logger.warning(
                                        "Video L2 content check failed (shot %d): %s",
                                        shot_id, content_detail,
                                    )
                                    try:
                                        os.remove(result.local_path)
                                    except Exception:
                                        pass
                                else:
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
        # 注入角色 canonical appearance 到视频 prompt
        # 三视图 anchor 在每集启动时从 DB 加载（episode_task._load_character_anchors），
        # 此处取 anchor.seed_prompt（首集 id_card 原文构建，跨集唯一基准），
        # 确保视频模型在动画过程中保持角色外观，防止 25 集间 drift。
        char_name = cs.get("character_name", "")
        if char_name:
            appearance = character_consistency.get_canonical_appearance(char_name)
            if appearance:
                parts.append(appearance)
            else:
                # anchor 未加载（如首集 DB 尚无记录）兜底：从 asset_library 取 id_card
                id_card = self._extract_id_card_from_context(cs)
                if id_card:
                    parts.append(id_card)
        return ", ".join(p for p in parts if p)

    def _extract_id_card_from_context(self, cs: dict) -> str:
        """从 classified_scene 关联的 asset_library 角色 id_card 兜底构建外貌描述。

        仅在 anchor 未加载时使用（首集 anchor 尚未持久化的瞬间）。
        """
        # asset_library 不在 cs 中，这里通过 character_name 无法直接定位；
        # 返回通用约束，让视频模型至少有"保持角色一致"的文本提示。
        char_name = cs.get("character_name", "")
        if not char_name:
            return ""
        return f"{char_name}, consistent character appearance, same person throughout"

    async def _check_video_content(
        self,
        video_path: str,
        source_image_path: str,
        scene_type: str,
        character_name: str = "",
    ) -> tuple[bool, str]:
        """检查视频内容质量，不通过则触发重试。

        1. 抽帧 + CLIP 相似度 vs 源图（所有视频）
        2. VLM 人物完整性检查（仅 KEY_SCENE）

        Returns:
            (passed, detail): passed=True 表示内容合格可进入合成
        """
        if not settings.VIDEO_CONTENT_CHECK_ENABLED:
            return True, "content check disabled"

        result: VideoContentResult = await video_content_checker.check_video(
            video_path=video_path,
            source_image_path=source_image_path,
            scene_type=scene_type,
            character_name=character_name,
        )
        if result.passed:
            return True, result.detail
        return False, result.detail

    # ================================================================
    # TTS 校验 + Checkpoint 恢复（音画对齐前置约束）
    # ================================================================

    @staticmethod
    def _validate_tts_result(result, text: str) -> tuple[bool, str]:
        """校验 TTS 结果质量。

        Returns:
            (is_valid, reason): is_valid=True 时 reason 为空；False 时 reason 说明失败原因
        """
        if result is None:
            return False, "result is None"
        if not getattr(result, "local_path", ""):
            return False, "local_path empty"
        if not os.path.isfile(result.local_path):
            return False, f"file not found: {result.local_path}"
        if os.path.getsize(result.local_path) < 1024:
            return False, f"file too small: {os.path.getsize(result.local_path)} bytes"
        dur = getattr(result, "duration_s", 0.0) or 0.0
        if dur < 0.5:
            return False, f"duration too short: {dur:.2f}s"
        # 时长与文本长度比例校验（5 字/秒为中文 TTS 基线）
        # ratio < 0.33 → 异常短（可能是截断）；ratio > 3.0 → 异常长（可能是噪声/重复）
        if text:
            expected_dur = max(len(text) / 5.0, 0.5)
            ratio = dur / expected_dur if expected_dur > 0 else 1.0
            if ratio < 0.33 or ratio > 3.0:
                return False, f"duration ratio out of range: {ratio:.2f} (dur={dur:.2f}s, expected={expected_dur:.2f}s)"
        return True, ""

    @staticmethod
    def _validate_video_result(result, shot_id: int) -> tuple[bool, str]:
        """校验视频生成结果的基本有效性。

        Returns:
            (is_valid, reason): is_valid=True 时 reason 为空；False 时 reason 说明失败原因
        """
        if result is None:
            return False, "result is None"
        if not getattr(result, "local_path", ""):
            return False, "local_path empty"
        local_path = result.local_path
        if not os.path.isfile(local_path):
            return False, f"file not found: {local_path}"
        file_size = os.path.getsize(local_path)
        if file_size < 10240:
            return False, f"file too small: {file_size} bytes"
        if not local_path.lower().endswith(".mp4"):
            return False, f"not an mp4 file: {local_path}"
        # ffprobe 验证文件是否为有效视频
        try:
            import subprocess
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", local_path],
                capture_output=True, text=True, timeout=15,
            )
            if r.returncode != 0:
                return False, f"ffprobe failed: {r.stderr[:200]}"
            dur = float(r.stdout.strip() or "0")
            if dur < 0.5:
                return False, f"duration too short: {dur:.2f}s"
        except Exception as e:
            return False, f"ffprobe exception: {e}"
        return True, ""

    async def _recover_tts_segment(
        self,
        seg: dict,
        ckpt,
        adapter,
    ) -> dict:
        """TTS 校验失败时的恢复流程：checkpoint 优先 → 重新生成。

        Args:
            seg: audio_segment dict（含 result/text/voice_id/shot_id 等）
            ckpt: CheckpointManager 实例
            adapter: TTS 适配器

        Returns:
            恢复后的 seg（result 已替换为有效 TTSResult 或仍为 None）
        """
        shot_id = seg.get("shot_id")
        text = seg.get("text", "")
        voice_id = seg.get("result").voice_id if seg.get("result") else "zh_male_ruyayichen_uranus_bigtts"

        # 1. 尝试 checkpoint 恢复
        if shot_id is not None:
            saved = ckpt.load("tts", shot_id)
            if saved and saved.get("local_path"):
                saved_path = saved.get("local_path", "")
                if os.path.isfile(saved_path) and os.path.getsize(saved_path) >= 1024:
                    saved_dur = saved.get("duration_s", 0.0) or 0.0
                    if saved_dur >= 0.5:
                        from app.resilience.adapters.audio_types import TTSResult
                        recovered = TTSResult(
                            audio_url=saved.get("audio_url", ""),
                            local_path=saved_path,
                            text=saved.get("text", text),
                            voice_id=saved.get("voice_id", voice_id),
                            duration_s=saved_dur,
                            model=saved.get("model", ""),
                            cost_usd=0.0,  # checkpoint 恢复不重复计费
                            word_timestamps=saved.get("word_timestamps", []),
                        )
                        # 校验 checkpoint 是否也通过质量检查
                        ok, _ = self._validate_tts_result(recovered, text)
                        if ok:
                            logger.info(
                                "TTS recovered from checkpoint: shot %s (dur=%.2fs)",
                                shot_id, recovered.duration_s,
                            )
                            return {**seg, "result": recovered}

        # 2. checkpoint 不可用 → 重新生成（最多 2 次重试）
        for attempt in range(2):
            try:
                # 兼容 shot narration（带 audio_scene）和 dialogue（纯文本）
                audio_scene = ""
                if "type" in seg and seg.get("type") == "shot_narration":
                    # narration 类型：尝试从 storyboard 取 audio_scene
                    pass
                text_prompt = text
                if audio_scene:
                    text_prompt = f"{text}（背景音：{audio_scene}）"

                speed = 0.95 if seg.get("type") == "shot_narration" else 1.0
                new_result = await adapter.synthesize(
                    text=text_prompt, voice_id=voice_id, speed=speed, pitch=0.0,
                )
                ok, reason = self._validate_tts_result(new_result, text)
                if ok:
                    logger.info(
                        "TTS regenerated: shot %s (attempt %d, dur=%.2fs)",
                        shot_id, attempt + 1, new_result.duration_s,
                    )
                    # 重新生成成功后写入 checkpoint
                    if shot_id is not None:
                        ckpt.save("tts", shot_id, {
                            "local_path": new_result.local_path,
                            "audio_url": new_result.audio_url,
                            "text": text,
                            "voice_id": voice_id,
                            "duration_s": new_result.duration_s,
                            "cost_usd": new_result.cost_usd,
                            "model": new_result.model,
                            "type": seg.get("type", ""),
                            "character": seg.get("character", ""),
                            "word_timestamps": new_result.word_timestamps,
                        })
                    return {**seg, "result": new_result}
                else:
                    logger.warning(
                        "TTS regen attempt %d failed for shot %s: %s",
                        attempt + 1, shot_id, reason,
                    )
            except Exception as e:
                logger.warning(
                    "TTS regen attempt %d exception for shot %s: %s",
                    attempt + 1, shot_id, e,
                )

        # 3. 恢复失败：返回原 seg（result 仍为无效）
        logger.error(
            "TTS recovery exhausted for shot %s: keeping invalid result",
            shot_id,
        )
        return seg

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
