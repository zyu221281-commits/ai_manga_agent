import asyncio
import logging
from typing import Optional
from app.agents.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


class ComposerPipeline:
    def __init__(self, context: PipelineContext, agent):
        self.context = context
        self.agent = agent
        self._import_settings()

    def _import_settings(self):
        from app.core.config import settings
        self.settings = settings
        self.ratio = self.context.key_scene_ratio if self.context.key_scene_ratio is not None else settings.VIDEO_KEY_SCENE_RATIO

    async def _stage_classify_scenes(self) -> None:
        classified = self.agent._classify_scenes(
            self.context.storyboard,
            self.context.image_prompts,
            self.context.script,
        )

        # Inject visual_spec into each classified_scene for deterministic rendering
        # 让 _generate_images 能用 PromptTemplateEngine 渲染确定性 prompt + 按 camera_angle 选 ref
        visual_specs_by_shot = {vs.get("shot_id"): vs for vs in self.context.visual_specs}
        for cs in classified:
            shot_id = cs.get("shot_id")
            if shot_id is not None and shot_id in visual_specs_by_shot:
                cs["visual_spec"] = visual_specs_by_shot[shot_id]

        self.context.classified_scenes = classified
        self.context.update_metadata("key_scenes", sum(1 for s in classified if s["type"] == KEY_SCENE))
        self.context.update_metadata("normal_scenes", sum(1 for s in classified if s["type"] == NORMAL_SCENE))
        logger.info("Scene classification: %d key, %d normal",
                    self.context.metadata["key_scenes"], self.context.metadata["normal_scenes"])

    async def _stage_generate_images(self) -> None:
        image_results = await self.agent._generate_images(
            self.context.classified_scenes,
            self.context.asset_library,
        )
        self.context.image_results = image_results
        self.context.add_cost(sum(img.cost_usd for img in image_results))
        self.context.update_metadata("images_generated", len(image_results))
        logger.info("Images: %d generated", len(image_results))

    async def _stage_generate_audio(self) -> None:
        audio_segments, scene_subtitle_data = await self.agent._generate_audio(
            self.context.script,
            self.context.asset_library,
            self.context.storyboard,
        )
        self.context.audio_segments = audio_segments
        self.context.subtitles = scene_subtitle_data
        self.context.add_cost(sum(a["result"].cost_usd for a in audio_segments if a["result"]))
        self.context.update_metadata("audio_segments", len(audio_segments))
        self.context.update_metadata("subtitle_entries", len(scene_subtitle_data))
        logger.info("TTS: %d segments generated", len(audio_segments))

    def _stage_pre_video_gate(self) -> bool:
        return self.agent._pre_video_gate(self.context.image_results, self.context.classified_scenes)

    async def _stage_content_gate(self) -> None:
        if not self.settings.CONTENT_GATE_ENABLED:
            return

        from app.quality.content_gate import content_gate

        try:
            gate_result = await content_gate.check_images(
                self.context.image_results,
                self.context.classified_scenes,
                self.context.asset_library,
            )
            self.context.update_metadata("content_gate", gate_result.to_dict())
            if gate_result.flagged_shots:
                logger.warning("ContentGate flagged %d shots: %s",
                              len(gate_result.flagged_shots), gate_result.flagged_shots)
        except Exception as e:
            logger.warning("ContentGate error (non-blocking): %s", e)
            self.context.update_metadata("content_gate", {"verdict": "error", "error": str(e)})
            self.context.record_error("content_gate", e, non_blocking=True)

    async def _stage_vqa_check(self) -> None:
        if not self.settings.VQA_ENABLED:
            return

        from app.quality.vqa_checker import vqa_checker

        try:
            vqa_result = await vqa_checker.check_key_scenes(
                self.context.image_results,
                self.context.classified_scenes,
            )
            self.context.update_metadata("vqa_check", vqa_result.to_dict())
            if vqa_result.flagged_shots:
                logger.warning("VQA flagged %d KEY scenes: %s",
                              len(vqa_result.flagged_shots), vqa_result.flagged_shots)
        except Exception as e:
            logger.warning("VQA error (non-blocking): %s", e)
            self.context.update_metadata("vqa_check", {"verdict": "error", "error": str(e)})
            self.context.record_error("vqa_check", e, non_blocking=True)

    async def _stage_generate_videos(self) -> None:
        shot_durations: dict[int, float] = {}
        for a in self.context.audio_segments:
            tts = a.get("result")
            if not tts or not tts.duration_s:
                continue
            sid = a.get("shot_id")
            if sid is not None:
                shot_durations[sid] = shot_durations.get(sid, 0.0) + tts.duration_s

        video_segments = await self.agent._generate_videos(
            self.context.classified_scenes,
            self.context.image_results,
            self.ratio,
            shot_durations=shot_durations,
        )
        self.context.video_segments = video_segments
        self.context.add_cost(sum(v.cost_usd for v in video_segments))
        self.context.update_metadata("videos_generated", len(video_segments))
        logger.info("Videos: %d generated", len(video_segments))

    def _stage_select_covers(self) -> None:
        covers = self.agent._select_cover_candidates(self.context.image_results)
        self.context.covers = covers
        self.context.update_metadata("covers", covers)

    async def _stage_compose_video(self) -> None:
        compose_result = await self.agent._compose_final_video(
            self.context.video_segments,
            self.context.audio_segments,
            self.context.subtitles,
            "",
            self.context.classified_scenes,
            self.context.episode_id,
            bgm_path="",
        )
        self.context.compose_result = compose_result
        self.context.add_cost(compose_result.get("extra_tts_cost", 0.0))
        self.context.update_metadata("composition", compose_result)

    def _stage_build_ai_label(self) -> None:
        ai_label = self.agent._build_ai_label(
            self.context.script,
            self.context.metadata,
            self.context.total_cost_usd,
        )
        self.context.ai_label = ai_label

    async def run(self) -> Optional[str]:
        await self._stage_classify_scenes()

        image_task = asyncio.create_task(self._stage_generate_images())
        audio_task = asyncio.create_task(self._stage_generate_audio())

        await image_task

        if not self._stage_pre_video_gate():
            return "Pre-video gate: image failure rate exceeds threshold"

        await audio_task

        await self._stage_content_gate()
        await self._stage_vqa_check()

        await self._stage_generate_videos()
        self._stage_select_covers()
        await self._stage_compose_video()
        self._stage_build_ai_label()

        logger.info(
            "Pipeline finished: %d images, %d videos, %d audio, cost=%.4f USD",
            self.context.metadata["images_generated"],
            self.context.metadata["videos_generated"],
            self.context.metadata["audio_segments"],
            self.context.total_cost_usd,
        )

        return None