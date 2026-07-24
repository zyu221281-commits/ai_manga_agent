"""Lightweight pipeline engine with cross-episode memory support."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from app.services.long_term_memory import (
    long_term_memory,
    extract_and_store_episode_memory,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineStep:
    name: str
    status: str = "pending"
    result: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    cost_usd: float = 0.0


@dataclass
class PipelineResult:
    episode_id: str
    success: bool
    steps: list[PipelineStep] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_duration_ms: float = 0.0
    output: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    foreshadowing_stats: Optional[dict[str, Any]] = None


class LightweightEngine:
    """In-process sequential pipeline for demo / development."""

    async def run(
        self,
        episode_id: Optional[str] = None,
        series_id: Optional[str] = None,
        creative_brief: Optional[dict] = None,
    ) -> PipelineResult:
        import time
        ep_id = episode_id or str(uuid.uuid4())
        total_start = time.monotonic()
        steps: list[PipelineStep] = []
        total_cost = 0.0

        state = {
            "episode_id": ep_id,
            "series_id": series_id or "",
            "episode_num": 1,
            "creative_brief": creative_brief or {},
            "retry_count": 0,
            "total_cost_usd": 0.0,
            "character_anchors": None,
            "previous_summary": None,
        }

        # Creative Director
        cd_result = await self._run_step("creative_director", state)
        steps.append(cd_result)
        if cd_result.result:
            state["creative_guidance"] = cd_result.result.get("creative_guidance", {})

        # Planner
        planner_result = await self._run_step("planner", state)
        steps.append(planner_result)
        if not planner_result.result:
            return PipelineResult(episode_id=ep_id, success=False, steps=steps, error=planner_result.error)
        total_cost += planner_result.cost_usd
        state["series_plan"] = planner_result.result

        # StoryCritic
        story_result = await self._run_step("story_critic", state)
        steps.append(story_result)
        total_cost += story_result.cost_usd

        # Writer
        writer_result = await self._run_step("writer", state)
        steps.append(writer_result)
        if not writer_result.result:
            return PipelineResult(episode_id=ep_id, success=False, steps=steps, error=writer_result.error)
        total_cost += writer_result.cost_usd
        state["script"] = writer_result.result.get("script")
        state["storyboard"] = writer_result.result.get("storyboard")
        state["image_prompts"] = writer_result.result.get("image_prompts")

        # Store in long-term memory
        script_data = writer_result.result.get("script", {})
        ep_plan = (state.get("series_plan", {}).get("episodes", []) or [{}])[0] if state.get("series_plan") else {}
        await extract_and_store_episode_memory(
            episode_num=state.get("episode_num", 1),
            script_data=script_data,
            episode_plan=ep_plan,
        )

        # Process foreshadowing
        foreshadowing_data = writer_result.result.get("foreshadowing", {})
        for resolved_key in foreshadowing_data.get("resolved", []):
            await long_term_memory.resolve_foreshadowing(str(resolved_key), state.get("episode_num", 1))
        for planted_desc in foreshadowing_data.get("planted", []):
            await long_term_memory.add_foreshadowing(
                key=f"ep{state.get('episode_num', 1)}_planted_{hash(str(planted_desc)) % 10000}",
                description=str(planted_desc),
                episode_num=state.get("episode_num", 1),
                expected_resolve_episode=state.get("episode_num", 1) + 5,
            )

        # AssetManager
        asset_result = await self._run_step("asset_manager", state)
        steps.append(asset_result)
        total_cost += asset_result.cost_usd
        state["asset_library"] = asset_result.result

        # Composer
        composer_result = await self._run_step("composer", state)
        steps.append(composer_result)
        if not composer_result.result:
            return PipelineResult(episode_id=ep_id, success=False, steps=steps, error=composer_result.error)
        total_cost += composer_result.cost_usd
        state["episode_asset"] = composer_result.result

        # Critic
        critic_result = await self._run_step("critic", state)
        steps.append(critic_result)
        total_cost += critic_result.cost_usd

        total_duration = (time.monotonic() - total_start) * 1000

        success = all(
            s.status in ("completed", "pending") for s in steps if s.name != "critic"
        ) and steps[-1].result.get("score", 0) >= 0.6

        fs_stats = await long_term_memory.get_foreshadowing_stats()

        return PipelineResult(
            episode_id=ep_id,
            success=success,
            steps=steps,
            total_cost_usd=total_cost,
            total_duration_ms=total_duration,
            output={"episode_asset": state.get("episode_asset"), "critic_score": state.get("critic_score")},
            foreshadowing_stats=fs_stats,
        )

    async def _run_step(self, agent_name: str, state: dict) -> PipelineStep:
        import time
        step = PipelineStep(name=agent_name, status="running")
        start = time.monotonic()

        try:
            from app.agents.creative_director import CreativeDirectorAgent
            from app.agents.planner import PlannerAgent
            from app.agents.story_critic import StoryCriticAgent
            from app.agents.writer import WriterAgent
            from app.agents.asset_manager import AssetManagerAgent
            from app.agents.composer import ComposerAgent
            from app.agents.critic import CriticAgent

            agent_map = {
                "creative_director": CreativeDirectorAgent,
                "planner": PlannerAgent,
                "story_critic": StoryCriticAgent,
                "writer": WriterAgent,
                "asset_manager": AssetManagerAgent,
                "composer": ComposerAgent,
                "critic": CriticAgent,
            }

            agent_cls = agent_map.get(agent_name)
            if agent_cls is None:
                step.status = "failed"
                step.error = f"Unknown agent: {agent_name}"
                return step

            agent = agent_cls(
                episode_id=state.get("episode_id"),
                series_id=state.get("series_id"),
            )
            result = await agent._run_with_tracking(state=state)

            step.duration_ms = (time.monotonic() - start) * 1000
            step.cost_usd = result.cost_usd

            if result.success:
                step.status = "completed"
                step.result = result.data
            else:
                step.status = "failed"
                step.error = result.error

            return step
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.duration_ms = (time.monotonic() - start) * 1000
            return step


lightweight_engine = LightweightEngine()
