"""Critic Agent - quality scoring + 3-tier gate with foreshadowing dimension."""

from __future__ import annotations

import json
from typing import Any

from app.agents.base import BaseAgent, AgentResult
from app.core.config import settings


CRITIC_PROMPT = """You are the director. Score this episode.

# Script
{script}

# Images
{images}

# Audio
{audio}

# Dimensions (0-1 each)
1. script_quality: narrative rhythm, conflict, hooks
2. visual_quality: visual appeal, style consistency
3. voice_quality: voice matching, expressiveness
4. consistency: character and style consistency
5. subtitle_quality: alignment, styling
6. bgm_quality: BGM matching
7. foreshadowing_continuity: whether prior foreshadowing is addressed

# Output (strict JSON)
{{
  "overall_score": 0.0-1.0,
  "dimensions": {{
    "script_quality": 0.0-1.0,
    "visual_quality": 0.0-1.0,
    "voice_quality": 0.0-1.0,
    "consistency": 0.0-1.0,
    "subtitle_quality": 0.0-1.0,
    "bgm_quality": 0.0-1.0,
    "foreshadowing_continuity": 0.0-1.0
  }},
  "verdict": "pass|review|retry",
  "retry_reason": "if retry: specific improvement suggestions",
  "highlights": ["episode strengths"],
  "issues": ["issues to fix"]
}}
"""


class CriticAgent(BaseAgent):
    """Critic Agent: 3-tier quality gate."""

    agent_name = "critic"

    @property
    def pass_threshold(self) -> float:
        return settings.CRITIC_PASS_THRESHOLD

    @property
    def review_threshold(self) -> float:
        return settings.CRITIC_REVIEW_THRESHOLD

    @property
    def max_retry(self) -> int:
        return settings.CRITIC_MAX_RETRY

    async def execute(
        self,
        episode_asset: dict[str, Any],
        retry_count: int = 0,
    ) -> AgentResult:
        script = json.dumps(episode_asset.get("script", {}), ensure_ascii=False, indent=2)[:3000]
        images = json.dumps(
            [{"prompt": img.get("prompt", "")} for img in episode_asset.get("images", [])],
            ensure_ascii=False,
        )[:2000]
        audio = json.dumps(
            [{"voice_id": a.get("voice_id", ""), "text": a.get("text", "")[:50]}
             for a in episode_asset.get("audio_segments", [])],
            ensure_ascii=False,
        )[:2000]

        prompt = CRITIC_PROMPT.format(
            script=script,
            images=images,
            audio=audio,
        )

        messages = [
            {"role": "system", "content": "You are the director. Output strict JSON."},
            {"role": "user", "content": prompt},
        ]

        try:
            result = await self._llm_json(
                messages=messages,
                model="qwen-vl-max",
                temperature=0.3,
                max_tokens=4096,
            )
        except Exception as e:
            return AgentResult(
                success=True,
                data={
                    "overall_score": 0.0,
                    "dimensions": {},
                    "verdict": "review",
                    "retry_reason": f"LLM evaluation failed: {e}",
                    "highlights": [],
                    "issues": [],
                },
                metadata={"error": str(e)},
            )

        score = result.get("overall_score", 0.0)
        decision = self._decide(score, retry_count)

        evaluation = {
            **result,
            "decision": decision,
            "retry_count": retry_count,
            "thresholds": {
                "pass": self.pass_threshold,
                "review": self.review_threshold,
                "max_retry": self.max_retry,
            },
        }

        if self.lineage_tracker:
            await self.lineage_tracker.record(
                episode_id=self.episode_id or "",
                artifact_type="critic_evaluation",
                artifact_data=evaluation,
                model_name="qwen-vl-max",
                model_params={"temperature": 0.3},
                trace_id=self.trace_id,
            )

        return AgentResult(
            success=decision == "pass",
            data=evaluation,
            cost_usd=0.003,
            metadata={
                "score": score,
                "decision": decision,
                "retry_count": retry_count,
            },
        )

    def _decide(self, score: float, retry_count: int) -> str:
        if score >= self.pass_threshold:
            return "pass"
        if score >= self.review_threshold:
            return "review"
        if retry_count < self.max_retry:
            return "retry"
        return "review"
