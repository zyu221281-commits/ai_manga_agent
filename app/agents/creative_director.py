"""Creative Director Agent — multi-concept exploration + creative guidance.

Takes a raw creative brief and generates 3-5 distinct creative concepts,
evaluates each for viral potential / differentiation / executability,
selects the best direction, and outputs creative guidance for the Planner.

V5: Creative Layer — the meta-level creative engine.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from app.agents.base import BaseAgent, AgentResult


CREATIVE_CONCEPT_PROMPT = """You are a top-tier manga/anime creative director with viral content instincts.

Generate 3-5 DISTINCT creative concepts from this brief. Each concept should have a UNIQUE angle
— don't just rephrase the same idea. Think: different tones, different twists, different emotional cores.

# Creative Brief
{creative_brief}

# Current Hot Trends (for creative reference)
{hot_trends}

# Requirements for EACH concept:
1. concept_name: A punchy name for this creative direction (≤15 chars)
2. tone: The dominant emotional tone (pick one primary + one secondary)
3. twist_pattern: The unique narrative trick/mechanic that makes this stand out
4. emotional_arc: The emotional journey across {episode_count} episodes (3-phase arc)
5. target_hook: The 15-word hook that makes viewers click
6. visual_style_keyword: 3-5 keywords defining the visual aesthetic
7. differentiation: Why this is different from typical stories in this genre
8. viral_potential: Estimated viral score 0-1 with brief justification

# Output (strict JSON):
{{
  "concepts": [
    {{
      "concept_name": "creative direction name",
      "tone": {{"primary": "dark|light|tense|warm|quirky|epic", "secondary": "..."}},
      "twist_pattern": "describe the unique narrative mechanic",
      "emotional_arc": {{"phase1": "early episodes", "phase2": "middle episodes", "phase3": "late episodes"}},
      "target_hook": "one-line hook",
      "visual_style_keyword": ["keyword1", "keyword2", "keyword3"],
      "differentiation": "why different",
      "viral_potential": 0.85
    }}
  ],
  "recommended_index": 0,
  "director_note": "Why this concept was chosen"
}}"""


CREATIVE_GUIDANCE_PROMPT = """As Creative Director, distill the selected concept into actionable guidance for the Planner agent.

# Selected Concept
{selected_concept}

# Original Brief
{creative_brief}

# Output (strict JSON) — this goes directly to Planner:
{{
  "tone": "primary emotional tone",
  "twist_frequency": "high|medium|low",
  "twist_types": ["identity_reveal", "power_inversion", "moral_dilemma", "unexpected_ally"],
  "character_arc_type": "redemption|corruption|discovery|transformation|resistance",
  "hook_style": "cliffhanger|mystery_box|emotional_punch|plot_twist|prophecy",
  "pacing": "fast|medium|slow_burn",
  "genre_fusion": "combined genre label",
  "vibe_keywords": ["keyword1", "keyword2", "keyword3"],
  "must_have_elements": ["element that MUST appear in the story"]
}}"""


class CreativeDirectorAgent(BaseAgent):
    """Creative Director: generates + evaluates multiple creative concepts,
    selects the best one, and produces actionable guidance for Planner.

    Pipeline: CreativeBrief → CreativeDirector → Planner → ...
    """

    agent_name = "creative_director"

    async def execute(
        self,
        creative_brief: dict[str, Any],
        hot_trends: Optional[list[str]] = None,
        num_concepts: int = 4,
    ) -> AgentResult:
        """Generate creative concepts and select best direction.

        Args:
            creative_brief: Raw creative brief from RPA layer
            hot_trends: Current trend keywords
            num_concepts: Number of concepts to generate (3-5)

        Returns:
            AgentResult.data = {{
                "concepts": [...],
                "selected": {{...}},
                "creative_guidance": {{...}},
            }}
        """
        # Phase 1: Generate multiple creative concepts
        concepts = await self._generate_concepts(creative_brief, hot_trends, num_concepts)
        if not concepts:
            return AgentResult(success=False, error="No concepts generated")

        # Phase 2: Evaluate and select best concept
        selected_idx, director_note = await self._evaluate_concepts(
            concepts, creative_brief
        )
        selected = concepts[selected_idx] if 0 <= selected_idx < len(concepts) else concepts[0]

        # Phase 3: Generate creative guidance for Planner
        guidance = await self._generate_guidance(selected, creative_brief)

        result = {
            "concepts": concepts,
            "selected": selected,
            "creative_guidance": guidance,
            "director_note": director_note,
            "total_concepts": len(concepts),
        }

        return AgentResult(
            success=True,
            data=result,
            cost_usd=0.05,
            metadata={
                "concepts_count": len(concepts),
                "selected": selected.get("concept_name", ""),
                "viral_score": selected.get("viral_potential", 0),
            },
        )

    async def _generate_concepts(
        self,
        creative_brief: dict[str, Any],
        hot_trends: Optional[list[str]],
        num_concepts: int,
    ) -> list[dict]:
        """Phase 1: Generate 3-5 distinct creative concepts."""
        # 集数：从 brief 读取，回退到 settings.DEFAULT_TOTAL_EPISODES
        from app.core.config import settings
        try:
            episode_count = int(creative_brief.get("episode_count", settings.DEFAULT_TOTAL_EPISODES))
        except (TypeError, ValueError):
            episode_count = settings.DEFAULT_TOTAL_EPISODES
        episode_count = max(1, min(episode_count, settings.MAX_TOTAL_EPISODES))

        prompt = CREATIVE_CONCEPT_PROMPT.format(
            creative_brief=json.dumps(creative_brief, ensure_ascii=False, indent=2),
            hot_trends=json.dumps(hot_trends or [], ensure_ascii=False),
            episode_count=episode_count,
        )
        prompt += f"\n\nPlease generate exactly {num_concepts} distinct concepts."

        messages = [
            {"role": "system", "content": "You are a creative director with deep understanding of viral manga content. Output strict JSON."},
            {"role": "user", "content": prompt},
        ]

        try:
            result = await self._llm_json(
                messages=messages,
                model="deepseek-v4-pro",
                temperature=0.9,
                max_tokens=8192,
            )
        except Exception as e:
            self.logger.error("Concept generation failed: %s", e)
            return []

        return result.get("concepts", [])

    async def _evaluate_concepts(
        self,
        concepts: list[dict],
        creative_brief: dict[str, Any],
    ) -> tuple[int, str]:
        """Phase 2: Select the best concept via LLM evaluation."""
        eval_prompt = f"""Rate these creative concepts and pick the best one.

# Original Brief
{json.dumps(creative_brief, ensure_ascii=False)[:500]}

# Concepts
{json.dumps([{"idx": i, "name": c.get("concept_name", ""), "tone": c.get("tone", {}), "differentiation": c.get("differentiation", ""), "viral_potential": c.get("viral_potential", 0)} for i, c in enumerate(concepts)], ensure_ascii=False)}

# Output (strict JSON):
{{"best_idx": 0, "reason": "why this concept is best"}}"""

        try:
            result = await self._llm_json(
                messages=[
                    {"role": "system", "content": "You are a creative director. Pick the single most viral-worthy concept."},
                    {"role": "user", "content": eval_prompt},
                ],
                model="deepseek-v4-pro",
                temperature=0.5,
                max_tokens=1024,
            )
            return result.get("best_idx", 0), result.get("reason", "")
        except Exception as e:
            self.logger.warning("Concept evaluation failed: %s, using heuristic", e)
            best = max(
                range(len(concepts)),
                key=lambda i: concepts[i].get("viral_potential", 0),
            )
            return best, f"Heuristic selection (viral_score={concepts[best].get('viral_potential', 0)})"

    async def _generate_guidance(
        self,
        selected_concept: dict[str, Any],
        creative_brief: dict[str, Any],
    ) -> dict[str, Any]:
        """Phase 3: Convert creative concept into actionable Planner guidance."""
        prompt = CREATIVE_GUIDANCE_PROMPT.format(
            selected_concept=json.dumps(selected_concept, ensure_ascii=False, indent=2),
            creative_brief=json.dumps(creative_brief, ensure_ascii=False, indent=2)[:1000],
        )

        try:
            return await self._llm_json(
                messages=[
                    {"role": "system", "content": "You are a creative director translating concepts into production guidance. Output strict JSON."},
                    {"role": "user", "content": prompt},
                ],
                model="deepseek-v4-pro",
                temperature=0.6,
                max_tokens=2048,
            )
        except Exception as e:
            self.logger.warning("Guidance generation failed: %s, using defaults", e)
            return {
                "tone": selected_concept.get("tone", {}).get("primary", "neutral"),
                "twist_frequency": "medium",
                "twist_types": ["identity_reveal"],
                "character_arc_type": "discovery",
                "hook_style": "cliffhanger",
                "pacing": "medium",
                "genre_fusion": "fantasy",
                "vibe_keywords": [],
                "must_have_elements": [],
            }
