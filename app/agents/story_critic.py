"""StoryCritic Agent：大纲吸引力评估（ 前置质量门）

 文档 6.2：
- Planner → StoryCritic → Writer
- 双模型投票：DeepSeek + Qwen 各打一次分取平均
- 评估 5 维度：conflict_density / twist_frequency / character_arc / hook_design / topic_match
- 单集 score < 0.7 → 重写该集（≤2 次）
- 整体 score < 0.5 → 整体重写（≤2 次）
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

from app.agents.base import BaseAgent, AgentResult


STORY_CRITIC_PROMPT = """你是漫剧爆款策划师，请评估以下 {episode_count} 集漫剧大纲的吸引力。

# 大纲
{outline_json}

# 评估要求
对每一集单独打分（0-1），并给出整体评分。从 5 个维度评估：
1. conflict_density: 剧情冲突密度（每集是否有明确冲突点）
2. twist_frequency: 反转频次（每 5 集至少 1 次反转，集数不足 5 集时按比例折算）
3. character_arc: 角色弧光（主角是否有成长轨迹）
4. hook_design: 钩子设计（每集结尾是否有悬念）
5. topic_match: 题材热度（与当前爆款题材的匹配度）

# 输出格式（严格 JSON）
{{
  "outline_score": 0.0-1.0,
  "episode_scores": [
    {{"episode_num": 1, "score": 0.0-1.0, "weak_dimensions": ["conflict_density"]}}
  ],
  "rewrite_episodes": [3, 17, 42],
  "suggestion": "改进建议（≤200 字）"
}}
"""


class StoryCriticAgent(BaseAgent):
    """StoryCritic Agent：大纲评估 + 重写决策。"""

    agent_name = "story_critic"

    # 阈值（可从 config 读取覆盖）
    EPISODE_PASS_THRESHOLD = 0.7
    OUTLINE_PASS_THRESHOLD = 0.5
    MAX_REWRITE = 2

    async def execute(
        self,
        series_plan: dict[str, Any],
        hot_topics: Optional[list[str]] = None,
        rewrite_count: int = 0,
    ) -> AgentResult:
        """评估大纲质量。

        Args:
            series_plan: Planner 输出的 SeriesPlan
            hot_topics: 当前爆款题材（可选注入）
            rewrite_count: 已重写次数
        """
        if rewrite_count >= self.MAX_REWRITE:
            return AgentResult(
                success=True,
                data={
                    "outline_score": 0.0,
                    "episode_scores": [],
                    "rewrite_episodes": [],
                    "suggestion": "Max rewrites reached. Push to human review.",
                    "decision": "human_review",
                },
                metadata={"rewrite_exhausted": True},
            )

        episodes = series_plan.get("episodes", [])
        if not episodes:
            return AgentResult(success=False, error="No episodes to evaluate")

        outline_json = json.dumps(
            [{
                "episode_num": ep["episode_num"],
                "title": ep["title"],
                "plot_summary": ep["plot_summary"],
                "key_conflict": ep.get("key_conflict", ""),
                "hook": ep.get("hook", ""),
            } for ep in episodes],
            ensure_ascii=False,
        )

        prompt = STORY_CRITIC_PROMPT.format(
            outline_json=outline_json,
            episode_count=len(episodes),
        )
        messages = [
            {"role": "system", "content": "你是专业漫剧质量评审，输出严格的 JSON 格式。"},
            {"role": "user", "content": prompt},
        ]

        # 双模型投票：DeepSeek + Qwen
        try:
            results = await asyncio.gather(
                self._llm_json(messages=messages, model="deepseek--pro", temperature=0.3),
                self._llm_json(messages=messages, model="qwen3.7-max", temperature=0.3),
                return_exceptions=True,
            )
        except Exception as e:
            return AgentResult(success=False, error=f"Voting failed: {e}")

        # 处理结果
        scores = []
        for r in results:
            if isinstance(r, Exception):
                continue
            s = r.get("outline_score", 0.0)
            if isinstance(s, (int, float)):
                scores.append(float(s))

        if not scores:
            return AgentResult(success=False, error="All models failed")

        avg_score = sum(scores) / len(scores)
        episode_scores = self._collect_episode_scores(results)
        weak_episodes = self._identify_weak_episodes(episode_scores)

        # 决策
        decision = self._make_decision(avg_score, weak_episodes, rewrite_count)

        evaluation = {
            "outline_score": avg_score,
            "episode_scores": episode_scores,
            "rewrite_episodes": weak_episodes,
            "suggestion": self._collect_suggestions(results),
            "decision": decision,
            "models_used": len(scores),
        }

        # 记录血缘
        if self.lineage_tracker:
            await self.lineage_tracker.record(
                episode_id=self.episode_id or "series",
                artifact_type="outline_evaluation",
                artifact_data=evaluation,
                model_name="deepseek--pro+qwen3.7-max",
                model_params={"temperature": 0.3, "voting_models": 2},
                trace_id=self.trace_id,
            )

        return AgentResult(
            success=decision != "rewrite_all",
            data=evaluation,
            cost_usd=0.03,  # ~$0.03 per evaluation
            metadata={"avg_score": avg_score, "decision": decision},
        )

    def _collect_episode_scores(self, results: list) -> list[dict]:
        """从多个模型结果收集单集评分并取平均。"""
        all_scores: dict[int, list[float]] = {}
        for r in results:
            if isinstance(r, Exception):
                continue
            for ep in r.get("episode_scores", []):
                num = ep.get("episode_num", 0)
                score = ep.get("score", 0)
                if isinstance(score, (int, float)):
                    if num not in all_scores:
                        all_scores[num] = []
                    all_scores[num].append(float(score))

        return [
            {
                "episode_num": num,
                "score": sum(s) / len(s),
                "weak_dimensions": [],
            }
            for num, s in sorted(all_scores.items())
        ]

    def _identify_weak_episodes(self, episode_scores: list[dict]) -> list[int]:
        """识别低于阈值的集数。"""
        return [
            ep["episode_num"]
            for ep in episode_scores
            if ep["score"] < self.EPISODE_PASS_THRESHOLD
        ]

    def _make_decision(self, avg_score: float, weak_episodes: list[int], rewrite_count: int) -> str:
        if avg_score >= self.OUTLINE_PASS_THRESHOLD and not weak_episodes:
            return "pass"
        if weak_episodes and rewrite_count < self.MAX_REWRITE:
            return "rewrite_episodes"
        if avg_score < self.OUTLINE_PASS_THRESHOLD and rewrite_count < self.MAX_REWRITE:
            return "rewrite_all"
        return "human_review"

    def _collect_suggestions(self, results: list) -> str:
        for r in results:
            if isinstance(r, Exception):
                continue
            s = r.get("suggestion", "")
            if s:
                return s
        return "No suggestions available"
