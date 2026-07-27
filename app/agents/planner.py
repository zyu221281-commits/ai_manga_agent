"""Planner Agent：系列大纲 + 调度 + 记忆注入

V4 文档 6.1：
- 输入: CreativeBrief（来自 RPA 采集层）
- 输出: SeriesPlan（系列大纲 + 调度顺序 + 记忆摘要）
- LLM: DeepSeek-V4-Pro
- 成本上报: 每次 LLM 调用记录到 cost_ledger
- 集数来源（优先级）：creative_brief["episode_count"] > settings.DEFAULT_TOTAL_EPISODES
"""

from __future__ import annotations

import json
from typing import Any, Optional


from app.agents.base import BaseAgent, AgentResult


PLANNER_PROMPT = """你是漫剧编剧策划师，请根据以下创意简报，生成一份 {episode_count} 集的漫剧大纲。

# 创意简报
{creative_brief}

# 历史爆款特征参考
{hot_trends}

# 输出要求
生成严格的 JSON 格式，包含 {episode_count} 集大纲，每集包含：
- episode_num: 集数 (1-{episode_count})
- title: 单集标题 (≤15字)
- plot_summary: 剧情摘要 (≤100字)
- key_conflict: 核心冲突点
- hook: 结尾钩子/悬念
- emotion_tags: 情绪标签列表
- characters_appeared: 本集出场角色列表
- estimated_duration: 预估时长（秒）

要求：
1. 前 3 集必须有强力钩子，确保用户留存
2. 每 5 集至少 1 次反转
3. 主角有清晰成长弧线
4. 每集结尾留悬念
5. 整体节奏紧凑，避免灌水
"""


class PlannerAgent(BaseAgent):
    """Planner Agent：生成系列大纲 + 调度顺序。"""

    agent_name = "planner"

    async def execute(
        self,
        creative_brief: dict[str, Any],
        hot_trends: Optional[list[str]] = None,
        creative_guidance: Optional[dict[str, Any]] = None,
    ) -> AgentResult:
        """执行大纲规划。

        Args:
            creative_brief: RPA 采集层输出的创意简报
                - 可选字段 episode_count: 用户指定集数（不传则使用 settings.DEFAULT_TOTAL_EPISODES）
            hot_trends: 热度趋势关键词列表（可选）
            creative_guidance: Creative Director 输出的创意指导方针（可选）
        """
        # 集数来源（优先级）：
        #   1. creative_brief["episode_count"]（用户输入）
        #   2. settings.DEFAULT_TOTAL_EPISODES（默认 30）
        # 上限保护：MAX_TOTAL_EPISODES（避免误输入导致 token 爆炸）
        from app.core.config import settings
        default_eps = settings.DEFAULT_TOTAL_EPISODES
        max_eps = settings.MAX_TOTAL_EPISODES
        try:
            episode_count = int(creative_brief.get("episode_count", default_eps))
        except (TypeError, ValueError):
            episode_count = default_eps
        episode_count = max(1, min(episode_count, max_eps))

        guidance_text = ""
        if creative_guidance:
            guidance_text = "\n# 创意指导方针（来自 Creative Director）\n" + json.dumps(creative_guidance, ensure_ascii=False, indent=2)
        prompt = PLANNER_PROMPT.format(
            creative_brief=json.dumps(creative_brief, ensure_ascii=False, indent=2),
            hot_trends=json.dumps(hot_trends or [], ensure_ascii=False),
            episode_count=episode_count,
        ) + guidance_text

        messages = [
            {"role": "system", "content": "你是专业的漫剧策划师，输出严格的 JSON 格式。"},
            {"role": "user", "content": prompt},
        ]

        try:
            plan_json = await self._llm_json(
                messages=messages,
                model="deepseek-v4-pro",
                temperature=0.8,
                max_tokens=32768,
            )
        except Exception as e:
            return AgentResult(success=False, error=f"LLM call failed: {e}")

        # 解析并验证
        episodes = plan_json.get("episodes", []) if isinstance(plan_json, dict) else []
        if len(episodes) < episode_count:
            # 尝试从不同 key 获取
            for key in ["outline", "series_plan", "episode_list"]:
                if key in plan_json and isinstance(plan_json[key], list):
                    episodes = plan_json[key]
                    break

        series_plan = {
            "total_episodes": len(episodes),
            "episodes": self._normalize_episodes(episodes),
            "schedule_order": list(range(1, len(episodes) + 1)),
            "global_hooks": plan_json.get("global_hooks", []),
            "character_arcs": plan_json.get("character_arcs", {}),
            "theme": creative_brief.get("theme", ""),
            "genre": creative_brief.get("genre", ""),
        }

        # 记录血缘
        if self.lineage_tracker:
            await self.lineage_tracker.record(
                episode_id=self.episode_id or "series",
                artifact_type="series_plan",
                artifact_data=series_plan,
                model_name="deepseek-v4-pro",
                model_params={"temperature": 0.8, "max_tokens": 32768},
                trace_id=self.trace_id,
            )

        return AgentResult(
            success=True,
            data=series_plan,
            cost_usd=self._estimate_cost(episodes),
            metadata={"episodes_count": len(episodes)},
        )

    def _normalize_episodes(self, raw_episodes: list[dict]) -> list[dict]:
        """标准化单集大纲格式。"""
        normalized = []
        for i, ep in enumerate(raw_episodes):
            normalized.append({
                "episode_num": ep.get("episode_num", i + 1),
                "title": ep.get("title", f"第{i + 1}集"),
                "plot_summary": ep.get("plot_summary", ""),
                "key_conflict": ep.get("key_conflict", ""),
                "hook": ep.get("hook", ""),
                "emotion_tags": ep.get("emotion_tags", []),
                "characters_appeared": ep.get("characters_appeared", []),
                "estimated_duration": ep.get("estimated_duration", 60),
            })
        return normalized

    def _estimate_cost(self, episodes: list[dict]) -> float:
        """估算 Planner 成本（~30K tokens input + ~10K tokens output）。"""
        return 30000 / 1_000_000 * 0.27 + 10000 / 1_000_000 * 1.10  # ~$0.02
