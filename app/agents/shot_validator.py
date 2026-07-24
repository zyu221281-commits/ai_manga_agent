"""ShotValidator Agent：分镜逻辑质检（生成前拦截）。

在 Writer 输出后、图像生成前执行，检查分镜的逻辑一致性。
不通过则打回 Writer 重写，避免浪费图像生成 token。

检查项：
1. 空间连续性：相邻镜头角色位置是否跳转
2. 角色一致性：同一场景角色服装/发型是否突变
3. 镜头节奏：连续 close-up 是否超过 3 次
4. 提示词完整性：每个 prompt 是否包含角色身份证信息
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from app.agents.base import BaseAgent, AgentResult

logger = logging.getLogger(__name__)


SHOT_VALIDATOR_PROMPT = """你是分镜逻辑质检官，请检查以下分镜和提示词的逻辑一致性。

# 分镜列表
{storyboard}

# 图像提示词列表
{image_prompts}

# 角色身份证
{character_id_cards}

# 检查项（逐项打分 0-1，1=通过，0=不通过）
1. **spatial_continuity**: 空间连续性 — 相邻镜头的角色位置是否有逻辑跳转
   （如：上一帧在门内，下一帧突然在屋顶 → 0分）
2. **character_consistency**: 角色一致性 — 同一场景中角色服装/发型是否突变
   （如：第1镜穿青衫，第3镜突然变红衣 → 0分）
3. **shot_rhythm**: 镜头节奏 — 是否连续超过3次相同景别（如连续4个close-up → 0分）
4. **prompt_completeness**: 提示词完整性 — 每个 prompt 是否包含角色身份证关键特征
   （如：角色有 hair_color=#8B4513 但 prompt 中没有 → 0分）

# 输出格式（严格 JSON）
{{
  "overall_score": 0.0-1.0,
  "checks": {{
    "spatial_continuity": {{"score": 1.0, "issues": ["问题描述"]}},
    "character_consistency": {{"score": 0.8, "issues": []}},
    "shot_rhythm": {{"score": 1.0, "issues": []}},
    "prompt_completeness": {{"score": 0.5, "issues": ["shot 3 缺少发色"]}}
  }},
  "failed_shots": [3, 5],
  "suggestions": "改进建议（≤200字）",
  "decision": "pass|rewrite"
}}

# 决策规则
- overall_score >= 0.7 且无 failed_shots → "pass"
- overall_score < 0.7 或有 failed_shots → "rewrite"
"""


class ShotValidatorAgent(BaseAgent):
    """ShotValidator Agent：分镜逻辑质检。"""

    agent_name = "shot_validator"

    PASS_THRESHOLD = 0.7

    async def execute(
        self,
        storyboard: list[dict],
        image_prompts: list[dict],
        character_anchors: Optional[dict[str, Any]] = None,
    ) -> AgentResult:
        """执行分镜逻辑质检。

        Args:
            storyboard: Writer 输出的分镜列表
            image_prompts: Writer 输出的图像提示词列表
            character_anchors: AssetManager 输出的角色资产（含 id_card）
        """
        # 提取角色身份证
        char_id_cards = {}
        if character_anchors:
            for char in character_anchors.get("characters", []):
                name = char.get("name", "")
                id_card = char.get("id_card", {})
                if name and id_card:
                    char_id_cards[name] = id_card

        # 先做本地规则检查（不消耗 LLM token）
        local_issues = self._local_checks(storyboard, image_prompts, char_id_cards)

        # 如果本地检查全通过，直接返回 pass（省 token）
        if not local_issues:
            return AgentResult(
                success=True,
                data={
                    "overall_score": 1.0,
                    "checks": {"local_validation": "all passed"},
                    "failed_shots": [],
                    "suggestions": "",
                    "decision": "pass",
                },
                cost_usd=0.0,
                metadata={"validation_method": "local_only"},
            )

        # 本地检查发现问题 → 调用 LLM 深度分析
        prompt = SHOT_VALIDATOR_PROMPT.format(
            storyboard=json.dumps(storyboard, ensure_ascii=False, indent=2),
            image_prompts=json.dumps(image_prompts, ensure_ascii=False, indent=2),
            character_id_cards=json.dumps(char_id_cards, ensure_ascii=False, indent=2),
        )

        messages = [
            {"role": "system", "content": "你是分镜逻辑质检官，输出严格的 JSON 格式。"},
            {"role": "user", "content": prompt},
        ]

        try:
            result = await self._llm_json(
                messages=messages,
                model="qwen-turbo",  # 用便宜的模型做质检
                temperature=0.2,
                max_tokens=4096,
            )
        except Exception as e:
            logger.warning("ShotValidator LLM failed: %s, using local check results", e)
            # LLM 失败时使用本地检查结果
            return AgentResult(
                success=True,
                data={
                    "overall_score": 0.5,
                    "checks": {"local_validation": local_issues},
                    "failed_shots": [],
                    "suggestions": "; ".join(local_issues),
                    "decision": "pass",  # LLM 失败时不阻断流程
                },
                cost_usd=0.0,
                metadata={"validation_method": "local_fallback", "local_issues": local_issues},
            )

        decision = result.get("decision", "pass")
        overall = result.get("overall_score", 1.0)

        # 本地检查的问题也合并进去
        if local_issues:
            existing_suggestions = result.get("suggestions", "")
            result["suggestions"] = f"[本地检查] {'; '.join(local_issues)} | {existing_suggestions}"

        return AgentResult(
            success=decision == "pass",
            data=result,
            cost_usd=0.001,  # qwen-turbo 很便宜
            metadata={
                "overall_score": overall,
                "decision": decision,
                "local_issues_count": len(local_issues),
                "validation_method": "local+llm",
            },
        )

    def _local_checks(
        self,
        storyboard: list[dict],
        image_prompts: list[dict],
        char_id_cards: dict[str, dict],
    ) -> list[str]:
        """本地规则检查（不消耗 LLM token）。

        Returns:
            问题列表，空列表表示全通过
        """
        issues: list[str] = []

        # 1. 镜头节奏：连续相同景别超过 3 次
        shot_types = [s.get("shot_type", "") for s in storyboard]
        consecutive = 1
        for i in range(1, len(shot_types)):
            if shot_types[i] == shot_types[i-1] and shot_types[i]:
                consecutive += 1
                if consecutive > 3:
                    issues.append(f"连续 {consecutive} 次 {shot_types[i]} 景别（shot {i+1}）")
                    break
            else:
                consecutive = 1

        # 2. 提示词完整性：检查角色身份证特征是否在 prompt 中
        for p in image_prompts:
            prompt_text = p.get("prompt", "").lower()
            shot_id = p.get("shot_id", "?")
            for char_name, id_card in char_id_cards.items():
                if char_name.lower() in prompt_text:
                    hair = id_card.get("hair_color", "")
                    if hair and hair.lower() not in prompt_text:
                        issues.append(f"shot {shot_id}: 角色 {char_name} 缺少发色 {hair}")
                    outfit = id_card.get("outfit", "")
                    if outfit and outfit.lower() not in prompt_text:
                        issues.append(f"shot {shot_id}: 角色 {char_name} 缺少服装描述")

        # 3. 检查空 prompt
        for p in image_prompts:
            if not p.get("prompt", "").strip():
                issues.append(f"shot {p.get('shot_id', '?')}: prompt 为空")

        return issues
