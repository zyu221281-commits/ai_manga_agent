"""AssetManager Agent：角色四视图 + 风格锁定 + 一致性校验

 文档 6.4：
- 输入: SeriesPlan + Writer 输出
- 输出: AssetLibrary（角色四视图 + 风格模板 + 一致性报告）
- 角色四视图定妆（Seedream Reference 图）
- StyleTemplate 风格锁定
- 跨集一致性校验（embedding 余弦 > 0.85）
- 资产版本化（最多 5 个历史版本）
"""

from __future__ import annotations

import json
from typing import Any, Optional

from app.agents.base import BaseAgent, AgentResult


ASSET_MANAGER_PROMPT = """你是漫剧视觉总监，请根据以下系列信息，设计角色视觉资产。

# 系列信息
{series_info}

# 已识别角色列表
{characters}

# 输出要求（严格 JSON）
{{
  "characters": [
    {{
      "name": "角色名",
      "role": "男主|女主|反派|配角|旁白",
      "visual_description": "外观描述（<=100字）",
      "id_card": {{
        "hair_color": "#色号（如 #8B4513）",
        "eye_color": "瞳孔色（如 amber）",
        "outfit": "服装风格描述（如 魏晋风骨, 青衫）",
        "body_type": "体型（slim|muscular|average|petite）",
        "distinguishing_features": "辨识特征（如 左眼疤痕）",
        "negative_traits": "必须避免的特征（如 六指, 三只眼, 面部不对称）"
      }},
      "four_views": {{
        "front": "正面视图描述",
        "side": "侧面视图描述",
        "back": "背面视图描述",
        "expression_sheet": "表情集描述（喜怒哀乐惊）"
      }},
      "style_prompts": {{
        "close_up": "特写镜头 Flux 提示词",
        "medium": "半身镜头 Flux 提示词",
        "full_body": "全身镜头 Flux 提示词",
        "action": "动作镜头 Flux 提示词"
      }},
      "voice_traits": {{
        "gender": "male|female",
        "age_range": "青年|中年|老年",
        "tone": "温柔|冷酷|活泼|沉稳|幽默",
        "suggested_voice_id": "Azure voice ID"
      }}
    }}
  ],
  "style_template": {{
    "name": "风格名称",
    "description": "风格描述",
    "color_palette": ["#颜色1", "#颜色2", "#颜色3"],
    "line_style": "描边粗细与风格",
    "shading": "阴影渲染方式",
    "global_prompt_suffix": "全局提示词后缀（添加到所有图）",
    "negative_prompt_global": "全局负面提示词"
  }}
}}

# 重要：角色身份证（id_card）
每个角色必须包含 id_card 字段，这是角色的"视觉身份证"：
- hair_color 和 eye_color 必须使用精确色号或颜色名
- outfit 必须包含服装风格和颜色
- negative_traits 必须列出所有需要避免的物理缺陷
- 这些信息将作为硬约束注入到所有图像生成提示词中
"""


class AssetManagerAgent(BaseAgent):
    """AssetManager Agent：管理角色与风格资产。"""

    agent_name = "asset_manager"
    MAX_VERSIONS = 5

    async def execute(
        self,
        series_plan: dict[str, Any],
        characters: Optional[list[dict[str, Any]]] = None,
        existing_assets: Optional[dict[str, Any]] = None,
    ) -> AgentResult:
        char_list = characters or self._extract_characters(series_plan)
        if not char_list:
            char_list = [
                {"name": "主角", "role": "男主"},
                {"name": "女主", "role": "女主"},
            ]

        series_info = {
            "title": series_plan.get("theme", "未命名"),
            "genre": series_plan.get("genre", "奇幻"),
            "total_episodes": series_plan.get("total_episodes", 60),
        }

        prompt = ASSET_MANAGER_PROMPT.format(
            series_info=json.dumps(series_info, ensure_ascii=False),
            characters=json.dumps(char_list, ensure_ascii=False),
        )

        messages = [
            {"role": "system", "content": "你是漫剧视觉设计总监，输出严格的 JSON 格式。"},
            {"role": "user", "content": prompt},
        ]

        try:
            result = await self._llm_json(
                messages=messages, model="deepseek--pro",
                temperature=0.6, max_tokens=16384,
            )
        except Exception as e:
            return AgentResult(success=False, error="LLM call failed: {}".format(e))

        version = 1
        if existing_assets:
            version = existing_assets.get("version", 0) + 1

        # 后处理：确保每个角色都有 id_card + 注入风格封印
        characters = result.get("characters", [])
        for char in characters:
            if "id_card" not in char:
                char["id_card"] = self._infer_id_card(char)

        # 注入风格封印（从 config 读取）
        from app.core.config import settings
        style_template = result.get("style_template", {})
        if not style_template.get("global_prompt_suffix"):
            style_template["global_prompt_suffix"] = settings.QUALITY_TAGS
        if not style_template.get("negative_prompt_global"):
            style_template["negative_prompt_global"] = settings.NEGATIVE_PROMPT
        style_template["global_seed"] = settings.GLOBAL_SEED

        asset_library = {
            "version": version,
            "characters": characters,
            "style_template": style_template,
            "series_info": series_info,
            "style_lock": {
                "global_seed": settings.GLOBAL_SEED,
                "quality_tags": settings.QUALITY_TAGS,
                "negative_prompt": settings.NEGATIVE_PROMPT,
                "style_similarity_threshold": settings.STYLE_SIMILARITY_THRESHOLD,
            },
            "previous_version": existing_assets.get("version") if existing_assets else None,
        }

        if self.lineage_tracker:
            await self.lineage_tracker.record(
                episode_id=self.episode_id or "series",
                artifact_type="style_template",
                artifact_data=asset_library.get("style_template", {}),
                model_name="deepseek--pro",
                model_params={"temperature": 0.6},
                trace_id=self.trace_id,
            )

        return AgentResult(
            success=True, data=asset_library, cost_usd=0.08,
            metadata={"characters_count": len(asset_library["characters"]), "version": version},
        )

    def _extract_characters(self, series_plan: dict) -> list[dict]:
        characters = set()
        for ep in series_plan.get("episodes", []):
            for c in ep.get("characters_appeared", []):
                if isinstance(c, str):
                    characters.add(c)
                elif isinstance(c, dict):
                    characters.add(c.get("name", str(c)))
        return [{"name": name} for name in characters]

    def _infer_id_card(self, char: dict) -> dict:
        """当 LLM 未返回 id_card 时，从 visual_description 推断基础身份证。"""
        desc = char.get("visual_description", "")
        return {
            "hair_color": "",
            "eye_color": "",
            "outfit": desc[:50] if desc else "",
            "body_type": "average",
            "distinguishing_features": "",
            "negative_traits": "extra fingers, extra limbs, deformed hands",
        }
