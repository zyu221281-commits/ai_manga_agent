"""VisualDescriptor + PromptTemplateEngine — structured visual description (normalized text)

设计目标：
- 在 Writer 和 Composer 之间加一层结构化视觉描述提取
- 将每个 shot 的角色外貌/场景元素/构图信息抽取为标准化 JSON（visual_spec）
- 用确定性 Jinja2 模板渲染为最终 prompt — 而非字符串拼接
- 角色 appearance 100% 从 character_anchors.id_card 原文复制（零变体）
- 仅 pose / expression / position 由每个 shot 独立描述（唯一自由度）
- 纯规则引擎，不调 LLM，零 API 成本

Schema 定义见下文。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from jinja2 import Environment, BaseLoader

logger = logging.getLogger(__name__)


# ================================================================
# 默认 Jinja2 模板（可被 PromptTemplateEngine.__init__ 覆盖）
# ================================================================

DEFAULT_PROMPT_TEMPLATE = """\
[{{ composition.shot_type }} shot, {{ composition.camera_angle }} angle]{% if composition.camera_motion %}, {{ composition.camera_motion }}{% endif %}{% if composition.depth %}, {{ composition.depth }}{% endif %}.
{% for char in characters %}Character "{{ char.name }}": {{ char.appearance.hair }}{% if char.appearance.eyes %}, {{ char.appearance.eyes }}{% endif %}{% if char.appearance.outfit %}, {{ char.appearance.outfit }}{% endif %}{% if char.appearance.body %}, {{ char.appearance.body }}{% endif %}{% if char.appearance.distinctive %}, {{ char.appearance.distinctive }}{% endif %}, {{ char.pose }}{% if char.expression %}, {{ char.expression }}{% endif %}{% if char.position %}, {{ char.position }}{% endif %}. {% endfor %}\
{% if scene.environment %}Scene: {{ scene.environment }}{% if scene.lighting %}, {{ scene.lighting }}{% endif %}{% if scene.color_tone %}, {{ scene.color_tone }}{% endif %}{% if scene.time %}, {{ scene.time }}{% endif %}{% if scene.weather %}, {{ scene.weather }}{% endif %}.{% endif %} \
Style: {{ style_suffix }}."""

# negative prompt 模板：全局负面词 + 角色级负面特征
DEFAULT_NEGATIVE_TEMPLATE = """\
{{ global_negative }}{% for char in characters %}{% if char.appearance.negative %}, {{ char.appearance.negative }}{% endif %}{% endfor %}"""


# ================================================================
# VisualDescriptor：从 storyboard + id_card 提取 visual_spec
# ================================================================

class VisualDescriptor:
    """从 Writer 输出 + character_anchors 构建结构化视觉描述。

    纯规则引擎，不调 LLM。角色 appearance 100% 从 id_card 原文复制。
    """

    def __init__(self, quality_tags: str = "", negative_global: str = ""):
        """可注入风格封印参数（默认从 settings 读取）。"""
        if not quality_tags or not negative_global:
            try:
                from app.core.config import settings
                quality_tags = quality_tags or settings.QUALITY_TAGS
                negative_global = negative_global or settings.NEGATIVE_PROMPT
            except Exception:
                pass
        self._quality_tags = quality_tags
        self._negative_global = negative_global

    # ---- 单 shot 构建 ----

    def build_visual_spec(
        self,
        shot: dict,
        character_anchors: dict,
        scene_data: dict,
        global_style: dict | None = None,
    ) -> dict:
        """返回 visual_spec JSON。角色 appearance 100% 从 id_card 原文复制。

        Args:
            shot: storyboard 中的一个 shot dict
            character_anchors: asset_library["characters"]（list[dict]）或
                               {name: char_dict} 映射
            scene_data: script["scenes"] 中对应的 scene dict
            global_style: style_template（可选，用于读取 global_prompt_suffix）
        """
        # 归一化 character_anchors 为 {name: char_dict}
        char_map = self._normalize_anchors(character_anchors)

        # 提取本 shot 涉及的角色
        shot_chars = self._extract_shot_characters(shot, scene_data, char_map)

        # 构建场景结构
        vc = scene_data.get("visual_context", {}) if scene_data else {}
        scene_block = {
            "environment": vc.get("environment", ""),
            "lighting": vc.get("lighting", ""),
            "color_tone": vc.get("color_tone", ""),
            "time": vc.get("time", ""),
            "weather": vc.get("weather", ""),
        }

        # 构图信息（camera_angle 由 Writer 在 storyboard 中输出）
        composition = {
            "shot_type": shot.get("shot_type", "medium"),
            "camera_angle": shot.get("camera_angle", "front"),
            "camera_motion": shot.get("camera_motion", ""),
            "depth": shot.get("depth", ""),
        }

        # 风格后缀：优先用 global_style.global_prompt_suffix，否则用 QUALITY_TAGS
        style_suffix = self._quality_tags
        if global_style and global_style.get("global_prompt_suffix"):
            style_suffix = global_style["global_prompt_suffix"]

        return {
            "shot_id": shot.get("shot_id"),
            "scene_id": shot.get("scene_id"),
            "characters": shot_chars,
            "scene": scene_block,
            "composition": composition,
            "style_suffix": style_suffix,
            "global_negative": self._negative_global,
        }

    # ---- 批量构建 ----

    def build_all(
        self,
        storyboard: list[dict],
        script: dict,
        asset_library: dict,
    ) -> list[dict]:
        """批量构建所有 shot 的 visual_specs。

        Args:
            storyboard: Writer 输出的 storyboard list
            script: Writer 输出的 script dict
            asset_library: AssetManager 输出的 asset_library
        """
        characters = asset_library.get("characters", []) if asset_library else []
        style_template = asset_library.get("style_template", {}) if asset_library else {}

        # scene_id -> scene_data 映射
        scenes_by_id: dict[int, dict] = {}
        for s in (script.get("scenes", []) if script else []):
            sid = s.get("scene_id")
            if sid is not None:
                scenes_by_id[sid] = s

        specs: list[dict] = []
        for shot in (storyboard or []):
            scene_data = scenes_by_id.get(shot.get("scene_id"), {})
            try:
                spec = self.build_visual_spec(
                    shot=shot,
                    character_anchors=characters,
                    scene_data=scene_data,
                    global_style=style_template,
                )
                specs.append(spec)
            except Exception as e:
                logger.warning(
                    "VisualDescriptor: shot %s spec build failed: %s",
                    shot.get("shot_id", "?"), e,
                )
                # 降级：最小 spec（保留 shot_id 让下游能继续）
                specs.append({
                    "shot_id": shot.get("shot_id"),
                    "scene_id": shot.get("scene_id"),
                    "characters": [],
                    "scene": {},
                    "composition": {
                        "shot_type": shot.get("shot_type", "medium"),
                        "camera_angle": shot.get("camera_angle", "front"),
                        "camera_motion": shot.get("camera_motion", ""),
                        "depth": "",
                    },
                    "style_suffix": self._quality_tags,
                    "global_negative": self._negative_global,
                    "_build_error": str(e),
                })
        return specs

    # ---- 内部辅助 ----

    @staticmethod
    def _normalize_anchors(character_anchors: Any) -> dict[str, dict]:
        """归一化为 {name: char_dict} 映射。"""
        if not character_anchors:
            return {}
        # 已经是 dict 形式 {name: char_dict}
        if isinstance(character_anchors, dict):
            # 检查 value 是否是 char dict（含 name 或 id_card）
            result = {}
            for k, v in character_anchors.items():
                if isinstance(v, dict) and ("id_card" in v or "name" in v):
                    name = v.get("name", k)
                    result[name] = v
                else:
                    # 退化为 key 作为 name
                    result[k] = v if isinstance(v, dict) else {"name": k}
            return result
        # list[dict] 形式
        if isinstance(character_anchors, list):
            result = {}
            for c in character_anchors:
                if isinstance(c, dict):
                    name = c.get("name", "")
                    if name:
                        result[name] = c
            return result
        return {}

    @staticmethod
    def _extract_shot_characters(
        shot: dict,
        scene_data: dict,
        char_map: dict[str, dict],
    ) -> list[dict]:
        """从 shot/scene 提取涉及的角色，并从 id_card 复制 appearance 字段。"""
        # 角色名来源（优先级）：
        # 1. shot.characters（若 Writer 输出）
        # 2. scene_data.dialogue[].character
        # 3. scene_data.characters（若存在）
        # 4. 主角（char_map 中第一个，作为兜底）
        names: list[str] = []

        shot_chars = shot.get("characters") or []
        if isinstance(shot_chars, list):
            for c in shot_chars:
                if isinstance(c, str):
                    names.append(c)
                elif isinstance(c, dict):
                    n = c.get("name")
                    if n:
                        names.append(n)

        if not names and scene_data:
            for d in scene_data.get("dialogue", []) or []:
                if isinstance(d, dict):
                    n = d.get("character")
                    if n:
                        names.append(n)

        if not names and scene_data:
            sc = scene_data.get("characters")
            if isinstance(sc, list):
                for c in sc:
                    if isinstance(c, str):
                        names.append(c)
                    elif isinstance(c, dict):
                        n = c.get("name")
                        if n:
                            names.append(n)

        # 去重保序
        seen = set()
        unique_names = []
        for n in names:
            if n not in seen:
                seen.add(n)
                unique_names.append(n)

        # 兜底：若仍无角色，但 char_map 有，取第一个（主角）
        if not unique_names and char_map:
            first_name = next(iter(char_map.keys()), None)
            if first_name:
                unique_names = [first_name]

        # 构建 character 块，appearance 100% 从 id_card 复制
        result = []
        for name in unique_names:
            char = char_map.get(name)
            if not char:
                # 角色不在 anchors 中：仅保留 name，appearance 为空
                result.append({
                    "name": name,
                    "appearance": {},
                    "pose": shot.get("pose", ""),
                    "expression": shot.get("expression", ""),
                    "position": shot.get("position", ""),
                })
                continue

            id_card = char.get("id_card", {}) or {}
            appearance = {
                "hair": id_card.get("hair_color", ""),
                "eyes": id_card.get("eye_color", ""),
                "outfit": id_card.get("outfit", ""),
                "body": id_card.get("body_type", ""),
                "distinctive": id_card.get("distinguishing_features", ""),
                "negative": id_card.get("negative_traits", ""),
            }
            result.append({
                "name": name,
                "appearance": appearance,
                "pose": shot.get("pose", ""),
                "expression": shot.get("expression", ""),
                "position": shot.get("position", ""),
            })
        return result


# ================================================================
# PromptTemplateEngine：visual_spec JSON -> prompt string
# ================================================================

class PromptTemplateEngine:
    """确定性模板引擎，visual_spec JSON -> prompt string。

    使用 Jinja2 渲染，模板可自定义。
    """

    def __init__(self, template_str: str | None = None, negative_template_str: str | None = None):
        """可选传入自定义 Jinja2 模板。"""
        self._env = Environment(loader=BaseLoader(), autoescape=False, trim_blocks=True, lstrip_blocks=True)
        self._prompt_tpl = self._env.from_string(template_str or DEFAULT_PROMPT_TEMPLATE)
        self._negative_tpl = self._env.from_string(negative_template_str or DEFAULT_NEGATIVE_TEMPLATE)

    def render_prompt(self, visual_spec: dict) -> str:
        """渲染正向 prompt。"""
        return self._prompt_tpl.render(**visual_spec).strip()

    def render_negative(self, visual_spec: dict) -> str:
        """渲染 negative_prompt，聚合全局负面词 + 角色级负面特征。"""
        return self._negative_tpl.render(**visual_spec).strip()

    def render_both(self, visual_spec: dict) -> tuple[str, str]:
        """返回 (prompt, negative_prompt)。"""
        return self.render_prompt(visual_spec), self.render_negative(visual_spec)


# 模块级单例（与 character_consistency 模式一致）
visual_descriptor = VisualDescriptor()
prompt_template_engine = PromptTemplateEngine()
