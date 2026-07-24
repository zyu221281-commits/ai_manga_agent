"""Writer Agent：剧本 + 分镜 + 提示词三件套

 文档 6.3：
- 输入: EpisodePlan + 角色锚点 + 前情摘要 + 伏笔上下文
- 输出: {script, storyboard, image_prompts, foreshadowing} 四件套
- 三件套一次性产出，避免多次 LLM 调用浪费 token
- 提示词格式: Flux 格式: [character], [action], [scene], [style:anime_comic], [quality_tags]
"""

from __future__ import annotations

import json
from typing import Any, Optional

from app.agents.base import BaseAgent, AgentResult


WRITER_PROMPT = """你是漫剧编剧+分镜师，请根据以下单集大纲，生成完整的剧本、分镜和图像提示词。

# 单集大纲
{episode_plan}

# 角色锚点（含角色身份证 — 必须严格遵守）
{character_anchors}

# 前情摘要
{previous_summary}

# 未回收伏笔（必须在合适的时机回收）
{foreshadowing_context}

# 风格设定（风格封印 — 不准偏离）
{style_template}

# 创作约束（紧箍咒）
1. **世界观锚点（硬约束）**：每个 image_prompt 必须包含角色身份证中的关键特征（发色、瞳色、服装）。
   如果角色有 id_card.hair_color=#8B4513，则 prompt 中必须出现 "brown hair (#8B4513)"。
   如果角色有 id_card.negative_traits="六指, 三只眼"，则 negative_prompt 中必须包含这些词。
2. **叙事节奏锁（软约束）**：你只能按照大纲的剧情走向生成分镜，不许自由发挥剧情。
   你的创意权仅限于"画面构图和光影细节"，叙事权由大纲锁定。
3. **风格封印（参数固化）**：所有 prompt 必须包含 style_template.global_prompt_suffix。
   negative_prompt 必须包含 style_template.negative_prompt_global。
4. **旁白质量约束（硬约束）**：narration 是给音频生成用的语音叙事文本，不是画面描述。
   - 正确示例：「城市已经沦陷三天了，幸存者们在废墟中搜寻着食物。」
   - 错误示例：「跟拍一个奔跑中跌倒的市民，他惊恐地回头望。」
   - 旁白应该是「听到的」，不是「看到的」。每个 scene 必须有 narration（除非纯对话）。
5. **narration 职责分离（硬约束）**：storyboard 中每个 shot 的 narration 字段是给音频生成用的语音叙事文本。
   - 必须是"听到的"，不是"看到的"
   - 有 dialogue 的 shot 可留空（dialogue 已覆盖音频）
   - 无 dialogue 的 shot 必须填写 narration（保证每个镜头都有音频）
   - 禁止把 description（画面描述）直接复制到 narration
   - **字数与时长严格匹配**：narration 字数必须 ≤ duration_s × 5（中文约 5 字/秒）。
     示例：6s shot → ≤30字；8s shot → ≤40字；5s shot → ≤25字。
     超出字数会导致 TTS 音频时长超过 shot 时长，破坏视频节奏。
6. **音频场景描述（audio_scene，硬约束）**：每个 shot 必须包含 audio_scene 字段，描述该镜头的声音氛围，供 Seed Audio 1.0 生成背景音/BGM/环境音效。
   - audio_scene 是自然语言描述，包括：环境声、BGM 风格、情绪氛围
   - 正确示例：「冷风呼啸，断弦颤音，低沉悬疑的背景音乐」
   - 错误示例：「画面显示一把剑」（这是视觉描述，不是听觉描述）
   - audio_scene 与 narration 是独立字段：narration 是要被"说出来"的文本，audio_scene 是"背景声音"的描述
   - 即使 shot 有 dialogue，也建议填写 audio_scene（为对白提供声音氛围）
   - 注意：Seed Audio 1.0 会根据 audio_scene 描述自动生成包含背景音/BGM/环境音的完整音频场景，无需手动编排时间线

# 输出要求（严格 JSON）
{{
  "script": {{
    "title": "本集标题",
    "scenes": [
      {{
        "scene_id": 1,
        "duration_s": 15,
        "emotion": "紧张|搞笑|感动|悬疑|治愈|悲伤",
        "visual_context": {{
          "environment": "场景环境描述",
          "lighting": "光照描述",
          "color_tone": "色调描述",
          "time": "时间"
        }},
        "dialogue": [
          {{"character": "角色名", "line": "台词", "expression": "表情"}}
        ],
        "narration": "旁白（TTS配音用的语音叙事文本，非画面描述。用自然口语讲述，禁止镜头术语和画面描述词。纯对话场景可留空）",
        "sound_effect": "音效描述"
      }}
    ]
  }},
  "storyboard": [
    {{
      "shot_id": 1,
      "scene_id": 1,
      "duration_s": 5,
      "shot_type": "close-up|medium|wide|extreme-close-up",
      "camera_motion": "static|zoom-in|zoom-out|pan-left|pan-right|dolly-in|dolly-out",
      "description": "画面描述（≤50字，给图像生成用，描述视觉构图）",
      "narration": "该镜头的语音旁白（给音频生成用，必须是听觉叙事不是画面描述。有 dialogue 的镜头可留空）",
      "audio_scene": "声音场景描述（供 Seed Audio 生成背景音/BGM/环境音效。示例：'冷风呼啸，断弦颤音，低沉悬疑的背景音乐'）",
      "transition": "cut|fade|dissolve|wipe"
    }}
  ],
  "image_prompts": [
    {{
      "shot_id": 1,
      "prompt": "[角色名 with id_card特征], [动作描述], 场景:[environment, lighting, color_tone], [风格:anime_comic], [quality_tags]",
      "negative_prompt": "[id_card.negative_traits], 低质量画面描述",
      "scene_type": "KEY_SCENE|NORMAL_SCENE"
    }}
  ],
  "foreshadowing": {{
    "planted": ["本集新埋下的伏笔"],
    "resolved": ["本集回收的前集伏笔 key"]
  }}
}}

# Flux 提示词格式
[character with hair_color, eye_color, outfit], [action], [scene], [style:anime_comic], [quality_tags]
quality_tags 选用：high quality, detailed, vibrant colors, cinematic lighting, sharp focus
"""


class WriterAgent(BaseAgent):
    """Writer Agent：剧本 + 分镜 + 提示词 + 伏笔四件套一次性产出。"""

    agent_name = "writer"

    async def execute(
        self,
        episode_plan: dict[str, Any],
        character_anchors: Optional[dict[str, Any]] = None,
        previous_summary: Optional[str] = None,
        style_template: Optional[dict[str, Any]] = None,
        foreshadowing_context: Optional[str] = None,
    ) -> AgentResult:
        prompt = WRITER_PROMPT.format(
            episode_plan=json.dumps(episode_plan, ensure_ascii=False, indent=2),
            character_anchors=json.dumps(character_anchors or {}, ensure_ascii=False),
            previous_summary=previous_summary or "新系列第一集，无前情。",
            foreshadowing_context=foreshadowing_context or "当前无未回收伏笔。",
            style_template=json.dumps(style_template or {}, ensure_ascii=False),
        )

        messages = [
            {"role": "system", "content": "你是专业的漫剧编剧和分镜师，输出严格的 JSON 格式。"},
            {"role": "user", "content": prompt},
        ]

        try:
            result = await self._llm_json(
                messages=messages,
                model="deepseek--pro",
                temperature=0.7,
                max_tokens=16384,
            )
        except Exception as e:
            return AgentResult(success=False, error=f"LLM call failed: {e}")

        if not all(k in result for k in ("script", "storyboard", "image_prompts")):
            return AgentResult(
                success=False,
                error=f"Missing required fields in response: {list(result.keys())}",
            )

        result["episode_plan"] = episode_plan
        result["character_anchors"] = character_anchors
        result["style_template"] = style_template

        result["image_prompts"] = self._validate_prompts(
            result["image_prompts"], result["script"].get("scenes", []), result.get("storyboard", []),
            character_anchors=character_anchors,
        )

        # narration 字数后处理：确保 TTS 时长不超过 shot 时长
        if "storyboard" in result:
            result["storyboard"] = self._trim_narration_to_duration(result["storyboard"])

        if self.lineage_tracker:
            await self.lineage_tracker.record(
                episode_id=self.episode_id or "",
                artifact_type="script",
                artifact_data={"scenes_count": len(result["script"].get("scenes", []))},
                model_name="deepseek--pro",
                model_params={"temperature": 0.7, "max_tokens": 16384},
                trace_id=self.trace_id,
            )

        foreshadowing = result.get("foreshadowing", {})

        return AgentResult(
            success=True,
            data=result,
            cost_usd=0.04,
            metadata={
                "scenes_count": len(result["script"].get("scenes", [])),
                "storyboard_shots": len(result["storyboard"]),
                "prompts_count": len(result["image_prompts"]),
                "foreshadowing_planted": len(foreshadowing.get("planted", [])),
                "foreshadowing_resolved": len(foreshadowing.get("resolved", [])),
            },
        )

    @staticmethod
    def _trim_narration_to_duration(storyboard: list[dict]) -> list[dict]:
        """后处理：确保 narration 字数 <= duration_s × 5（中文约 5 字/秒）。

        超出字数会截断到句子边界（优先保留完整句子），避免 TTS 时长超过 shot 时长。
        - 计算：max_chars = max(int(duration_s * 5), 10)
        - 仅处理非空 narration（空 narration 跳过）
        - 若截断，记录 warning 日志
        """
        import logging
        logger = logging.getLogger(__name__)
        CHARS_PER_SEC = 5
        trimmed_count = 0
        for shot in storyboard:
            narration = (shot.get("narration") or "").strip()
            if not narration:
                continue
            duration = float(shot.get("duration_s", 0) or 0)
            if duration <= 0:
                continue
            max_chars = max(int(duration * CHARS_PER_SEC), 10)
            # 去除标点符号计入字数（中文标点不算）
            import re
            punctuation = r"[，。！？、；：""''（）【】《》,.!?;:\"']+"
            text_chars = re.sub(punctuation, "", narration)
            if len(text_chars) <= max_chars:
                continue
            # 截断到子句边界：在 [。！？；，,] 处分割，保留分隔符
            # 这样既能保留完整句子，也能在逗号处截断长句
            clauses = re.split(r"([。！？；，,])", narration)
            trimmed = ""
            current_len = 0
            for i in range(0, len(clauses) - 1, 2):
                clause = clauses[i] + (clauses[i + 1] if i + 1 < len(clauses) else "")
                clause_chars = re.sub(punctuation, "", clause)
                if current_len + len(clause_chars) > max_chars:
                    break
                trimmed += clause
                current_len += len(clause_chars)
            # 若没有任何子句合适（首句就超长），直接按字数截断
            if not trimmed:
                trimmed = narration[:max_chars]
            if len(trimmed) < len(narration):
                logger.warning(
                    "Shot %s: narration %d字 -> %d字 (duration=%.0fs, max=%d字)",
                    shot.get("shot_id", "?"), len(text_chars),
                    len(re.sub(punctuation, "", trimmed)), duration, max_chars,
                )
                shot["narration"] = trimmed
                trimmed_count += 1
        if trimmed_count:
            logger.info("narration 字数校准: %d 个 shot 已截断以匹配时长", trimmed_count)
        return storyboard

    def _validate_prompts(
        self, prompts: list[dict], scenes: list[dict], storyboard=None,
        character_anchors: dict = None,
    ) -> list[dict]:
        if isinstance(prompts, dict):
            prompts = list(prompts.values())
        shot_to_scene = {}
        if storyboard:
            for s in storyboard:
                sid = s.get("shot_id")
                if sid is not None:
                    shot_to_scene[sid] = s.get("scene_id")
        vc_map = {}
        for s in scenes:
            scid = s.get("scene_id")
            if scid is not None:
                vc_map[scid] = s.get("visual_context", {})

        # 构建角色身份证映射
        char_id_cards = {}
        if character_anchors:
            for char in character_anchors.get("characters", []):
                name = char.get("name", "")
                id_card = char.get("id_card", {})
                if name and id_card:
                    char_id_cards[name] = id_card

        # 风格封印
        from app.core.config import settings
        quality_tags = settings.QUALITY_TAGS
        negative_global = settings.NEGATIVE_PROMPT

        for p in prompts:
            prompt_text = p.get("prompt", "").strip()
            if not prompt_text:
                p["prompt"] = quality_tags
            if "anime_comic" not in p["prompt"]:
                p["prompt"] += ", anime comic style"
            if quality_tags and quality_tags not in p["prompt"]:
                p["prompt"] += ", " + quality_tags

            # 注入角色身份证特征
            for char_name, id_card in char_id_cards.items():
                if char_name.lower() in p["prompt"].lower():
                    # 发色
                    hair = id_card.get("hair_color", "")
                    if hair and hair not in p["prompt"]:
                        p["prompt"] += f", hair color {hair}"
                    # 瞳色
                    eye = id_card.get("eye_color", "")
                    if eye and eye not in p["prompt"]:
                        p["prompt"] += f", {eye} eyes"
                    # 服装
                    outfit = id_card.get("outfit", "")
                    if outfit and outfit not in p["prompt"]:
                        p["prompt"] += f", {outfit}"
                    # 负面特征
                    neg_traits = id_card.get("negative_traits", "")
                    if neg_traits:
                        existing_neg = p.get("negative_prompt", "")
                        if neg_traits not in existing_neg:
                            p["negative_prompt"] = f"{neg_traits}, {existing_neg}".strip(", ")

            # 确保负面提示词包含全局负面
            neg = p.get("negative_prompt", "")
            if negative_global and negative_global not in neg:
                p["negative_prompt"] = f"{negative_global}, {neg}".strip(", ")

            # 注入场景视觉上下文
            shot_id = p.get("shot_id")
            scene_id = shot_to_scene.get(shot_id) if shot_id is not None else None
            if scene_id is not None:
                ctx = vc_map.get(scene_id, {})
                env = ctx.get("environment", "")
                if env and env not in p["prompt"]:
                    p["prompt"] = p["prompt"].rstrip(", ") + ", " + env
        return prompts
