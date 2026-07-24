"""单集状态（Pydantic BaseModel）

LangGraph 工作流：
  顶层图: CreativePhase子图 → Writer∥AssetManager并行 → Composer → QualityGate子图
  CreativePhase子图: CreativeDirector → GapAnalysis → Planner → StoryCritic
  QualityGate子图: Critic → SafetyCheck → T0-T3分层路由 → END/DLQ

使用 Pydantic BaseModel 替代 TypedDict: 运行时校验 + IDE 补全 + 字段缺失当场报错。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional

from pydantic import BaseModel, Field


def _last_value(left: str, right: str) -> str:
    """last_value reducer: 返回非空的最右侧值。

    用于 status/error_message 字段，允许子图和并行节点同时写入，
    后写入的非空值生效，空值不覆盖已有值。
    """
    return right if right else left


def _last_float(left: float, right: float) -> float:
    """last_value reducer for float: 总是返回最右侧值（包括 0.0）。

    用于 critic_score/outline_score 等评估字段，允许 retry 循环中多次写入。
    """
    return right


class EpisodeState(BaseModel):
    """单集生产工作流的状态定义。

    所有字段有默认值，runtime 自动校验类型。
    """

    # ---- 标识 ----
    episode_id: str = ""
    series_id: str = ""
    episode_num: int = 1
    trace_id: str = ""

    # ---- 创意阶段输入 ----
    creative_brief: dict[str, Any] = Field(default_factory=dict)
    creative_guidance: dict[str, Any] = Field(default_factory=dict)
    hot_trends: list[str] = Field(default_factory=list)
    style_template: dict[str, Any] = Field(default_factory=dict)

    # ---- CreativeDirector 输出 ----
    creative_concepts: list[dict[str, Any]] = Field(default_factory=list)

    # ---- Planner 输出 ----
    series_plan: dict[str, Any] = Field(default_factory=dict)

    # ---- StoryCritic 输出 ----
    outline_score: Annotated[float, _last_float] = 0.0
    rewrite_episodes: list[int] = Field(default_factory=list)
    critic_evaluation: dict[str, Any] = Field(default_factory=dict)

    # ---- Writer 输入 & 输出 ----
    episode_plan: dict[str, Any] = Field(default_factory=dict)
    character_anchors: dict[str, Any] = Field(default_factory=dict)
    previous_summary: str = ""
    foreshadowing_context: str = ""
    script: dict[str, Any] = Field(default_factory=dict)
    storyboard: list[dict[str, Any]] = Field(default_factory=list)
    image_prompts: list[dict[str, Any]] = Field(default_factory=list)

    # ---- AssetManager 输出 ----
    asset_library: dict[str, Any] = Field(default_factory=dict)

    # ---- Composer 输出 ----
    episode_asset: dict[str, Any] = Field(default_factory=dict)

    # ---- Critic 输出 ----
    # 使用 reducer，允许 retry 循环中多次写入（最后一次生效）
    critic_score: Annotated[float, _last_float] = 0.0
    critic_decision: Annotated[str, _last_value] = "review"
    review_decision: Annotated[str, _last_value] = ""

    # ---- 安全审核 ----
    safety_checks: dict[str, bool] = Field(default_factory=dict)

    # ---- 控制 ----
    # status/error_message 使用 last_value reducer，允许子图和并行节点同时写入（最后值生效）
    status: Annotated[str, _last_value] = "pending"
    retry_count: int = 0
    error_message: Annotated[str, _last_value] = ""
    # total_cost_usd 使用累加 reducer，允许并行节点（writer ∥ asset_manager）同时写入
    total_cost_usd: Annotated[float, operator.add] = 0.0

    # ---- 质量门分层 ----
    quality_tier: Annotated[str, _last_value] = ""
    # ---- Composer 子图中间态 (P3: streaming progress) ----
    composer_classified_scenes: list[dict[str, Any]] = Field(default_factory=list)
    composer_image_paths: list[str] = Field(default_factory=list)
    composer_image_results: list[dict[str, Any]] = Field(default_factory=list)
    composer_video_paths: list[str] = Field(default_factory=list)
    composer_video_results: list[dict[str, Any]] = Field(default_factory=list)
    composer_audio_ready: bool = False
    composer_audio_segments: list[dict[str, Any]] = Field(default_factory=list)
    composer_subtitle_data: list[dict[str, Any]] = Field(default_factory=list)
    composer_bgm_track: str = ""
    # composer_step 使用 last_value reducer，允许子图多次执行/并行节点同时写入
    composer_step: Annotated[str, _last_value] = ""
    composer_content_gate: dict[str, Any] = Field(default_factory=dict)
    composer_vqa_check: dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "forbid"


# ---- 辅助结构（用于中断 payload） ----

class GapAnalysisResult(BaseModel):
    """创意缺口分析结果。"""
    should_interrupt: bool = False
    confidence: float = 0.0
    checks: dict[str, bool] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list)
    present: list[str] = Field(default_factory=list)


class CreativeGatePayload(BaseModel):
    """Creative Gate 中断时推送给用户的数据。"""
    type: str = "creative_gate"
    understanding: dict[str, Any] = Field(default_factory=dict)
    gaps: list[dict[str, Any]] = Field(default_factory=list)
    direction_previews: list[dict[str, Any]] = Field(default_factory=list)
    can_skip_all: bool = True
    auto_resolve_policy: str = ""


class QualityGatePayload(BaseModel):
    """Quality Gate 中断时推送给用户的数据。"""
    type: str = "quality_review"
    episode_id: str = ""
    episode_num: int = 1
    decision_tier: str = ""
    auto_resolve_at: Optional[str] = None
    summary: dict[str, Any] = Field(default_factory=dict)
    preview: dict[str, Any] = Field(default_factory=dict)
    actions: list[dict[str, Any]] = Field(default_factory=list)
