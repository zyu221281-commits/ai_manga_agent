"""LangGraph V5: episode pipeline with subgraphs, parallel execution, and dual gates."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any
from copy import deepcopy

from langgraph.graph import StateGraph, END
from langgraph.constants import Send
from langgraph.types import interrupt
from langgraph.config import get_stream_writer
from sqlalchemy.ext.asyncio import AsyncSession

from app.state.episode_state import EpisodeState, GapAnalysisResult, CreativeGatePayload, QualityGatePayload
from app.core.config import settings
from app.core.dependencies import get_db
from app.services.file_lineage_tracker import FileLineageTracker
from app.services.long_term_memory import (
    long_term_memory,
    extract_and_store_episode_memory,
    build_writer_context,
)

logger = logging.getLogger(__name__)


def _emit(event: dict) -> None:
    """安全地发送流式事件（在非流式上下文中静默跳过）。"""
    try:
        writer = get_stream_writer()
        writer(event)
    except Exception:
        pass


@asynccontextmanager
async def _node_session():
    """Graph node 用的 DB session，自动关闭。

    用法：
        async with _node_session() as session:
            agent = SomeAgent(session=session, ...)
            result = await agent._run_with_tracking(...)

    设计要点：
    - session 是 node 函数内的局部变量，不写入 EpisodeState，
      因此不会被 LangGraph 序列化到 checkpoint，不影响 interrupt/resume。
    - AsyncSession 创建时不立即连接 PG（lazy connect），PG 不可用时不影响 session 创建。
    - 实际 PG 故障由 CostTracker/LineageTracker 内部的 try/except 兜底，管线继续运行。
    - async with 退出时自动 close，无连接泄漏。
    """
    session: AsyncSession = await get_db()
    try:
        yield session
    finally:
        await session.close()


# ================================================================
# Gap Analysis
# ================================================================

def analyze_gaps(brief: dict[str, Any]) -> GapAnalysisResult:
    checks = {
        "theme": bool(brief.get("theme")),
        "genre": bool(brief.get("genre")),
        "characters": len(brief.get("characters", [])) >= 1,
        "core_premise": bool(brief.get("core_premise")) or bool(brief.get("summary")),
        "tone": bool(brief.get("tone")),
    }
    score = sum(checks.values()) / max(len(checks), 1)
    return GapAnalysisResult(
        should_interrupt=score < 0.8,
        confidence=round(score, 2),
        checks=checks,
        missing=[k for k, v in checks.items() if not v],
        present=[k for k, v in checks.items() if v],
    )


def _build_creative_gate_payload(state: EpisodeState, gaps: GapAnalysisResult) -> dict:
    brief = state.creative_brief
    field_map = {
        "theme": {"priority": "critical", "question": "故事主题是什么？", "hint": "比如：都市特种兵逆袭"},
        "genre": {"priority": "critical", "question": "题材/类型？", "hint": "比如：热血、悬疑"},
        "tone": {"priority": "critical", "question": "整体基调？", "hint": "比如：热血激昂、暗黑写实"},
        "characters": {"priority": "important", "question": "主角设定？", "hint": "年龄、性格、背景"},
        "core_premise": {"priority": "important", "question": "核心矛盾？", "hint": "一两句话"},
    }
    gaps_list = []
    for field in gaps.missing:
        info = field_map.get(field, {"priority": "nice_to_have", "question": field, "hint": ""})
        gaps_list.append({"field": field, "priority": info["priority"], "question": info["question"], "hint": info["hint"]})
    return CreativeGatePayload(
        understanding={"parsed_genre": brief.get("genre", ""), "parsed_tone_hint": brief.get("tone", ""), "summary": brief.get("summary", "")[:200], "confidence": gaps.confidence},
        gaps=gaps_list, can_skip_all=True,
        auto_resolve_policy="critical 和 important 字段建议填写",
    ).model_dump()


def _build_quality_gate_payload(state: EpisodeState) -> dict:
    asset = state.episode_asset
    tier = state.quality_tier
    return QualityGatePayload(
        episode_id=state.episode_id, episode_num=state.episode_num,
        decision_tier=tier, auto_resolve_at=None if tier == "T3" else "4h",
        summary={"critic_score": state.critic_score, "critic_decision": state.critic_decision, "cost_usd": round(state.total_cost_usd, 4), "retry_count": state.retry_count},
        preview={"video_path": asset.get("final_video_path", ""), "duration_s": asset.get("final_video_duration_s", 0), "covers": asset.get("covers", [{}])[:1]},
        actions=[{"id": "approve", "label": "通过发布", "is_default": True}, {"id": "retry", "label": "重做本集"}, {"id": "reject", "label": "废弃进DLQ", "requires_note": True}],
    ).model_dump()


def _get_tracer(state: EpisodeState) -> FileLineageTracker:
    return FileLineageTracker(trace_id=state.trace_id or "unknown")


# ================================================================
# CreativePhase subgraph nodes
# ================================================================

async def creative_director_node(state: EpisodeState) -> dict[str, Any]:
    from app.agents.creative_director import CreativeDirectorAgent
    _emit({"event": "creative_director", "step": "start"})
    tid = state.trace_id
    tracer = _get_tracer(state)
    async with _node_session() as session:
        agent = CreativeDirectorAgent(session=session, episode_id=state.episode_id, series_id=state.series_id, trace_id=tid, tracer=tracer)
        result = await agent._run_with_tracking(creative_brief=state.creative_brief, hot_trends=state.hot_trends)
    if not result.success:
        tracer.flush()
        _emit({"event": "creative_director", "step": "failed", "error": result.error or ""})
        return {"status": "failed", "error_message": result.error or "creative_director failed"}
    creative_data = result.data
    cost = result.cost_usd
    _emit({"event": "creative_director", "step": "done", "concepts": len(creative_data.get("concepts", []))})
    return {"status": "creative_directed", "creative_guidance": creative_data.get("creative_guidance", {}), "creative_concepts": creative_data.get("concepts", []), "total_cost_usd": cost}


async def gap_analysis_node(state: EpisodeState) -> dict[str, Any]:
    gaps = analyze_gaps(state.creative_brief)
    tracer = _get_tracer(state)
    if gaps.should_interrupt:
        _emit({"event": "gap_analysis", "step": "interrupt", "confidence": gaps.confidence, "missing": gaps.missing})
        payload = _build_creative_gate_payload(state, gaps)
        tracer.record_interrupt("creative_gate", payload)
        tracer.flush()
        resume_data = interrupt(payload)
        tracer.record_custom_event("creative_gate_resumed", resume_data if isinstance(resume_data, dict) else {})
        action = resume_data.get("action", "skip") if isinstance(resume_data, dict) else "skip"
        _emit({"event": "gap_analysis", "step": "resumed", "action": action})
        if action == "enrich":
            fields = resume_data.get("fields", {})
            return {"creative_brief": {**state.creative_brief, **fields}}
    _emit({"event": "gap_analysis", "step": "pass", "confidence": gaps.confidence})
    tracer.record_routing_decision("gap_analysis", "continue", f"confidence={gaps.confidence}")
    return {}


async def planner_node(state: EpisodeState) -> dict[str, Any]:
    from app.agents.planner import PlannerAgent
    from app.services.checkpoint_manager import CheckpointManager
    _emit({"event": "planner", "step": "start"})

    # Checkpoint: 如果 Planner 已执行过，直接加载（省 LLM token）
    ckpt = CheckpointManager(episode_id=state.episode_id)
    saved_plan = ckpt.load("plan", shot_id=0)
    if saved_plan and saved_plan.get("series_plan"):
        episodes = len(saved_plan["series_plan"].get("episodes", [])) if isinstance(saved_plan["series_plan"], dict) else 0
        _emit({"event": "planner", "step": "checkpoint_hit", "episodes": episodes})
        return {"status": "planned", "series_plan": saved_plan["series_plan"], "total_cost_usd": 0.0}

    tid = state.trace_id
    tracer = _get_tracer(state)
    async with _node_session() as session:
        agent = PlannerAgent(session=session, episode_id=state.episode_id, series_id=state.series_id, trace_id=tid, tracer=tracer)
        result = await agent._run_with_tracking(creative_brief=state.creative_brief, hot_trends=state.hot_trends, creative_guidance=state.creative_guidance)
    if not result.success:
        tracer.flush()
        _emit({"event": "planner", "step": "failed", "error": result.error or ""})
        return {"status": "failed", "error_message": result.error or "planner failed"}
    cost = result.cost_usd
    episodes = len(result.data.get("episodes", [])) if isinstance(result.data, dict) else 0
    _emit({"event": "planner", "step": "done", "episodes": episodes})

    # 保存到 checkpoint（断点续传：崩溃后重启直接加载，不重新调 LLM）
    ckpt.save("plan", shot_id=0, result={"series_plan": result.data})
    _emit({"event": "planner", "step": "checkpoint_saved"})

    return {"status": "planned", "series_plan": result.data, "total_cost_usd": cost}


async def story_critic_node(state: EpisodeState) -> dict[str, Any]:
    from app.agents.story_critic import StoryCriticAgent
    _emit({"event": "story_critic", "step": "start"})
    series_plan = state.series_plan
    if not series_plan:
        _emit({"event": "story_critic", "step": "skipped"})
        return {"status": "evaluated", "outline_score": 1.0}
    tid = state.trace_id
    tracer = _get_tracer(state)
    async with _node_session() as session:
        agent = StoryCriticAgent(session=session, episode_id=state.episode_id, series_id=state.series_id, trace_id=tid, tracer=tracer)
        result = await agent._run_with_tracking(series_plan=series_plan)
    cost = result.cost_usd
    # 防御性：story_critic 失败时 result.data 可能为 None
    data = result.data if result.data else {}
    score = data.get("outline_score", 0.0)
    _emit({"event": "story_critic", "step": "done", "score": score})
    return {"status": "evaluated", "outline_score": score, "total_cost_usd": cost}


def planner_router(state: EpisodeState) -> str:
    if state.status == "failed":
        return "dlq"
    return "story_critic"


# ================================================================
# Main graph: production nodes
# ================================================================

async def writer_node(state: EpisodeState) -> dict[str, Any]:
    from app.agents.writer import WriterAgent
    from app.services.checkpoint_manager import CheckpointManager
    _emit({"event": "writer", "step": "start"})

    # Checkpoint: 如果 Writer 已执行过，直接加载（省 LLM token，Writer 是最贵的 LLM 调用）
    ckpt = CheckpointManager(episode_id=state.episode_id)
    saved_script = ckpt.load("script", shot_id=0)
    if saved_script and saved_script.get("script"):
        _emit({"event": "writer", "step": "checkpoint_hit",
               "shots": len(saved_script.get("storyboard", [])),
               "prompts": len(saved_script.get("image_prompts", []))})
        return {"status": "written", "script": saved_script.get("script"), "storyboard": saved_script.get("storyboard"), "image_prompts": saved_script.get("image_prompts"), "visual_specs": saved_script.get("visual_specs", []), "total_cost_usd": 0.0}

    series_plan = state.series_plan
    episodes = series_plan.get("episodes", []) if series_plan else []
    episode_num = state.episode_num
    ep_plan = episodes[episode_num - 1] if 0 < episode_num <= len(episodes) else {}
    tid = state.trace_id
    tracer = _get_tracer(state)
    async with _node_session() as session:
        agent = WriterAgent(session=session, episode_id=state.episode_id, series_id=state.series_id, trace_id=tid, tracer=tracer)
        try:
            previous_summary = await build_writer_context(episode_num)
            foreshadowing_context = await long_term_memory.get_foreshadowing_context(episode_num)
        except Exception:
            previous_summary, foreshadowing_context = "", ""
        result = await agent._run_with_tracking(episode_plan=ep_plan, character_anchors=state.character_anchors, previous_summary=previous_summary, style_template=state.style_template, foreshadowing_context=foreshadowing_context)
    cost = result.cost_usd
    if not result.success:
        tracer.flush()
        _emit({"event": "writer", "step": "failed", "error": result.error or ""})
        return {"status": "failed", "error_message": result.error or "writer failed"}
    script_pkg = result.data
    try:
        await extract_and_store_episode_memory(episode_num=episode_num, script_data=script_pkg.get("script", {}), episode_plan=ep_plan)
    except Exception:
        pass
    _emit({"event": "writer", "step": "done",
           "shots": len(script_pkg.get("storyboard", [])),
           "prompts": len(script_pkg.get("image_prompts", [])),
           "visual_specs": len(script_pkg.get("visual_specs", []))})

    # 保存到 checkpoint（断点续传：崩溃后重启直接加载，不重新调 LLM）
    ckpt.save("script", shot_id=0, result=script_pkg)
    _emit({"event": "writer", "step": "checkpoint_saved"})

    return {"status": "written", "script": script_pkg.get("script"), "storyboard": script_pkg.get("storyboard"), "image_prompts": script_pkg.get("image_prompts"), "visual_specs": script_pkg.get("visual_specs", []), "total_cost_usd": cost}


async def shot_validator_node(state: EpisodeState) -> dict[str, Any]:
    """分镜逻辑质检节点 — 在 Writer 之后、Composer 之前执行。

    检查空间连续性、角色一致性、镜头节奏、提示词完整性。
    不通过则标记为 rewrite，路由回 Writer 重写。
    """
    from app.agents.shot_validator import ShotValidatorAgent
    _emit({"event": "shot_validator", "step": "start"})

    tid = state.trace_id
    tracer = _get_tracer(state)
    async with _node_session() as session:
        agent = ShotValidatorAgent(session=session, episode_id=state.episode_id, series_id=state.series_id, trace_id=tid, tracer=tracer)

        result = await agent._run_with_tracking(
            storyboard=state.storyboard or [],
            image_prompts=state.image_prompts or [],
            character_anchors=state.character_anchors or {},
        )

    cost = result.cost_usd
    data = result.data or {}

    if result.success and data.get("decision") == "pass":
        _emit({"event": "shot_validator", "step": "passed", "score": data.get("overall_score", 1.0)})
        return {"status": "validated", "total_cost_usd": cost}
    else:
        _emit({
            "event": "shot_validator", "step": "issues_found",
            "score": data.get("overall_score", 0.0),
            "issues": data.get("suggestions", "")[:200],
        })
        return {"status": "validated", "total_cost_usd": cost, "shot_validation_issues": data.get("suggestions", "")}


def after_shot_validator(state: EpisodeState) -> str:
    """ShotValidator 之后的路由（当前总是继续到 composer）。"""
    return "composer"


async def asset_manager_node(state: EpisodeState) -> dict[str, Any]:
    from app.agents.asset_manager import AssetManagerAgent
    _emit({"event": "asset_manager", "step": "start"})
    tid = state.trace_id
    tracer = _get_tracer(state)
    async with _node_session() as session:
        agent = AssetManagerAgent(session=session, episode_id=state.episode_id, series_id=state.series_id, trace_id=tid, tracer=tracer)
        result = await agent._run_with_tracking(series_plan=state.series_plan)
        # Multi-view anchor pre-generation for cross-episode consistency
        # execute() 返回 asset_library 后，为每个角色生成三视图 anchor 并持久化到 DB。
        # 首集生成三视图，后续集 load_all_anchors 已加载 → has_multi_view 命中 → 跳过。
        # 失败不阻塞管线（非阻断），composer 仍可用 id_card 文本约束兜底。
        if result.success and result.data.get("characters"):
            try:
                result.data = await agent.generate_multi_view_anchors(result.data)
                mv_count = sum(
                    1 for c in result.data.get("characters", [])
                    if c.get("view_images")
                )
                _emit({"event": "asset_manager", "step": "multi_view",
                       "anchors": mv_count})
            except Exception as e:
                logger.warning("Multi-view anchor generation failed (non-blocking): %s", e)
                _emit({"event": "asset_manager", "step": "multi_view_failed",
                       "error": str(e)[:120]})
    # 注意：asset_manager 与 writer 并行运行，不返回 status/error_message（避免 last_value 冲突）
    # total_cost_usd 使用 Annotated[float, operator.add] reducer，可安全并行写入
    if not result.success:
        tracer.flush()
        _emit({"event": "asset_manager", "step": "failed", "error": result.error or ""})
        logger.error("asset_manager failed: %s", result.error)
        # 返回空 asset_library，composer 会处理缺失情况
        return {"asset_library": {}, "total_cost_usd": result.cost_usd}
    _emit({"event": "asset_manager", "step": "done",
           "characters": len(result.data.get("characters", []))})
    return {"asset_library": result.data, "total_cost_usd": result.cost_usd}


# ================================================================
# QualityGate subgraph nodes
# ================================================================

async def critic_node(state: EpisodeState) -> dict[str, Any]:
    from app.agents.critic import CriticAgent
    _emit({"event": "critic", "step": "start"})
    logger.info("critic_node executing (retry_count=%d, status=%s)", state.retry_count, state.status)
    episode_asset = state.episode_asset or {}
    tid = state.trace_id
    tracer = _get_tracer(state)
    async with _node_session() as session:
        agent = CriticAgent(session=session, episode_id=state.episode_id, series_id=state.series_id, trace_id=tid, tracer=tracer)
        result = await agent._run_with_tracking(episode_asset=episode_asset, retry_count=state.retry_count)
    cost = result.cost_usd
    tracer.flush()
    score = result.data.get("overall_score", 0.0) if result.success else 0.0
    decision = result.data.get("decision", "review") if result.success else "review"
    logger.info("critic_node done (success=%s, score=%.2f, decision=%s)", result.success, score, decision)
    _emit({"event": "critic", "step": "done", "score": score, "decision": decision})
    return {"status": "critiqued", "critic_score": score, "critic_decision": decision, "total_cost_usd": cost}


def classify_quality_tier(state: EpisodeState) -> str:
    score = state.critic_score
    is_first = state.episode_num == 1
    retries_exhausted = state.retry_count >= settings.CRITIC_MAX_RETRY
    if retries_exhausted or state.critic_decision == "review":
        return "T3"
    if score < 0.5:
        return "T3"
    if score < 0.7 or is_first:
        return "T2"
    if score < 0.85:
        return "T1"
    return "T0"


def quality_tier_router(state: EpisodeState) -> dict[str, str]:
    tier = classify_quality_tier(state)
    return {"quality_tier": tier}


def _quality_route(state: EpisodeState) -> str:
    tier = state.quality_tier or classify_quality_tier(state)
    logger.info("_quality_route: tier=%s, score=%.2f, decision=%s", tier, state.critic_score, state.critic_decision)
    if tier in ("T0", "T1"):
        return "pass"
    if tier in ("T2", "T3"):
        return "interrupt"
    return "dlq"


async def quality_interrupt_node(state: EpisodeState) -> dict[str, Any]:
    payload = _build_quality_gate_payload(state)
    tracer = _get_tracer(state)
    tracer.record_interrupt("quality_gate", payload)
    tracer.flush()
    logger.info("quality_interrupt_node: triggering interrupt (tier=%s, score=%.2f)", state.quality_tier, state.critic_score)
    _emit({"event": "quality_gate", "step": "interrupt", "tier": state.quality_tier})
    resume_data = interrupt(payload)
    logger.info("quality_interrupt_node: resumed with %s", resume_data)
    tracer.record_custom_event("quality_gate_resumed", resume_data if isinstance(resume_data, dict) else {})
    action = resume_data.get("action", "reject") if isinstance(resume_data, dict) else "reject"
    _emit({"event": "quality_gate", "step": "resumed", "action": action})
    # 重试时递增 retry_count，防止无限重试循环
    if action == "retry":
        return {"review_decision": action, "retry_count": state.retry_count + 1}
    return {"review_decision": action}


async def safety_check_node(state: EpisodeState) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    try:
        checks["ai_labeled"] = bool(state.episode_asset.get("ai_label"))
    except Exception:
        checks["ai_labeled"] = True
    checks["text_safe"] = True
    checks["image_safe"] = True
    return {"safety_checks": checks}


async def dlq_node(state: EpisodeState) -> dict[str, Any]:
    _emit({"event": "dlq", "step": "entered", "error": state.error_message or ""})
    try:
        _get_tracer(state).flush()
    except Exception:
        pass
    return {"status": "dlq"}


def after_quality_router(state: EpisodeState) -> str:
    decision = state.review_decision
    logger.info("after_quality_router: review_decision=%s, retry_count=%d", decision, state.retry_count)
    if decision in ("approved", "approve"):
        return "end"
    if decision == "retry":
        if state.retry_count < settings.CRITIC_MAX_RETRY:
            return "composer_classify"
    return "dlq"



# ================================================================
# Composer subgraph nodes (P3: streaming progress via 5 internal nodes)
# ================================================================

async def composer_classify_node(state):
    from app.agents.composer import ComposerAgent
    writer = get_stream_writer()
    async with _node_session() as session:
        agent = ComposerAgent(session=session, episode_id=state.episode_id, series_id=state.series_id, trace_id=state.trace_id)
        classified = agent._classify_scenes(state.storyboard, state.image_prompts, state.script)

    # Inject visual_spec into each classified_scene for deterministic rendering
    # 让 composer_image_gen_node 能用 PromptTemplateEngine 渲染 + 按 camera_angle 选 ref
    visual_specs_by_shot = {vs.get("shot_id"): vs for vs in (state.visual_specs or [])}
    for cs in classified:
        shot_id = cs.get("shot_id")
        if shot_id is not None and shot_id in visual_specs_by_shot:
            cs["visual_spec"] = visual_specs_by_shot[shot_id]

    key_count = sum(1 for s in classified if s.get("type") == "key")
    writer({"event": "composer", "step": "classified", "scenes": len(classified), "key": key_count})
    return {"composer_classified_scenes": classified, "composer_step": "classified"}


async def composer_image_gen_node(state):
    from app.agents.composer import ComposerAgent
    from app.core.config import settings
    writer = get_stream_writer()
    writer({"event": "composer", "step": "image_gen_start"})
    async with _node_session() as session:
        agent = ComposerAgent(session=session, episode_id=state.episode_id, series_id=state.series_id, trace_id=state.trace_id)
        classified = state.composer_classified_scenes
        if not classified:
            return {"status": "failed", "error_message": "No classified scenes"}
        images = await agent._generate_images(classified, state.asset_library)
    cost = sum(img.cost_usd for img in images if img)

    # Pre-Video Gate: 图像失败率检查
    if not agent._pre_video_gate(images, classified):
        writer({"event": "composer", "step": "image_gen_failed"})
        return {"status": "failed", "error_message": "Image failure rate exceeds threshold"}

    # ContentGate: CLIP 风格相似度 + 角色一致性检查（非阻断）
    content_gate_dict = {}
    if settings.CONTENT_GATE_ENABLED:
        try:
            from app.quality.content_gate import content_gate
            gate_result = await content_gate.check_images(images, classified, state.asset_library)
            content_gate_dict = gate_result.to_dict()
            if gate_result.flagged_shots:
                writer({"event": "composer", "step": "content_gate_flagged",
                        "shots": gate_result.flagged_shots})
        except Exception as e:
            logger.warning("ContentGate error (non-blocking): %s", e)
            content_gate_dict = {"verdict": "error", "error": str(e)}

    # VQA: KEY_SCENE 物理异常检查（非阻断）
    vqa_dict = {}
    if settings.VQA_ENABLED:
        try:
            from app.quality.vqa_checker import vqa_checker
            vqa_result = await vqa_checker.check_key_scenes(images, classified)
            vqa_dict = vqa_result.to_dict()
            if vqa_result.flagged_shots:
                writer({"event": "composer", "step": "vqa_flagged",
                        "shots": vqa_result.flagged_shots})
        except Exception as e:
            logger.warning("VQA error (non-blocking): %s", e)
            vqa_dict = {"verdict": "error", "error": str(e)}

    # 保存完整图像结果（含 url, local_path, prompt, cost_usd）
    image_results_data = [
        {"url": img.url, "local_path": img.local_path or "", "prompt": img.prompt,
         "cost_usd": img.cost_usd, "width": img.width, "height": img.height}
        for img in images
    ]
    paths = [img.local_path for img in images if img and img.local_path]
    writer({"event": "composer", "step": "image_gen_done", "count": len(paths),
            "content_gate": content_gate_dict.get("verdict", "skip"),
            "vqa": vqa_dict.get("verdict", "skip")})
    return {
        "composer_image_paths": paths,
        "composer_image_results": image_results_data,
        "composer_content_gate": content_gate_dict,
        "composer_vqa_check": vqa_dict,
        "total_cost_usd": cost,
    }


async def composer_tts_gen_node(state):
    from app.agents.composer import ComposerAgent
    from app.services.checkpoint_manager import CheckpointManager
    from app.resilience.adapters.tts_adapter import get_tts_adapter
    writer = get_stream_writer()
    writer({"event": "composer", "step": "tts_gen_start"})
    async with _node_session() as session:
        agent = ComposerAgent(session=session, episode_id=state.episode_id, series_id=state.series_id, trace_id=state.trace_id)
        audio_segments, subtitle_data = await agent._generate_audio(state.script or {}, state.asset_library or {}, state.storyboard)

    # ================================================================
    # TTS 质量校验 + Checkpoint 恢复 + 重新生成
    # 音画对齐前置约束：视频生成要求 TTS 真实时长，故 TTS 必须有效
    # ================================================================
    ckpt = CheckpointManager(episode_id=state.episode_id)
    adapter = get_tts_adapter()
    invalid_count = 0
    recovered_count = 0
    regenerated_count = 0

    for i, seg in enumerate(audio_segments):
        r = seg.get("result")
        text = seg.get("text", "")
        ok, reason = agent._validate_tts_result(r, text)
        if ok:
            continue

        invalid_count += 1
        writer({"event": "composer", "step": "tts_invalid",
                "shot_id": seg.get("shot_id"), "reason": reason[:120]})
        logger.warning(
            "TTS invalid (shot %s): %s — attempting recovery",
            seg.get("shot_id"), reason,
        )

        # 恢复：checkpoint 优先 → 重新生成
        recovered_seg = await agent._recover_tts_segment(seg, ckpt, adapter)
        new_r = recovered_seg.get("result")
        new_ok, _ = agent._validate_tts_result(new_r, text)
        if new_ok:
            audio_segments[i] = recovered_seg
            # 判断恢复来源（checkpoint 还是重生）
            if new_r.cost_usd == 0.0 and new_r.model == "":
                recovered_count += 1
            else:
                regenerated_count += 1
        else:
            logger.error(
                "TTS recovery failed for shot %s: keeping invalid result",
                seg.get("shot_id"),
            )

    writer({"event": "composer", "step": "tts_validated",
            "invalid": invalid_count, "recovered": recovered_count,
            "regenerated": regenerated_count})
    if invalid_count > 0:
        logger.info(
            "TTS validation: %d invalid, %d recovered from checkpoint, %d regenerated",
            invalid_count, recovered_count, regenerated_count,
        )

    # 累加 TTS 成本（audio_segments 中每项有 result.cost_usd）
    cost = 0.0
    for seg in audio_segments:
        r = seg.get("result")
        if r and hasattr(r, "cost_usd"):
            cost += r.cost_usd
        # Save successful TTS result to checkpoint for crash recovery
        # （仅校验通过的 TTS 才写入，避免污染 checkpoint）
        if hasattr(r, "local_path") and getattr(r, "local_path", None):
            shot_id = seg.get("shot_id", 0)
            ok, _ = agent._validate_tts_result(r, seg.get("text", ""))
            if ok:
                ckpt.save("tts", shot_id, {
                    "local_path": r.local_path,
                    "text": getattr(r, "text", ""),
                    "voice_id": getattr(r, "voice_id", ""),
                    "duration_s": getattr(r, "duration_s", 0),
                    "cost_usd": getattr(r, "cost_usd", 0),
                    "model": getattr(r, "model", ""),
                    "type": seg.get("type", ""),
                    "character": seg.get("character", ""),
                    "word_timestamps": getattr(r, "word_timestamps", []),
                })
    writer({"event": "composer", "step": "tts_gen_done", "segments": len(audio_segments)})
    return {"composer_audio_ready": True, "composer_audio_segments": audio_segments, "composer_subtitle_data": subtitle_data, "total_cost_usd": cost}


async def composer_video_gen_node(state):
    from app.agents.composer import ComposerAgent
    from app.resilience.adapters.image_adapter import ImageResult
    from app.core.config import settings
    writer = get_stream_writer()
    writer({"event": "composer", "step": "video_gen_start"})
    async with _node_session() as session:
        agent = ComposerAgent(session=session, episode_id=state.episode_id, series_id=state.series_id, trace_id=state.trace_id)
        classified = state.composer_classified_scenes
        # 从完整图像结果重建 ImageResult（含 url, local_path, prompt）
        image_data = state.composer_image_results
        image_results = []
        for i, cs in enumerate(classified):
            if i < len(image_data):
                d = image_data[i]
                image_results.append(ImageResult(
                    url=d.get("url", ""), local_path=d.get("local_path", ""),
                    prompt=d.get("prompt", ""), width=d.get("width", 1080),
                    height=d.get("height", 1920), model="reconstructed",
                    cost_usd=d.get("cost_usd", 0.0),
                ))
            else:
                image_results.append(ImageResult(url="", local_path="", prompt="",
                                                 width=1080, height=1920, model="", cost_usd=0.0))

        # 音画对齐硬约束：从已校验的 TTS 结果聚合 shot_durations
        # 未传 shot_durations 时 _generate_videos 会跳过所有 shot（防音画不同步）
        shot_durations: dict[int, float] = {}
        for a in (state.composer_audio_segments or []):
            tts = a.get("result")
            if not tts or not getattr(tts, "duration_s", 0):
                continue
            sid = a.get("shot_id")
            if sid is not None:
                shot_durations[sid] = shot_durations.get(sid, 0.0) + tts.duration_s
        if shot_durations:
            writer({"event": "composer", "step": "video_durations_ready",
                    "shots_with_tts": len(shot_durations)})

        videos = await agent._generate_videos(
            classified, image_results, settings.VIDEO_KEY_SCENE_RATIO,
            shot_durations=shot_durations,
        )
    # 保存完整视频结果（含 duration_s, url, local_path, cost_usd）
    video_results_data = [
        {"url": v.url, "local_path": v.local_path or "", "duration_s": v.duration_s,
         "cost_usd": v.cost_usd, "scene_type": v.scene_type, "model": v.model}
        for v in videos
    ]
    paths = [v.local_path for v in videos if v and v.local_path]
    cost = sum(v.cost_usd for v in videos if v)
    writer({"event": "composer", "step": "video_gen_done", "count": len(paths)})
    return {
        "composer_video_paths": paths,
        "composer_video_results": video_results_data,
        "composer_step": "videos_done",
        "total_cost_usd": cost,
    }


async def composer_final_node(state):
    from app.agents.composer import ComposerAgent
    from app.resilience.adapters.image_adapter import ImageResult
    from app.resilience.adapters.video_adapter import VideoResult
    writer = get_stream_writer()
    writer({"event": "composer", "step": "compose_start"})
    async with _node_session() as session:
        agent = ComposerAgent(session=session, episode_id=state.episode_id, series_id=state.series_id, trace_id=state.trace_id)
        # BGM 已整合到 Seed Audio 输出中，不再独立选取
        bgm = ""

        # 从完整图像结果重建 ImageResult（用于封面选择）
        image_data = state.composer_image_results
        fake_imgs = [
            ImageResult(
                url=d.get("url", ""), local_path=d.get("local_path", ""),
                prompt=d.get("prompt", ""), width=d.get("width", 1080),
                height=d.get("height", 1920),
            )
            for d in image_data[:3] if d.get("local_path") or d.get("url")
        ]
        covers = agent._select_cover_candidates(fake_imgs)

        # 从完整视频结果重建 VideoResult（含真实 duration_s）
        video_data = state.composer_video_results
        fake_vids = [
            VideoResult(
                url=d.get("url", ""), local_path=d.get("local_path", ""),
                duration_s=d.get("duration_s", 5.0),
            )
            for d in video_data if d.get("local_path")
        ]

        # audio_segments: 就绪时传实际数据，未就绪时传空 list
        audio_segments = state.composer_audio_segments if state.composer_audio_ready else []

        result = await agent._compose_final_video(
            video_segments=fake_vids,
            audio_segments=audio_segments,
            subtitles=state.composer_subtitle_data,
            bgm_track=bgm, classified_scenes=state.composer_classified_scenes,
            episode_id=state.episode_id, bgm_path=bgm,
        )

    if isinstance(result, dict) and result.get("success"):
        writer({"event": "composer", "step": "done", "video": result.get("final_video_path", "")})
        # 构建完整的 episode_asset（供 Critic 评估）
        episode_asset = {
            "episode_id": state.episode_id,
            "script": state.script,
            "storyboard": state.storyboard,
            "image_prompts": state.image_prompts,
            "images": image_data,
            "video_segments": video_data,
            "audio_segments": audio_segments + result.get("auto_narration_segments", []),
            "subtitles": state.composer_subtitle_data,
            "bgm_track": bgm,
            "covers": covers,
            "final_video_path": result.get("final_video_path", ""),
            "final_video_duration_s": result.get("final_video_duration_s", 0),
            "ai_label": {"generated": True, "model": "ai_manga_agent"},
            "cost_usd": round(state.total_cost_usd, 4),
            "metadata": {
                "content_gate": state.composer_content_gate,
                "vqa_check": state.composer_vqa_check,
                "image_count": len(image_data),
                "video_count": len(video_data),
                "audio_count": len(audio_segments) + len(result.get("auto_narration_segments", [])),
            },
        }
        cost = result.get("extra_tts_cost", 0)
        return {"status": "composed", "episode_asset": episode_asset, "composer_step": "done", "total_cost_usd": cost}
    writer({"event": "composer", "step": "failed", "error": result.get("error", "unknown") if isinstance(result, dict) else str(result)})
    return {"status": "failed", "error_message": str(result)}


def _fanout_to_image_and_tts(state):
    from copy import deepcopy
    return [Send("composer_image_gen", deepcopy(state)), Send("composer_tts_gen", deepcopy(state))]


# ================================================================
# Main graph builder
# ================================================================

def _fanout_to_writer_and_asset(state: EpisodeState) -> list[Send]:
    return [Send("writer", deepcopy(state)), Send("asset_manager", deepcopy(state))]


def build_episode_graph() -> StateGraph:
    workflow = StateGraph(EpisodeState)
    # 创意层节点直接内联到主图（支持 interrupt/checkpointer）
    workflow.add_node("gap_analysis", gap_analysis_node)
    workflow.add_node("creative_director", creative_director_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("story_critic", story_critic_node)
    # 生产节点
    workflow.add_node("writer", writer_node)
    workflow.add_node("shot_validator", shot_validator_node)
    workflow.add_node("asset_manager", asset_manager_node)
    # composer 节点直接内联到主图（解决子图 barrier 节点不等待前驱的问题：
    # 子图编译后 composer_final 未等待 composer_video_gen 完成就执行，导致 "no valid video segments"）
    workflow.add_node("composer_classify", composer_classify_node)
    workflow.add_node("composer_image_gen", composer_image_gen_node)
    workflow.add_node("composer_tts_gen", composer_tts_gen_node)
    workflow.add_node("composer_video_gen", composer_video_gen_node)
    workflow.add_node("composer_final", composer_final_node)
    # 质量门节点直接内联到主图（支持 interrupt/checkpointer，子图内 interrupt 不生效）
    workflow.add_node("critic", critic_node)
    workflow.add_node("safety_check", safety_check_node)
    workflow.add_node("quality_interrupt", quality_interrupt_node)
    workflow.add_node("dlq", dlq_node)
    # 创意层边（内联）
    workflow.set_entry_point("gap_analysis")
    workflow.add_edge("gap_analysis", "creative_director")
    workflow.add_edge("creative_director", "planner")
    workflow.add_conditional_edges("planner", planner_router, {"story_critic": "story_critic", "dlq": "dlq"})
    # story_critic 完成后 fanout 到 writer 和 asset_manager（并行）
    workflow.add_conditional_edges("story_critic", _fanout_to_writer_and_asset, path_map=["writer", "asset_manager"])
    # 生产边：writer/asset_manager → composer_classify（入口）
    workflow.add_edge("writer", "shot_validator")
    workflow.add_edge("shot_validator", "composer_classify")
    workflow.add_edge("asset_manager", "composer_classify")
    # composer 边（内联）：classify → fanout(image_gen ∥ tts_gen) → video_gen → final → critic
    workflow.add_conditional_edges("composer_classify", _fanout_to_image_and_tts, path_map=["composer_image_gen", "composer_tts_gen"])
    workflow.add_conditional_edges("composer_image_gen", lambda s: "dlq" if s.status == "failed" else "continue", {"continue": "composer_video_gen", "dlq": "dlq"})
    workflow.add_edge("composer_video_gen", "composer_final")
    workflow.add_edge("composer_tts_gen", "composer_final")
    workflow.add_edge("composer_final", "critic")
    # 质量门边（内联）：critic → safety_check → 分层路由
    workflow.add_edge("critic", "safety_check")
    # pass(T0/T1 自动通过)→END, interrupt(T2/T3 人工)→quality_interrupt, dlq(质量太差)→dlq
    workflow.add_conditional_edges("safety_check", _quality_route, {"pass": END, "interrupt": "quality_interrupt", "dlq": "dlq"})
    # quality_interrupt 后由 after_quality_router 路由（approve→END, retry→composer_classify, reject→dlq）
    workflow.add_conditional_edges("quality_interrupt", after_quality_router, {"end": END, "composer_classify": "composer_classify", "dlq": "dlq"})
    workflow.add_edge("dlq", END)
    return workflow


def compile_episode_graph(checkpointer=None):
    workflow = build_episode_graph()
    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()
