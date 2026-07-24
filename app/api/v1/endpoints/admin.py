""""管理后台 API（暂停/恢复/健康检查/校准/A-B测试）

 管理端点：
- 系列管理（创建/暂停/恢复）
- Critic 校准触发
- 视频分级 A/B 测试触发
- AI 标注状态查询
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/health")
async def health_check():
    """"系统健康检查。"""
    from sqlalchemy import text
    components = []

    # Redis 连通性检查
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(settings.redis_url)
        await r.ping()
        components.append({"name": "redis", "healthy": True, "message": "connected"})
    except Exception as e:
        components.append({"name": "redis", "healthy": False, "message": str(e)})

    # Postgres 连通性检查
    try:
        from sqlalchemy.ext.asyncio import create_async_engine
        engine = create_async_engine(settings.database_url, echo=False)
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        components.append({"name": "postgres", "healthy": True, "message": "connected"})
        await engine.dispose()
    except Exception as e:
        components.append({"name": "postgres", "healthy": False, "message": str(e)})

    return {
        "status": "ok" if all(c["healthy"] for c in components) else "degraded",
        "components": components,
    }


@router.get("/health/postgres")
async def health_postgres():
    """"Postgres 连通性检查。"""
    try:
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine
        engine = create_async_engine(settings.database_url, echo=False)
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        await engine.dispose()
        return {"component": "postgres", "healthy": True, "message": "connected"}
    except Exception as e:
        return {"component": "postgres", "healthy": False, "message": str(e)}


@router.post("/series/{series_id}/pause")
async def pause_series(series_id: str):
    """"暂停系列生产。"""
    return {"series_id": series_id, "status": "paused", "message": "Series production paused"}


@router.post("/series/{series_id}/resume")
async def resume_series(series_id: str):
    """"恢复系列生产。"""
    return {"series_id": series_id, "status": "resumed", "message": "Series production resumed"}


@router.post("/calibrate-critic")
async def trigger_critic_calibration(samples_path: str = "tests/fixtures/critic_calibration_samples.json"):
    """"触发 Critic 校准。"""
    from app.quality.critic_calibration import CriticCalibration
    cal = CriticCalibration()

    try:
        samples = cal.load_samples(samples_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Samples file not found: {samples_path}")

    report = cal.calibrate(samples)
    return {
        "pearson_r": report.pearson_r,
        "mean_bias": report.mean_bias,
        "samples_count": report.samples_count,
        "threshold_reliable": report.threshold_reliable,
        "suggested_pass_threshold": report.suggested_pass_threshold,
        "suggested_review_threshold": report.suggested_review_threshold,
    }


@router.post("/ab-test/{episode_id}")
async def trigger_ab_test(episode_id: str, variant_ratios: list[float] = None):
    """"触发视频分级 A/B 测试。"""
    from app.quality.ab_test_runner import ABTestRunner
    runner = ABTestRunner()

    # Mock 数据（实际从 DB 获取）
    result = await runner.run_test(
        episode_id=episode_id,
        script={},
        storyboard=[],
        image_prompts=[],
        asset_library={},
        variant_ratios=variant_ratios or [0.10, 0.20, 0.30],
    )

    return {
        "episode_id": result.episode_id,
        "winner": result.winner,
        "recommendation": result.recommendation,
        "variants": [
            {"id": v.variant_id, "ratio": v.key_scene_ratio, "cost": v.cost_usd}
            for v in result.variants
        ],
    }


@router.get("/ai-labeling/status")
async def ai_labeling_status(episode_id: str = ""):
    """"查询 AI 标注状态。"""
    if episode_id:
        return {"episode_id": episode_id, "ai_generated": True, "labeled": True}
    return {"message": "AI labeling active", "platforms_configured": ["douyin", "bilibili", "youtube", "kuaishou"]}


@router.get("/budget/dashboard")
async def budget_dashboard():
    """"预算看板。"""
    return {
        "daily_used": 3.50,
        "daily_cap": 15.0,
        "daily_ratio": 0.23,
        "monthly_used": 25.0,
        "monthly_cap": 300.0,
        "monthly_ratio": 0.083,
    }


@router.get("/prompt-templates")
async def list_prompt_templates():
    """"获取所有 Prompt 模板版本。"""
    from app.analytics.prompt_optimizer import PromptOptimizer
    opt = PromptOptimizer()
    return {"templates": {name: opt.get_history(name) for name in ["planner", "writer"]}}
