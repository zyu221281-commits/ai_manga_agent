"""Episode task: Celery + LangGraph  production entry point.

Replaces the old manual _run_pipeline with graph.ainvoke().
Supports GraphInterrupt for Creative Gate and Quality Gate.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

from langgraph.errors import GraphInterrupt

from app.tasks.celery_app import celery_app
from app.resilience.idempotency import idempotency_lock
from app.state.episode_state import EpisodeState
from app.state.graph_builder import compile_episode_graph
from app.services.file_lineage_tracker import FileLineageTracker

# 触发 app.core.__init__ 设置 Windows EventLoop 策略
# （SelectorEventLoop，兼容 psycopg async / AsyncPostgresSaver）
import app.core  # noqa: F401

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"


def _ensure_output_dirs():
    for sub in ["images", "videos", "audio", "traces"]:
        (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)


def _ensure_checkpointer_tables():
    """幂等创建 LangGraph checkpoint 表（仅在使用 AsyncPostgresSaver 时需要）。

    绕过 AsyncPostgresSaver.setup() 的 CREATE INDEX CONCURRENTLY 事务冲突。
    PG 不可用时静默失败，由 checkpointer 自身在连接时报错（不阻塞 worker 启动）。
    """
    try:
        from app.db.init_pg_tables import setup_langgraph_tables

        setup_langgraph_tables()
    except Exception as e:
        logger.warning("setup_langgraph_tables skipped: %s", e)


@celery_app.task(bind=True, max_retries=2, default_retry_delay=120, acks_late=True, name="app.tasks.episode_task.run_episode_task")
def run_episode_task(self, series_id: str, episode_num: int, creative_brief: Optional[dict] = None, series_plan: Optional[dict] = None, asset_library: Optional[dict] = None, checkpointer=None) -> dict:
    episode_id = f"{series_id}_ep_{episode_num}"
    trace_id = f"celery_{series_id}_{episode_num}"
    acquired, holder = asyncio.run(idempotency_lock.try_acquire(episode_id))
    if not acquired:
        return {"episode_id": episode_id, "status": "skipped", "reason": "idempotency_lock"}
    try:
        _ensure_output_dirs()
        # 使用 AsyncPostgresSaver 等 checkpointer 时，先幂等创建表
        # （绕过 AsyncPostgresSaver.setup() 的事务冲突）
        if checkpointer is not None:
            _ensure_checkpointer_tables()
        initial_state = EpisodeState(episode_id=episode_id, series_id=series_id, episode_num=episode_num, trace_id=trace_id, creative_brief=creative_brief or {}, series_plan=series_plan or {}, asset_library=asset_library or {})
        graph = compile_episode_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": episode_id}}
        # celery task 是 sync 函数，用 asyncio.run() 驱动 ainvoke
        # （与上方 idempotency_lock.try_acquire / release 一致）
        final_state = asyncio.run(graph.ainvoke(initial_state, config))
        try:
            tracer = FileLineageTracker(trace_id=trace_id)
            tracer.flush()
        except Exception:
            pass
        result = {"episode_id": episode_id, "series_id": series_id, "episode_num": episode_num, "status": final_state.status, "critic_score": final_state.critic_score, "critic_decision": final_state.critic_decision, "total_cost_usd": round(final_state.total_cost_usd, 4), "trace_id": trace_id}
        logger.info("Episode %s done: score=%.2f decision=%s cost=$%.4f", episode_id, final_state.critic_score, final_state.critic_decision, final_state.total_cost_usd)
        return result
    except GraphInterrupt as e:
        interrupt_payload = e.args[0] if e.args else {}
        gate_type = interrupt_payload.get("type", "unknown")
        logger.info("Episode %s interrupted at %s gate", episode_id, gate_type)
        return {"episode_id": episode_id, "status": f"awaiting_{gate_type}", "thread_id": episode_id, "interrupt_payload": interrupt_payload, "trace_id": trace_id}
    except Exception as exc:
        logger.error("Episode %s failed: %s", episode_id, exc)
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)
        return {"episode_id": episode_id, "status": "failed", "error": str(exc)[:500], "retries_exhausted": True}
    finally:
        asyncio.run(idempotency_lock.release(episode_id, holder))


@celery_app.task(bind=True, max_retries=1, name="app.tasks.episode_task.retry_episode_task")
def retry_episode_task(self, series_id: str, episode_num: int, retry_reason: str = "", series_plan: Optional[dict] = None, asset_library: Optional[dict] = None) -> str:
    return run_episode_task.apply_async(args=[series_id, episode_num], kwargs={"creative_brief": None, "series_plan": series_plan, "asset_library": asset_library}, queue="episodes", priority=7).id
