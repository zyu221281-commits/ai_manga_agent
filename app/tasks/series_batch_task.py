"""系列批量任务 + 幂等键 + 并行度=3

V4 韧性加固：
- N 集 → N 个独立 Celery 链
- episode_id 作为幂等键
- 重投不重复执行
- 总集数：默认从 settings.DEFAULT_TOTAL_EPISODES 读取（30），可在调用时覆盖
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from celery import group

from app.tasks.celery_app import celery_app
from app.tasks.episode_task import run_episode_task
from app.resilience.idempotency import idempotency_lock
from app.resilience.task_splitter import TaskSplitter

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    max_retries=1,
    default_retry_delay=300,
    name="app.tasks.series_batch_task.create_series_tasks",
)
def create_series_tasks(self, series_id: str, total_episodes: int = None):
    """为整个系列创建所有单集任务。

    Args:
        series_id: 系列 ID
        total_episodes: 总集数。None 时使用 settings.DEFAULT_TOTAL_EPISODES（默认 30）
    """
    from app.core.config import settings
    if total_episodes is None:
        total_episodes = settings.DEFAULT_TOTAL_EPISODES
    total_episodes = max(1, min(int(total_episodes), settings.MAX_TOTAL_EPISODES))

    splitter = TaskSplitter()
    plan = splitter.split(series_id, total_episodes)

    logger.info(
        "Creating %d episode tasks for series %s (%d parallel groups)",
        total_episodes, series_id, len(plan.parallel_groups),
    )

    # 按并行组创建任务链
    for group_num, episode_nums in enumerate(plan.parallel_groups):
        task_signatures = [
            run_episode_task.si(series_id, ep_num)
            for ep_num in episode_nums
        ]
        # 并行组执行
        group(*task_signatures).apply_async(
            queue="episodes",
            task_id=f"series_{series_id}_group_{group_num}",
        )
        logger.debug("Dispatched group %d: episodes %s", group_num, episode_nums)

    return {
        "series_id": series_id,
        "total_episodes": total_episodes,
        "parallel_groups": len(plan.parallel_groups),
    }


@celery_app.task(
    bind=True,
    max_retries=2,
    default_retry_delay=120,
    name="app.tasks.series_batch_task.dispatch_episode",
)
def dispatch_episode_task(
    self,
    series_id: str,
    episode_num: int,
    creative_brief: Optional[dict[str, Any]] = None,
):
    """分发单个单集任务（带幂等检查）。

    同一 episode_id 在锁有效期内不会被重复执行。
    """
    episode_id = f"{series_id}_ep_{episode_num}"

    # 幂等检查
    is_locked = idempotency_lock.is_locked(episode_id)
    if is_locked:
        logger.info("Episode %s already running, skipping", episode_id)
        return {"episode_id": episode_id, "status": "already_running"}

    # 提交单集任务
    result = run_episode_task.apply_async(
        args=[series_id, episode_num],
        kwargs={"creative_brief": creative_brief},
        queue="episodes",
        task_id=f"episode_{episode_id}",
    )

    return {
        "episode_id": episode_id,
        "celery_task_id": result.id,
        "status": "dispatched",
    }


@celery_app.task(
    bind=True,
    name="app.tasks.series_batch_task.get_series_progress",
)
def get_series_progress(self, series_id: str) -> dict[str, Any]:
    """查询系列完成进度。"""
    # 在完整实现中从 PG episodes 表查询
    from app.core.config import settings
    return {
        "series_id": series_id,
        "total": settings.DEFAULT_TOTAL_EPISODES,
        "completed": 0,
        "failed": 0,
        "running": 0,
        "progress": 0.0,
    }
