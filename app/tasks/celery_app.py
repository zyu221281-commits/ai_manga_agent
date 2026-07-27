""""Celery + Beat 配置（含 DLQ）

V4 任务调度：
- Celery 任务粒度调度（不参与单集内部流程）
- N 集 → N 个 Celery task → 各自启动 LangGraph 工作流
- 并行度 = 3，episode_id 作为幂等键
- 重试 2 次失败 → DLQ
- N 默认从 settings.DEFAULT_TOTAL_EPISODES 读取（默认 30）
"""

from __future__ import annotations

import logging

from celery import Celery
from celery.schedules import crontab

from app.core.config import settings

logger = logging.getLogger(__name__)

# Celery 实例
celery_app = Celery(
    "ai_manga_agent",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[
        "app.tasks.episode_task",
        "app.tasks.series_batch_task",
    ],
)

# 额外配置
celery_app.conf.update(
    # 任务配置
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Shanghai",
    enable_utc=True,

    # 并发
    worker_concurrency=3,                # 同时最多 3 集
    worker_prefetch_multiplier=1,

    # 重试
    task_acks_late=True,                 # 执行完成后才 ACK
    task_reject_on_worker_lost=True,     # Worker 丢失时重试
    task_default_retry_delay=60,         # 重试延迟 60s
    task_max_retries=2,                  # 最多重试 2 次

    # DLQ
    task_queue_max_priority=10,
    task_default_priority=5,

    # 结果存储
    result_expires=86400,                # 24h

    # Rate limit
    task_annotations={
        "app.tasks.episode_task.run_episode": {"rate_limit": "3/m"},
    },
)

# Beat 定时任务
celery_app.conf.beat_schedule = {
    "health-check": {
        "task": "app.tasks.checkpoint.health_check_task",
        "schedule": crontab(minute="*/15"),  # 每 15 分钟
        "options": {"queue": "maintenance"},
    },
    "cost-summary": {
        "task": "app.tasks.checkpoint.cost_summary_task",
        "schedule": crontab(hour=9, minute=0),  # 早 9 点
        "options": {"queue": "maintenance"},
    },
    "backup-daily": {
        "task": "app.tasks.checkpoint.backup_task",
        "schedule": crontab(hour=2, minute=0),  # 凌晨 2 点
        "options": {"queue": "maintenance"},
    },
}

# 队列定义
celery_app.conf.task_routes = {
    "app.tasks.episode_task.*": {"queue": "episodes"},
    "app.tasks.series_batch_task.*": {"queue": "series"},
    "app.tasks.checkpoint.*": {"queue": "maintenance"},
}

# DLQ 配置：失败任务自动路由到 DLQ
celery_app.conf.task_queues = None  # 自动创建


# 用于测试/开发的便捷配置覆盖
def configure_for_testing():
    celery_app.conf.update(
        task_always_eager=True,
        task_eager_propagates=True,
    )
