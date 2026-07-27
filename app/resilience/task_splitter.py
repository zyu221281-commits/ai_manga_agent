"""任务拆分器（V4 韧性加固）

N 集 → N 个独立 Celery 链，单集失败不阻塞其他集，可独立重试。
总集数默认从 settings.DEFAULT_TOTAL_EPISODES 读取（默认 30）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EpisodeTask:
    """单集任务描述。"""
    episode_num: int
    series_id: str
    episode_id: Optional[str] = None
    title: Optional[str] = None
    episode_plan: Optional[dict] = None
    dependencies: list[int] = field(default_factory=list)  # 依赖的 episode_num
    priority: int = 0  # 0=normal, 1=high


@dataclass
class SeriesTaskPlan:
    """系列任务计划。"""
    series_id: str
    total_episodes: int
    episodes: list[EpisodeTask]
    parallel_groups: list[list[int]]  # 每个子列表是并行组
    schedule_order: list[int]  # 最终执行顺序


class TaskSplitter:
    """将系列拆分为独立任务，计算调度顺序。

    V4 策略：
    - 每集一个独立 Celery task
    - 并行度 = 3（同时最多 3 集运行）
    - 依赖链：第 n 集的 Planner 可能依赖第 n-1 集的前情摘要
    """

    MAX_PARALLEL = 3

    def split(self, series_id: str, total_episodes: int = None) -> SeriesTaskPlan:
        """拆分系列为独立任务。

        Args:
            series_id: 系列 ID
            total_episodes: 总集数。None 时使用 settings.DEFAULT_TOTAL_EPISODES（默认 30）
        """
        if total_episodes is None:
            from app.core.config import settings
            total_episodes = settings.DEFAULT_TOTAL_EPISODES
        total_episodes = max(1, int(total_episodes))
        episodes = []
        for i in range(1, total_episodes + 1):
            deps = [i - 1] if i > 1 else []
            episodes.append(EpisodeTask(
                episode_num=i,
                series_id=series_id,
                title=f"Episode {i}",
                dependencies=deps,
                priority=0,
            ))

        # 按依赖分组：每 MAX_PARALLEL 集为一组
        parallel_groups = []
        schedule_order = []
        for i in range(1, total_episodes + 1, self.MAX_PARALLEL):
            group = list(range(i, min(i + self.MAX_PARALLEL, total_episodes + 1)))
            parallel_groups.append(group)
            schedule_order.extend(group)

        return SeriesTaskPlan(
            series_id=series_id,
            total_episodes=total_episodes,
            episodes=episodes,
            parallel_groups=parallel_groups,
            schedule_order=schedule_order,
        )

    def get_next_batch(
        self,
        plan: SeriesTaskPlan,
        completed: set[int],
        running: set[int],
    ) -> list[EpisodeTask]:
        """获取下一批可执行的任务（最多 MAX_PARALLEL 个）。

        Args:
            plan: 系列任务计划
            completed: 已完成的 episode_num 集合
            running: 正在运行的 episode_num 集合
        """
        available = []
        for ep in plan.episodes:
            if ep.episode_num in completed or ep.episode_num in running:
                continue
            if all(dep in completed for dep in ep.dependencies):
                available.append(ep)
                if len(available) >= self.MAX_PARALLEL - len(running):
                    break
        return available

    def is_episode_ready(self, plan: SeriesTaskPlan, episode_num: int, completed: set[int]) -> bool:
        """判断单集是否可以执行。"""
        ep = next((e for e in plan.episodes if e.episode_num == episode_num), None)
        if ep is None:
            return False
        return all(dep in completed for dep in ep.dependencies)

    def get_progress(self, plan: SeriesTaskPlan, completed: set[int]) -> float:
        """获取系列完成进度。"""
        return len(completed) / plan.total_episodes if plan.total_episodes > 0 else 0.0
