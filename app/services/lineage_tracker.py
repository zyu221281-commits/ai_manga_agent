"""数据血缘追踪（ 新增）

每步产出自动记录 lineage 到 data_lineage 表：
- 剧本 → 分镜 → 提示词 → 图像 → 视频 → 音频 → 字幕 → 封面
- 通过 parent_lineage_ids 串联上下游
- 记录 Prompt 版本、模型参数、seed，支持 Bad Case 复现
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


ARTIFACT_TYPES = [
    "creative_brief", "series_plan", "outline_evaluation",
    "script", "storyboard", "image_prompt",
    "character_view", "style_template",
    "image", "video", "audio", "subtitle", "cover",
]


class LineageTracker:
    """数据血缘追踪器。

    Usage:
        tracker = LineageTracker(session)
        script_id = await tracker.record(
            episode_id="ep_1",
            artifact_type="script",
            artifact_data={"content": "..."},
            prompt_template_id="pt_1",
            model_name="deepseek--pro",
            model_params={"temperature": 0.7},
            seed=42,
            trace_id="trace_123",
            parent_ids=[storyboard_id],
        )
    """

    def __init__(self, session: AsyncSession):
        self._session = session

    async def record(
        self,
        episode_id: str,
        artifact_type: str,
        artifact_data: dict[str, Any],
        prompt_template_id: Optional[str] = None,
        model_name: Optional[str] = None,
        model_params: Optional[dict] = None,
        seed: Optional[int] = None,
        trace_id: Optional[str] = None,
        cost_record_id: Optional[str] = None,
        parent_ids: Optional[list[str]] = None,
    ) -> str:
        """写入一条血缘记录。返回 lineage_id。

        PG 不可用时静默降级：返回空 lineage_id，但不影响主管线。
        """
        lineage_id = str(uuid.uuid4())
        try:
            sql = text("""
                INSERT INTO data_lineage (
                    id, episode_id, artifact_type, artifact_data,
                    prompt_template_id, model_name, model_params, seed,
                    trace_id, cost_record_id, parent_lineage_ids
                ) VALUES (
                    :id, :episode_id, :artifact_type, :artifact_data,
                    :prompt_template_id, :model_name, :model_params, :seed,
                    :trace_id, :cost_record_id, :parent_lineage_ids
                )
            """)
            await self._session.execute(sql, {
                "id": lineage_id,
                "episode_id": episode_id,
                "artifact_type": artifact_type,
                "artifact_data": json.dumps(artifact_data),
                "prompt_template_id": prompt_template_id,
                "model_name": model_name,
                "model_params": json.dumps(model_params) if model_params else None,
                "seed": seed,
                "trace_id": trace_id,
                "cost_record_id": cost_record_id,
                "parent_lineage_ids": parent_ids or [],
            })
            await self._session.commit()
        except Exception as e:
            logger.warning(
                "Failed to record lineage to PG (ep=%s, type=%s): %s",
                episode_id, artifact_type, e,
            )
            # 返回空字符串让上游链路 parent_ids 引用失效但不阻塞
            return ""
        return lineage_id

    async def get_episode_lineage(self, episode_id: str) -> list[dict]:
        """获取单集完整血缘链。"""
        sql = text("""
            SELECT * FROM data_lineage
            WHERE episode_id = :eid
            ORDER BY created_at
        """)
        result = await self._session.execute(sql, {"eid": episode_id})
        return [dict(row._mapping) for row in result]

    async def get_artifact_lineage(self, lineage_id: str) -> Optional[dict]:
        """获取单个产物的血缘信息。"""
        sql = text("SELECT * FROM data_lineage WHERE id = :lid")
        result = await self._session.execute(sql, {"lid": lineage_id})
        row = result.fetchone()
        return dict(row._mapping) if row else None

    async def get_reproduce_params(self, lineage_id: str) -> Optional[dict]:
        """获取复现参数（Bad Case 排查用）。"""
        sql = text("""
            SELECT
                artifact_type, prompt_template_id, model_name,
                model_params, seed, trace_id
            FROM data_lineage
            WHERE id = :lid
        """)
        result = await self._session.execute(sql, {"lid": lineage_id})
        row = result.fetchone()
        if not row:
            return None
        r = dict(row._mapping)
        r["model_params"] = json.loads(r["model_params"]) if r["model_params"] else None
        return r

    async def get_prompt_episodes(self, prompt_template_id: str) -> list[str]:
        """查询某 Prompt 版本影响了哪些集。"""
        sql = text("""
            SELECT DISTINCT episode_id FROM data_lineage
            WHERE prompt_template_id = :pid
        """)
        result = await self._session.execute(sql, {"pid": prompt_template_id})
        return [row[0] for row in result]
