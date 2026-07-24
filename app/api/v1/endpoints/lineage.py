"""数据血缘查询 API（ 新增 — Bad Case 排查）

提供单集完整血缘链查询 + 错误诊断 + 成本分解。
所有数据来源：output/traces/{trace_id}.jsonl（无需 DB）。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.services.file_lineage_tracker import FileLineageTracker

router = APIRouter(prefix="/lineage", tags=["lineage"])


@router.get("/traces")
async def list_traces():
    """列出所有追踪记录。"""
    traces = FileLineageTracker.list_traces()
    return {"count": len(traces), "traces": traces}


@router.get("/traces/{trace_id}")
async def get_trace(trace_id: str):
    """获取单条追踪的完整记录。"""
    entries = FileLineageTracker.load(trace_id)
    if not entries:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
    return {"trace_id": trace_id, "entries": entries, "count": len(entries)}


@router.get("/traces/{trace_id}/summary")
async def get_trace_summary(trace_id: str):
    """获取追踪摘要（Agent 序列、成本、耗时、错误）。"""
    summary = FileLineageTracker.summary(trace_id)
    if "error" in summary:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
    return summary


@router.get("/traces/{trace_id}/errors")
async def get_trace_errors(trace_id: str):
    """获取追踪中的错误信息。"""
    entries = FileLineageTracker.load(trace_id)
    if not entries:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")

    errors = []
    for e in entries:
        if e["step"] == "end" and not e.get("success", True):
            errors.append({
                "agent": e["agent_name"],
                "error": e.get("error"),
                "timestamp": e.get("timestamp"),
                "cost_usd": e.get("cost_usd", 0),
            })

    return {"trace_id": trace_id, "errors": errors, "error_count": len(errors)}


@router.get("/traces/{trace_id}/cost")
async def get_trace_cost(trace_id: str):
    """获取追踪的成本分解。"""
    entries = FileLineageTracker.load(trace_id)
    if not entries:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")

    cost_breakdown = []
    total = 0.0
    for e in entries:
        if e["step"] == "end":
            c = e.get("cost_usd", 0)
            total += c
            cost_breakdown.append({
                "agent": e["agent_name"],
                "cost_usd": c,
                "duration_ms": e.get("duration_ms", 0),
            })

    return {"trace_id": trace_id, "total_cost_usd": round(total, 6), "breakdown": cost_breakdown}