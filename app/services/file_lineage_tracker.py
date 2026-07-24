"""File-level data lineage tracker with graph event support ().

Extended with: graph-level events (node start/end, routing decisions,
interrupt triggers, custom events), Prometheus metrics emission, and
model_calls tracking for Bad Case reproduction.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"
TRACES_DIR = OUTPUT_DIR / "traces"


class FileLineageTracker:
    """File-level data lineage tracker with graph event support.

    Usage:
        tracer = FileLineageTracker(trace_id="trace_001")

        # Agent-level (unchanged API)
        tracer.record_start("planner", input_keys=["creative_brief"])
        tracer.record_end("planner", output_keys=["series_plan"], cost_usd=0.02,
                          duration_ms=1500, model_calls=[...])

        # Graph-level ( new)
        tracer.record_node_start("creative_director")
        tracer.record_node_end("creative_director")
        tracer.record_routing_decision("critic_router", "retry", "score=0.45")
        tracer.record_interrupt("creative_gate", {"missing": ["tone"]})

        tracer.flush()
    """

    def __init__(self, trace_id: str, output_dir: Optional[Path] = None):
        self.trace_id = trace_id
        self._dir = output_dir or TRACES_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._entries: list[dict] = []
        self._filepath = self._dir / f"{trace_id}.jsonl"

    # === Agent-level events ===

    def record_start(
        self, agent_name: str,
        input_keys: Optional[list[str]] = None,
        input_shapes: Optional[dict[str, Any]] = None,
    ):
        self._entries.append({
            "trace_id": self.trace_id, "level": "agent",
            "agent_name": agent_name, "step": "start",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_keys": input_keys or [],
            "input_shapes": input_shapes or {},
        })

    def record_end(
        self, agent_name: str,
        output_keys: Optional[list[str]] = None,
        output_shapes: Optional[dict[str, Any]] = None,
        cost_usd: float = 0.0, duration_ms: float = 0.0,
        success: bool = True, error: Optional[str] = None,
        model_name: Optional[str] = None,
        model_params: Optional[dict] = None,
        artifact_type: Optional[str] = None,
        model_calls: Optional[list[dict[str, Any]]] = None,
    ):
        entry = {
            "trace_id": self.trace_id, "level": "agent",
            "agent_name": agent_name, "step": "end",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "output_keys": output_keys or [],
            "output_shapes": output_shapes or {},
            "cost_usd": round(cost_usd, 6),
            "duration_ms": round(duration_ms, 0),
            "success": success, "error": error,
            "model_name": model_name, "model_params": model_params,
            "artifact_type": artifact_type,
            "model_calls": model_calls or [],
        }
        self._entries.append(entry)
        self._emit_prometheus(agent_name, duration_ms)

    # === Graph-level events () ===

    def record_node_start(self, node_name: str, run_id: str = "", metadata: Optional[dict] = None):
        self._entries.append({
            "trace_id": self.trace_id, "level": "graph",
            "step": "node_start", "node_name": node_name,
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        })

    def record_node_end(self, node_name: str, run_id: str = "", metadata: Optional[dict] = None):
        self._entries.append({
            "trace_id": self.trace_id, "level": "graph",
            "step": "node_end", "node_name": node_name,
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        })

    def record_routing_decision(self, router_name: str, decision: str, reason: str = ""):
        self._entries.append({
            "trace_id": self.trace_id, "level": "graph",
            "step": "routing", "router_name": router_name,
            "decision": decision, "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def record_interrupt(self, gate_name: str, payload: dict[str, Any]):
        self._entries.append({
            "trace_id": self.trace_id, "level": "graph",
            "step": "interrupt", "gate_name": gate_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload_summary": {
                k: _summarize_value(v) for k, v in payload.items()
                if k != "actions"
            },
        })

    def record_custom_event(self, event_name: str, data: dict[str, Any]):
        self._entries.append({
            "trace_id": self.trace_id, "level": "graph",
            "step": "custom_event", "event_name": event_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        })

    # === I/O ===

    def flush(self):
        if not self._entries:
            return
        with open(self._filepath, "w", encoding="utf-8") as f:
            for entry in self._entries:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        self._entries = []

    @property
    def filepath(self) -> Path:
        return self._filepath

    @classmethod
    def load(cls, trace_id: str, output_dir: Optional[Path] = None) -> list[dict]:
        d = output_dir or TRACES_DIR
        fp = d / f"{trace_id}.jsonl"
        if not fp.exists():
            return []
        entries = []
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        return entries

    @classmethod
    def list_traces(cls, output_dir: Optional[Path] = None) -> list[str]:
        d = output_dir or TRACES_DIR
        if not d.exists():
            return []
        return sorted([f.stem for f in d.glob("*.jsonl")])

    @classmethod
    def summary(cls, trace_id: str, output_dir: Optional[Path] = None) -> dict:
        entries = cls.load(trace_id, output_dir)
        if not entries:
            return {"trace_id": trace_id, "error": "no data"}
        agents, total_cost, total_duration, errors = [], 0.0, 0.0, []
        for e in entries:
            if e.get("level") == "agent" and e.get("step") == "end":
                agents.append({
                    "name": e["agent_name"], "cost_usd": e.get("cost_usd", 0),
                    "duration_ms": e.get("duration_ms", 0),
                    "success": e.get("success", True), "error": e.get("error"),
                    "model_calls": len(e.get("model_calls", [])),
                })
                total_cost += e.get("cost_usd", 0)
                total_duration += e.get("duration_ms", 0)
                if not e.get("success"):
                    errors.append({"agent": e["agent_name"], "error": e.get("error")})
        return {
            "trace_id": trace_id, "agents": agents,
            "total_cost_usd": round(total_cost, 6),
            "total_duration_s": round(total_duration / 1000, 1),
            "agent_count": len(agents), "errors": errors,
        }

    @staticmethod
    def _emit_prometheus(agent_name: str, duration_ms: float):
        try:
            from app.observability.metrics import agent_duration_seconds
            agent_duration_seconds.labels(agent=agent_name).observe(duration_ms / 1000.0)
        except Exception:
            pass


def _summarize_value(v: Any) -> Any:
    if isinstance(v, dict):
        return f"dict({len(v)} keys)"
    if isinstance(v, list):
        return f"list({len(v)} items)"
    if isinstance(v, str) and len(v) > 200:
        return v[:200] + "..."
    return v
