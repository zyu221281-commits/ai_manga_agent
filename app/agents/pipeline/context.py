from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PipelineContext:
    script: dict[str, Any]
    storyboard: list[dict]
    image_prompts: list[dict]
    asset_library: dict[str, Any]
    episode_id: str
    key_scene_ratio: Optional[float] = None
    # One visual_spec per shot for deterministic prompt rendering
    visual_specs: list[dict] = field(default_factory=list)

    classified_scenes: list[dict] = field(default_factory=list)
    image_results: list[Any] = field(default_factory=list)
    audio_segments: list[dict] = field(default_factory=list)
    video_segments: list[Any] = field(default_factory=list)
    subtitles: list[dict] = field(default_factory=list)
    covers: list[dict] = field(default_factory=list)
    compose_result: dict = field(default_factory=dict)
    ai_label: dict = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    errors: list[dict] = field(default_factory=list)

    def add_cost(self, cost: float) -> None:
        self.total_cost_usd += cost

    def record_error(self, stage: str, error: Exception, non_blocking: bool = False) -> None:
        self.errors.append({
            "stage": stage,
            "error": str(error),
            "non_blocking": non_blocking,
        })
        if not non_blocking:
            self.metadata.setdefault("blocking_errors", []).append(str(error))

    def update_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value