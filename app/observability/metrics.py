"""Prometheus 指标暴露

 可观测性 12.2：核心指标定义。
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# --- 成本类 ---
cost_per_minute_usd = Gauge(
    "ama_cost_per_minute_usd", "Current cost rate per minute"
)
budget_remaining_usd = Gauge(
    "ama_budget_remaining_usd", "Remaining budget", ["scope"]
)
cost_per_episode_usd = Gauge(
    "ama_cost_per_episode_usd", "Cost per episode in USD"
)

# --- 任务类 ---
episode_success_rate = Gauge(
    "ama_episode_success_rate", "Episode success rate"
)
agent_duration_seconds = Histogram(
    "ama_agent_duration_seconds", "Agent execution duration", ["agent"]
)
dlq_queue_depth = Gauge(
    "ama_dlq_queue_depth", "DLQ queue depth"
)

# --- API 类 ---
api_rate_limit_count = Counter(
    "ama_api_rate_limit_count", "API rate limit hits", ["api"]
)
api_error_rate = Gauge(
    "ama_api_error_rate", "API error rate", ["api"]
)
cache_hit_rate = Gauge(
    "ama_cache_hit_rate", "Cache hit rate"
)

# --- StoryCritic 类 ---
story_critic_outline_score = Gauge(
    "ama_story_critic_outline_score", "StoryCritic outline score"
)
story_critic_rewrite_count = Counter(
    "ama_story_critic_rewrite_count", "Number of rewrites triggered"
)

# --- 视频分级类 ---
video_strategy_ratio = Gauge(
    "ama_video_strategy_ratio", "Video key scene ratio"
)

# --- 人工看板类 ---
review_pending_count = Gauge(
    "ama_review_pending_count", "Pending reviews", ["source"]
)
review_sla_breach_total = Counter(
    "ama_review_sla_breach_total", "SLA breaches"
)

# --- 下架自愈类 ---
takedown_recovery_count = Counter(
    "ama_takedown_recovery_count", "Takedown recovery events", ["level"]
)

# --- Critic 校准类 ---
critic_calibration_pearson_r = Gauge(
    "ama_critic_calibration_pearson_r", "Critic calibration Pearson r"
)

# --- AI 标注类 ---
ai_labeling_applied_total = Counter(
    "ama_ai_labeling_applied_total", "AI labeling applied count", ["platform"]
)


def init_metrics():
    """初始化指标（在 FastAPI app startup 调用）。"""
    pass  # prometheus_client 自动注册
