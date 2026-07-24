"""结构化日志配置

 可观测性 12.1：structlog + trace_id 贯穿 LangGraph。
"""

from __future__ import annotations

import logging
import structlog


def configure_logging(level: str = "INFO"):
    """配置结构化日志。

    Args:
        level: 日志级别 (DEBUG/INFO/WARN/ERROR)
    """
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def get_logger(name: str = "ai_manga_agent", **context) -> structlog.BoundLogger:
    """获取结构化的 logger 实例。

    Usage:
        log = get_logger("agent.planner", episode_id="ep_1", trace_id="trace_123")
        log.info("episode_started", cost_usd=0.5, duration_ms=1200)
    """
    return structlog.get_logger(name).bind(**context)
