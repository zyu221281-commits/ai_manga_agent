"""错误分类与处理策略

 错误处理策略：
- 可恢复错误：指数退避重试（自动）
- 账户级错误：停止该平台分发 + 告警
- 系统级错误：暂停全局 + 告警 + 人工介入
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

from app.core.exceptions import (
    AMABaseError,
    APIAuthError,
    APIGatewayError,
    APIRateLimitError,
    APITimeoutError,
    APIError,
    BudgetExhaustedError,
    BudgetHardStopError,
    SafetyCheckFailed,
)

logger = logging.getLogger(__name__)


class ErrorLevel(str, Enum):
    RECOVERABLE = "recoverable"     # 自动重试
    ACCOUNT = "account"             # 停止该平台
    SYSTEM = "system"               # 暂停全局


class ErrorCategory(str, Enum):
    API_RATE_LIMIT = "api_rate_limit"
    API_AUTH = "api_auth"
    API_TIMEOUT = "api_timeout"
    API_GATEWAY = "api_gateway"
    BUDGET = "budget"
    SAFETY = "safety"
    AGENT = "agent"
    WORKFLOW = "workflow"
    DATA = "data"
    RPA = "rpa"
    UNKNOWN = "unknown"


def classify_error(error: Exception) -> tuple[ErrorCategory, ErrorLevel, Optional[str]]:
    """分类错误，返回 (category, level, suggested_action)。

    Suggested actions:
    - retry: 自动重试
    - fallback: 切换到备用方案
    - pause_account: 暂停该平台
    - pause_global: 暂停全局
    - manual_review: 人工审核
    - skip: 跳过当前任务
    """
    mapping = {
        APIRateLimitError: (ErrorCategory.API_RATE_LIMIT, ErrorLevel.RECOVERABLE, "retry"),
        APIAuthError: (ErrorCategory.API_AUTH, ErrorLevel.SYSTEM, "pause_global"),
        APITimeoutError: (ErrorCategory.API_TIMEOUT, ErrorLevel.RECOVERABLE, "retry"),
        APIGatewayError: (ErrorCategory.API_GATEWAY, ErrorLevel.RECOVERABLE, "fallback"),
        BudgetExhaustedError: (ErrorCategory.BUDGET, ErrorLevel.SYSTEM, "pause_global"),
        BudgetHardStopError: (ErrorCategory.BUDGET, ErrorLevel.SYSTEM, "pause_global"),
        SafetyCheckFailed: (ErrorCategory.SAFETY, ErrorLevel.SYSTEM, "manual_review"),
    }

    # Walk MRO to find matching error class
    for exc_type in type(error).__mro__:
        if exc_type in mapping:
            return mapping[exc_type]

    if isinstance(error, APIError):
        return ErrorCategory.API_GATEWAY, ErrorLevel.RECOVERABLE, "fallback"
    if isinstance(error, AMABaseError):
        return ErrorCategory.UNKNOWN, ErrorLevel.RECOVERABLE, "skip"

    return ErrorCategory.UNKNOWN, ErrorLevel.SYSTEM, "pause_global"


def handle_error(error: Exception, context: str = "") -> str:
    """统一错误处理入口，记录日志并返回处理建议。

    Returns: 处理建议字符串（retry / fallback / pause_global / manual_review / skip）
    """
    category, level, action = classify_error(error)
    logger.error(
        "[%s] %s: %s (level=%s, action=%s, context=%s)",
        category.value,
        type(error).__name__,
        str(error)[:200],
        level.value,
        action,
        context,
    )
    return action or "skip"
