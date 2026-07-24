"""异常定义

 系统所需的所有自定义异常，用于错误分类与处理策略（error_handler.py）。
"""

from __future__ import annotations


class AMABaseError(Exception):
    """AI Manga Agent 基础异常。"""


# ================================================================
# 预算相关
# ================================================================

class BudgetExhaustedError(AMABaseError):
    """预算不足，无法启动新任务。"""
    def __init__(self, message: str = "Budget exhausted"):
        super().__init__(message)


class BudgetHardStopError(AMABaseError):
    """预算熔断，停止所有正在进行的任务。"""
    def __init__(self, message: str = "Hard stop: budget 100% used"):
        super().__init__(message)


# ================================================================
# Agent 相关
# ================================================================

class AgentError(AMABaseError):
    """Agent 执行错误。"""


class PlannerError(AgentError):
    """Planner Agent 失败。"""


class StoryCriticError(AgentError):
    """StoryCritic Agent 评估失败。"""


class WriterError(AgentError):
    """Writer Agent 失败。"""


class AssetManagerError(AgentError):
    """AssetManager Agent 失败。"""


class ComposerError(AgentError):
    """Composer Agent 失败。"""


class CriticError(AgentError):
    """Critic Agent 失败。"""


class CriticScoreBelowThreshold(AgentError):
    """Critic 评分低于阈值，触发重试。"""
    def __init__(self, score: float, threshold: float):
        self.score = score
        self.threshold = threshold
        super().__init__(f"Critic score {score} below threshold {threshold}")


class RetryExhaustedError(AgentError):
    """重试耗尽，任务进 DLQ。"""
    def __init__(self, agent: str, retry_count: int, last_score: float = 0.0):
        self.agent = agent
        self.retry_count = retry_count
        self.last_score = last_score
        super().__init__(f"{agent} retries exhausted ({retry_count}), last score={last_score}")


# ================================================================
# API 相关
# ================================================================

class APIError(AMABaseError):
    """外部 API 错误。"""
    def __init__(self, api: str, status_code: int = 0, message: str = ""):
        self.api = api
        self.status_code = status_code
        super().__init__(f"{api} API error {status_code}: {message}")


class APIRateLimitError(APIError):
    """API 限流。"""


class APIAuthError(APIError):
    """API 认证失败。"""


class APITimeoutError(APIError):
    """API 超时。"""


class APIGatewayError(APIError):
    """API 网关错误（4xx/5xx）。"""


# ================================================================
# 安全审核相关
# ================================================================

class SafetyCheckFailed(AMABaseError):
    """内容安全审核不通过。"""
    def __init__(self, check_type: str, reason: str = ""):
        self.check_type = check_type
        super().__init__(f"Safety check failed ({check_type}): {reason}")


class TextSafetyFailed(SafetyCheckFailed):
    """文本审核不通过。"""


class ImageSafetyFailed(SafetyCheckFailed):
    """图片 NSFW 审核不通过。"""


class PolicyViolation(SafetyCheckFailed):
    """平台政策违规。"""


class AILabelingError(AMABaseError):
    """AI 内容标注失败。"""


# ================================================================
# 工作流相关
# ================================================================

class WorkflowError(AMABaseError):
    """LangGraph 工作流错误。"""


class CheckpointError(WorkflowError):
    """Checkpoint 读写失败。"""


class TaskAlreadyRunning(WorkflowError):
    """同一 episode_id 已有正在运行的任务。"""


class InvalidStateTransition(WorkflowError):
    """非法状态迁移。"""


# ================================================================
# 数据相关
# ================================================================

class DataError(AMABaseError):
    """数据处理错误。"""


class CacheError(DataError):
    """缓存错误。"""


class AssetNotFoundError(DataError):
    """资产未找到。"""


# ================================================================
# RPA 相关
# ================================================================

class RPAError(AMABaseError):
    """RPA 自动化错误。"""


class PublishingError(RPAError):
    """分发失败。"""


class CollectionError(RPAError):
    """采集失败。"""


class TakedownRecoveryError(RPAError):
    """下架恢复失败。"""


# ================================================================
# 合规相关
# ================================================================

class ComplianceError(AMABaseError):
    """合规错误。"""


class IllegalAccountOperation(ComplianceError):
    """非法的账户操作（如尝试使用 IP 隔离）。"""
