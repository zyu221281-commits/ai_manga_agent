"""告警管理（AlertManager → 钉钉/飞书 webhook）"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    WARN = "WARN"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    name: str
    severity: AlertSeverity
    message: str
    source: str = "ai_manga_agent"


class AlertManager:
    """告警管理器。

    Usage:
        alerts = AlertManager()
        await alerts.send(Alert("BudgetAlert80", AlertSeverity.WARN, "Budget at 80%"))
    """

    def __init__(self, webhook_url: str = ""):
        self._webhook_url = webhook_url or settings.ALERT_WEBHOOK_URL
        self._http = httpx.AsyncClient(timeout=10.0)

    async def send(self, alert: Alert) -> bool:
        """发送告警到 webhook。"""
        if not self._webhook_url:
            logger.warning("Alert (no webhook): [%s] %s", alert.severity.value, alert.message)
            return False

        payload = {
            "msgtype": "markdown",
            "markdown": {
                "title": f"[{alert.severity.value}] {alert.name}",
                "text": (
                    f"## [{alert.severity.value}] {alert.name}\n"
                    f"> {alert.message}\n\n"
                    f"**Source**: {alert.source}\n"
                ),
            },
        }

        try:
            resp = await self._http.post(self._webhook_url, json=payload)
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.error("Failed to send alert: %s", e)
            return False

    async def close(self):
        await self._http.aclose()
