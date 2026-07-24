"""密钥管理（KMS 抽象）

 设计：
- 本地开发：从 .env 读取（git 忽略）
- 生产预留：阿里云 KMS / AWS Secrets Manager（接口已抽象）
- Key 轮换：每 90 天提醒（Demo 仅提醒，不强制）
- 访问审计：记录谁/什么时候取了哪个 Key
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


logger = logging.getLogger(__name__)


@dataclass
class SecretAccessRecord:
    secret_name: str
    accessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "env"


class SecretProvider(ABC):
    """密钥提供者抽象接口。"""

    @abstractmethod
    async def get_secret(self, name: str) -> Optional[str]:
        ...

    @abstractmethod
    async def list_secrets(self) -> list[str]:
        ...


class EnvSecretProvider(SecretProvider):
    """本地开发：从环境变量 / .env 读取。"""

    async def get_secret(self, name: str) -> Optional[str]:
        val = os.getenv(name)
        if val:
            self._audit(name, "env")
        return val

    async def list_secrets(self) -> list[str]:
        sensitive_keys = [
            "DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY", "SILICONFLOW_API_KEY",
            "ARK_API_KEY", "AZURE_SPEECH_KEY", "VOLCENGINE_TTS_KEY",
            "FANQIE_API_KEY", "QUARK_API_KEY", "DOUYIN_OPEN_KEY",
            "BILIBILI_OPEN_KEY", "YOUTUBE_API_KEY", "KUAISHOU_OPEN_KEY",
            "ALIYUN_SAFETY_KEY",
        ]
        return [k for k in sensitive_keys if os.getenv(k)]

    def _audit(self, name: str, source: str):
        """在正式项目中写入 audit_events 表。"""
        logger.debug("Secret accessed: %s from %s", name, source)


class SecretsManager:
    """密钥管理器，封装提供者切换。

    Usage:
        sec = SecretsManager()
        api_key = await sec.get("DEEPSEEK_API_KEY")
    """

    def __init__(self, provider: Optional[SecretProvider] = None):
        self._provider = provider or EnvSecretProvider()
        self._access_log: list[SecretAccessRecord] = []

    async def get(self, name: str) -> Optional[str]:
        secret = await self._provider.get_secret(name)
        if secret:
            self._access_log.append(SecretAccessRecord(secret_name=name))
        return secret

    async def get_required(self, name: str) -> str:
        """获取必须存在的密钥，不存在则抛出异常。"""
        secret = await self.get(name)
        if not secret:
            raise ValueError(
                f"Required secret '{name}' not found. "
                f"Set it in .env or configure KMS provider."
            )
        return secret

    async def list_names(self) -> list[str]:
        return await self._provider.list_secrets()

    def get_access_log(self) -> list[SecretAccessRecord]:
        return list(self._access_log)

    def is_dev_mode(self) -> bool:
        return isinstance(self._provider, EnvSecretProvider)


# 模块级单例
secrets_manager = SecretsManager()
