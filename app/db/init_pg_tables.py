"""LangGraph AsyncPostgresSaver 表结构初始化（幂等）。

为什么需要这个模块：
    `AsyncPostgresSaver.setup()` 内部使用 `CREATE INDEX CONCURRENTLY`，
    该语句不能在事务中执行，会抛出 `ActiveSqlTransaction` 错误。
    本模块用同步 psycopg + autocommit 手动创建表与索引，
    绕过事务冲突，且全部 SQL 都是 IF NOT EXISTS，可重复执行。

使用方式：
    # 生产入口（Celery worker / API 启动时）
    from app.db.init_pg_tables import setup_langgraph_tables
    setup_langgraph_tables()

    # 自定义连接串
    setup_langgraph_tables(conninfo="host=... user=... password=... dbname=...")
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# LangGraph PostgresSaver 表结构（与 langgraph.checkpoint.postgres.base.MIGRATIONS 一致）
# 全部使用 IF NOT EXISTS，幂等可重复执行
# 注意：checkpoint 列是 JSONB 不是 BYTEA（旧版 schema 是 BYTEA，新版改了）
_LANGGRAPH_TABLE_SQL = [
    # 版本迁移记录表
    "CREATE TABLE IF NOT EXISTS checkpoint_migrations (v INTEGER PRIMARY KEY)",
    # 主 checkpoint 表（checkpoint 列是 JSONB NOT NULL，不是 BYTEA）
    """
    CREATE TABLE IF NOT EXISTS checkpoints (
        thread_id TEXT NOT NULL,
        checkpoint_ns TEXT NOT NULL DEFAULT '',
        checkpoint_id TEXT NOT NULL,
        parent_checkpoint_id TEXT,
        type TEXT,
        checkpoint JSONB NOT NULL,
        metadata JSONB NOT NULL DEFAULT '{}',
        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
    )
    """,
    # channel blobs（blob 可空，type 非空）
    """
    CREATE TABLE IF NOT EXISTS checkpoint_blobs (
        thread_id TEXT NOT NULL,
        checkpoint_ns TEXT NOT NULL DEFAULT '',
        channel TEXT NOT NULL,
        version TEXT NOT NULL,
        type TEXT NOT NULL,
        blob BYTEA,
        PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
    )
    """,
    # 写入队列（blob 非空）
    """
    CREATE TABLE IF NOT EXISTS checkpoint_writes (
        thread_id TEXT NOT NULL,
        checkpoint_ns TEXT NOT NULL DEFAULT '',
        checkpoint_id TEXT NOT NULL,
        task_id TEXT NOT NULL,
        idx INTEGER NOT NULL,
        channel TEXT NOT NULL,
        type TEXT,
        blob BYTEA NOT NULL,
        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
    )
    """,
    # 兼容性 ALTER（幂等）
    "ALTER TABLE checkpoint_blobs ALTER COLUMN blob DROP NOT NULL",
    "ALTER TABLE checkpoint_writes ADD COLUMN IF NOT EXISTS task_path TEXT NOT NULL DEFAULT ''",
    # 普通索引（非 CONCURRENTLY，可在事务内创建，绕过 setup() 的事务冲突）
    "CREATE INDEX IF NOT EXISTS checkpoints_thread_id_idx ON checkpoints (thread_id)",
    "CREATE INDEX IF NOT EXISTS checkpoint_blobs_thread_id_idx ON checkpoint_blobs (thread_id)",
    "CREATE INDEX IF NOT EXISTS checkpoint_writes_thread_id_idx ON checkpoint_writes (thread_id)",
]

# DROP 语句（仅供测试/重置用，生产环境不要调用）
_DROP_TABLE_SQL = [
    "DROP TABLE IF EXISTS checkpoint_writes CASCADE",
    "DROP TABLE IF EXISTS checkpoint_blobs CASCADE",
    "DROP TABLE IF EXISTS checkpoints CASCADE",
    "DROP TABLE IF EXISTS checkpoint_migrations CASCADE",
]


def _build_conninfo() -> str:
    """从 settings 构建 psycopg conninfo 字符串。"""
    from app.core.config import settings

    return (
        f"host={settings.POSTGRES_HOST} "
        f"port={settings.POSTGRES_PORT} "
        f"user={settings.POSTGRES_USER} "
        f"password={settings.POSTGRES_PASSWORD} "
        f"dbname={settings.POSTGRES_DB}"
    )


def setup_langgraph_tables(conninfo: Optional[str] = None) -> None:
    """幂等创建 LangGraph AsyncPostgresSaver 所需的表与索引。

    Args:
        conninfo: psycopg 连接串。为 None 时从 app.core.config.settings 读取。
                  格式：`host=... port=... user=... password=... dbname=...`

    Raises:
        ImportError: 未安装 psycopg
        psycopg.OperationalError: PG 连接失败
    """
    if conninfo is None:
        conninfo = _build_conninfo()

    import psycopg

    # autocommit=True 让每条 SQL 独立提交，避免事务包裹
    # （CREATE INDEX CONCURRENTLY 在事务中会失败，虽然这里用的是普通 INDEX，
    #   但 autocommit 模式更安全，也便于将来切换到 CONCURRENTLY 优化大表迁移）
    conn = psycopg.connect(conninfo, autocommit=True)
    try:
        with conn.cursor() as cur:
            for sql in _LANGGRAPH_TABLE_SQL:
                cur.execute(sql)
        logger.info("LangGraph checkpoint tables ensured (idempotent)")
    finally:
        conn.close()


def drop_langgraph_tables(conninfo: Optional[str] = None) -> None:
    """DROP 所有 LangGraph checkpoint 表（仅供测试/重置用）。

    警告：会删除所有 checkpoint 数据。生产环境不要调用。

    使用场景：
    - 测试环境切换 langgraph 版本后，旧 schema 不兼容时重置
    - 开发环境清空 checkpoint 重新开始
    """
    if conninfo is None:
        conninfo = _build_conninfo()

    import psycopg

    conn = psycopg.connect(conninfo, autocommit=True)
    try:
        with conn.cursor() as cur:
            for sql in _DROP_TABLE_SQL:
                cur.execute(sql)
        logger.warning("LangGraph checkpoint tables DROPPED (data lost)")
    finally:
        conn.close()


if __name__ == "__main__":
    # 命令行直接执行：python -m app.db.init_pg_tables
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    setup_langgraph_tables()
    print("OK: LangGraph checkpoint tables ready")
