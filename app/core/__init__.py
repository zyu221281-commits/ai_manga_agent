"""核心配置包。

模块导入时设置 Windows EventLoop 策略，避免 psycopg async 与 ProactorEventLoop
冲突（影响所有入口：Celery worker、API、测试脚本等）。

必须在任何 asyncio 代码执行前完成设置，因此放在包 __init__ 顶部。
"""

import sys

if sys.platform == "win32":
    import asyncio

    # Windows 默认 ProactorEventLoop 不支持 psycopg async 的 fd 监听
    # 必须切到 SelectorEventLoop 才能使用 AsyncPostgresSaver / psycopg_pool
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
