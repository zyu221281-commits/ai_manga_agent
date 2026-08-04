"""Pydantic Settings（含预算 + 视频分级比例 + Critic 阈值配置）

所有配置从 .env 文件读取，支持环境变量覆盖。
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置，从项目根目录 .env 文件加载。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # === 基础设施：PostgreSQL ===
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "root"
    POSTGRES_PASSWORD: str = "123456"  # 仅限开发环境，生产环境必须通过 .env 覆盖
    POSTGRES_DB: str = "mydb"

    # === 基础设施：Redis ===
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # === 基础设施：MinIO ===
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"  # 仅限开发环境，生产环境必须通过 .env 覆盖
    MINIO_SECURE: bool = False
    MINIO_UPLOAD_ENABLED: bool = True  # 媒体资产 fire-and-forget 上传开关（失败不阻塞管线）

    # === 预算配置（USD）===
    COST_BUDGET_PER_EPISODE_USD: float = 1.5
    COST_DAILY_CAP_USD: float = 15.0
    COST_MONTHLY_CAP_USD: float = 300.0
    COST_ALERT_THRESHOLD: float = 0.8
    COST_HARD_STOP_THRESHOLD: float = 1.0

    # === 视频分级策略 ===
    VIDEO_KEY_SCENE_RATIO: float = 0.20
    VIDEO_KEY_SCENE_MODEL: str = "volcengine-video"
    VIDEO_AB_TEST_ENABLED: bool = True

    # === Critic 质量门 ===
    CRITIC_PASS_THRESHOLD: float = 0.8
    CRITIC_REVIEW_THRESHOLD: float = 0.6
    CRITIC_MAX_RETRY: int = 2
    CRITIC_HUMAN_REVIEW_TIMEOUT_H: int = 4

    # === 应用配置 ===
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    ALERT_WEBHOOK_URL: str = ""

    # === 系列集数配置 ===
    # 默认总集数（用户未输入时使用）。原先硬编码为 60，现改为可配置，默认 30。
    # 上限保护：避免用户误输入超大值导致 LLM token 爆炸。
    DEFAULT_TOTAL_EPISODES: int = 30
    MAX_TOTAL_EPISODES: int = 200

    # === 安全配置 ===
    # CORS 允许的源列表，逗号分隔（如 "http://localhost:3000,https://app.example.com"）
    # 为空时：development 允许所有源（不携带凭证），production 拒绝所有跨域请求
    CORS_ORIGINS: str = ""
    # BGM 文件目录（相对项目根目录）
    BGM_DIR: str = "assets/bgm"
    # SFX 音效库目录（相对项目根目录，供 audio_timeline 的 sound_effect 引用）
    SFX_DIR: str = "assets/sfx"

    # === 媒体生成单价（USD，可通过 .env 覆盖）===
    # 集中管理所有媒体生成的单价，避免分散硬编码
    IMAGE_COST_PER_UNIT: float = 0.02       # 每张图片（Seedream 5.0 Pro）
    VIDEO_COST_PER_SECOND: float = 0.30     # 视频每秒（Seedance）
    VIDEO_POLL_MAX_WAIT_S: int = 600          # Seedance 任务轮询最大等待秒数
    TTS_COST_PER_CHAR: float = 0.000016     # TTS 每字符（seed-tts-2.0）

    # === 风格封印（锁定创作参数，防止 Agent 随机创新）===
    GLOBAL_SEED: int = 42                   # 全局随机种子（0 = 随机）
    QUALITY_TAGS: str = "anime comic style, high quality, detailed, vibrant colors, cinematic lighting"
    NEGATIVE_PROMPT: str = "low quality, deformed, bad anatomy, extra fingers, extra limbs, blurry, watermark, text"
    STYLE_SIMILARITY_THRESHOLD: float = 0.85  # CLIP 风格相似度阈值

    # === 阶段4: 内容质检（生成后拦截）===
    # ContentGate: CLIP 风格相似度 + 角色一致性检查（图像→视频之间）
    CONTENT_GATE_ENABLED: bool = True
    # VQA: 轻量视觉问答，仅检查 KEY_SCENE 物理异常
    VQA_ENABLED: bool = True
    VQA_MODEL: str = "qwen-vl-max"           # 多模态 LLM 模型
    VQA_MAX_IMAGES: int = 5                  # 单集最多检查的 KEY_SCENE 数量（成本控制）

    # === 视频内容质检 ===
    VIDEO_CONTENT_CHECK_ENABLED: bool = True
    VIDEO_CONTENT_CHECK_CLIP_THRESHOLD: float = 0.72  # 视频帧与源图 CLIP 相似度阈值
    VIDEO_CONTENT_CHECK_VLM_ENABLED: bool = True       # KEY_SCENE VLM 人物完整性检查

    # === API Keys（从 .env 读取，Demo 阶段可为空）===
    DEEPSEEK_API_KEY: str = ""
    DASHSCOPE_API_KEY: str = ""
    SILICONFLOW_API_KEY: str = ""
    ARK_API_KEY: str = ""

    AZURE_SPEECH_KEY: str = ""
    AZURE_SPEECH_REGION: str = "eastasia"
    VOLCENGINE_TTS_KEY: str = ""
    FANQIE_API_KEY: str = ""
    QUARK_API_KEY: str = ""
    DOUYIN_OPEN_KEY: str = ""


    # === 模型标识符（集中管理，避免分散硬编码）===
    ARK_LLM_MODEL: str = "deepseek-v4-pro"
    ARK_IMAGE_MODEL: str = "doubao-seedream-5-0-pro-260628"
    ARK_VIDEO_MODEL: str = "doubao-seedance-1-0-pro-fast-251015"
    ARK_TTS_MODEL: str = "seed-tts-2.0"          # legacy TTS（保留作为 fallback）
    ARK_AUDIO_MODEL: str = "seed-audio-1.0"      # Seed Audio 1.0（整合 TTS + BGM）
    VIDEO_DURATION_S: int = 5

    # === TTS / Seed Audio 配置 ===
    TTS_SPEAKER: str = "zh_female_gaolengyujie_uranus_bigtts"
    # Seed Audio 1.0 API（同步 HTTP，返回 base64 音频 + 字级字幕）
    SEED_AUDIO_API_URL: str = "https://openspeech.bytedance.com/api/v3/tts/create"
    SEED_AUDIO_COST_PER_SECOND: float = 0.02      # Seed Audio 按秒计费（估算值）
    # 音频生成 provider：seed_audio（默认，整合 TTS+BGM）| tts（legacy seed-tts-2.0）
    AUDIO_PROVIDER: str = "seed_audio"

    # === 派生连接串 ===
    @property
    def database_url(self) -> str:
        """SQLAlchemy 用的 PG 连接串（psycopg3 dialect）。"""
        return (
            f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def redis_url(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    @property
    def cost_hard_stop_enabled(self) -> bool:
        """Demo 默认不强制熔断，仅告警。生产可设 APP_ENV=production 开启。"""
        return self.APP_ENV == "production"

    @property
    def cors_origins_list(self) -> list[str]:
        """解析 CORS_ORIGINS 为列表，空字符串返回空列表。"""
        if not self.CORS_ORIGINS:
            return []
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def bgm_dir_abs(self) -> str:
        """BGM 目录的绝对路径。"""
        from pathlib import Path
        p = Path(self.BGM_DIR)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent.parent.parent / self.BGM_DIR
        return str(p)

    @property
    def sfx_dir_abs(self) -> str:
        """SFX 音效库目录的绝对路径。"""
        from pathlib import Path
        p = Path(self.SFX_DIR)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent.parent.parent / self.SFX_DIR
        return str(p)

    def validate_production_security(self) -> list[str]:
        """生产环境安全检查，返回警告消息列表（空列表表示通过）。"""
        warnings: list[str] = []
        if self.APP_ENV != "production":
            return warnings

        insecure_defaults = [
            ("POSTGRES_PASSWORD", self.POSTGRES_PASSWORD, ["123456", "password", ""]),
            ("MINIO_SECRET_KEY", self.MINIO_SECRET_KEY, ["minioadmin", "password", ""]),
        ]
        for name, value, bad_values in insecure_defaults:
            if value in bad_values:
                warnings.append(
                    f"安全警告: {name} 在生产环境中使用了不安全的默认值，请通过环境变量或 .env 设置强密码"
                )

        if not self.CORS_ORIGINS:
            warnings.append(
                "安全警告: CORS_ORIGINS 未配置，生产环境将拒绝所有跨域请求"
            )

        return warnings


@lru_cache
def get_settings() -> Settings:
    """单例 Settings，避免重复读取 .env。"""
    return Settings()


# 模块级单例，供直接 import 使用
settings = get_settings()
