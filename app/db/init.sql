-- AI Manga Agent V4 — 数据库初始化脚本
-- 由 docker-compose 的 pg-init 服务执行（postgres healthy 后启动）
-- 幂等设计：IF NOT EXISTS，可重复执行

-- ================================================================
-- cost_ledger: 成本流水表（每条 LLM/图像/视频/TTS 调用一行）
-- ================================================================
CREATE TABLE IF NOT EXISTS cost_ledger (
    id              VARCHAR(36) PRIMARY KEY,
    episode_id      VARCHAR(128),
    series_id       VARCHAR(128),
    model           VARCHAR(128) NOT NULL,
    operation       VARCHAR(64)  NOT NULL,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    unit_count      DOUBLE PRECISION,
    cost_usd        DOUBLE PRECISION NOT NULL,
    trace_id        VARCHAR(64),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_cost_ledger_episode    ON cost_ledger(episode_id);
CREATE INDEX IF NOT EXISTS idx_cost_ledger_series     ON cost_ledger(series_id);
CREATE INDEX IF NOT EXISTS idx_cost_ledger_trace      ON cost_ledger(trace_id);
CREATE INDEX IF NOT EXISTS idx_cost_ledger_created_at ON cost_ledger(created_at);

-- ================================================================
-- data_lineage: 数据血缘追踪表
-- ================================================================
CREATE TABLE IF NOT EXISTS data_lineage (
    id                  VARCHAR(36) PRIMARY KEY,
    episode_id          VARCHAR(128) NOT NULL,
    artifact_type       VARCHAR(64)  NOT NULL,
    artifact_data       JSONB NOT NULL,
    prompt_template_id  VARCHAR(128),
    model_name          VARCHAR(128),
    model_params        JSONB,
    seed                INTEGER,
    trace_id            VARCHAR(64),
    cost_record_id      VARCHAR(36),
    parent_lineage_ids  JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_data_lineage_episode   ON data_lineage(episode_id);
CREATE INDEX IF NOT EXISTS idx_data_lineage_prompt    ON data_lineage(prompt_template_id);
CREATE INDEX IF NOT EXISTS idx_data_lineage_trace     ON data_lineage(trace_id);
CREATE INDEX IF NOT EXISTS idx_data_lineage_artifact  ON data_lineage(artifact_type);

-- ================================================================
-- pending_reviews: 人工审核任务表（Critic 临界区 / DLQ 推送）
-- ================================================================
CREATE TABLE IF NOT EXISTS pending_reviews (
    id              VARCHAR(36) PRIMARY KEY,
    episode_id      VARCHAR(128) NOT NULL,
    source          VARCHAR(32)  NOT NULL,  -- critic / dlq
    critic_score    DOUBLE PRECISION,
    reason          VARCHAR(1000) NOT NULL DEFAULT '',
    status          VARCHAR(16)  NOT NULL DEFAULT 'pending',  -- pending / decided
    decision        VARCHAR(16),  -- approve / reject / edit
    decided_by      VARCHAR(64),
    decided_at      TIMESTAMPTZ,
    sla_deadline    TIMESTAMPTZ NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_pending_reviews_episode ON pending_reviews(episode_id);
CREATE INDEX IF NOT EXISTS idx_pending_reviews_status  ON pending_reviews(status);

-- ================================================================
-- character_anchors: persistent character anchors for cross-episode consistency
-- 由 SQLAlchemy ORM 自动 create_all；此处的 ALTER 用于已有部署升级
-- ================================================================
CREATE TABLE IF NOT EXISTS character_anchors (
    id                  SERIAL PRIMARY KEY,
    name                VARCHAR(200) UNIQUE NOT NULL,
    role                VARCHAR(100) DEFAULT '',
    seed_prompt         TEXT DEFAULT '',
    seed_image_url      TEXT DEFAULT '',
    seed_image_path     TEXT DEFAULT '',
    seed                INTEGER DEFAULT 42,
    traits_json         TEXT DEFAULT '[]',
    generation_count    INTEGER DEFAULT 0,
    -- 多视角 anchor（front / side / back）
    front_image_path    TEXT DEFAULT '',
    side_image_path     TEXT DEFAULT '',
    back_image_path     TEXT DEFAULT '',
    front_image_url     TEXT DEFAULT '',
    side_image_url      TEXT DEFAULT '',
    back_image_url      TEXT DEFAULT ''
);

-- 已有部署的兼容性迁移（IF NOT EXISTS 等价：DO 块检查列存在性）
DO $$
BEGIN
    -- multi-view anchor fields
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='character_anchors' AND column_name='front_image_path') THEN
        ALTER TABLE character_anchors ADD COLUMN front_image_path TEXT DEFAULT '';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='character_anchors' AND column_name='side_image_path') THEN
        ALTER TABLE character_anchors ADD COLUMN side_image_path TEXT DEFAULT '';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='character_anchors' AND column_name='back_image_path') THEN
        ALTER TABLE character_anchors ADD COLUMN back_image_path TEXT DEFAULT '';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='character_anchors' AND column_name='front_image_url') THEN
        ALTER TABLE character_anchors ADD COLUMN front_image_url TEXT DEFAULT '';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='character_anchors' AND column_name='side_image_url') THEN
        ALTER TABLE character_anchors ADD COLUMN side_image_url TEXT DEFAULT '';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='character_anchors' AND column_name='back_image_url') THEN
        ALTER TABLE character_anchors ADD COLUMN back_image_url TEXT DEFAULT '';
    END IF;
END $$;

-- ================================================================
-- 验证
-- ================================================================
DO $$
BEGIN
    RAISE NOTICE 'AI Manga Agent V4 schema initialized: cost_ledger, data_lineage, pending_reviews, character_anchors (multi-view)';
END $$;
