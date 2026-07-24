# AI Manga Agent

> 垂直漫剧超级工厂 + RPA 全自动化生产系统

## 项目简介

AI Manga Agent 是一个基于 LangGraph 的 6-Agent 协作管线，能够将用户的创意简报自动转换为可发布的漫剧视频内容。

### 核心特性

- **创意到视频的全链路自动化**：从创意简报 → 剧本创作 → 分镜设计 → 图像生成 → 视频合成 → 质量审核
- **6-Agent 协作管线**：Creative Director → Planner → Story Critic → Writer → Shot Validator → Composer
- **双质量门禁**：ContentGate（生成前）+ QualityGate（生成后）
- **断点续传**：基于 Checkpoint 的故障恢复机制
- **成本可观测**：完整的 LLM/图像/视频成本追踪与预算控制

## 快速开始

### 环境要求

- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- MinIO（可选，用于媒体资产存储）
- CUDA 12.1（用于 NudeNet NSFW 检测）

### 安装与运行

```bash
# 1. 复制环境变量模板
cp .env.example .env

# 2. 填入 API Key（至少需要一个 LLM API Key）
# 编辑 .env 文件，填入 DEEPSEEK_API_KEY 或 DASHSCOPE_API_KEY

# 3. 使用 Docker Compose 启动基础设施
docker-compose up -d

# 4. 初始化数据库
make migrate

# 5. 启动应用
make run
```

### 手动启动

```bash
# 创建 Conda 环境
conda env create -f environment.yml
conda activate ai_manga_agent

# 启动 FastAPI 服务
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/v1/series/` | POST | 创建系列 |
| `/api/v1/episode/` | POST | 创建单集 |
| `/api/v1/episode/{id}/start` | POST | 启动生产管线 |
| `/api/v1/cost/` | GET | 预算看板 |
| `/api/v1/review/` | GET | 审核队列 |
| `/ws` | WebSocket | 实时进度推送 |

## 架构设计

### Agent 职责

1. **Creative Director**：创意探索与方向选择
2. **Planner**：系列大纲规划（60集）
3. **Story Critic**：大纲吸引力评估
4. **Writer**：剧本 + 分镜 + 提示词生成
5. **Shot Validator**：分镜逻辑质检
6. **Composer**：图像→视频→音频合成

### 质量门禁

- **ContentGate**：CLIP 风格相似度 + 角色一致性检查
- **VQA Checker**：KEY_SCENE 物理异常检测
- **Critic**：多维度质量评分与人工审核路由

## 配置说明

主要配置项（在 `.env` 中设置）：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `APP_ENV` | 运行环境 | development |
| `COST_BUDGET_PER_EPISODE_USD` | 单集预算 | 1.5 |
| `VIDEO_KEY_SCENE_RATIO` | 关键场景比例 | 0.20 |
| `CRITIC_PASS_THRESHOLD` | 质检通过阈值 | 0.8 |
| `AUDIO_PROVIDER` | 音频生成器 | seed_audio |

## 目录结构

```
app/
├── agents/          # Agent 实现
│   ├── creative_director.py
│   ├── planner.py
│   ├── story_critic.py
│   ├── writer.py
│   ├── shot_validator.py
│   └── composer.py
├── api/             # API 端点
├── core/            # 核心配置与工具
├── services/        # 业务服务
├── state/           # LangGraph 状态与图定义
├── quality/         # 质量控制模块
└── resilience/      # 韧性与适配器
```

## 许可证

MIT License
