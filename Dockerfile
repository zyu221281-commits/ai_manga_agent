# 基于 CUDA 12.1 + Conda 的镜像
# 全云端策略：LLM/图像/视频全部走 API，仅 ONNX Runtime for NudeNet 需要 GPU

FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends `
    wget `
    curl `
    git `
    ffmpeg `
    libgl1-mesa-glx `
    libglib2.0-0 `
    libsm6 `
    libxext6 `
    libxrender-dev `
    libgomp1 `
    ca-certificates `
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh `
    && /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR `
    && rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# 复制 environment.yml 并创建环境
WORKDIR /app
COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy
ENV PATH=$CONDA_DIR/envs/ai_manga_agent/bin:$PATH

# 复制应用代码
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY .env.example .env

# 创建模型缓存目录
RUN mkdir -p /app/models/nudenet

# 入口
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
