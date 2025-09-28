# 使用Python 3.10官方镜像作为基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    libsndfile1 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# 升级pip并配置国内镜像源
RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.extra-index-url https://pypi.org/simple && \
    pip config set global.timeout 300 && \
    pip config set global.retries 5


# 复制requirements.txt并安装Python依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p /app/.chroma /app/received_images

# 设置权限
RUN chmod +x /app/websocket_service.py

# 暴露端口
EXPOSE 9000

# 设置环境变量默认值
ENV SERVICE_HOST=0.0.0.0
ENV SERVICE_PORT=9000
ENV SERVICE_SCHEME=http
ENV USE_OPENAI_CLIENT=true
ENV HISTORY_ENABLE=true
ENV LOG_ENABLE=true

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9000/health || exit 1

# 启动命令
CMD ["python", "websocket_service_optimized.py"]
