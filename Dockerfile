# AgentForge Docker 镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 关闭 Python 字节码与缓冲，便于容器日志实时输出
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 安装依赖（先拷贝 requirements 以利用 Docker 层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝项目源码
COPY . .

# 运行时数据目录
RUN mkdir -p /app/data /app/logs

EXPOSE 8000 8501

# 默认启动 FastAPI 服务
CMD ["python", "-m", "uvicorn", "agentforge.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]
