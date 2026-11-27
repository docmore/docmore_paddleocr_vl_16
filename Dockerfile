# syntax=docker/dockerfile:1
# 构建上下文 = 本项目目录本身：
#   docker build -t docmore-paddleocr-vl-16:0.1.0 .
#
# docmore-utility 走 PyPI（>=0.2.0 已发布），无需工作区根目录作上下文。
# 运行时需能访问 PaddleOCR-VL 1.6 端点；PP-DocLayoutV3 模型（~124 MB）不随镜像
# 打包，通过 volume 挂载并用 LAYOUT_MODEL_DIR 指过去：
#
#   docker run -d -p 18910:18910 \
#       -e OPENAI_API_BASE=http://<paddleocr-host>:8113/v1 \
#       -v /path/to/PP-DocLayoutV3-onnx:/models/PP-DocLayoutV3-onnx:ro \
#       -e LAYOUT_MODEL_DIR=/models/PP-DocLayoutV3-onnx \
#       docmore-paddleocr-vl-16:0.1.0

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# OpenCV (opencv-python-headless) 需要 libGL/libglib 的系统库；headless 已
# 自带大部分依赖，slim 镜像仍缺 libglib2.0-0。
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
        ".[server,detect,docling,layout]"

RUN useradd --create-home appuser \
 && chown -R appuser:appuser /app
USER appuser

# OpenAI 兼容服务，绑定 0.0.0.0:18910（可用
# PADDLEOCR_VL_16_SERVER_* 环境变量覆盖，
# 见 docmore.paddleocr_vl_16.server.config.ServeSettings）
EXPOSE 18910

CMD ["docmore-paddleocr-vl-16-serve"]