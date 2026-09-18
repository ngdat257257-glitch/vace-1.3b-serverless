# Base Image từ RunPod PyTorch CUDA chính thức
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV VACE_ROOT=/workspace/VACE
ENV MODEL_DIR=/models/Wan2.1-VACE-1.3B

WORKDIR /workspace

# 1. Cài đặt các thư viện hệ thống cần cho xử lý video và đồ họa
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Cài đặt dependencies Python
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /workspace/requirements.txt

# 3. Clone mã nguồn VACE chính thức từ Alibaba Tongyi Lab
RUN git clone https://github.com/ali-vilab/VACE.git /workspace/VACE && \
    cd /workspace/VACE && \
    pip install --no-cache-dir -e . 2>/dev/null || true

# 4. Copy handler và script kiểm tra
COPY handler.py /workspace/handler.py

# 5. Khởi động worker RunPod Serverless
CMD ["python", "-u", "/workspace/handler.py"]
