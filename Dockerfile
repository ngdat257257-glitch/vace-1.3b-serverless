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

# 2. Cài đặt dependencies Python (Bao gồm easydict, ftfy, timm, onnxruntime, decord...)
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /workspace/requirements.txt

# 3. Clone mã nguồn VACE và cài đặt Wan2.1
RUN git clone https://github.com/ali-vilab/VACE.git /workspace/VACE && \
    cd /workspace/VACE && \
    pip install --no-cache-dir -e . 2>/dev/null || true && \
    pip install --no-cache-dir git+https://github.com/Wan-Video/Wan2.1.git 2>/dev/null || true

# 4. Tải trước weights model Wan2.1-VACE-1.3B (~3.5GB) để khởi động worker tức thì
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-VACE-1.3B', local_dir='/models/Wan2.1-VACE-1.3B', resume_download=True)" || \
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ali-vilab/Wan2.1-VACE-1.3B', local_dir='/models/Wan2.1-VACE-1.3B', resume_download=True)" || true

# 5. Copy handler
COPY handler.py /workspace/handler.py

# 6. Khởi động worker RunPod Serverless
CMD ["python", "-u", "/workspace/handler.py"]
