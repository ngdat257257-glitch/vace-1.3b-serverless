#!/usr/bin/env bash
# ==============================================================================
# SCRIPT TẢI MODEL VACE 1.3B VÀO THƯ MỤC /models
# Có thể chạy trên Pod Runpod hoặc lúc Build Docker
# ==============================================================================
set -e

DEST_DIR="${1:-/models/Wan2.1-VACE-1.3B}"
mkdir -p "$DEST_DIR"

echo "📥 Bắt đầu tải model Wan2.1-VACE-1.3B về: $DEST_DIR"

# Dùng huggingface-cli tải toàn bộ checkpoint
pip install -q huggingface_hub

python3 -c "
from huggingface_hub import snapshot_download
print('Đang tải weights từ HuggingFace (ali-vilab/Wan2.1-VACE-1.3B)...')
snapshot_download(
    repo_id='ali-vilab/Wan2.1-VACE-1.3B',
    local_dir='$DEST_DIR',
    local_dir_use_symlinks=False,
    resume_download=True
)
print('✅ Tải model hoàn tất!')
"
