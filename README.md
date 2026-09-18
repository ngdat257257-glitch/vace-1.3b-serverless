# HƯỚNG DẪN TRIỂN KHAI VACE 1.3B TRÊN RUNPOD SERVERLESS (PYTHON THUẦN)

## 1. Cấu trúc thư mục
- `handler.py`: Script Python tiếp nhận request từ RunPod Serverless API, thực thi inference VACE 1.3B và trả về kết quả MP4 dạng Base64.
- `Dockerfile`: File đóng gói Docker image nhẹ nhất có thể, chỉ bao gồm Torch + CUDA + VACE.
- `requirements.txt`: Các thư viện Python cần thiết.
- `download_model.sh`: Script tải weights model `Wan2.1-VACE-1.3B` từ Hugging Face.
- `test_payload.json`: File mẫu input để test trên RunPod Console.

---

## 2. Các bước triển khai

### Bước 2.1: Build & Đẩy Docker Image lên Docker Hub
Chạy các lệnh sau trong Terminal máy của bạn (hoặc máy chủ có cài Docker):

```bash
cd /Users/ngdat/web/runpod_workers/vace_1.3b

# Đăng nhập Docker Hub (nếu chưa đăng nhập)
docker login

# Đặt tên image theo tài khoản Docker của bạn (ví dụ: ngdat)
# Bạn có thể build trực tiếp:
docker build -t your_docker_username/vace-1.3b-serverless:latest .

# Đẩy image lên Docker Hub
docker push your_docker_username/vace-1.3b-serverless:latest
```

---

### Bước 2.2: Tạo Serverless Endpoint trên RunPod Console
1. Đăng nhập vào [RunPod Console](https://www.runpod.io/console/serverless).
2. Chọn menu **Serverless** -> Nhấn **New Endpoint**.
3. Điền các thông số:
   - **Endpoint Name**: `vace-1-3b-worker`
   - **Docker Image Name**: `your_docker_username/vace-1.3b-serverless:latest`
   - **Container Disk**: `20 GB`
   - **GPU Selection**: Chọn các dòng GPU 24GB như **RTX 3090 (24GB)** hoặc **RTX 4090 (24GB)** hoặc **L4 (24GB)**.
   - **Active Workers**: `0` (để khi không có ai dùng thì worker tắt hoàn toàn, không tốn tiền).
   - **Max Workers**: `2` (hoặc tùy quy mô người dùng của bạn).
   - **Idle Timeout**: `60` giây (sau 60s không có job thì tự giải phóng GPU).
4. Nhấn **Create**.
5. Copy **Endpoint ID** vừa tạo (ví dụ: `v0a1b2c3d4e5`).

---

### Bước 2.3: Cấu hình vào Web PHP
Thêm vào file `.env` hoặc cập nhật trong `config/runpod.php`:

```env
RUNPOD_ENDPOINT_VACE_1_3B=v0a1b2c3d4e5
```

---

### Bước 2.4: Kiểm tra thử nghiệm (Testing)
Bạn có thể test trực tiếp trên tab **Requests / Test** của Endpoint trên RunPod Console bằng cách dán nội dung từ file `test_payload.json`.
