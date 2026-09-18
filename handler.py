import os
import sys
import time
import uuid
import base64
import shutil
import tempfile
import subprocess
import requests
from pathlib import Path
import runpod

# Thêm đường dẫn VACE vào sys.path
VACE_ROOT = os.getenv("VACE_ROOT", "/workspace/VACE")
if VACE_ROOT not in sys.path and os.path.exists(VACE_ROOT):
    sys.path.append(VACE_ROOT)

MODEL_DIR = os.getenv("MODEL_DIR", "/models/Wan2.1-VACE-1.3B")

def ensure_model_weights():
    global MODEL_DIR
    target = MODEL_DIR or "/models/Wan2.1-VACE-1.3B"
    # Kiểm tra nếu thư mục model chưa có hoặc trống
    if not os.path.exists(target) or not any(Path(target).iterdir()):
        if os.path.exists("/workspace/models/Wan2.1-VACE-1.3B"):
            MODEL_DIR = "/workspace/models/Wan2.1-VACE-1.3B"
            return MODEL_DIR
        print(f">>> [VACE] Đang tự động tải weights model Wan2.1-VACE-1.3B về {target}...")
        os.makedirs(target, exist_ok=True)
        from huggingface_hub import snapshot_download
        try:
            snapshot_download(repo_id="Wan-AI/Wan2.1-VACE-1.3B", local_dir=target, resume_download=True)
        except Exception as ex:
            print(f">>> Thử lại tải từ ali-vilab: {ex}")
            snapshot_download(repo_id="ali-vilab/Wan2.1-VACE-1.3B", local_dir=target, resume_download=True)
        print(">>> [VACE] Tải weights model thành công!")
    MODEL_DIR = target
    return MODEL_DIR

print(f"=== [VACE 1.3B WORKER] Khởi động ===")
print(f"VACE Root: {VACE_ROOT}")
print(f"Model Directory: {MODEL_DIR}")

def save_input_asset(asset_data: str, target_path: str, is_base64: bool = False):
    """Lưu asset từ URL hoặc chuỗi Base64 ra file vật lý"""
    if is_base64 or asset_data.startswith("data:") or len(asset_data) > 1000 and not asset_data.startswith("http"):
        # Xử lý Base64
        if "," in asset_data:
            asset_data = asset_data.split(",", 1)[1]
        raw_bytes = base64.b64decode(asset_data)
        with open(target_path, "wb") as f:
            f.write(raw_bytes)
    elif asset_data.startswith("http://") or asset_data.startswith("https://"):
        # Tải từ URL
        res = requests.get(asset_data, stream=True, timeout=60)
        res.raise_for_status()
        with open(target_path, "wb") as f:
            for chunk in res.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        # Đường dẫn file nội bộ
        if os.path.exists(asset_data):
            shutil.copyfile(asset_data, target_path)
        else:
            raise FileNotFoundError(f"Không tìm thấy file nguồn: {asset_data}")

def handler(job):
    """
    Hàm tiếp nhận request từ RunPod Serverless API
    Payload mẫu từ PHP:
    {
        "input": {
            "character_image": "<url hoặc base64>",
            "driving_video": "<url hoặc base64>",
            "prompt": "a realistic character dancing smoothly, high quality",
            "task": "swap_anything", // hoặc "depth", "frameref"
            "num_frames": 49,
            "fps": 16
        }
    }
    """
    job_input = job.get("input", {})
    if not job_input:
        return {"status": "error", "message": "Payload đầu vào (job.input) rỗng!"}

    char_data = job_input.get("character_image") or job_input.get("image") or job_input.get("image_base64")
    video_data = job_input.get("driving_video") or job_input.get("video") or job_input.get("video_base64")
    prompt = job_input.get("prompt", "a person dancing gracefully, smooth motion, high quality photorealistic")
    task = job_input.get("task", "swap_anything") # swap_anything, depth, frameref
    num_frames = int(job_input.get("num_frames", 49))
    fps = int(job_input.get("fps", 16))

    if not char_data:
        return {"status": "error", "message": "Thiếu ảnh nhân vật (character_image)"}

    # Tạo thư mục tạm cho job
    job_id = job.get("id", uuid.uuid4().hex[:8])
    temp_dir = tempfile.mkdtemp(prefix=f"vace_job_{job_id}_")
    
    input_char_path = os.path.join(temp_dir, "character.png")
    input_video_path = os.path.join(temp_dir, "driving.mp4") if video_data else None
    output_video_path = os.path.join(temp_dir, f"output_{job_id}.mp4")

    start_time = time.time()

    try:
        print(f"[{job_id}] Đang lưu trữ file đầu vào...")
        save_input_asset(char_data, input_char_path)

        if video_data and input_video_path:
            save_input_asset(video_data, input_video_path)

        # Đảm bảo model checkpoint đã sẵn sàng
        ckpt_dir = ensure_model_weights()

        # Ánh xạ task hợp lệ cho vace_pipeline.py
        # Các task được hỗ trợ: depth, depthv2, pose, inpainting, frameref
        valid_tasks = ["depth", "depthv2", "pose", "pose_body", "inpainting", "frameref"]
        if task not in valid_tasks:
            task = "depth" if input_video_path else "frameref"

        print(f"[{job_id}] Đang thực hiện inference VACE 1.3B (Task: {task}, Ckpt: {ckpt_dir})...")

        # Chuẩn bị lệnh gọi pipeline VACE
        pipeline_script = os.path.join(VACE_ROOT, "vace", "vace_pipeline.py")
        if not os.path.exists(pipeline_script):
            pipeline_script = os.path.join(VACE_ROOT, "vace_pipeline.py")

        cmd = [
            sys.executable,
            pipeline_script,
            "--base", "wan",
            "--model_name", "vace-1.3B",
            "--task", task,
            "--prompt", prompt,
            "--output_dir", temp_dir,
            "--num_frames", str(num_frames),
            "--fps", str(fps),
            "--ckpt_dir", ckpt_dir
        ]

        # Gán tham số ảnh và video
        if input_video_path and os.path.exists(input_video_path):
            cmd.extend(["--video", input_video_path])
        
        cmd.extend(["--image", input_char_path])

        print(f"[{job_id}] Thực thi lệnh: {' '.join(cmd)}")
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        print(f"[{job_id}] Output stdout:\n{process.stdout[-500:] if process.stdout else ''}")

        # Tìm file video xuất ra
        generated_files = list(Path(temp_dir).glob("*.mp4"))
        final_video = None
        for f in generated_files:
            if "output" in f.name or f.name.endswith(".mp4") and f.name != "driving.mp4":
                final_video = str(f)
                break

        if not final_video or not os.path.exists(final_video):
            # Fallback nếu tên file xuất khác
            for f in generated_files:
                if f.name != "driving.mp4":
                    final_video = str(f)
                    break

        if not final_video or not os.path.exists(final_video):
            return {
                "status": "error",
                "message": "Quá trình inference hoàn tất nhưng không tìm thấy file video MP4 kết quả!",
                "logs": process.stderr[-1000:] if process.stderr else ""
            }

        file_size = os.path.getsize(final_video)
        print(f"[{job_id}] Video tạo thành công! Dung lượng: {file_size / (1024*1024):.2f} MB")

        # Đọc dữ liệu video ra Base64 trả về cho PHP Web
        with open(final_video, "rb") as vf:
            video_base64 = base64.b64encode(vf.read()).decode("utf-8")

        elapsed_time = round(time.time() - start_time, 2)

        return {
            "status": "success",
            "video_base64": video_base64,
            "format": "mp4",
            "execution_time_seconds": elapsed_time,
            "job_id": job_id
        }

    except subprocess.CalledProcessError as cpe:
        print(f"[{job_id}] LỖI SUBPROCESS: {cpe.stderr}")
        return {
            "status": "error",
            "message": f"VACE Inference thất bại: {cpe.stderr[-1000:] if cpe.stderr else str(cpe)}",
            "stdout": cpe.stdout[-1000:] if cpe.stdout else ""
        }
    except Exception as e:
        print(f"[{job_id}] LỖI NGOẠI LỆ: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }
    finally:
        # Dọn dẹp thư mục tạm
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

if __name__ == "__main__":
    print(">>> Sẵn sàng lắng nghe jobs từ RunPod Serverless API...")
    runpod.serverless.start({"handler": handler})
