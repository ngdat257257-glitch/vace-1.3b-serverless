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

# 1. Cấu hình đường dẫn
VACE_ROOT = os.getenv("VACE_ROOT", "/workspace/VACE")
WAN_ROOT = os.getenv("WAN_ROOT", "/workspace/Wan2.1")
MODEL_DIR = os.getenv("MODEL_DIR", "/models/Wan2.1-VACE-1.3B")

def setup_torch_xpu_compat():
    """Tạo universal mock torch.xpu để tương thích hoàn toàn diffusers với PyTorch 2.2"""
    sitecustomize_path = "/workspace/sitecustomize.py"
    hook_code = """import sys

class _DummyMethod:
    def __call__(self, *args, **kwargs):
        return None
    def __bool__(self):
        return False
    def __int__(self):
        return 0
    def __getattr__(self, name):
        return self

class _MockXPU:
    def __getattr__(self, name):
        if name == "is_available":
            return lambda: False
        if name in ("device_count", "current_device"):
            return lambda: 0
        return _DummyMethod()

try:
    import torch
    if not hasattr(torch, "xpu"):
        torch.xpu = _MockXPU()
except Exception:
    pass
"""
    try:
        with open(sitecustomize_path, "w") as f:
            f.write(hook_code)
        print(">>> [VACE] Đã tạo /workspace/sitecustomize.py với universal _MockXPU")
    except Exception as e:
        print(f">>> [VACE WARNING] Không thể ghi sitecustomize.py: {e}")

    try:
        import torch
        if not hasattr(torch, "xpu"):
            class _DummyMethod:
                def __call__(self, *args, **kwargs):
                    return None
                def __bool__(self):
                    return False
                def __int__(self):
                    return 0
                def __getattr__(self, name):
                    return self

            class _MockXPU:
                def __getattr__(self, name):
                    if name == "is_available":
                        return lambda: False
                    if name in ("device_count", "current_device"):
                        return lambda: 0
                    return _DummyMethod()

            torch.xpu = _MockXPU()
            print(">>> [VACE] Đã patch torch.xpu universal trong tiến trình hiện tại")
    except Exception:
        pass

def ensure_wan_package():
    """Đảm bảo thư viện wan của Wan2.1 sẵn sàng trong môi trường Python"""
    global WAN_ROOT
    setup_torch_xpu_compat()
    if not os.path.exists(WAN_ROOT) or not os.path.exists(os.path.join(WAN_ROOT, "wan")):
        print(f">>> [VACE] Đang clone Wan2.1 về {WAN_ROOT}...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/Wan-Video/Wan2.1.git", WAN_ROOT],
                check=True,
                capture_output=True,
                text=True
            )
            print(">>> [VACE] Clone Wan2.1 thành công!")
        except Exception as e:
            print(f">>> [VACE WARNING] Lỗi khi clone Wan2.1: {e}")

    # Cập nhật sys.path
    for p in ["/workspace", WAN_ROOT, VACE_ROOT]:
        if p not in sys.path and os.path.exists(p):
            sys.path.insert(0, p)

    # Cập nhật biến môi trường PYTHONPATH
    current_pp = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"/workspace:{VACE_ROOT}:{WAN_ROOT}:{current_pp}"

def ensure_model_weights():
    """Đảm bảo weights Wan2.1-VACE-1.3B tồn tại"""
    global MODEL_DIR
    target = MODEL_DIR or "/models/Wan2.1-VACE-1.3B"

    # Kiểm tra xem có sẵn ở /models hay /workspace/models
    if not os.path.exists(target) or not any(Path(target).iterdir()):
        if os.path.exists("/workspace/models/Wan2.1-VACE-1.3B") and any(Path("/workspace/models/Wan2.1-VACE-1.3B").iterdir()):
            MODEL_DIR = "/workspace/models/Wan2.1-VACE-1.3B"
            target = MODEL_DIR
        else:
            print(f">>> [VACE] Đang tải weights model Wan2.1-VACE-1.3B về {target}...")
            os.makedirs(target, exist_ok=True)
            from huggingface_hub import snapshot_download
            try:
                snapshot_download(repo_id="Wan-AI/Wan2.1-VACE-1.3B", local_dir=target, resume_download=True)
            except Exception as ex:
                print(f">>> Thử lại tải từ ali-vilab: {ex}")
                snapshot_download(repo_id="ali-vilab/Wan2.1-VACE-1.3B", local_dir=target, resume_download=True)
            print(">>> [VACE] Tải weights model thành công!")

    MODEL_DIR = target

    # Symlink vào VACE_ROOT/models/Wan2.1-VACE-1.3B để hỗ trợ đường dẫn tương đối
    try:
        vace_models_dir = os.path.join(VACE_ROOT, "models")
        os.makedirs(vace_models_dir, exist_ok=True)
        vace_target = os.path.join(vace_models_dir, "Wan2.1-VACE-1.3B")
        if not os.path.exists(vace_target):
            os.symlink(target, vace_target)
    except Exception as e:
        print(f">>> [VACE] Bỏ qua symlink models: {e}")

    return MODEL_DIR

def download_file_if_missing(url: str, local_path: str):
    """Tải file từ URL nếu chưa có hoặc dung lượng không hợp lệ"""
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
        return
    print(f">>> Đang tải {os.path.basename(local_path)}...")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    res = requests.get(url, stream=True, timeout=180)
    res.raise_for_status()
    tmp_path = local_path + ".tmp"
    with open(tmp_path, "wb") as f:
        for chunk in res.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
    os.rename(tmp_path, local_path)
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f">>> Tải thành công {os.path.basename(local_path)} ({size_mb:.1f} MB)")

def ensure_annotators(task: str):
    """Tải model annotator tương ứng cho task (DWPose / Depth)"""
    annotators_dir = os.path.join(VACE_ROOT, "models", "VACE-Annotators")
    os.makedirs(annotators_dir, exist_ok=True)

    if task in ["pose", "pose_body"]:
        pose_dir = os.path.join(annotators_dir, "pose")
        download_file_if_missing(
            "https://huggingface.co/ali-vilab/VACE-Annotators/resolve/main/pose/dw-ll_ucoco_384.onnx",
            os.path.join(pose_dir, "dw-ll_ucoco_384.onnx")
        )
        download_file_if_missing(
            "https://huggingface.co/ali-vilab/VACE-Annotators/resolve/main/pose/yolox_l.onnx",
            os.path.join(pose_dir, "yolox_l.onnx")
        )
    elif task in ["depth", "depthv2"]:
        depth_dir = os.path.join(annotators_dir, "depth")
        download_file_if_missing(
            "https://huggingface.co/ali-vilab/VACE-Annotators/resolve/main/depth/dpt_hybrid-midas-501f0c75.pt",
            os.path.join(depth_dir, "dpt_hybrid-midas-501f0c75.pt")
        )

def save_input_asset(asset_data: str, target_path: str):
    """Lưu asset từ URL, Base64 hoặc file cục bộ"""
    if not asset_data:
        return
    if asset_data.startswith("data:") or (len(asset_data) > 1000 and not asset_data.startswith("http")):
        if "," in asset_data:
            asset_data = asset_data.split(",", 1)[1]
        raw_bytes = base64.b64decode(asset_data)
        with open(target_path, "wb") as f:
            f.write(raw_bytes)
    elif asset_data.startswith("http://") or asset_data.startswith("https://"):
        res = requests.get(asset_data, stream=True, timeout=60)
        res.raise_for_status()
        with open(target_path, "wb") as f:
            for chunk in res.iter_content(chunk_size=8192):
                f.write(chunk)
    elif os.path.exists(asset_data):
        shutil.copyfile(asset_data, target_path)
    else:
        raise FileNotFoundError(f"Không tìm thấy file nguồn: {asset_data}")

def handler(job):
    """
    RunPod Serverless Handler cho VACE 1.3B
    """
    job_input = job.get("input", {})
    if not job_input:
        return {"status": "error", "message": "Payload đầu vào (job.input) rỗng!"}

    char_data = job_input.get("character_image") or job_input.get("image") or job_input.get("image_base64")
    video_data = job_input.get("driving_video") or job_input.get("video") or job_input.get("video_base64")
    prompt = job_input.get("prompt", "a person dancing gracefully, smooth movements, synchronized motion, photorealistic, cinematic lighting")
    task = job_input.get("task", "pose")
    raw_frames = int(job_input.get("num_frames", 49))
    fps = int(job_input.get("fps", 16))

    if not char_data:
        return {"status": "error", "message": "Thiếu dữ liệu ảnh nhân vật (character_image)"}

    # Wan2.1 yêu cầu frame_num theo công thức 4n + 1 (17, 33, 49, 65, 81...)
    safe_frames = max(17, min(81, raw_frames))
    num_frames = ((safe_frames - 1) // 4) * 4 + 1

    job_id = job.get("id", uuid.uuid4().hex[:8])
    temp_dir = tempfile.mkdtemp(prefix=f"vace_{job_id}_")

    input_char_path = os.path.join(temp_dir, "character.png")
    input_video_path = os.path.join(temp_dir, "driving.mp4") if video_data else None

    start_time = time.time()
    try:
        ensure_wan_package()
        ckpt_dir = ensure_model_weights()

        print(f"[{job_id}] Đang lưu trữ file đầu vào...")
        save_input_asset(char_data, input_char_path)
        if video_data and input_video_path:
            save_input_asset(video_data, input_video_path)

        # Chọn task phù hợp
        if input_video_path and os.path.exists(input_video_path):
            if task in ["swap_anything", "dance", "dancing", "animate_anything"]:
                task = "pose"
            elif task not in ["pose", "depth", "flow", "scribble"]:
                task = "pose"
        else:
            task = "frameref"

        print(f"[{job_id}] Kiểm tra annotators cho task '{task}'...")
        ensure_annotators(task)

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
            "--save_dir", temp_dir,
            "--frame_num", str(num_frames),
            "--save_fps", str(fps),
            "--ckpt_dir", ckpt_dir,
            "--offload_model", "True"
        ]

        if task == "frameref":
            cmd.extend(["--mode", "firstframe", "--image", input_char_path])
        else:
            if input_video_path and os.path.exists(input_video_path):
                cmd.extend(["--video", input_video_path])
            cmd.extend([
                "--image", input_char_path,
                "--src_ref_images", input_char_path
            ])

        print(f"[{job_id}] Thực thi lệnh: {' '.join(cmd)}")
        setup_torch_xpu_compat()
        sub_env = os.environ.copy()
        sub_env["PYTHONPATH"] = f"/workspace:{VACE_ROOT}:{WAN_ROOT}:{sub_env.get('PYTHONPATH', '')}"

        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=VACE_ROOT,
            env=sub_env,
            check=True
        )

        print(f"[{job_id}] Stdout cuối:\n{process.stdout[-600:] if process.stdout else ''}")

        # Tìm video kết quả: out_video.mp4
        final_video = None
        candidates = [
            os.path.join(temp_dir, "out_video.mp4"),
            os.path.join(temp_dir, f"output_{job_id}.mp4"),
        ]
        for c in candidates:
            if os.path.exists(c) and os.path.getsize(c) > 1000:
                final_video = c
                break

        if not final_video:
            for f in Path(temp_dir).rglob("*.mp4"):
                if f.name not in ["driving.mp4", "src_video.mp4", "src_mask.mp4"] and f.stat().st_size > 1000:
                    final_video = str(f)
                    break

        if not final_video or not os.path.exists(final_video):
            return {
                "status": "error",
                "message": "Quá trình hoàn tất nhưng không tìm thấy file video đầu ra out_video.mp4!",
                "stdout": process.stdout[-1500:] if process.stdout else "",
                "stderr": process.stderr[-1500:] if process.stderr else ""
            }

        file_size_mb = os.path.getsize(final_video) / (1024 * 1024)
        print(f"[{job_id}] Video kết quả: {final_video} ({file_size_mb:.2f} MB)")

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
            "message": f"VACE Inference thất bại: {cpe.stderr[-1500:] if cpe.stderr else str(cpe)}",
            "stdout": cpe.stdout[-1000:] if cpe.stdout else ""
        }
    except Exception as e:
        print(f"[{job_id}] LỖI NGOẠI LỆ: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    print("=== [VACE 1.3B WORKER] Khởi động ===")
    ensure_wan_package()
    ensure_model_weights()
    print(">>> Sẵn sàng lắng nghe jobs từ RunPod Serverless API...")
    runpod.serverless.start({"handler": handler})
