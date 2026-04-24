import os
import shutil
import tempfile
from pathlib import Path

import boto3
import requests
import pipeline_module as pipeline

# ===== ENV =====
API_BASE_URL = os.getenv("API_BASE_URL", "").rstrip("/")
INTERNAL_CALLBACK_TOKEN = os.getenv("INTERNAL_CALLBACK_TOKEN", "")

R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET = os.getenv("R2_BUCKET", "")
R2_PUBLIC_BASE_URL = os.getenv("R2_PUBLIC_BASE_URL", "").rstrip("/")


# ===== CALLBACK =====
def callback(payload: dict):
    if not API_BASE_URL:
        print("WARN: missing API_BASE_URL, skip callback:", payload)
        return

    try:
        requests.post(
            f"{API_BASE_URL}/api/internal/job-callback",
            headers={
                "authorization": f"Bearer {INTERNAL_CALLBACK_TOKEN}",
                "content-type": "application/json",
            },
            json=payload,
            timeout=30,
        )
    except Exception as e:
        print("CALLBACK ERROR:", repr(e))


# ===== R2 STORAGE =====
def get_s3():
    endpoint = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def upload_to_r2(local_path: str, key: str) -> str:
    s3 = get_s3()

    s3.upload_file(
        Filename=local_path,
        Bucket=R2_BUCKET,
        Key=key,
        ExtraArgs={"ContentType": "video/mp4"},
    )

    if R2_PUBLIC_BASE_URL:
        return f"{R2_PUBLIC_BASE_URL}/{key}"

    return key


# ===== MAIN HANDLER =====
def handler(event):
    input_data = event.get("input", {}) or {}
    job_id = str(input_data.get("job_id", "")).strip()

    if not job_id:
        return {"ok": False, "error": "Missing job_id"}

    workdir = Path(tempfile.mkdtemp(prefix=f"{job_id}_"))

    try:
        # STEP 1: nhận job
        callback({
            "job_id": job_id,
            "status": "running",
            "step": "job_received",
            "progress_pct": 3,
            "eta_sec": 120
        })

        # STEP 2: chạy pipeline
        result = pipeline.run_job_serverless(
            job_config=input_data,
            job_id=job_id,
            base_dir=str(workdir),
            progress_callback=callback,
        )

        local_video = result.get("video_path")

        if not local_video or not os.path.exists(local_video):
            raise Exception("Video file not found after pipeline")

        # STEP 3: upload R2
        key = f"videos/{job_id}/output.mp4"
        public_url = upload_to_r2(local_video, key)

        # STEP 4: callback success
        callback({
            "job_id": job_id,
            "status": "completed",
            "step": "done",
            "progress_pct": 100,
            "eta_sec": 0,
            "result_video_key": key,
            "result_video_url": public_url,
        })

        return {
            "ok": True,
            "job_id": job_id,
            "video_key": key,
            "video_url": public_url
        }

    except Exception as e:
        # STEP FAIL
        callback({
            "job_id": job_id,
            "status": "failed",
            "step": "error",
            "progress_pct": 100,
            "error_message": str(e),
        })

        return {
            "ok": False,
            "job_id": job_id,
            "error": str(e)
        }

    finally:
        shutil.rmtree(workdir, ignore_errors=True)