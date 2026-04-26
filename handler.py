import runpod
import os
import shutil
import tempfile
from pathlib import Path

import boto3
import requests
import pipeline_module as pipeline


def get_env():
    return {
        "API_BASE_URL": os.getenv("API_BASE_URL", "").strip().rstrip("/"),
        "INTERNAL_CALLBACK_TOKEN": os.getenv("INTERNAL_CALLBACK_TOKEN", "").strip(),

        "R2_ACCOUNT_ID": os.getenv("R2_ACCOUNT_ID", "").strip(),
        "R2_ACCESS_KEY_ID": os.getenv("R2_ACCESS_KEY_ID", "").strip(),
        "R2_SECRET_ACCESS_KEY": os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
        "R2_BUCKET": (
            os.getenv("R2_BUCKET_NAME", "").strip()
            or os.getenv("R2_BUCKET", "").strip()
        ),
        "R2_PUBLIC_BASE_URL": (
            os.getenv("R2_PUBLIC_BASE_URL", "").strip().rstrip("/")
            or os.getenv("R2_PUBLIC_URL", "").strip().rstrip("/")
        ),
    }


def validate_env(env):
    missing = []

    for k in [
        "API_BASE_URL",
        "INTERNAL_CALLBACK_TOKEN",
        "R2_ACCOUNT_ID",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET",
    ]:
        if not env.get(k):
            missing.append(k)

    if missing:
        raise ValueError(f"Missing environment variables: {missing}")


def callback(payload: dict, env: dict):
    api_base_url = env.get("API_BASE_URL", "")
    token = env.get("INTERNAL_CALLBACK_TOKEN", "")

    if not api_base_url:
        print("WARN: missing API_BASE_URL, skip callback:", payload)
        return

    if not token:
        print("WARN: missing INTERNAL_CALLBACK_TOKEN, skip callback:", payload)
        return

    try:
        resp = requests.post(
            f"{api_base_url}/api/internal/job-callback",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )

        print("CALLBACK STATUS:", resp.status_code)
        print("CALLBACK RESPONSE:", resp.text[:500])

    except Exception as e:
        print("CALLBACK ERROR:", repr(e))


def get_s3(env: dict):
    endpoint = f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=env["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def make_public_url(env: dict, key: str) -> str:
    base_url = env.get("R2_PUBLIC_BASE_URL", "").strip().rstrip("/")

    if not base_url:
        return key

    return f"{base_url}/{key}"


def upload_to_r2(local_path: str, key: str, env: dict) -> str:
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local video file not found: {local_path}")

    s3 = get_s3(env)

    s3.upload_file(
        Filename=local_path,
        Bucket=env["R2_BUCKET"],
        Key=key,
        ExtraArgs={
            "ContentType": "video/mp4",
            "CacheControl": "public, max-age=31536000",
        },
    )

    return make_public_url(env, key)


def handler(event):
    env = get_env()

    input_data = event.get("input", {}) or {}
    job_id = str(input_data.get("job_id", "")).strip()

    if not job_id:
        return {
            "ok": False,
            "error": "Missing job_id",
        }

    workdir = Path(tempfile.mkdtemp(prefix=f"{job_id}_"))

    try:
        validate_env(env)

        print("JOB START:", job_id)
        print("API_BASE_URL:", env["API_BASE_URL"])
        print("R2_BUCKET:", env["R2_BUCKET"])
        print("R2_PUBLIC_BASE_URL:", env.get("R2_PUBLIC_BASE_URL") or "[EMPTY]")

        callback({
            "job_id": job_id,
            "status": "running",
            "step": "job_received",
            "progress_pct": 3,
            "eta_sec": 120,
        }, env)

        def progress_callback(payload: dict):
            callback(payload, env)

        result = pipeline.run_job_serverless(
            job_config=input_data,
            job_id=job_id,
            base_dir=str(workdir),
            progress_callback=progress_callback,
        )

        local_video = result.get("video_path")

        if not local_video or not os.path.exists(local_video):
            raise Exception(f"Video file not found after pipeline: {local_video}")

        key = f"videos/{job_id}/output.mp4"
        public_url = upload_to_r2(local_video, key, env)

        callback({
            "job_id": job_id,
            "status": "completed",
            "step": "done",
            "progress_pct": 100,
            "eta_sec": 0,
            "result_video_key": key,
            "result_video_url": public_url,
        }, env)

        print("JOB COMPLETED:", job_id)
        print("VIDEO KEY:", key)
        print("VIDEO URL:", public_url)

        return {
            "ok": True,
            "job_id": job_id,
            "video_key": key,
            "video_url": public_url,
            "r2_public_base_url": env.get("R2_PUBLIC_BASE_URL") or "",
        }

    except Exception as e:
        error_message = str(e)

        print("JOB FAILED:", job_id)
        print("ERROR:", error_message)

        callback({
            "job_id": job_id,
            "status": "failed",
            "step": "error",
            "progress_pct": 100,
            "error_message": error_message,
        }, env)

        return {
            "ok": False,
            "job_id": job_id,
            "error": error_message,
        }

    finally:
        shutil.rmtree(workdir, ignore_errors=True)


runpod.serverless.start({
    "handler": handler
})