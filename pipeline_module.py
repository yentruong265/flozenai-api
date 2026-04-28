# pipeline_module.py — Production serverless version for RunPod
# Stable high-quality version: SDXL Base only, dynamic image settings, compact prompts,
# per-scene audio, preview/download use the same final output. No Lightning env required.
# Converted from the tested notebook.
# Entry point used by handler.py: run_job_serverless(job_config, job_id, base_dir, progress_callback)

import os
import torch
from pathlib import Path
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

DRIVE_ROOT = os.getenv("JOB_ROOT", "/tmp/easyai_jobs")
PENDING_DIR = os.path.join(DRIVE_ROOT, "pending")
RUNNING_DIR = os.path.join(DRIVE_ROOT, "running")
COMPLETED_DIR = os.path.join(DRIVE_ROOT, "completed")
FAILED_DIR = os.path.join(DRIVE_ROOT, "failed")

_PROGRESS_CALLBACK = None

def init_job_dirs(base_dir=None):
    global DRIVE_ROOT, PENDING_DIR, RUNNING_DIR, COMPLETED_DIR, FAILED_DIR
    DRIVE_ROOT = str(base_dir or os.getenv("JOB_ROOT", "/tmp/easyai_jobs"))
    PENDING_DIR = os.path.join(DRIVE_ROOT, "pending")
    RUNNING_DIR = os.path.join(DRIVE_ROOT, "running")
    COMPLETED_DIR = os.path.join(DRIVE_ROOT, "completed")
    FAILED_DIR = os.path.join(DRIVE_ROOT, "failed")
    for d in [PENDING_DIR, RUNNING_DIR, COMPLETED_DIR, FAILED_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)
    return {"DRIVE_ROOT": DRIVE_ROOT, "PENDING_DIR": PENDING_DIR, "RUNNING_DIR": RUNNING_DIR, "COMPLETED_DIR": COMPLETED_DIR, "FAILED_DIR": FAILED_DIR}

init_job_dirs(DRIVE_ROOT)

# ===== CELL 3 =====
# B3 — JSON helpers + normalize job config (final reviewed)

import os
import json
import time
import shutil
import traceback
import uuid
from datetime import datetime
from pathlib import Path


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_json(path, data):
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_job():
    files = sorted([
        f for f in os.listdir(PENDING_DIR)
        if f.endswith(".json") and os.path.isfile(os.path.join(PENDING_DIR, f))
    ])
    return files[0] if files else None


def write_status(job_dir, job_id, status, extra=None):
    payload = {
        "job_id": job_id,
        "status": status,
        "updated_at": now_str()
    }
    if isinstance(extra, dict):
        payload.update(extra)

    try:
        write_json(os.path.join(job_dir, "status.json"), payload)
    except Exception as e:
        print("WARN: could not write local status.json:", repr(e))

    global _PROGRESS_CALLBACK
    if _PROGRESS_CALLBACK:
        try:
            _PROGRESS_CALLBACK(payload)
        except Exception as e:
            print("WARN: progress callback failed:", repr(e))


def _safe_str(v, default=""):
    if v is None:
        return default
    return str(v).strip()


def _safe_int(v, default):
    try:
        if v in ("", None):
            return default
        return int(float(v))
    except Exception:
        return default


def _safe_float(v, default):
    try:
        if v in ("", None):
            return default
        return float(v)
    except Exception:
        return default


def _safe_bool(v, default=False):
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return True
    if s in {"0", "false", "no", "n"}:
        return False
    return default


def _clamp_int(v, default, min_value=None, max_value=None):
    x = _safe_int(v, default)
    if x is None:
        return default
    if min_value is not None:
        x = max(min_value, x)
    if max_value is not None:
        x = min(max_value, x)
    return x


def _clamp_float(v, default, min_value=None, max_value=None):
    x = _safe_float(v, default)
    if x is None:
        return default
    if min_value is not None:
        x = max(min_value, x)
    if max_value is not None:
        x = min(max_value, x)
    return x


def _normalize_style_value(style_value: str) -> str:
    style_value = _safe_str(style_value).lower()
    if style_value in {"auto", "tự động"}:
        return ""
    return style_value


def _normalize_aspect_ratio(aspect_ratio: str) -> str:
    aspect_ratio = _safe_str(aspect_ratio, "16:9")
    if aspect_ratio not in {"16:9", "9:16"}:
        return "16:9"
    return aspect_ratio


def _infer_dimensions_from_aspect_ratio(aspect_ratio: str):
    if aspect_ratio == "9:16":
        return 432, 768
    return 768, 432


def normalize_job_config(job_config, fallback_job_id=None):
    job_config = job_config or {}

    job_id = (
        _safe_str(job_config.get("job_id"))
        or fallback_job_id
        or f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    )

    prompt = _safe_str(job_config.get("prompt"))
    story_text = _safe_str(job_config.get("story_text"))
    if not story_text:
        story_text = prompt

    # style từ portal / source khác
    style = _normalize_style_value(job_config.get("style"))
    video_style_preset = _normalize_style_value(job_config.get("video_style_preset"))

    if not video_style_preset and style:
        video_style_preset = style
    if not style and video_style_preset:
        style = video_style_preset

    # single voice only
    primary_voice = (
        _safe_str(job_config.get("primary_voice")).lower()
        or _safe_str(job_config.get("voice")).lower()
        or _safe_str(job_config.get("tts_voice")).lower()
        or "fable"
    )

    # timing tổng
    target_total_sec = job_config.get("target_total_sec", None)
    if target_total_sec not in ("", None):
        target_total_sec = _safe_int(target_total_sec, None)
    else:
        target_total_sec = None

    target_total_video_sec = job_config.get("target_total_video_sec", None)
    if target_total_video_sec in ("", None):
        target_total_video_sec = target_total_sec
    else:
        target_total_video_sec = _safe_int(target_total_video_sec, target_total_sec)

    duration_mode = _safe_str(job_config.get("duration_mode")) or _safe_str(job_config.get("video_length_mode")) or "auto"
    video_length_mode = _safe_str(job_config.get("video_length_mode")) or duration_mode

    meta = job_config.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}

    lock_style_from_portal = _safe_bool(job_config.get("lock_style_from_portal"), True)

    tts_engine = _safe_str(job_config.get("tts_engine"), "").lower()
    if tts_engine not in {"", "openai", "edge"}:
        tts_engine = ""

    # aspect ratio + size
    aspect_ratio = _normalize_aspect_ratio(job_config.get("aspect_ratio"))
    default_w, default_h = _infer_dimensions_from_aspect_ratio(aspect_ratio)

    width = _clamp_int(job_config.get("width"), default_w, min_value=256)
    height = _clamp_int(job_config.get("height"), default_h, min_value=256)
    fps = _clamp_int(job_config.get("fps"), 24, min_value=8, max_value=60)

    # ép về bội số của 8
    width = width - (width % 8)
    height = height - (height % 8)

    # nếu width/height bị lệch hẳn với ratio mà không hợp lệ, fallback về mặc định ratio
    if width <= 0 or height <= 0:
        width, height = default_w, default_h

    normalized = {
        "job_id": job_id,
        "job_type": _safe_str(job_config.get("job_type"), "text_to_video"),

        "prompt": prompt,
        "story_text": story_text,
        "negative_prompt": _safe_str(job_config.get("negative_prompt")),

        "aspect_ratio": aspect_ratio,
        "width": width,
        "height": height,
        "fps": fps,

        "duration_per_scene": _clamp_float(job_config.get("duration_per_scene"), 3.0, min_value=0.5, max_value=20.0),
        "target_words_per_scene": _clamp_int(job_config.get("target_words_per_scene"), 40, min_value=8, max_value=120),
        "target_total_sec": target_total_sec,
        "target_total_video_sec": target_total_video_sec,
        "max_scene_duration": _clamp_float(job_config.get("max_scene_duration"), 12.0, min_value=2.0, max_value=30.0),
        "duration_mode": duration_mode,
        "video_length_mode": video_length_mode,

        # style
        "style": style,
        "video_style_preset": video_style_preset,
        "lock_style_from_portal": lock_style_from_portal,

        # single voice
        "voice": primary_voice,
        "primary_voice": primary_voice,
        "tts_voice": _safe_str(job_config.get("tts_voice")).lower() or primary_voice,
        "tts_engine": tts_engine,

        # tts controls
        "tts_rate": _safe_str(job_config.get("tts_rate"), "+0%"),
        "tts_pitch": _safe_str(job_config.get("tts_pitch"), "+0Hz"),
        "speech_speed": _clamp_float(job_config.get("speech_speed"), 1.20, min_value=0.7, max_value=1.5),
        "trim_audio_start": _clamp_float(job_config.get("trim_audio_start"), 0.03, min_value=0.0, max_value=0.5),
        "trim_audio_end": _clamp_float(job_config.get("trim_audio_end"), 0.16, min_value=0.0, max_value=0.8),

        # image generation — stable quality/speed default.
        # No Lightning env required. SDXL Base is slower than Lightning but much more stable.
        # These are only initial defaults; generate_image() still applies aspect-aware safeguards.
        "num_inference_steps": _clamp_int(job_config.get("num_inference_steps"), 16, min_value=8, max_value=28),
        "guidance_scale": _clamp_float(job_config.get("guidance_scale"), 6.0, min_value=4.0, max_value=8.0),
        "seed": _safe_int(job_config.get("seed"), 42),

        # retry
        "retry_count": _clamp_int(job_config.get("retry_count"), 0, min_value=0, max_value=20),
        "max_retry": _clamp_int(job_config.get("max_retry"), 2, min_value=0, max_value=20),

        "created_at": _safe_str(job_config.get("created_at"), now_str()),
        "meta": meta,
    }

    return normalized


# ===== CELL 5 =====
# B5 — SDXL image generation (PRO+, aspect-ratio-aware, anti-crop for 9:16)

import os
import re
import gc
import time
import torch
import threading
import requests
from PIL import Image

SDXL_MODEL_ID = os.getenv("SDXL_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0").strip()
IMAGE_ACCELERATION = "none"  # SDXL Base only. Ignore Lightning env for stability.
_IMAGE_PIPE = None
_IMAGE_PIPE_LOCK = threading.Lock()
_IMAGE_ACCELERATION_APPLIED = "none"

# Optional stock image mode.
# Priority order for each scene:
#   1) local stock assets, if you have STOCK_ASSET_DIR
#   2) Pexels API, if PEXELS_API_KEY is set
#   3) SDXL AI fallback
STOCK_ASSET_DIR = os.getenv("STOCK_ASSET_DIR", os.path.join(os.getenv("JOB_ROOT", "/tmp/easyai_jobs"), "stock_assets")).strip()
STOCK_METADATA_FILE = os.getenv("STOCK_METADATA_FILE", os.path.join(STOCK_ASSET_DIR, "stock_metadata.json")).strip()
ENABLE_STOCK_ASSETS = os.getenv("ENABLE_STOCK_ASSETS", "1").strip().lower() in {"1", "true", "yes", "y"}
ENABLE_STOCK_FETCH = os.getenv("ENABLE_STOCK_FETCH", "1").strip().lower() in {"1", "true", "yes", "y"}
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "").strip()
PEXELS_PER_PAGE = int(os.getenv("PEXELS_PER_PAGE", "10"))
STOCK_MIN_MATCH_SCORE = float(os.getenv("STOCK_MIN_MATCH_SCORE", "0.18"))

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, worst quality, low detail, noisy, jpeg artifacts, "
    "bad anatomy, deformed, distorted face, asymmetrical face, ugly eyes, crossed eyes, "
    "bad hands, fused fingers, extra fingers, missing fingers, extra limbs, missing limbs, "
    "duplicate subject, cloned face, cropped, cut off, out of frame, watermark, text, logo, subtitles, "
    "oversaturated, flat lighting, plastic skin, unrealistic skin, uncanny face, "
    "cartoon, anime, illustration, painting, storybook, 2d, vector art"
)


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def shorten_prompt_for_sdxl(prompt: str, max_chars: int = 380, max_words: int = None) -> str:
    """Compact SDXL prompt by both characters and words to avoid CLIP truncation."""
    prompt = " ".join((prompt or "").strip().split())
    if not prompt:
        return "clear coherent scene"

    prompt = re.sub(r"[\"“”]", "", prompt)
    prompt = prompt.replace("\n", " ")
    prompt = re.sub(r"\s+,", ",", prompt)
    prompt = re.sub(r",\s*,+", ", ", prompt).strip(" ,")

    replacements = {
        "ultra realistic cinematic photography": "cinematic realistic photo",
        "documentary realism": "realistic documentary",
        "real life everyday scene": "realistic daily scene",
        "candid moment": "natural moment",
        "modern real environment": "real environment",
        "natural human behavior": "natural people",
        "high dynamic range": "HDR",
        "professional color grading": "professional color",
        "shallow depth of field": "soft depth of field",
    }
    for k, v in replacements.items():
        prompt = re.sub(re.escape(k), v, prompt, flags=re.IGNORECASE)

    if max_words is not None:
        try:
            mw = int(max_words)
            if mw > 0:
                words = prompt.split()
                if len(words) > mw:
                    prompt = " ".join(words[:mw]).strip(" ,")
        except Exception:
            pass

    if max_chars is not None and len(prompt) > int(max_chars):
        prompt = prompt[:int(max_chars)]
        if "," in prompt:
            prompt = prompt.rsplit(",", 1)[0]
        prompt = prompt.strip(" ,")

    return prompt or "clear coherent scene"


def _cleanup_existing_pipe():
    global _IMAGE_PIPE
    if _IMAGE_PIPE is not None:
        try:
            del _IMAGE_PIPE
        except Exception:
            pass
        _IMAGE_PIPE = None
    free_memory()


def _safe_image_size(width: int, height: int):
    width = int(width)
    height = int(height)

    width = max(256, width)
    height = max(256, height)

    if width % 8 != 0:
        width = width - (width % 8)
    if height % 8 != 0:
        height = height - (height % 8)

    return width, height


def _is_vertical_frame(width: int, height: int) -> bool:
    return int(height) > int(width)


def _enhance_prompt_for_aspect_ratio(prompt: str, width: int, height: int) -> str:
    prompt = shorten_prompt_for_sdxl(prompt, max_chars=240, max_words=48)

    if _is_vertical_frame(width, height):
        prompt = ", ".join([
            prompt,
            "vertical composition",
            "portrait framing",
            "subject centered",
            "safe margins top and bottom",
            "full subject comfortably inside frame",
            "camera pulled back slightly",
            "balanced spacing",
            "mobile friendly framing",
            "avoid extreme close-up",
        ])
    else:
        prompt = ", ".join([
            prompt,
            "landscape composition",
            "wide cinematic framing",
            "balanced left-right spacing",
        ])

    return shorten_prompt_for_sdxl(prompt, max_chars=240, max_words=48)


def _enhance_negative_prompt_for_aspect_ratio(negative_prompt: str, width: int, height: int) -> str:
    negative_prompt = (negative_prompt or DEFAULT_NEGATIVE_PROMPT).strip()

    if _is_vertical_frame(width, height):
        negative_prompt = ", ".join([
            negative_prompt,
            "cropped head",
            "cut off body",
            "cut off feet",
            "top cropped",
            "bottom cropped",
            "subject too close",
            "zoomed in",
            "extreme close-up",
            "bad portrait framing",
            "off-center subject",
            "subject filling entire frame",
        ])

    return shorten_prompt_for_sdxl(negative_prompt, max_chars=260, max_words=55)


def _is_turbo_model() -> bool:
    return "turbo" in (SDXL_MODEL_ID or "").lower()


def _use_lightning_lora() -> bool:
    return False


def _is_lightning_active() -> bool:
    return False


def _get_guidance_scale(width: int, height: int, guidance_scale: float) -> float:
    """
    Stable SDXL Base guidance.
    Lower than the old 7.2+ setting for speed/control balance,
    but high enough to avoid the loose/glitchy look from Lightning/Turbo.
    """
    g = float(guidance_scale)
    if _is_turbo_model():
        return max(0.0, min(g, 1.0))

    if _is_vertical_frame(width, height):
        g = max(g, 6.2)
    else:
        g = max(g, 5.8)
    return min(g, 7.2)


def _get_num_inference_steps(width: int, height: int, steps: int) -> int:
    """
    Stable dynamic steps.
    - Landscape: 16-18 is the speed/quality sweet spot on RTX 4090.
    - Vertical: add a little quality to reduce crop/anatomy issues, but do not force 30.
    """
    s = int(steps)
    if _is_turbo_model():
        return min(max(s, 4), 16)
    if _is_vertical_frame(width, height):
        s = max(s, 18)
    else:
        s = max(s, 16)
    return min(s, 24)


def load_image_pipe(force_reload: bool = False):
    global _IMAGE_PIPE, _IMAGE_ACCELERATION_APPLIED

    if _IMAGE_PIPE is not None and not force_reload:
        return _IMAGE_PIPE

    with _IMAGE_PIPE_LOCK:
        if _IMAGE_PIPE is not None and not force_reload:
            return _IMAGE_PIPE

        if force_reload:
            _cleanup_existing_pipe()
        else:
            free_memory()

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print("🔄 Loading SDXL:", SDXL_MODEL_ID)

        pipe = StableDiffusionXLPipeline.from_pretrained(
            SDXL_MODEL_ID,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if torch.cuda.is_available() else None,
        )

        _IMAGE_ACCELERATION_APPLIED = "none"

        try:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                use_karras_sigmas=True,
            )
        except Exception:
            try:
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            except Exception:
                pass

        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass

        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass

        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        else:
            pipe = pipe.to("cpu")

        _IMAGE_PIPE = pipe
        print(f"✅ SDXL ready | acceleration={_IMAGE_ACCELERATION_APPLIED}")
        return _IMAGE_PIPE

def save_image_safely(image: Image.Image, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if image.mode != "RGB":
        image = image.convert("RGB")

    image.save(out_path, format="PNG")


def generate_image(
    prompt,
    out_path,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    width=768,
    height=432,
    num_inference_steps=28,
    guidance_scale=7.0,
    seed=None,
    retries=2,
    scene_id=None,
):
    width, height = _safe_image_size(width, height)

    # aspect-aware prompt / negative prompt
    prompt = _enhance_prompt_for_aspect_ratio(prompt, width, height)
    negative_prompt = _enhance_negative_prompt_for_aspect_ratio(negative_prompt, width, height)

    num_inference_steps = _get_num_inference_steps(width, height, num_inference_steps)
    guidance_scale = _get_guidance_scale(width, height, guidance_scale)

    # Slightly improve quality for human/face scenes, where AI artifacts are most visible.
    human_prompt = _has_any_term(prompt, _PERSON_WORDS) if "_PERSON_WORDS" in globals() else False
    if human_prompt and not _is_turbo_model():
        num_inference_steps = min(num_inference_steps + 2, 24)
        guidance_scale = min(max(guidance_scale, 6.2), 7.2)

    retries = max(1, int(retries))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    last_err = None
    is_vertical = _is_vertical_frame(width, height)

    for attempt in range(1, retries + 1):
        try:
            pipe = load_image_pipe(force_reload=(attempt > 1))
            device = "cuda" if torch.cuda.is_available() else "cpu"

            generator = None
            if seed is not None:
                generator = torch.Generator(device=device).manual_seed(int(seed) + attempt - 1)

            run_prompt = prompt
            run_negative_prompt = negative_prompt
            run_guidance_scale = guidance_scale
            run_steps = num_inference_steps

            # Retry strategy cho 9:16:
            # lần retry sau, ép model "lùi camera" mạnh hơn một chút
            if is_vertical and attempt >= 2:
                run_prompt = shorten_prompt_for_sdxl(
                    run_prompt + ", camera pulled back more, full body clearly visible, extra space above and below subject",
                    max_chars=240
                )
                run_negative_prompt = shorten_prompt_for_sdxl(
                    run_negative_prompt + ", face too close, torso cropped, feet cropped, oversized subject",
                    max_chars=260
                )
                run_guidance_scale = min(run_guidance_scale + 0.3, 12.0)
                run_steps = min(run_steps + 2, 80)

            if scene_id is not None:
                print(
                    f"🎨 Generating scene {int(scene_id):02d} | "
                    f"attempt {attempt} | size={width}x{height} | "
                    f"{'vertical' if is_vertical else 'landscape'}"
                )
            else:
                print(
                    f"🎨 Generating image | attempt {attempt} | size={width}x{height} | "
                    f"{'vertical' if is_vertical else 'landscape'}"
                )

            with torch.inference_mode():
                result = pipe(
                    prompt=run_prompt,
                    negative_prompt=run_negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=run_steps,
                    guidance_scale=run_guidance_scale,
                    num_images_per_prompt=1,
                    generator=generator,
                    output_type="pil",
                )

            image = result.images[0]
            save_image_safely(image, out_path)
            free_memory()
            return out_path

        except torch.cuda.OutOfMemoryError as e:
            last_err = e
            print("⚠️ CUDA OOM, retry with forced reload:", repr(e))
            _cleanup_existing_pipe()
            time.sleep(1.2)

        except Exception as e:
            last_err = e
            print("⚠️ generate_image retry because:", repr(e))
            free_memory()
            time.sleep(1.0)

    raise RuntimeError(f"generate_image failed: {repr(last_err)}")


# ===== CELL 6 =====
# B6 — Planner + TTS + motion engine (FINAL reviewed, aspect-ratio-aware, single-voice, ready for full narration)

import os
import re
import gc
import cv2
import json
import math
import shutil
import random
import asyncio
import nest_asyncio
nest_asyncio.apply()
import numpy as np
import torch
import edge_tts

from pathlib import Path
from typing import Dict, List, Any
from PIL import Image
from moviepy import VideoClip
from openai import OpenAI

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"
DTYPE = torch.float16 if USE_GPU else torch.float32

# KHÔNG hard-code API key trong code
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_VIDEO_PLANNER_MODEL", "gpt-4.1-mini").strip()
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts").strip()
ENABLE_AI_B6 = os.getenv("ENABLE_AI_B6", "1").strip() == "1"

DEFAULT_PRIMARY_VOICE = "fable"
DEFAULT_STYLE_PRESET = "cinematic_realistic"

STYLE_PRESETS = {
    "warm_storybook": {
        "name": "warm_storybook",
        "prompt_style": (
            "storybook illustration, warm tones, soft light, painterly texture, "
            "expressive composition, emotional storytelling, beautiful framing"
        ),
        "negative_style": (
            "photorealistic, realistic photography, live action, harsh flash, ugly composition, "
            "low detail, flat background"
        ),
        "motion_mode": "gentle",
        "duration_profile": "standard",
    },
    "zen_soft": {
        "name": "zen_soft",
        "prompt_style": (
            "calm minimal illustration, elegant composition, pastel tones, soft painterly look, "
            "peaceful atmosphere, balanced framing"
        ),
        "negative_style": (
            "photorealistic, chaotic frame, harsh contrast, messy scene, aggressive motion, clutter"
        ),
        "motion_mode": "slow",
        "duration_profile": "slow",
    },
    "dramatic_cinematic": {
        "name": "dramatic_cinematic",
        "prompt_style": (
            "photorealistic cinematic frame, dramatic light, realistic proportions, "
            "strong atmosphere, 35mm cinema"
        ),
        "negative_style": (
            "cartoon, anime, drawing, painting, 2d, cel shading, low realism, flat image, childish illustration"
        ),
        "motion_mode": "dramatic",
        "duration_profile": "standard",
    },
    "watercolor_poetic": {
        "name": "watercolor_poetic",
        "prompt_style": (
            "poetic watercolor painting, dreamy composition, soft brush texture, artistic paper feel, "
            "gentle storytelling, lyrical atmosphere"
        ),
        "negative_style": (
            "photorealistic, hard flash, live action, 3d render, noisy scene, ugly proportions"
        ),
        "motion_mode": "gentle",
        "duration_profile": "slow",
    },
    "cinematic_realistic": {
        "name": "cinematic_realistic",
        "prompt_style": (
            "cinematic realistic photo, realistic daily scene, natural people, "
            "35mm photo, natural light, clear subject, professional color"
        ),
        "negative_style": (
            "cartoon, anime, illustration, drawing, painting, storybook, 2d, stylized face, "
            "watercolor, vector art, flat color, cel shading, unreal skin, plastic face, fantasy illustration"
        ),
        "motion_mode": "standard",
        "duration_profile": "standard",
    },
}

BASE_NEGATIVE_PROMPT = (
    "blurry, low quality, bad anatomy, deformed face, asymmetrical face, ugly eyes, "
    "bad hands, fused fingers, extra fingers, missing fingers, extra limbs, missing limbs, "
    "duplicate subject, cloned face, cropped, cut off, out of frame, text, watermark, logo, subtitles, "
    "oversaturated, messy composition, distorted body, broken perspective, uncanny face, plastic skin"
)

OPENAI_TTS_VOICES = {
    "alloy", "ash", "ballad", "coral", "echo", "fable",
    "nova", "onyx", "sage", "shimmer", "verse"
}

EDGE_TTS_DEFAULT_VOICE = os.getenv("EDGE_TTS_DEFAULT_VOICE", "vi-VN-HoaiMyNeural").strip()


def clamp(v, vmin, vmax):
    return max(vmin, min(v, vmax))


def clear_folder(folder):
    folder = Path(folder)
    if not folder.exists():
        return
    for item in folder.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item, ignore_errors=True)


def sanitize_tts_text(text: str, max_chars: int = 4000) -> str:
    text = (text or "").strip()
    text = text.replace("*", " ")
    text = text.replace("•", " ")
    text = re.sub(r"\s+", " ", text)
    return text[:max_chars].strip()


def get_openai_client():
    if not OPENAI_API_KEY:
        raise ValueError("Missing OPENAI_API_KEY")
    return OpenAI(api_key=OPENAI_API_KEY)


def normalize_openai_voice(voice: str) -> str:
    voice = (voice or DEFAULT_PRIMARY_VOICE).strip().lower()
    return voice if voice in OPENAI_TTS_VOICES else DEFAULT_PRIMARY_VOICE


def resolve_single_voice(job_config=None) -> str:
    if isinstance(job_config, dict):
        for key in ["primary_voice", "voice", "tts_voice"]:
            vv = str(job_config.get(key, "") or "").strip().lower()
            if vv:
                return normalize_openai_voice(vv)
    return DEFAULT_PRIMARY_VOICE


def normalize_style_preset(style_value: str) -> str:
    """
    Normalize style from frontend into one of STYLE_PRESETS.
    This keeps the user's selected style whenever possible.
    """
    raw = str(style_value or "").strip().lower()
    raw = raw.replace("-", "_").replace(" ", "_")

    aliases = {
        "auto": "",
        "tu_dong": "",
        "tự_động": "",

        "realistic": "cinematic_realistic",
        "cinematic": "cinematic_realistic",
        "cinematic_realism": "cinematic_realistic",
        "photo_realistic": "cinematic_realistic",
        "photorealistic": "cinematic_realistic",
        "cinematic_realistic": "cinematic_realistic",

        "dramatic": "dramatic_cinematic",
        "drama": "dramatic_cinematic",
        "dramatic_cinematic": "dramatic_cinematic",

        "zen": "zen_soft",
        "soft_zen": "zen_soft",
        "zen_soft": "zen_soft",

        "storybook": "warm_storybook",
        "warm": "warm_storybook",
        "warm_storybook": "warm_storybook",

        "watercolor": "watercolor_poetic",
        "poetic": "watercolor_poetic",
        "watercolor_poetic": "watercolor_poetic",
    }

    normalized = aliases.get(raw, raw)
    return normalized if normalized in STYLE_PRESETS else DEFAULT_STYLE_PRESET




def shorten_prompt_for_sdxl(prompt: str, max_chars: int = 380, max_words: int = None) -> str:
    """Compact SDXL prompt by both characters and words to avoid CLIP truncation."""
    prompt = " ".join((prompt or "").strip().split())
    if not prompt:
        return "clear coherent scene"

    prompt = re.sub(r"[\"“”]", "", prompt)
    prompt = prompt.replace("\n", " ")
    prompt = re.sub(r"\s+,", ",", prompt)
    prompt = re.sub(r",\s*,+", ", ", prompt).strip(" ,")

    replacements = {
        "ultra realistic cinematic photography": "cinematic realistic photo",
        "documentary realism": "realistic documentary",
        "real life everyday scene": "realistic daily scene",
        "candid moment": "natural moment",
        "modern real environment": "real environment",
        "natural human behavior": "natural people",
        "high dynamic range": "HDR",
        "professional color grading": "professional color",
        "shallow depth of field": "soft depth of field",
    }
    for k, v in replacements.items():
        prompt = re.sub(re.escape(k), v, prompt, flags=re.IGNORECASE)

    if max_words is not None:
        try:
            mw = int(max_words)
            if mw > 0:
                words = prompt.split()
                if len(words) > mw:
                    prompt = " ".join(words[:mw]).strip(" ,")
        except Exception:
            pass

    if max_chars is not None and len(prompt) > int(max_chars):
        prompt = prompt[:int(max_chars)]
        if "," in prompt:
            prompt = prompt.rsplit(",", 1)[0]
        prompt = prompt.strip(" ,")

    return prompt or "clear coherent scene"


def chunk_story(story_text: str, target_words: int = 40) -> List[str]:
    text = re.sub(r"\s+", " ", (story_text or "").strip())
    if not text:
        return []

    sentences = re.split(r"(?<=[\.\!\?\…])\s+|\n+", text)
    chunks = []
    current = []
    current_words = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        words = len(sent.split())

        if current and current_words + words > target_words:
            chunks.append(" ".join(current).strip())
            current = [sent]
            current_words = words
        else:
            current.append(sent)
            current_words += words

    if current:
        chunks.append(" ".join(current).strip())

    merged = []
    for c in chunks:
        if merged and len(c.split()) < max(8, target_words // 3):
            merged[-1] = (merged[-1] + " " + c).strip()
        else:
            merged.append(c)

    return [c for c in merged if c][:18]



def force_scene_chunks_by_words(story_text: str, min_scenes: int = 4, max_scenes: int = 8) -> List[str]:
    """
    Fallback splitter for short / one-paragraph prompts.
    It ensures production video has multiple scenes even when chunk_story returns only 1 chunk.
    """
    text = re.sub(r"\s+", " ", (story_text or "").strip())
    if not text:
        return []

    # Respect explicit user scene markers first.
    explicit = re.split(
        r"(?:^|\s)(?:cảnh\s*\d+[:：\-]|scene\s*\d+[:：\-])",
        text,
        flags=re.IGNORECASE
    )
    explicit = [x.strip(" .,:;|-") for x in explicit if x and x.strip(" .,:;|-")]
    if len(explicit) >= 2:
        return explicit[:max_scenes]

    # Try sentence split.
    sentences = re.split(r"(?<=[\.\!\?\…])\s+|\n+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) >= min_scenes:
        return sentences[:max_scenes]

    words = text.split()
    if len(words) <= 12:
        return [text]

    target_scenes = min(max_scenes, max(min_scenes, min(8, max(2, len(words) // 18))))
    words_per_scene = max(8, int(len(words) / target_scenes + 0.999))

    chunks = []
    for i in range(0, len(words), words_per_scene):
        chunk = " ".join(words[i:i + words_per_scene]).strip()
        if chunk:
            chunks.append(chunk)

    # If the last chunk is too short, merge it.
    if len(chunks) > 1 and len(chunks[-1].split()) < 6:
        chunks[-2] = (chunks[-2] + " " + chunks[-1]).strip()
        chunks.pop()

    return chunks[:max_scenes]


def clean_ai_scene_list(ai_plan: Dict[str, Any], min_scenes: int = 4, max_scenes: int = 8) -> List[Dict[str, Any]]:
    scenes = (ai_plan or {}).get("scenes", [])
    if not isinstance(scenes, list):
        return []

    cleaned = []
    for idx, s in enumerate(scenes, 1):
        if not isinstance(s, dict):
            continue
        item = dict(s)
        item["scene_id"] = int(item.get("scene_id") or idx)
        cleaned.append(item)

    return cleaned[:max_scenes]


def get_ai_scene_narration(scene: Dict[str, Any], fallback_text: str = "") -> str:
    """
    Extract narration from AI scene. The planner should return narration_text,
    but this function is defensive for production.
    """
    for key in ["narration_text", "voice_text", "source_chunk", "script", "caption"]:
        val = str(scene.get(key, "") or "").strip()
        if val:
            return sanitize_tts_text(val, max_chars=700)

    subject = str(scene.get("main_subject", "") or "").strip()
    action = str(scene.get("action", "") or "").strip()
    location = str(scene.get("location", "") or "").strip()
    mood = str(scene.get("mood", "") or "").strip()
    combined = " ".join([subject, action, location, mood]).strip()
    return sanitize_tts_text(combined or fallback_text, max_chars=700)


def local_pick_style(story_text: str) -> str:
    text = (story_text or "").lower()

    if any(k in text for k in ["thiền", "an yên", "bình yên", "tĩnh lặng", "tâm an"]):
        return "zen_soft"
    if any(k in text for k in ["kịch tính", "trận chiến", "căng thẳng", "dramatic", "đối đầu", "nguy hiểm"]):
        return "dramatic_cinematic"
    if any(k in text for k in ["thơ", "mộng", "poetic", "watercolor", "mơ màng", "lãng đãng"]):
        return "watercolor_poetic"
    if any(k in text for k in ["vlog", "news", "doanh nhân", "tài chính", "marketing", "realistic", "phỏng vấn", "cinematic", "đời thực", "đời sống"]):
        return "cinematic_realistic"

    return DEFAULT_STYLE_PRESET


def is_style_locked(job_config: Dict[str, Any]) -> bool:
    v = str(job_config.get("lock_style_from_portal", "1")).strip().lower()
    return v in {"1", "true", "yes", "y"}


def is_vertical_aspect(job_config: Dict[str, Any]) -> bool:
    aspect_ratio = str(job_config.get("aspect_ratio", "") or "").strip()
    if aspect_ratio == "9:16":
        return True

    try:
        width = int(job_config.get("width", 768))
        height = int(job_config.get("height", 432))
        return height > width
    except Exception:
        return False


def infer_camera_language(style_name: str, scene_plan: Dict[str, Any], chunk: str, is_vertical: bool = False) -> Dict[str, str]:
    shot = str(scene_plan.get("shot", "") or "").strip()
    lighting = str(scene_plan.get("lighting", "") or "").strip()
    mood = str(scene_plan.get("mood", "") or "").strip()

    text = chunk.lower()

    if not shot:
        if is_vertical:
            if any(k in text for k in ["đi bộ", "đứng", "người", "cậu bé", "cô gái", "ông lão", "phụ nữ", "đàn ông"]):
                shot = "portrait medium-long shot"
            elif any(k in text for k in ["gần", "khuôn mặt", "ánh mắt", "nước mắt", "cảm xúc"]):
                shot = "portrait close-up shot"
            else:
                shot = "portrait medium shot"
        else:
            if any(k in text for k in ["gần", "khuôn mặt", "suy nghĩ", "nước mắt", "ánh mắt", "cảm xúc"]):
                shot = "close-up shot"
            elif any(k in text for k in ["đường phố", "đám đông", "không gian", "toàn cảnh", "bên ngoài"]):
                shot = "wide shot"
            elif style_name == "cinematic_realistic":
                shot = "eye-level medium shot"
            elif style_name == "dramatic_cinematic":
                shot = "cinematic medium shot"
            else:
                shot = "medium shot"

    if not lighting:
        if style_name == "cinematic_realistic":
            lighting = "natural daylight" if any(k in text for k in ["ban ngày", "sáng", "buổi sáng", "trưa"]) else "natural ambient light"
        elif style_name == "dramatic_cinematic":
            lighting = "dramatic moody lighting"
        else:
            lighting = "natural soft light"

    if not mood:
        if style_name == "cinematic_realistic":
            mood = "authentic everyday mood"
        elif style_name == "dramatic_cinematic":
            mood = "tense emotional mood"
        elif style_name == "zen_soft":
            mood = "peaceful calm mood"
        else:
            mood = "clear emotionally engaging mood"

    return {
        "shot": shot,
        "lighting": lighting,
        "mood": mood,
    }


def get_composition_hints(is_vertical: bool, style_name: str, scene_plan: Dict[str, Any]) -> List[str]:
    subject = str(scene_plan.get("main_subject", "") or "").lower()
    action = str(scene_plan.get("action", "") or "").lower()

    if is_vertical:
        hints = [
            "vertical composition",
            "portrait framing",
            "subject centered",
            "safe margins top and bottom",
            "balanced spacing",
            "mobile friendly framing",
            "avoid cropped head",
            "avoid cut off feet",
            "avoid subject too close to camera",
            "keep full subject comfortably inside frame",
        ]

        if any(k in subject or k in action for k in ["man", "woman", "boy", "girl", "person", "people", "ông", "bà", "cô", "cậu", "người", "phụ nữ", "đàn ông"]):
            hints.extend([
                "camera pulled back slightly",
                "full body or medium-long portrait framing",
                "subject fully visible",
            ])
        else:
            hints.extend([
                "clear central subject",
                "portrait-safe composition",
            ])
        return hints

    return [
        "landscape composition",
        "wide cinematic framing",
        "balanced left-right spacing",
    ]




# ===== Scene grounding + AI artifact reduction helpers =====

_PERSON_WORDS = {
    "person", "people", "man", "woman", "boy", "girl", "child", "children", "old man", "old woman",
    "father", "mother", "son", "daughter", "monk", "teacher", "customer", "seller",
    "người", "ông", "bà", "cô", "cậu", "bé", "trẻ", "con trai", "con gái", "cha", "mẹ",
    "nhà sư", "thiền sư", "khách hàng", "người bán", "phụ nữ", "đàn ông", "thanh niên"
}

_ABSTRACT_WORDS = {
    "số phận", "may mắn", "bài học", "ý nghĩa", "hạnh phúc", "khổ đau", "vô thường", "nhân quả",
    "fate", "luck", "lesson", "meaning", "happiness", "suffering", "karma", "impermanence"
}

_VIETNAMESE_HINT_WORDS = {
    "và", "là", "của", "một", "những", "người", "không", "trong", "cho", "với",
    "tôi", "bạn", "câu", "chuyện", "video", "đời", "sống", "thiền", "phật", "tâm"
}

_STOCK_FRIENDLY_WORDS = {
    "daily life", "realistic daily scene", "office", "business", "family", "nature", "city", "street",
    "food", "travel", "shopping", "school", "student", "teacher", "home", "work", "coffee",
    "đời sống", "văn phòng", "gia đình", "thiên nhiên", "đường phố", "thành phố", "học tập",
    "làm việc", "mua sắm", "du lịch", "quán cà phê", "nhà", "trường học", "kinh doanh"
}

_AI_REQUIRED_WORDS = {
    "ancient", "myth", "fantasy", "magic", "dragon", "heaven", "hell", "deity", "buddha",
    "ma quỷ", "quỷ", "thần", "phật", "cổ trang", "truyền thuyết", "huyền bí", "tâm linh",
    "future", "sci-fi", "spaceship", "robot", "cyberpunk", "surreal"
}


def detect_language(text: str) -> str:
    """Very small language detector for routing narration. Keeps Vietnamese narration in Vietnamese."""
    t = (text or "").strip().lower()
    if not t:
        return "unknown"
    if re.search(r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", t):
        return "vi"
    words = set(re.findall(r"\b\w+\b", t))
    if len(words & _VIETNAMESE_HINT_WORDS) >= 2:
        return "vi"
    return "en"


def looks_english_text(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    if detect_language(t) == "vi":
        return False
    common_en = {"the", "a", "an", "and", "is", "are", "with", "in", "on", "to", "of", "for", "from"}
    words = set(re.findall(r"\b[a-z]+\b", t))
    return len(words & common_en) >= 2


def repair_narration_language(narration: str, fallback_text: str, source_language: str) -> str:
    narration = sanitize_tts_text(narration, max_chars=700)
    fallback_text = sanitize_tts_text(fallback_text, max_chars=700)
    if source_language == "vi" and looks_english_text(narration) and fallback_text:
        return fallback_text
    return narration or fallback_text


def tokenize_for_match(text: str):
    t = (text or "").lower()
    t = re.sub(r"[^0-9a-zA-Zăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ\s_-]", " ", t)
    return {w for w in re.split(r"[\s_-]+", t) if len(w) >= 3}


def _has_any_term(text: str, terms) -> bool:
    t = (text or "").lower()
    return any(term in t for term in terms)


def scene_has_human(scene_plan: Dict[str, Any], narration: str = "") -> bool:
    joined = " ".join([
        str(narration or ""),
        str(scene_plan.get("main_subject", "") or ""),
        str(scene_plan.get("action", "") or ""),
        str(scene_plan.get("location", "") or ""),
        " ".join([str(x) for x in (scene_plan.get("details", []) or [])]),
    ]).lower()
    return _has_any_term(joined, _PERSON_WORDS)


def scene_is_abstract(narration: str, scene_plan: Dict[str, Any]) -> bool:
    joined = " ".join([
        str(narration or ""),
        str(scene_plan.get("main_subject", "") or ""),
        str(scene_plan.get("action", "") or ""),
    ]).lower()
    return _has_any_term(joined, _ABSTRACT_WORDS)


def scene_requires_ai(narration: str, scene_plan: Dict[str, Any], video_style_preset: str = "") -> bool:
    joined = " ".join([
        str(narration or ""),
        str(video_style_preset or ""),
        str(scene_plan.get("main_subject", "") or ""),
        str(scene_plan.get("action", "") or ""),
        str(scene_plan.get("location", "") or ""),
        " ".join([str(x) for x in (scene_plan.get("details", []) or [])]),
    ]).lower()
    if video_style_preset in {"warm_storybook", "watercolor_poetic", "zen_soft"}:
        return True
    return _has_any_term(joined, _AI_REQUIRED_WORDS)


def scene_stock_friendly(narration: str, scene_plan: Dict[str, Any], video_style_preset: str = "") -> bool:
    if scene_requires_ai(narration, scene_plan, video_style_preset):
        return False
    if video_style_preset not in {"cinematic_realistic", "dramatic_cinematic", ""}:
        return False
    joined = " ".join([
        str(narration or ""),
        str(scene_plan.get("main_subject", "") or ""),
        str(scene_plan.get("action", "") or ""),
        str(scene_plan.get("location", "") or ""),
        " ".join([str(x) for x in (scene_plan.get("details", []) or [])]),
    ]).lower()
    return _has_any_term(joined, _STOCK_FRIENDLY_WORDS) or not scene_is_abstract(narration, scene_plan)


def build_stock_query(narration: str, scene_plan: Dict[str, Any], is_vertical: bool = False) -> str:
    parts = [
        scene_plan.get("main_subject", ""),
        scene_plan.get("action", ""),
        scene_plan.get("location", ""),
        scene_plan.get("mood", ""),
        narration,
        "vertical" if is_vertical else "landscape",
    ]
    return shorten_prompt_for_sdxl(" ".join([str(p) for p in parts if p]), max_chars=180, max_words=32)


def _load_stock_metadata():
    items = []
    if not ENABLE_STOCK_ASSETS or not STOCK_ASSET_DIR or not os.path.isdir(STOCK_ASSET_DIR):
        return []

    if STOCK_METADATA_FILE and os.path.exists(STOCK_METADATA_FILE):
        try:
            data = read_json(STOCK_METADATA_FILE)
            if isinstance(data, dict):
                data = data.get("items", []) or data.get("assets", []) or []
            if isinstance(data, list):
                for row in data:
                    if not isinstance(row, dict):
                        continue
                    rel = str(row.get("file") or row.get("path") or "").strip()
                    if not rel:
                        continue
                    path = rel if os.path.isabs(rel) else os.path.join(STOCK_ASSET_DIR, rel)
                    if os.path.exists(path):
                        tags = row.get("tags", [])
                        if isinstance(tags, str):
                            tags = [tags]
                        items.append({
                            "path": path,
                            "tags": [str(x).lower() for x in tags],
                            "title": str(row.get("title", "") or ""),
                            "category": str(row.get("category", "") or ""),
                        })
        except Exception as e:
            print("WARN: could not read stock metadata:", repr(e))

    if not items:
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        for root, _, files in os.walk(STOCK_ASSET_DIR):
            for fn in files:
                if os.path.splitext(fn)[1].lower() in exts:
                    path = os.path.join(root, fn)
                    rel = os.path.relpath(path, STOCK_ASSET_DIR)
                    tags = re.split(r"[\s_\-/.]+", os.path.splitext(rel)[0].lower())
                    items.append({"path": path, "tags": tags, "title": rel, "category": os.path.basename(root)})
    return items


def find_stock_asset(stock_query: str, scene_plan: Dict[str, Any], is_vertical: bool = False):
    items = _load_stock_metadata()
    if not items:
        return None
    q_tokens = tokenize_for_match(stock_query)
    best = None
    best_score = 0.0
    for item in items:
        tag_text = " ".join(item.get("tags", []) + [item.get("title", ""), item.get("category", "")])
        t_tokens = tokenize_for_match(tag_text)
        if not t_tokens:
            continue
        overlap = len(q_tokens & t_tokens)
        score = overlap / max(6, len(q_tokens))
        if is_vertical and ({"vertical", "portrait", "reels", "shorts"} & t_tokens):
            score += 0.08
        if (not is_vertical) and ({"landscape", "wide", "horizontal"} & t_tokens):
            score += 0.08
        if score > best_score:
            best_score = score
            best = item
    if best and best_score >= STOCK_MIN_MATCH_SCORE:
        return {**best, "match_score": round(best_score, 3), "query": stock_query}
    return None


def prepare_stock_image(src_path: str, out_path: str, width: int, height: int):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img = Image.open(src_path).convert("RGB")
    src_w, src_h = img.size
    target_ratio = width / max(height, 1)
    src_ratio = src_w / max(src_h, 1)
    if src_ratio > target_ratio:
        new_w = int(src_h * target_ratio)
        x1 = max(0, (src_w - new_w) // 2)
        img = img.crop((x1, 0, x1 + new_w, src_h))
    else:
        new_h = int(src_w / max(target_ratio, 1e-6))
        y1 = max(0, (src_h - new_h) // 2)
        img = img.crop((0, y1, src_w, y1 + new_h))
    img = img.resize((width, height), Image.LANCZOS)
    save_image_safely(img, out_path)
    return out_path


def fetch_pexels_photo(stock_query: str, is_vertical: bool = False):
    """Fetch one suitable Pexels photo URL for stock-friendly scenes.
    Returns metadata dict or None. Does not raise.
    """
    if not ENABLE_STOCK_FETCH or not PEXELS_API_KEY:
        return None

    query = shorten_prompt_for_sdxl(stock_query or "daily life", max_chars=120, max_words=18)
    if not query:
        query = "daily life"

    orientation = "portrait" if is_vertical else "landscape"
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": PEXELS_API_KEY}
    params = {
        "query": query,
        "orientation": orientation,
        "per_page": max(1, min(int(PEXELS_PER_PAGE), 20)),
    }
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=8)
        if resp.status_code != 200:
            print(f"WARN: Pexels API status={resp.status_code} query={query!r}")
            return None
        data = resp.json()
        photos = data.get("photos", []) or []
        if not photos:
            return None

        # Prefer larger, non-empty images. Use deterministic first result to reduce randomness.
        photo = photos[0]
        src = photo.get("src", {}) or {}
        image_url = src.get("large2x") or src.get("large") or src.get("original")
        if not image_url:
            return None
        return {
            "url": image_url,
            "query": query,
            "photographer": photo.get("photographer", ""),
            "pexels_id": photo.get("id", ""),
        }
    except Exception as e:
        print("WARN: Pexels fetch failed:", repr(e))
        return None


def prepare_pexels_image(image_url: str, out_path: str, width: int, height: int):
    """Download and crop a Pexels image to target frame."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        r = requests.get(image_url, timeout=15)
        r.raise_for_status()
        tmp_path = out_path + ".download"
        with open(tmp_path, "wb") as f:
            f.write(r.content)
        prepare_stock_image(tmp_path, out_path, width, height)
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return out_path
    except Exception as e:
        print("WARN: Pexels image download failed:", repr(e))
        return None


def _clean_prompt_piece(value: str, max_words: int = 12) -> str:
    value = re.sub(r"\s+", " ", str(value or "")).strip(" ,.;:")
    if not value:
        return ""
    words = value.split()
    if len(words) > max_words:
        value = " ".join(words[:max_words]).strip(" ,.;:")
    return value


def _scene_anchor_from_plan(scene_plan: Dict[str, Any], narration: str, is_vertical: bool) -> str:
    """Build a short, concrete visual anchor from structured planner fields."""
    subject = _clean_prompt_piece(scene_plan.get("main_subject", ""), 10)
    action = _clean_prompt_piece(scene_plan.get("action", ""), 12)
    location = _clean_prompt_piece(scene_plan.get("location", ""), 10)
    expression = _clean_prompt_piece(scene_plan.get("expression", ""), 8)
    shot = _clean_prompt_piece(scene_plan.get("shot", ""), 8)
    lighting = _clean_prompt_piece(scene_plan.get("lighting", ""), 8)
    mood = _clean_prompt_piece(scene_plan.get("mood", ""), 8)

    if not subject:
        subject = "main person" if scene_has_human(scene_plan, narration) else "main subject"
    if not action:
        action = "visible action matching narration"
    if not location:
        location = "realistic setting matching narration"

    if is_vertical and not shot:
        shot = "portrait medium shot"
    elif not shot:
        shot = "eye-level medium shot"

    parts = [
        subject,
        action,
        location,
        expression,
        shot,
        lighting or "natural light",
        mood,
    ]
    return ", ".join([p for p in parts if p])


def build_grounded_visual_prompt(
    base_prompt: str,
    narration: str,
    video_style: Dict[str, Any],
    scene_plan: Dict[str, Any],
    is_vertical: bool = False,
) -> str:
    """
    Make image prompt more grounded:
    - prioritize visible subject/action/location from the narration/scene plan
    - avoid abstract/symbolic prompts
    - add human anatomy safeguards only when people appear
    - keep prompt short enough to reduce CLIP truncation
    """
    style_name = (video_style.get("name") or "").strip().lower()
    anchor = _scene_anchor_from_plan(scene_plan, narration, is_vertical)

    if style_name == "cinematic_realistic":
        style_terms = "cinematic realistic photo, natural light, realistic faces, clear subject"
    elif style_name == "dramatic_cinematic":
        style_terms = "photorealistic cinematic frame, dramatic light, realistic proportions"
    elif style_name == "zen_soft":
        style_terms = "calm soft visual, peaceful composition, gentle light"
    elif style_name == "warm_storybook":
        style_terms = "warm storybook illustration, clean composition, expressive subject"
    elif style_name == "watercolor_poetic":
        style_terms = "poetic watercolor, soft brush texture, clear subject"
    else:
        style_terms = str(video_style.get("prompt_style", "") or "")

    human_terms = ""
    if scene_has_human(scene_plan, narration):
        human_terms = (
            "one clear human subject, natural face, correct anatomy, realistic hands, "
            "normal fingers, detailed eyes"
        )

    abstract_guard = ""
    if scene_is_abstract(narration, scene_plan):
        abstract_guard = "show a concrete real moment, not abstract symbolism"

    if is_vertical:
        composition = "portrait framing, subject centered, safe margins, no cropped head or feet"
    else:
        composition = "landscape framing, balanced composition, eye-level camera"

    # Keep AI prompt, but only as support after the structured anchor.
    base_prompt = shorten_prompt_for_sdxl(base_prompt, max_chars=160, max_words=32)
    narration_hint = shorten_prompt_for_sdxl(f"matches narration: {narration}", max_chars=100, max_words=18)

    prompt = ", ".join([
        anchor,
        style_terms,
        human_terms,
        abstract_guard,
        composition,
        base_prompt,
        narration_hint,
    ])
    return shorten_prompt_for_sdxl(prompt, max_chars=260, max_words=58)


def build_scene_negative_prompt(global_negative_prompt: str, scene_plan: Dict[str, Any], narration: str, is_vertical: bool = False) -> str:
    """Add scene-specific negatives without making the negative prompt too long."""
    extras = []
    if scene_has_human(scene_plan, narration):
        extras.extend([
            "bad face", "uncanny face", "asymmetrical eyes", "bad hands",
            "fused fingers", "extra fingers", "missing fingers", "extra arms", "extra legs"
        ])
    if is_vertical:
        extras.extend([
            "cropped head", "cut off feet", "subject too close", "extreme close-up", "off-center subject"
        ])
    if scene_is_abstract(narration, scene_plan):
        extras.extend(["abstract symbols", "floating objects", "surreal unrelated scene"])

    neg = ", ".join([global_negative_prompt or "", ", ".join(extras)])
    return shorten_prompt_for_sdxl(neg, max_chars=300, max_words=65)


def validate_and_repair_scene_plan(scene_plan: Dict[str, Any], narration: str, is_vertical: bool = False) -> Dict[str, Any]:
    """Defensive repair for incomplete AI planner output."""
    sp = dict(scene_plan or {})
    if not str(sp.get("main_subject", "") or "").strip():
        sp["main_subject"] = "main person" if scene_has_human(sp, narration) else "main subject"
    if not str(sp.get("action", "") or "").strip():
        sp["action"] = "visible action matching the narration"
    if not str(sp.get("location", "") or "").strip():
        sp["location"] = "realistic setting matching the narration"
    if not str(sp.get("shot", "") or "").strip():
        sp["shot"] = "portrait medium shot" if is_vertical else "eye-level medium shot"
    if not str(sp.get("lighting", "") or "").strip():
        sp["lighting"] = "natural soft light"
    return sp


def sanitize_scene_plan(scene_plan: Dict[str, Any], style_name: str, chunk: str, is_vertical: bool = False) -> Dict[str, Any]:
    scene_plan = dict(scene_plan or {})

    subject = str(scene_plan.get("main_subject", "") or "").strip()
    location = str(scene_plan.get("location", "") or "").strip()
    details = scene_plan.get("details", []) or []

    if not isinstance(details, list):
        details = []

    details = [str(d).strip() for d in details if str(d).strip()][:5]

    if style_name == "cinematic_realistic":
        banned = ["fantasy", "magical", "anime", "cartoon", "illustration", "painting", "storybook"]
        text = " ".join([subject, location, " ".join(details)]).lower()
        if any(b in text for b in banned):
            subject = ""
            location = ""
            details = []

    inferred = infer_camera_language(style_name, scene_plan, chunk, is_vertical=is_vertical)

    return {
        "main_subject": subject,
        "location": location,
        "details": details,
        "shot": inferred["shot"],
        "lighting": inferred["lighting"],
        "mood": inferred["mood"],
        "motion_mode": str(scene_plan.get("motion_mode", "") or "").strip().lower() or "gentle",
        "action": str(scene_plan.get("action", "") or "").strip(),
        "expression": str(scene_plan.get("expression", "") or "").strip(),
        "time_of_day": str(scene_plan.get("time_of_day", "") or "").strip(),
    }


def build_visual_prompt(
    chunk: str,
    video_style: Dict[str, Any],
    scene_plan: Dict[str, Any],
    style_locked: bool = True,
    is_vertical: bool = False
) -> str:
    style_name = (video_style.get("name") or "").strip().lower()
    scene_plan = sanitize_scene_plan(scene_plan, style_name, chunk, is_vertical=is_vertical)

    subject = scene_plan.get("main_subject", "")
    location = scene_plan.get("location", "")
    mood = scene_plan.get("mood", "")
    shot = scene_plan.get("shot", "")
    lighting = scene_plan.get("lighting", "")
    details = scene_plan.get("details", []) or []
    details_text = ", ".join(details[:4])
    action = scene_plan.get("action", "")
    expression = scene_plan.get("expression", "")
    time_of_day = scene_plan.get("time_of_day", "")
    composition_hints = get_composition_hints(is_vertical, style_name, scene_plan)

    chunk_short = re.sub(r"\s+", " ", chunk).strip()[:140]
    base_style = video_style.get("prompt_style", "")

    if style_name == "cinematic_realistic":
        parts = [
            "cinematic realistic photo",
            "realistic daily scene",
            "natural people",
            "35mm photo",
            "natural light",
            "clear subject",
            subject or "ordinary people",
            action or "natural everyday activity",
            expression or "authentic facial expression",
            location or "real daily life setting",
            shot or ("portrait medium-long shot" if is_vertical else "eye-level medium shot"),
            lighting or "natural daylight",
            time_of_day,
            mood or "authentic everyday mood",
            *composition_hints,
            details_text,
            f"scene context: {chunk_short}",
        ]
    elif style_name == "dramatic_cinematic":
        parts = [
            "cinematic live action frame",
            "photorealistic",
            "dramatic realism",
            "strong atmosphere",
            "expressive camera language",
            "film still",
            subject or "main subject",
            action or "meaningful dramatic action",
            expression or "strong emotional expression",
            location or "dramatic real setting",
            shot or ("portrait cinematic shot" if is_vertical else "cinematic medium shot"),
            lighting or "dramatic moody lighting",
            time_of_day,
            mood or "tense emotional mood",
            *composition_hints,
            details_text,
            f"scene context: {chunk_short}",
        ]
    else:
        parts = [
            subject or "main subject matching the narration",
            action or "meaningful action matching the narration",
            expression,
            location or "appropriate setting",
            shot or ("portrait medium shot" if is_vertical else "medium shot"),
            lighting or "natural soft light",
            time_of_day,
            mood or "clear and emotionally engaging",
            *composition_hints,
            details_text,
            base_style,
            f"scene context: {chunk_short}",
        ]

    prompt = ", ".join([p for p in parts if p])
    return build_grounded_visual_prompt(
        base_prompt=prompt,
        narration=chunk,
        video_style=video_style,
        scene_plan=scene_plan,
        is_vertical=is_vertical,
    )


def build_scene_profile_from_plan(scene_plan: Dict[str, Any], idx: int, video_style: Dict[str, Any]) -> str:
    motion_mode = (scene_plan.get("motion_mode") or video_style.get("motion_mode") or "gentle").strip().lower()
    if motion_mode not in {"gentle", "slow", "dramatic", "standard"}:
        motion_mode = "gentle"
    return motion_mode


def _extract_text_from_responses_api(resp) -> str:
    texts = []
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            t = getattr(c, "text", None)
            if t:
                texts.append(t)
    return "\n".join(texts).strip()


def plan_video_with_ai(
    story_text: str,
    scene_chunks: List[str],
    job_config: Dict[str, Any],
    locked_style: str = ""
) -> Dict[str, Any]:
    """
    Ask OpenAI to become the video director:
    - understand the user's requested content,
    - split it into multiple visually different scenes,
    - generate prompt-ready scene plans,
    - keep the exact frontend style when style is locked.
    """
    if not ENABLE_AI_B6:
        raise RuntimeError("AI planner disabled")

    client = get_openai_client()

    locked_style = normalize_style_preset(locked_style) if locked_style else ""
    style_locked = bool(locked_style)
    is_vertical = is_vertical_aspect(job_config)

    user_style_text = locked_style or "auto"
    aspect_text = "9:16 vertical TikTok/Reels/Shorts" if is_vertical else "16:9 landscape YouTube/web"

    min_scenes = int(job_config.get("min_scenes", 3) or 3)
    max_scenes = int(job_config.get("max_scenes", 10) or 10)
    min_scenes = max(2, min(min_scenes, 10))
    max_scenes = max(min_scenes, min(max_scenes, 12))

    prompt = f"""
You are FlozenAI's production video director and prompt engineer.

Return ONLY valid JSON.
No markdown.
No explanation.
No code fence.

USER REQUEST:
{story_text}

FRONTEND SELECTED STYLE:
{user_style_text}

STYLE LOCKED:
{style_locked}

ASPECT RATIO:
{aspect_text}

CURRENT ROUGH TEXT CHUNKS:
{json.dumps(scene_chunks, ensure_ascii=False)}

LANGUAGE RULES:
A. Detect the language of USER REQUEST.
B. narration_text MUST be in the SAME language as USER REQUEST.
C. If USER REQUEST is Vietnamese, every narration_text MUST be Vietnamese. Do not translate narration_text into English.
D. visual_prompt and stock_query MUST be concise English for image search/generation.

IMPORTANT STYLE RULES:
1. If STYLE LOCKED is true, video_style_preset MUST be exactly "{locked_style}".
2. Do not override the user's selected frontend style.
3. Every scene must visually follow the selected style preset.
4. Allowed style presets: {list(STYLE_PRESETS.keys())}.

IMPORTANT PLANNING RULES:
5. Understand the user request and turn it into a coherent short video.
6. Choose the scene count naturally based on the content and target length, within {min_scenes} to {max_scenes} scenes.
7. For short videos under 45 seconds, usually create 4-7 scenes, but use your judgment.
8. Even if the input is one paragraph, you MUST still create multiple coherent scenes.
9. Each scene must represent a different moment, action, camera framing, or visual idea.
10. Each scene must include narration_text. This is the spoken narration for that scene.
11. narration_text must follow the user's requested content; do not invent unrelated facts.
12. Keep each narration_text short and natural: ideally 8-16 words, one short sentence.
13. No scene narration should feel longer than about 4.5 seconds when spoken; split long narration into multiple scenes.
14. CRITICAL: visual_prompt MUST directly and literally represent narration_text.
15. visual_prompt must be written in concise English for SDXL.
16. Do NOT create generic, symbolic, unrelated, abstract, or decorative visuals.
17. If narration_text mentions a person, visual_prompt must show that same person type, visible emotion, and visible action.
18. If narration_text mentions an action, visual_prompt must show that visible action, not only a portrait.
19. If narration_text mentions a place, product, object, or situation, visual_prompt must include it clearly.
20. Each visual_prompt must be concrete: subject + visible action + location/background + camera framing + mood + lighting.
21. Keep visual_prompt compact: maximum 35-50 English words. Avoid long repeated style phrases.
21b. Choose visual_source as "stock" for realistic daily-life/business/nature/family/office/travel scenes that can use existing photos.
21c. Choose visual_source as "ai" for unique characters, Buddhist/fantasy/spiritual/ancient scenes, product-specific scenes, or anything hard to find in stock assets.
21d. Provide stock_query in English whenever visual_source is "stock".
22. Avoid repeating the same subject/action/location across scenes unless the story requires continuity.
23. Use physically possible scenes. Avoid impossible body poses, floating objects, random symbols, unrelated fantasy elements.
24. If the narration is abstract, convert it into one concrete visible moment that represents the meaning.
25. Each scene must include a specific subject, visible action, location/background, shot, lighting, mood, and details.
26. Keep visual prompts practical and literal, not abstract.
20. If the user asks for a product/review/sales video, scenes should follow: hook, product close-up, benefit/use case, trust/social proof, call-to-action.
21. If the user asks for a story video, scenes should follow narrative progression: opening, conflict/context, turning point, insight, ending.
22. If aspect ratio is 9:16, use portrait-safe framing: centered subject, safe margins, avoid cropped head/feet.
27. Do not add text, subtitles, watermark, logo, or words inside generated images.
28. For every scene, narration_text and visual_prompt must describe the SAME moment.
29. For scenes with humans, prefer one clear subject or a small natural group; avoid crowds unless necessary.
30. For humans, include natural face, correct anatomy, realistic hands, normal fingers, and clear eyes in visual_prompt when relevant.

STYLE-SPECIFIC RULES:
19. For cinematic_realistic: use real-life photography, documentary realism, natural humans, realistic locations, natural lighting. Avoid illustration/cartoon/anime/painting.
20. For dramatic_cinematic: use photorealistic cinematic frames, dramatic lighting, strong emotion.
21. For zen_soft: use calm, peaceful, minimal soft visual language.
22. For warm_storybook: use warm storybook illustration.
23. For watercolor_poetic: use poetic watercolor visual language.

Return JSON exactly with this schema:
{{
  "video_style_preset": "{locked_style if style_locked else 'one of allowed style presets'}",
  "num_inference_steps": 16,
  "guidance_scale": 6.0,
  "global_negative_prompt_addon": "",
  "director_notes": "short note",
  "scenes": [
    {{
      "scene_id": 1,
      "narration_text": "spoken narration in the same language as USER REQUEST",
      "visual_source": "stock or ai",
      "stock_query": "English search query for stock/local asset, or empty string",
      "visual_prompt": "English SDXL prompt, concrete and style-consistent",
      "main_subject": "short specific visual subject",
      "action": "short visible action",
      "expression": "visible emotion if relevant",
      "location": "specific realistic/appropriate setting",
      "details": ["detail 1", "detail 2", "detail 3"],
      "shot": "camera shot/framing",
      "lighting": "lighting condition",
      "time_of_day": "morning / afternoon / night / neutral",
      "mood": "emotional tone",
      "motion_mode": "gentle"
    }}
  ]
}}
"""

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
        temperature=0.30,
    )

    text = _extract_text_from_responses_api(resp)
    if not text:
        text = str(resp)

    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    data = json.loads(text)

    if "scenes" not in data or not isinstance(data["scenes"], list):
        raise ValueError("AI planner JSON missing scenes")

    if style_locked:
        data["video_style_preset"] = locked_style
    else:
        data["video_style_preset"] = normalize_style_preset(data.get("video_style_preset", ""))

    # Production guardrail: enforce minimum useful scene count.
    scenes = clean_ai_scene_list(data, min_scenes=min_scenes, max_scenes=max_scenes)
    if len(scenes) < 2:
        raise ValueError(f"AI planner returned too few scenes: {len(scenes)}")

    data["scenes"] = scenes
    return data


def create_adaptive_video_plan(
    story_text: str,
    job_config: Dict[str, Any],
    target_words_per_scene: int = 40
) -> Dict[str, Any]:
    """
    Create a production-grade video plan.

    Key fix:
    - OpenAI planner is allowed to define the REAL number of scenes.
    - If AI creates 4-8 scenes, we use those scenes directly.
    - The frontend-selected style is respected via lock_style_from_portal.
    - If OpenAI fails, fallback still creates multiple scenes.
    """
    initial_chunks = chunk_story(story_text, target_words=target_words_per_scene)
    if not initial_chunks:
        initial_chunks = force_scene_chunks_by_words(story_text, min_scenes=4, max_scenes=8)
    if not initial_chunks:
        raise ValueError("No scene chunks generated")

    source_language = detect_language(story_text)
    used_ai = False
    ai_plan = None
    director_notes = ""

    portal_style_raw = (
        str(job_config.get("video_style_preset", "") or "").strip()
        or str(job_config.get("style", "") or "").strip()
    )
    portal_style = normalize_style_preset(portal_style_raw) if portal_style_raw else ""

    style_locked = is_style_locked(job_config)
    is_vertical = is_vertical_aspect(job_config)

    min_scenes = int(job_config.get("min_scenes", 3) or 3)
    max_scenes = int(job_config.get("max_scenes", 10) or 10)
    min_scenes = max(2, min(min_scenes, 10))
    max_scenes = max(min_scenes, min(max_scenes, 12))

    if ENABLE_AI_B6 and OPENAI_API_KEY:
        try:
            ai_plan = plan_video_with_ai(
                story_text=story_text,
                scene_chunks=initial_chunks,
                job_config={
                    **job_config,
                    "min_scenes": min_scenes,
                    "max_scenes": max_scenes,
                },
                locked_style=portal_style if style_locked and portal_style else ""
            )
            used_ai = True
            director_notes = ai_plan.get("director_notes", "")
        except Exception as e:
            print("⚠️ AI planner fallback:", repr(e))

    # Style selection: keep frontend style if locked, otherwise let AI choose.
    if style_locked and portal_style:
        requested_style = portal_style
    else:
        requested_style = (
            portal_style
            or (ai_plan or {}).get("video_style_preset")
            or local_pick_style(story_text)
        )

    video_style_preset = normalize_style_preset(requested_style)
    video_style = dict(STYLE_PRESETS[video_style_preset])

    # Stable high-quality production defaults.
    # SDXL Base only: no Lightning env required, no LoRA startup risk.
    # Keep scene count from OpenAI, but use safe image settings.
    default_steps = 18 if video_style_preset in {"cinematic_realistic", "dramatic_cinematic"} else 16
    if is_vertical:
        default_steps = max(default_steps, 18)
    default_guidance = 6.2 if video_style_preset in {"cinematic_realistic", "dramatic_cinematic"} else 5.8

    num_inference_steps = int((ai_plan or {}).get("num_inference_steps", job_config.get("num_inference_steps", default_steps)))
    guidance_scale = float((ai_plan or {}).get("guidance_scale", job_config.get("guidance_scale", default_guidance)))

    addon = (ai_plan or {}).get("global_negative_prompt_addon", "")
    style_negative = video_style.get("negative_style", "")

    if video_style_preset == "cinematic_realistic":
        style_negative += (
            ", illustration, cartoon, anime, painting, watercolor, storybook, stylized, "
            "vector art, 2d art, flat color, fantasy creature, unreal scene"
        )

    if is_vertical:
        style_negative += (
            ", cropped head, cut off body, cut off feet, subject too close, zoomed in, "
            "extreme close-up, bad portrait framing, off-center subject, top cropped, bottom cropped"
        )

    global_negative_prompt = f"{BASE_NEGATIVE_PROMPT}, {style_negative}, {addon}".strip(", ")

    selected_voice = resolve_single_voice(job_config=job_config)

    plan_scenes = clean_ai_scene_list(ai_plan or {}, min_scenes=min_scenes, max_scenes=max_scenes)

    # Critical fix:
    # If AI planner returns multiple scenes, those scenes become the source of truth.
    # Otherwise fallback creates multiple chunks from the story text.
    if used_ai and len(plan_scenes) >= 2:
        scene_source = []
        for idx, scene in enumerate(plan_scenes, 1):
            fallback = initial_chunks[min(idx - 1, len(initial_chunks) - 1)] if initial_chunks else story_text
            narration = get_ai_scene_narration(scene, fallback_text=fallback)
            narration = repair_narration_language(narration, fallback, source_language)
            if not narration:
                narration = fallback
            scene_source.append({
                "chunk": narration,
                "scene_plan": scene,
            })
    else:
        fallback_chunks = initial_chunks
        if len(fallback_chunks) < 2:
            fallback_chunks = force_scene_chunks_by_words(story_text, min_scenes=min_scenes, max_scenes=max_scenes)
        scene_source = [
            {
                "chunk": chunk,
                "scene_plan": {},
            }
            for chunk in fallback_chunks[:max_scenes]
        ]

    if not scene_source:
        raise ValueError("No scene source generated")

    full_narration_text = " ".join([x["chunk"] for x in scene_source if x.get("chunk")]).strip()
    if not full_narration_text:
        full_narration_text = sanitize_tts_text(story_text, max_chars=4000)

    scene_objects = []
    for i, item in enumerate(scene_source, 1):
        chunk = str(item.get("chunk", "") or "").strip()
        raw_scene_plan = item.get("scene_plan", {}) or {}

        scene_plan = sanitize_scene_plan(raw_scene_plan, video_style_preset, chunk, is_vertical=is_vertical)
        scene_plan = validate_and_repair_scene_plan(scene_plan, chunk, is_vertical=is_vertical)
        profile = build_scene_profile_from_plan(scene_plan, i, video_style)

        # Prefer AI's explicit visual_prompt, but ground it again with structured fields.
        # This reduces generic/unrelated scenes and common AI artifacts.
        ai_visual_prompt = str(raw_scene_plan.get("visual_prompt", "") or "").strip()
        if ai_visual_prompt:
            visual_prompt = build_grounded_visual_prompt(
                base_prompt=ai_visual_prompt,
                narration=chunk,
                video_style=video_style,
                scene_plan=scene_plan,
                is_vertical=is_vertical,
            )
        else:
            visual_prompt = build_visual_prompt(
                chunk=chunk,
                video_style=video_style,
                scene_plan=scene_plan,
                style_locked=style_locked,
                is_vertical=is_vertical
            )

        scene_negative_prompt = build_scene_negative_prompt(
            global_negative_prompt=global_negative_prompt,
            scene_plan=scene_plan,
            narration=chunk,
            is_vertical=is_vertical,
        )

        raw_visual_source = str(raw_scene_plan.get("visual_source", "") or "").strip().lower()
        if raw_visual_source not in {"stock", "ai"}:
            raw_visual_source = "stock" if scene_stock_friendly(chunk, scene_plan, video_style_preset) else "ai"
        if scene_requires_ai(chunk, scene_plan, video_style_preset):
            raw_visual_source = "ai"
        stock_query = str(raw_scene_plan.get("stock_query", "") or "").strip()
        if not stock_query:
            stock_query = build_stock_query(chunk, scene_plan, is_vertical=is_vertical)

        scene_objects.append({
            "scene_id": i,
            "profile": profile,
            "voice": selected_voice,
            "voice_text": chunk,
            "visual_source": raw_visual_source,
            "stock_query": stock_query,
            "visual_prompt": visual_prompt,
            "negative_prompt": scene_negative_prompt,
            "source_chunk": chunk,
            "scene_plan": scene_plan,
            "aspect_ratio": "9:16" if is_vertical else "16:9",
        })

    return {
        "used_ai_planner": used_ai,
        "source_language": source_language,
        "video_style_preset": video_style_preset,
        "video_style": video_style,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "director_notes": director_notes,
        "global_negative_prompt": global_negative_prompt,
        "selected_voice": selected_voice,
        "scene_objects": scene_objects,
        "full_narration_text": full_narration_text,
        "is_vertical": is_vertical,
    }


def synthesize_single_tts_openai(text, out_path, voice=None):
    text = sanitize_tts_text(text)
    if not text:
        raise ValueError("voice_text is empty")

    client = get_openai_client()
    voice = normalize_openai_voice(voice)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with client.audio.speech.with_streaming_response.create(
        model=OPENAI_TTS_MODEL,
        voice=voice,
        input=text,
    ) as response:
        response.stream_to_file(out_path)

    if not os.path.exists(out_path) or os.path.getsize(out_path) <= 0:
        raise RuntimeError(f"OpenAI TTS failed with selected voice: {voice}")

    return {
        "voice_used": voice,
        "text_len": len(text),
        "segments": 1,
        "engine": "openai_tts",
        "tts_mode": "single_voice",
    }


async def synthesize_single_tts_edge(text, out_path, rate="+0%", pitch="+0Hz", edge_voice=EDGE_TTS_DEFAULT_VOICE):
    text = sanitize_tts_text(text)
    if not text:
        raise ValueError("voice_text is empty")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    communicate = edge_tts.Communicate(text=text, voice=edge_voice, rate=rate, pitch=pitch)
    await communicate.save(out_path)

    if not os.path.exists(out_path) or os.path.getsize(out_path) <= 0:
        raise RuntimeError(f"Edge TTS failed with voice: {edge_voice}")

    return {
        "voice_used": edge_voice,
        "text_len": len(text),
        "segments": 1,
        "engine": "edge_tts",
        "tts_mode": "single_voice",
    }


async def save_tts(
    text,
    out_path,
    voice=None,
    rate=None,
    pitch=None,
):
    text = sanitize_tts_text(text)
    if not text:
        raise ValueError("voice_text is empty")

    rate = rate or "+0%"
    pitch = pitch or "+0Hz"
    if str(pitch).strip() == "0Hz":
        pitch = "+0Hz"
    selected_voice = normalize_openai_voice(voice or DEFAULT_PRIMARY_VOICE)

    if OPENAI_API_KEY:
        meta = synthesize_single_tts_openai(
            text=text,
            out_path=out_path,
            voice=selected_voice
        )
        meta["segment_plan"] = [{"voice": selected_voice, "text": text}]
        return meta

    meta = await synthesize_single_tts_edge(
        text=text,
        out_path=out_path,
        rate=rate,
        pitch=pitch,
        edge_voice=EDGE_TTS_DEFAULT_VOICE,
    )
    meta["segment_plan"] = [{"voice": EDGE_TTS_DEFAULT_VOICE, "text": text}]
    return meta



def run_async_safely(coro):
    """
    Run async coroutine safely inside Runpod/Jupyter/any environment
    where an event loop may already be running.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        nest_asyncio.apply(loop)

    return loop.run_until_complete(coro)


# Legacy helper: tạo 1 audio duy nhất cho toàn video; B7 mới không dùng helper này để tránh lệch timeline
async def generate_full_narration(plan: Dict[str, Any], output_dir: str):
    narration_text = sanitize_tts_text(plan.get("full_narration_text", ""), max_chars=4000)
    if not narration_text:
        raise ValueError("full_narration_text is empty")

    narration_path = os.path.join(output_dir, "full_narration.mp3")
    meta = await save_tts(
        text=narration_text,
        out_path=narration_path,
        voice=plan.get("selected_voice", DEFAULT_PRIMARY_VOICE),
    )
    return narration_path, meta


def get_scene_duration(audio_duration, scene_profile="standard", video_length_mode="auto", max_scene_duration=12.0):
    audio_duration = float(audio_duration)
    scene_profile = (scene_profile or "standard").lower()
    mode = (video_length_mode or "auto").lower()

    base = audio_duration + 0.04

    if scene_profile in {"slow", "gentle"}:
        base += 0.08
    elif scene_profile == "dramatic":
        base += 0.02

    if mode == "compact":
        base *= 0.98
    elif mode == "slow":
        base *= 1.05

    final = clamp(base, 0.85, max_scene_duration)
    return final, scene_profile


def _load_image_rgb(image_path, width, height):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((width, height), Image.LANCZOS)
    return np.array(img)


def _crop_resized(img, scale=1.0, x=0.5, y=0.5, out_w=768, out_h=432):
    h, w = img.shape[:2]
    zoom_w = max(out_w, int(round(w / max(scale, 1e-6))))
    zoom_h = max(out_h, int(round(h / max(scale, 1e-6))))

    cx = int(x * w)
    cy = int(y * h)

    x1 = max(0, min(w - zoom_w, cx - zoom_w // 2))
    y1 = max(0, min(h - zoom_h, cy - zoom_h // 2))
    crop = img[y1:y1 + zoom_h, x1:x1 + zoom_w]
    crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
    return crop


def make_motion_clip(image_path, duration, width, height, scene_profile="gentle", fps=24, video_style=None):
    base = _load_image_rgb(image_path, width, height)
    scene_profile = (scene_profile or "gentle").lower()

    # Natural pacing: reduce excessive Ken Burns movement.
    # Old values made the video feel slow/heavy and sometimes blurred.
    if scene_profile == "dramatic":
        max_zoom = 1.045
        drift = 0.018
    elif scene_profile == "slow":
        max_zoom = 1.018
        drift = 0.006
    elif scene_profile == "gentle":
        max_zoom = 1.026
        drift = 0.008
    else:
        max_zoom = 1.032
        drift = 0.010

    def make_frame(t):
        p = 0.0 if duration <= 0 else min(1.0, max(0.0, t / duration))
        ease = 0.5 - 0.5 * math.cos(math.pi * p)

        scale = 1.0 + (max_zoom - 1.0) * ease
        x = 0.5 + drift * math.sin(2 * math.pi * (0.35 * p + 0.10))
        y = 0.5 + drift * math.cos(2 * math.pi * (0.28 * p + 0.18))

        frame = _crop_resized(base, scale=scale, x=x, y=y, out_w=width, out_h=height)

        if video_style and video_style.get("name") in {"cinematic_realistic", "dramatic_cinematic"}:
            h, w = frame.shape[:2]
            yy, xx = np.ogrid[:h, :w]
            cx, cy = w / 2.0, h / 2.0
            dist = ((xx - cx) ** 2 / (cx ** 2) + (yy - cy) ** 2 / (cy ** 2))
            # Very light vignette only; avoid dark/blurred-looking back half.
            vignette = np.clip(1.0 - 0.006 * dist, 0.994, 1.0)
            frame = np.clip(frame.astype(np.float32) * vignette[..., None], 0, 255).astype(np.uint8)

        return frame

    return VideoClip(frame_function=make_frame, duration=float(duration)).with_fps(int(fps))


# ===== CELL 7 =====
# B7 — run_job FIXED FULL (single narration audio, no overlap, with progress, safe return)

from moviepy import (
    AudioFileClip,
    concatenate_videoclips,
    vfx,
)


def get_transition_duration(scene_profile: str, video_style_preset: str) -> float:
    scene_profile = (scene_profile or "standard").lower()
    video_style_preset = (video_style_preset or "").lower()

    base = 0.18
    if scene_profile == "dramatic":
        base = 0.10
    elif scene_profile == "slow":
        base = 0.28
    elif scene_profile == "gentle":
        base = 0.22

    if video_style_preset == "zen_soft":
        base += 0.05
    elif video_style_preset == "dramatic_cinematic":
        base -= 0.03

    return clamp(base, 0.08, 0.35)


def apply_visual_finish(clip, video_style_preset: str):
    video_style_preset = (video_style_preset or "").lower()
    effects = []

    if video_style_preset in {"cinematic_realistic", "dramatic_cinematic"}:
        effects.append(vfx.GammaCorrection(1.03))
    elif video_style_preset == "zen_soft":
        effects.append(vfx.GammaCorrection(1.01))

    if effects:
        return clip.with_effects(effects)
    return clip


def apply_scene_transitions(clip, scene_profile: str, video_style_preset: str, is_last_scene: bool = False):
    td = get_transition_duration(scene_profile, video_style_preset)
    td = min(td, 0.16)
    if is_last_scene:
        # Never fade out the final scene, otherwise the back half can look dim/blurred.
        out = clip.with_effects([vfx.FadeIn(min(td, 0.10))])
    else:
        out = clip.with_effects([vfx.FadeIn(td), vfx.FadeOut(td)])
    return out, td


def estimate_eta(elapsed_sec, processed, total, fallback_sec=60):
    try:
        elapsed_sec = float(elapsed_sec)
        processed = int(processed)
        total = int(total)
    except Exception:
        return fallback_sec

    if processed <= 0 or total <= 0:
        return fallback_sec

    avg = elapsed_sec / max(processed, 1)
    remain = max(0, total - processed)
    return int(avg * remain)


def allocate_scene_durations_from_narration(scene_objects, total_audio_duration, min_scene_sec=1.6, max_scene_sec=12.0):
    """
    Chia tổng audio duration cho các scene theo độ dài text từng chunk.
    """
    if not scene_objects:
        return []

    weights = []
    for s in scene_objects:
        txt = str(s.get("voice_text", "") or s.get("source_chunk", "") or "").strip()
        w = max(1, len(txt.split()))
        weights.append(w)

    total_weight = max(1, sum(weights))
    raw = [total_audio_duration * (w / total_weight) for w in weights]
    durations = [clamp(x, min_scene_sec, max_scene_sec) for x in raw]

    # scale lại để tổng khớp audio
    current_total = sum(durations)
    if current_total > 0:
        ratio = total_audio_duration / current_total
        durations = [clamp(d * ratio, min_scene_sec, max_scene_sec) for d in durations]

    return durations



def _get_image_max_workers(job_config=None) -> int:
    """
    Configurable image generation workers.

    Practical note:
    - On one small GPU, too many workers can cause CUDA OOM.
    - Default is 1 for safety.
    - Set IMAGE_MAX_WORKERS=2 only after the container is stable with your selected image model.
    """
    raw = None
    if isinstance(job_config, dict):
        raw = job_config.get("image_max_workers")
    if raw in ("", None):
        raw = os.getenv("IMAGE_MAX_WORKERS", "1")
    try:
        workers = int(float(raw))
    except Exception:
        workers = 1
    return max(1, min(workers, 3))


def _generate_one_scene_image(scene, img_path, width, height, num_inference_steps, guidance_scale, seed):
    generate_image(
        prompt=scene["visual_prompt"],
        out_path=img_path,
        negative_prompt=scene["negative_prompt"],
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed + int(scene["scene_id"]),
        retries=1 if (_is_turbo_model() or _use_lightning_lora()) else 2,
        scene_id=scene["scene_id"],
    )
    scene["visual_used"] = "ai"
    scene["visual_file"] = os.path.basename(img_path)
    return img_path


def _prepare_one_scene_visual(scene, img_path, width, height, num_inference_steps, guidance_scale, seed):
    """Use local stock or Pexels when available; otherwise fall back to SDXL."""
    visual_source = str(scene.get("visual_source", "ai") or "ai").lower()
    is_vertical = _is_vertical_frame(width, height)

    if visual_source == "stock":
        stock_query = scene.get("stock_query") or scene.get("visual_prompt") or scene.get("voice_text") or ""

        # 1) Try local stock asset folder first, if present.
        if ENABLE_STOCK_ASSETS:
            stock = find_stock_asset(stock_query, scene.get("scene_plan", {}) or {}, is_vertical=is_vertical)
            if stock:
                print(f"🖼️ Using local stock asset for scene {int(scene.get('scene_id', 0)):02d}: {stock.get('path')} | score={stock.get('match_score')}")
                prepare_stock_image(stock["path"], img_path, width, height)
                scene["visual_used"] = "stock_local"
                scene["stock_asset_path"] = stock.get("path")
                scene["stock_match_score"] = stock.get("match_score")
                scene["visual_file"] = os.path.basename(img_path)
                return img_path

        # 2) Try Pexels API if user has no local asset folder.
        pexels = fetch_pexels_photo(stock_query, is_vertical=is_vertical)
        if pexels:
            print(f"📸 Using Pexels stock for scene {int(scene.get('scene_id', 0)):02d}: query={pexels.get('query')!r}")
            ok = prepare_pexels_image(pexels["url"], img_path, width, height)
            if ok:
                scene["visual_used"] = "stock_pexels"
                scene["stock_query_used"] = pexels.get("query", "")
                scene["pexels_id"] = pexels.get("pexels_id", "")
                scene["pexels_photographer"] = pexels.get("photographer", "")
                scene["visual_file"] = os.path.basename(img_path)
                return img_path

        print(f"ℹ️ No suitable stock/Pexels image found for scene {int(scene.get('scene_id', 0)):02d}; falling back to AI image")

    return _generate_one_scene_image(scene, img_path, width, height, num_inference_steps, guidance_scale, seed)

def run_job(job_config, job_id):
    """
    Production run_job — FIX TIMELINE DRIFT.

    Key change:
    - OLD: create one full narration audio, then estimate scene durations.
    - NEW: create one audio per scene, then each clip uses its real audio duration.

    This prevents:
    - too many scenes in the first half,
    - one last scene being held for the second half,
    - final scene fade/blur covering the back half of the video.
    """
    job_dir = os.path.join(RUNNING_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    write_status(job_dir, job_id, "running", {
        "step": "init",
        "progress_pct": 1,
        "eta_sec": 120,
    })

    IMG_DIR = os.path.join(job_dir, "images")
    AUDIO_DIR = os.path.join(job_dir, "audio")
    OUT_DIR = os.path.join(job_dir, "outputs")
    META_DIR = os.path.join(job_dir, "meta")

    for d in [IMG_DIR, AUDIO_DIR, OUT_DIR, META_DIR]:
        os.makedirs(d, exist_ok=True)
        clear_folder(d)

    story_text = str(job_config.get("story_text", "") or "").strip()
    prompt = str(job_config.get("prompt", "") or "").strip()

    if not story_text:
        story_text = prompt

    if not story_text:
        raise ValueError("Missing story_text")

    width = int(job_config.get("width", 768))
    height = int(job_config.get("height", 432))
    fps = int(job_config.get("fps", 24))
    target_words_per_scene = int(job_config.get("target_words_per_scene", 40))
    max_scene_duration = float(job_config.get("max_scene_duration", 12.0))
    seed = int(job_config.get("seed", 42) or 42)

    write_status(job_dir, job_id, "running", {
        "step": "plan_video",
        "progress_pct": 6,
        "eta_sec": 90,
    })

    adaptive_plan = create_adaptive_video_plan(
        story_text=story_text,
        job_config=job_config,
        target_words_per_scene=target_words_per_scene,
    )

    video_style_preset = adaptive_plan["video_style_preset"]
    video_style = adaptive_plan["video_style"]
    num_inference_steps = int(adaptive_plan["num_inference_steps"])
    guidance_scale = float(adaptive_plan["guidance_scale"])
    # Hard safety clamp for production mode.
    # Lightning uses very few steps and guidance 0. Normal SDXL keeps moderate guidance.
    if _is_turbo_model() or _use_lightning_lora():
        num_inference_steps = min(max(num_inference_steps, 4), 8)
        guidance_scale = max(0.0, min(guidance_scale, 1.0))
    else:
        num_inference_steps = min(max(num_inference_steps, 8), 24)
        guidance_scale = max(1.0, min(guidance_scale, 8.0))
    scene_objects = adaptive_plan["scene_objects"]
    selected_voice = adaptive_plan.get(
        "selected_voice",
        job_config.get("primary_voice") or job_config.get("voice") or "fable"
    )

    if not scene_objects:
        raise ValueError("No scenes generated in adaptive_plan")

    total_scenes = len(scene_objects)

    write_json(os.path.join(META_DIR, "adaptive_plan.json"), {
        "used_ai_planner": adaptive_plan["used_ai_planner"],
        "source_language": adaptive_plan.get("source_language", "unknown"),
        "video_style_preset": video_style_preset,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "director_notes": adaptive_plan.get("director_notes", ""),
        "global_negative_prompt": adaptive_plan["global_negative_prompt"],
        "scene_count": total_scenes,
        "selected_voice": selected_voice,
        "full_narration_text": adaptive_plan.get("full_narration_text", ""),
        "timeline_mode": "per_scene_audio",
        "image_max_workers": _get_image_max_workers(job_config),
        "image_model": SDXL_MODEL_ID,
        "image_acceleration": IMAGE_ACCELERATION,
        "stock_assets_enabled": ENABLE_STOCK_ASSETS,
        "stock_asset_dir": STOCK_ASSET_DIR,
        "lightning_lora": "",
    })

    write_json(os.path.join(META_DIR, "scene_profiles.json"), scene_objects)

    write_status(job_dir, job_id, "running", {
        "step": "generate_images",
        "scene_count": total_scenes,
        "video_style_preset": video_style_preset,
        "used_ai_planner": adaptive_plan["used_ai_planner"],
        "selected_voice": selected_voice,
        "progress_pct": 12,
        "eta_sec": max(45, total_scenes * 12),
    })

    image_paths = [None] * total_scenes
    images_started_at = time.time()
    image_max_workers = _get_image_max_workers(job_config)

    # Workers are configurable but capped to avoid CUDA OOM on 24GB GPUs.
    # For SDXL Base, IMAGE_MAX_WORKERS=1 is safest; try 2 only on larger GPUs.
    if image_max_workers <= 1:
        for idx, scene in enumerate(scene_objects):
            img_path = os.path.join(IMG_DIR, f"scene_{scene['scene_id']:02d}.png")
            image_paths[idx] = _prepare_one_scene_visual(
                scene=scene,
                img_path=img_path,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )

            processed_images = sum(1 for x in image_paths if x)
            eta_sec = estimate_eta(time.time() - images_started_at, processed_images, total_scenes, fallback_sec=max(12, total_scenes * 4))
            progress_pct = int(12 + (processed_images / max(total_scenes, 1)) * 34)

            write_status(job_dir, job_id, "running", {
                "step": "generate_images",
                "scene_count": total_scenes,
                "processed_images": processed_images,
                "image_max_workers": image_max_workers,
                "video_style_preset": video_style_preset,
                "selected_voice": selected_voice,
                "progress_pct": progress_pct,
                "eta_sec": eta_sec,
            })
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        future_to_idx = {}
        with ThreadPoolExecutor(max_workers=image_max_workers) as executor:
            for idx, scene in enumerate(scene_objects):
                img_path = os.path.join(IMG_DIR, f"scene_{scene['scene_id']:02d}.png")
                fut = executor.submit(
                    _prepare_one_scene_visual,
                    scene,
                    img_path,
                    width,
                    height,
                    num_inference_steps,
                    guidance_scale,
                    seed,
                )
                future_to_idx[fut] = idx

            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                image_paths[idx] = fut.result()

                processed_images = sum(1 for x in image_paths if x)
                eta_sec = estimate_eta(time.time() - images_started_at, processed_images, total_scenes, fallback_sec=max(12, total_scenes * 4))
                progress_pct = int(12 + (processed_images / max(total_scenes, 1)) * 34)

                write_status(job_dir, job_id, "running", {
                    "step": "generate_images",
                    "scene_count": total_scenes,
                    "processed_images": processed_images,
                    "image_max_workers": image_max_workers,
                    "video_style_preset": video_style_preset,
                    "selected_voice": selected_voice,
                    "progress_pct": progress_pct,
                    "eta_sec": eta_sec,
                })

    if any(not p for p in image_paths):
        raise RuntimeError("Some scene images were not generated")

    write_status(job_dir, job_id, "running", {
        "step": "generate_tts_per_scene",
        "scene_count": total_scenes,
        "selected_voice": selected_voice,
        "progress_pct": 50,
        "eta_sec": max(20, total_scenes * 4),
    })

    audio_clips = []
    tts_scene_log = []
    tts_started_at = time.time()

    for idx, scene in enumerate(scene_objects, start=1):
        scene_id = int(scene.get("scene_id", idx))
        voice_text = sanitize_tts_text(scene.get("voice_text") or scene.get("source_chunk") or "", max_chars=900)
        if not voice_text:
            voice_text = sanitize_tts_text(story_text, max_chars=900)

        audio_path = os.path.join(AUDIO_DIR, f"scene_{scene_id:02d}.mp3")
        tts_meta = run_async_safely(save_tts(
            text=voice_text,
            out_path=audio_path,
            voice=scene.get("voice") or selected_voice,
            rate=job_config.get("tts_rate", "+0%"),
            pitch=job_config.get("tts_pitch", "+0Hz"),
        ))

        if not os.path.exists(audio_path) or os.path.getsize(audio_path) <= 0:
            raise RuntimeError(f"Scene TTS output missing or empty: {audio_path}")

        audio_clip = AudioFileClip(audio_path)
        duration = float(audio_clip.duration or 0)
        if duration <= 0:
            try:
                audio_clip.close()
            except Exception:
                pass
            raise RuntimeError(f"Scene audio duration is zero: {audio_path}")

        if max_scene_duration and duration > max_scene_duration + 3.0:
            print(f"WARN: scene {scene_id} audio duration {duration:.2f}s is longer than max_scene_duration {max_scene_duration:.2f}s. Keeping real audio duration to avoid sync drift.")

        audio_clips.append(audio_clip)
        tts_scene_log.append({
            "scene_id": scene_id,
            "audio_file": os.path.basename(audio_path),
            "audio_duration": round(duration, 3),
            "voice_used": tts_meta.get("voice_used", selected_voice),
            "tts_engine": tts_meta.get("engine", "unknown"),
            "text_len": len(voice_text),
            "word_count": len(voice_text.split()),
            "voice_text": voice_text,
        })

        eta_sec = estimate_eta(time.time() - tts_started_at, idx, total_scenes, fallback_sec=max(10, total_scenes * 3))
        progress_pct = int(50 + (idx / max(total_scenes, 1)) * 20)

        write_status(job_dir, job_id, "running", {
            "step": "generate_tts_per_scene",
            "generated_audio": idx,
            "scene_count": total_scenes,
            "selected_voice": selected_voice,
            "progress_pct": progress_pct,
            "eta_sec": eta_sec,
        })

    total_audio_duration = sum(float(a.duration or 0) for a in audio_clips)
    write_json(os.path.join(META_DIR, "tts_debug.json"), {
        "tts_mode": "per_scene_audio",
        "scene_count": total_scenes,
        "selected_voice": selected_voice,
        "total_audio_duration": round(total_audio_duration, 3),
        "scenes": tts_scene_log,
    })

    write_status(job_dir, job_id, "running", {
        "step": "build_video",
        "selected_voice": selected_voice,
        "progress_pct": 72,
        "eta_sec": max(12, total_scenes * 2),
    })

    video_clips = []
    scene_duration_log = []
    build_started_at = time.time()

    for idx, (scene, img_path, audio_clip) in enumerate(zip(scene_objects, image_paths, audio_clips), start=1):
        scene_id = int(scene.get("scene_id", idx))
        duration = max(float(audio_clip.duration or 0), 1.0)

        vclip = make_motion_clip(
            image_path=img_path,
            duration=duration,
            width=width,
            height=height,
            scene_profile=scene["profile"],
            fps=fps,
            video_style=video_style,
        )

        vclip = apply_visual_finish(vclip, video_style_preset)

        if idx < total_scenes:
            transition_duration = min(get_transition_duration(scene["profile"], video_style_preset), 0.16)
            vclip = vclip.with_effects([vfx.FadeIn(transition_duration), vfx.FadeOut(transition_duration)])
        else:
            transition_duration = min(get_transition_duration(scene["profile"], video_style_preset), 0.12)
            vclip = vclip.with_effects([vfx.FadeIn(transition_duration)])

        vclip = vclip.with_audio(audio_clip)
        video_clips.append(vclip)

        scene_duration_log.append({
            "scene_id": scene_id,
            "final_duration": round(float(duration), 3),
            "scene_profile": scene["profile"],
            "voice_used": scene.get("voice", selected_voice),
            "transition_duration": round(float(transition_duration), 3),
            "visual_source": scene.get("visual_source", "ai"),
            "visual_used": scene.get("visual_used", "ai"),
            "stock_query": scene.get("stock_query", ""),
            "stock_asset_path": scene.get("stock_asset_path", ""),
            "visual_prompt": scene["visual_prompt"],
            "negative_prompt": scene["negative_prompt"],
            "scene_plan": scene.get("scene_plan", {}),
        })

        eta_sec = estimate_eta(time.time() - build_started_at, idx, total_scenes, fallback_sec=max(10, total_scenes * 2))
        progress_pct = int(72 + (idx / max(total_scenes, 1)) * 20)

        write_status(job_dir, job_id, "running", {
            "step": "build_video",
            "built_scenes": idx,
            "scene_count": total_scenes,
            "selected_voice": selected_voice,
            "progress_pct": progress_pct,
            "eta_sec": eta_sec,
        })

    write_json(os.path.join(META_DIR, "scene_duration_log.json"), scene_duration_log)

    write_status(job_dir, job_id, "running", {
        "step": "build_video",
        "substep": "final_render",
        "selected_voice": selected_voice,
        "progress_pct": 97,
        "eta_sec": 8,
    })

    final_video = concatenate_videoclips(video_clips, method="compose")
    final_path = os.path.join(OUT_DIR, "final_story_video_adaptive.mp4")

    final_video.write_videofile(
        final_path,
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        preset=os.getenv("FFMPEG_PRESET", "veryfast"),
    )

    for v in video_clips:
        try:
            v.close()
        except Exception:
            pass

    for a in audio_clips:
        try:
            a.close()
        except Exception:
            pass

    try:
        final_video.close()
    except Exception:
        pass

    free_memory()

    write_status(job_dir, job_id, "completed", {
        "step": "done",
        "progress_pct": 100,
        "eta_sec": 0,
        "selected_voice": selected_voice,
        "video_path": f"completed/{job_id}/outputs/final_story_video_adaptive.mp4",
        "finished_at": now_str(),
    })

    return {
        "job_id": job_id,
        "status": "completed",
        "video_path": final_path,
        "relative_video_path": f"completed/{job_id}/outputs/final_story_video_adaptive.mp4",
        "scene_count": len(scene_objects),
        "prompt": prompt,
        "story_text": story_text,
        "video_style_preset": video_style_preset,
        "selected_voice": selected_voice,
        "used_ai_planner": adaptive_plan["used_ai_planner"],
        "timeline_mode": "per_scene_audio",
        "image_max_workers": _get_image_max_workers(job_config),
        "image_model": SDXL_MODEL_ID,
        "image_acceleration": IMAGE_ACCELERATION,
        "lightning_lora": "",
        "total_audio_duration": round(total_audio_duration, 3),
        "finished_at": now_str(),
    }



# ===== SERVERLESS ENTRYPOINT =====
def run_job_serverless(job_config, job_id, base_dir="/tmp/easyai", progress_callback=None):
    """
    Production entry point called by handler.py.

    Returns a dict containing video_path so handler.py can upload it to R2.
    """
    global _PROGRESS_CALLBACK

    init_job_dirs(base_dir)

    normalized = normalize_job_config(job_config or {}, fallback_job_id=job_id)
    normalized["job_id"] = str(job_id or normalized.get("job_id") or "").strip()

    if not normalized["job_id"]:
        raise ValueError("Missing job_id")

    old_callback = _PROGRESS_CALLBACK
    _PROGRESS_CALLBACK = progress_callback

    try:
        result = run_job(normalized, normalized["job_id"])

        video_path = result.get("video_path")
        if not video_path or not os.path.exists(video_path):
            raise RuntimeError(f"run_job completed but video_path is missing: {video_path}")

        return {
            **result,
            "ok": True,
            "job_id": normalized["job_id"],
            "video_path": video_path,
        }

    finally:
        _PROGRESS_CALLBACK = old_callback