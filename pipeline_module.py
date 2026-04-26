# pipeline_module.py — Production serverless version for RunPod
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
        "speech_speed": _clamp_float(job_config.get("speech_speed"), 1.05, min_value=0.7, max_value=1.5),
        "trim_audio_start": _clamp_float(job_config.get("trim_audio_start"), 0.03, min_value=0.0, max_value=0.5),
        "trim_audio_end": _clamp_float(job_config.get("trim_audio_end"), 0.16, min_value=0.0, max_value=0.8),

        # image generation
        "num_inference_steps": _clamp_int(job_config.get("num_inference_steps"), 24, min_value=8, max_value=80),
        "guidance_scale": _clamp_float(job_config.get("guidance_scale"), 6.5, min_value=1.0, max_value=20.0),
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
from PIL import Image

SDXL_MODEL_ID = os.getenv("SDXL_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0").strip()
_IMAGE_PIPE = None

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, worst quality, low detail, noisy, jpeg artifacts, "
    "bad anatomy, deformed, distorted face, ugly eyes, bad hands, extra fingers, extra limbs, "
    "duplicate subject, cropped, cut off, watermark, text, logo, oversaturated, flat lighting, "
    "plastic skin, unrealistic skin, cartoon, anime, illustration, painting, storybook, 2d, vector art"
)


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def shorten_prompt_for_sdxl(prompt: str, max_chars: int = 380) -> str:
    prompt = " ".join((prompt or "").strip().split())
    if not prompt:
        return "clear coherent scene"

    prompt = re.sub(r"[\"“”]", "", prompt)
    prompt = prompt.replace("\n", " ")
    prompt = re.sub(r"\s+,", ",", prompt)
    prompt = re.sub(r",\s*,+", ", ", prompt).strip(" ,")

    if len(prompt) > max_chars:
        prompt = prompt[:max_chars]
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
    prompt = shorten_prompt_for_sdxl(prompt, max_chars=380)

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

    return shorten_prompt_for_sdxl(prompt, max_chars=380)


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

    return shorten_prompt_for_sdxl(negative_prompt, max_chars=360)


def _get_guidance_scale(width: int, height: int, guidance_scale: float) -> float:
    g = float(guidance_scale)
    if _is_vertical_frame(width, height):
        g = max(g, 7.2)
    return min(g, 12.0)


def _get_num_inference_steps(width: int, height: int, steps: int) -> int:
    s = max(8, int(steps))
    if _is_vertical_frame(width, height):
        s = max(s, 30)
    return min(s, 80)


def load_image_pipe(force_reload: bool = False):
    global _IMAGE_PIPE

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

    try:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True
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
    print("✅ SDXL ready")
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
                    max_chars=380
                )
                run_negative_prompt = shorten_prompt_for_sdxl(
                    run_negative_prompt + ", face too close, torso cropped, feet cropped, oversized subject",
                    max_chars=360
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
            "cinematic live action frame, dramatic lighting, photorealistic, film still, "
            "realistic proportions, high detail, strong atmosphere, expressive camera language, 35mm cinema"
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
            "ultra realistic cinematic photography, documentary realism, real life everyday scene, "
            "candid moment, modern real environment, realistic face, natural skin texture, "
            "35mm film look, shallow depth of field, natural lighting, detailed composition, "
            "high dynamic range, professional color grading"
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
    "blurry, low quality, bad anatomy, extra fingers, extra limbs, duplicate subject, "
    "deformed face, cropped, text, watermark, logo, oversaturated, messy composition, "
    "bad hands, ugly eyes, distorted body, broken perspective"
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
    style_value = str(style_value or "").strip()
    return style_value if style_value in STYLE_PRESETS else DEFAULT_STYLE_PRESET


def shorten_prompt_for_sdxl(prompt: str, max_chars: int = 380) -> str:
    prompt = " ".join((prompt or "").strip().split())
    if not prompt:
        return "clear coherent scene"

    prompt = re.sub(r"[\"“”]", "", prompt)
    prompt = prompt.replace("\n", " ")
    prompt = re.sub(r"\s+,", ",", prompt)
    prompt = re.sub(r",\s*,+", ", ", prompt).strip(" ,")

    if len(prompt) > max_chars:
        prompt = prompt[:max_chars]
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
            "ultra realistic cinematic photography",
            "documentary realism",
            "real life everyday scene",
            "candid moment",
            "natural human behavior",
            "modern real environment",
            "35mm film look",
            "shallow depth of field",
            "high dynamic range",
            "realistic face",
            "natural skin texture",
            "non-illustration",
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
    return shorten_prompt_for_sdxl(prompt, max_chars=380)


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
    if not ENABLE_AI_B6:
        raise RuntimeError("AI planner disabled")

    client = get_openai_client()

    locked_style = (locked_style or "").strip()
    style_locked = bool(locked_style)
    is_vertical = is_vertical_aspect(job_config)

    prompt = f"""
You must return ONLY valid JSON.
No markdown.
No explanation.
No code fence.

The user selected style preset from portal: {locked_style or "auto"}.
Style locked from portal: {style_locked}.
Aspect ratio requested: {"9:16 vertical" if is_vertical else "16:9 landscape"}.

Important rules:
1. If style is locked, you must keep that exact style preset.
2. Do NOT change the style preset when style is locked.
3. You are allowed to plan scenes only inside that style.
4. For cinematic_realistic, every scene must look like real-life photography, everyday life, documentary-like realism.
5. For cinematic_realistic, avoid illustration, painting, cartoon, anime, watercolor, storybook, fantasy visuals.
6. Keep scenes visually concrete and practical, not abstract.
7. Each scene should include a specific subject, a visible action, a realistic location, a camera angle, a lighting condition, and an emotional tone.
8. Prefer short, concrete, visual phrases instead of abstract ideas.
9. If the story is normal daily life, choose realistic people, realistic actions, realistic places.
10. If aspect ratio is 9:16 vertical, plan scenes with portrait-safe framing.
11. For 9:16 vertical, avoid close framing that crops the head or feet.
12. For 9:16 vertical, prefer centered subject, full body or medium-long portrait framing, enough space above and below the subject.
13. For 9:16 vertical, do NOT describe scenes as extreme close-up unless the text clearly requires facial emotion.

Return JSON with this schema:
{{
  "video_style_preset": "one of {list(STYLE_PRESETS.keys())}",
  "num_inference_steps": 24,
  "guidance_scale": 6.5,
  "global_negative_prompt_addon": "",
  "director_notes": "",
  "scenes": [
    {{
      "scene_id": 1,
      "main_subject": "short specific visual subject",
      "action": "short visible action",
      "expression": "short visible facial emotion",
      "location": "short realistic location",
      "details": ["detail 1", "detail 2"],
      "shot": "camera shot",
      "lighting": "lighting condition",
      "time_of_day": "morning / afternoon / night / etc",
      "mood": "clear emotional tone",
      "motion_mode": "gentle"
    }}
  ]
}}

Story:
{story_text}

Scene chunks:
{json.dumps(scene_chunks, ensure_ascii=False)}
"""

    resp = client.responses.create(model=OPENAI_MODEL, input=prompt)
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

    return data


def create_adaptive_video_plan(
    story_text: str,
    job_config: Dict[str, Any],
    target_words_per_scene: int = 40
) -> Dict[str, Any]:
    scene_chunks = chunk_story(story_text, target_words=target_words_per_scene)
    if not scene_chunks:
        raise ValueError("No scene chunks generated")

    used_ai = False
    ai_plan = None
    director_notes = ""

    portal_style = (
        str(job_config.get("video_style_preset", "") or "").strip()
        or str(job_config.get("style", "") or "").strip()
    )

    style_locked = is_style_locked(job_config)
    is_vertical = is_vertical_aspect(job_config)

    if ENABLE_AI_B6 and OPENAI_API_KEY:
        try:
            ai_plan = plan_video_with_ai(
                story_text=story_text,
                scene_chunks=scene_chunks,
                job_config=job_config,
                locked_style=portal_style if style_locked else ""
            )
            used_ai = True
            director_notes = ai_plan.get("director_notes", "")
        except Exception as e:
            print("⚠️ AI planner fallback:", repr(e))

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

    default_steps = 28 if video_style_preset in {"cinematic_realistic", "dramatic_cinematic"} else 24
    default_guidance = 7.2 if (video_style_preset in {"cinematic_realistic", "dramatic_cinematic"} and is_vertical) else (
        7.0 if video_style_preset in {"cinematic_realistic", "dramatic_cinematic"} else 6.5
    )

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

    plan_scenes = (ai_plan or {}).get("scenes", [])
    selected_voice = resolve_single_voice(job_config=job_config)

    # 🔥 FULL narration text cho toàn video
    full_narration_text = " ".join(scene_chunks).strip()
    if not full_narration_text:
        full_narration_text = sanitize_tts_text(story_text, max_chars=4000)

    scene_objects = []
    for i, chunk in enumerate(scene_chunks, 1):
        raw_scene_plan = plan_scenes[i - 1] if i - 1 < len(plan_scenes) else {}
        scene_plan = sanitize_scene_plan(raw_scene_plan, video_style_preset, chunk, is_vertical=is_vertical)
        profile = build_scene_profile_from_plan(scene_plan, i, video_style)

        scene_objects.append({
            "scene_id": i,
            "profile": profile,
            "voice": selected_voice,
            "voice_text": chunk,   # giữ để debug / fallback
            "visual_prompt": build_visual_prompt(
                chunk=chunk,
                video_style=video_style,
                scene_plan=scene_plan,
                style_locked=style_locked,
                is_vertical=is_vertical
            ),
            "negative_prompt": global_negative_prompt,
            "source_chunk": chunk,
            "scene_plan": scene_plan,
            "aspect_ratio": "9:16" if is_vertical else "16:9",
        })

    return {
        "used_ai_planner": used_ai,
        "video_style_preset": video_style_preset,
        "video_style": video_style,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "director_notes": director_notes,
        "global_negative_prompt": global_negative_prompt,
        "selected_voice": selected_voice,
        "scene_objects": scene_objects,

        # 🔥 NEW: dùng cho B7 tạo 1 file audio duy nhất
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


# 🔥 NEW: helper để B7 gọi 1 audio duy nhất cho toàn video
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

    if scene_profile == "dramatic":
        max_zoom = 1.12
        drift = 0.07
    elif scene_profile == "slow":
        max_zoom = 1.05
        drift = 0.03
    elif scene_profile == "gentle":
        max_zoom = 1.07
        drift = 0.04
    else:
        max_zoom = 1.09
        drift = 0.05

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
            vignette = np.clip(1.0 - 0.12 * dist, 0.86, 1.0)
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


def apply_scene_transitions(clip, scene_profile: str, video_style_preset: str):
    td = get_transition_duration(scene_profile, video_style_preset)
    out = clip.with_effects([
        vfx.FadeIn(td),
        vfx.FadeOut(td),
    ])
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


def run_job(job_config, job_id):
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

    # ✅ FIX: fallback về prompt như B7 cũ
    if not story_text:
        story_text = prompt

    if not story_text:
        raise ValueError("Missing story_text")

    width = int(job_config.get("width", 768))
    height = int(job_config.get("height", 432))
    fps = int(job_config.get("fps", 24))
    target_words_per_scene = int(job_config.get("target_words_per_scene", 40))
    target_total_video_sec = job_config.get("target_total_video_sec", job_config.get("target_total_sec", None))
    max_scene_duration = float(job_config.get("max_scene_duration", 12.0))
    video_length_mode = str(job_config.get("video_length_mode", job_config.get("duration_mode", "auto")) or "auto").strip()
    seed = int(job_config.get("seed", 42) or 42)

    # =========================
    # 1. PLAN
    # =========================
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
        "video_style_preset": video_style_preset,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "director_notes": adaptive_plan.get("director_notes", ""),
        "global_negative_prompt": adaptive_plan["global_negative_prompt"],
        "scene_count": total_scenes,
        "selected_voice": selected_voice,
        "full_narration_text": adaptive_plan.get("full_narration_text", ""),
    })

    write_json(os.path.join(META_DIR, "scene_profiles.json"), scene_objects)

    # =========================
    # 2. GENERATE IMAGES
    # =========================
    write_status(job_dir, job_id, "running", {
        "step": "generate_images",
        "scene_count": total_scenes,
        "video_style_preset": video_style_preset,
        "used_ai_planner": adaptive_plan["used_ai_planner"],
        "selected_voice": selected_voice,
        "progress_pct": 12,
        "eta_sec": max(45, total_scenes * 12),
    })

    image_paths = []
    images_started_at = time.time()

    for scene in scene_objects:
        img_path = os.path.join(IMG_DIR, f"scene_{scene['scene_id']:02d}.png")

        generate_image(
            prompt=scene["visual_prompt"],
            out_path=img_path,
            negative_prompt=scene["negative_prompt"],
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed + scene["scene_id"],
            retries=2,
            scene_id=scene["scene_id"],
        )

        image_paths.append(img_path)

        processed_images = len(image_paths)
        eta_sec = estimate_eta(
            time.time() - images_started_at,
            processed_images,
            total_scenes,
            fallback_sec=max(20, total_scenes * 8),
        )
        progress_pct = int(12 + (processed_images / max(total_scenes, 1)) * 38)

        write_status(job_dir, job_id, "running", {
            "step": "generate_images",
            "scene_count": total_scenes,
            "processed_images": processed_images,
            "video_style_preset": video_style_preset,
            "selected_voice": selected_voice,
            "progress_pct": progress_pct,
            "eta_sec": eta_sec,
        })

    # =========================
    # 3. GENERATE ONE NARRATION AUDIO
    # =========================
    write_status(job_dir, job_id, "running", {
        "step": "generate_tts",
        "scene_count": total_scenes,
        "selected_voice": selected_voice,
        "progress_pct": 55,
        "eta_sec": max(20, total_scenes * 4),
    })

    narration_path, narration_meta = run_async_safely(
        generate_full_narration(adaptive_plan, AUDIO_DIR)
    )

    if not os.path.exists(narration_path) or os.path.getsize(narration_path) <= 0:
        raise RuntimeError(f"Full narration output missing or empty: {narration_path}")

    narration_clip = AudioFileClip(narration_path)
    total_audio_duration = float(narration_clip.duration or 0)

    if total_audio_duration <= 0:
        raise RuntimeError("Narration audio duration is zero")

    tts_debug = {
        "tts_mode": "single_full_narration",
        "voice_used": narration_meta.get("voice_used"),
        "tts_engine": narration_meta.get("engine", "unknown"),
        "text_len": narration_meta.get("text_len"),
        "audio_file": os.path.basename(narration_path),
        "audio_duration": round(total_audio_duration, 3),
        "scene_count": total_scenes,
    }
    write_json(os.path.join(META_DIR, "tts_debug.json"), tts_debug)

    # =========================
    # 4. BUILD VIDEO CLIPS (NO PER-SCENE AUDIO)
    # =========================
    write_status(job_dir, job_id, "running", {
        "step": "build_video",
        "selected_voice": selected_voice,
        "progress_pct": 80,
        "eta_sec": max(12, total_scenes * 2),
    })

    scene_durations = allocate_scene_durations_from_narration(
        scene_objects=scene_objects,
        total_audio_duration=total_audio_duration,
        min_scene_sec=1.6,
        max_scene_sec=max_scene_duration,
    )

    # nếu user vẫn ép total duration, scale lại theo target
    if target_total_video_sec is not None and scene_durations:
        current_total = sum(scene_durations)
        if current_total > 0:
            ratio = float(target_total_video_sec) / current_total
            scene_durations = [
                clamp(d * ratio, 1.0, max_scene_duration + 2.0)
                for d in scene_durations
            ]

    video_clips = []
    scene_duration_log = []
    build_started_at = time.time()

    for idx, (scene, img_path, duration) in enumerate(zip(scene_objects, image_paths, scene_durations), start=1):
        duration = max(float(duration), 1.0)

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
        vclip, transition_duration = apply_scene_transitions(
            vclip,
            scene_profile=scene["profile"],
            video_style_preset=video_style_preset,
        )

        video_clips.append(vclip)
        scene_duration_log.append({
            "scene_id": scene["scene_id"],
            "final_duration": round(float(duration), 2),
            "scene_profile": scene["profile"],
            "voice_used": scene.get("voice", selected_voice),
            "transition_duration": round(float(transition_duration), 3),
            "visual_prompt": scene["visual_prompt"],
            "negative_prompt": scene["negative_prompt"],
            "scene_plan": scene.get("scene_plan", {}),
        })

        eta_sec = estimate_eta(
            time.time() - build_started_at,
            idx,
            total_scenes,
            fallback_sec=max(10, total_scenes * 2),
        )
        progress_pct = int(80 + (idx / max(total_scenes, 1)) * 12)

        write_status(job_dir, job_id, "running", {
            "step": "build_video",
            "built_scenes": idx,
            "scene_count": total_scenes,
            "selected_voice": selected_voice,
            "progress_pct": progress_pct,
            "eta_sec": eta_sec,
        })

    write_json(os.path.join(META_DIR, "scene_duration_log.json"), scene_duration_log)

    # =========================
    # 5. CONCAT VIDEO + ATTACH ONE AUDIO
    # =========================
    write_status(job_dir, job_id, "running", {
        "step": "build_video",
        "substep": "final_render",
        "selected_voice": selected_voice,
        "progress_pct": 97,
        "eta_sec": 8,
    })

    final_video = concatenate_videoclips(video_clips, method="compose")
    final_video = final_video.with_audio(narration_clip)

    final_path = os.path.join(OUT_DIR, "final_story_video_adaptive.mp4")

    final_video.write_videofile(
        final_path,
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        threads=2,
        preset="medium",
    )

    try:
        narration_clip.close()
    except Exception:
        pass

    for v in video_clips:
        try:
            v.close()
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