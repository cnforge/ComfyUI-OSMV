"""Output formats and render presets.

LTX latents are 1/32 of the pixel size and the multi-stage pipeline upsamples the latent x2
per stage. The union-control IC-LoRA (pose guides for twins and motion) encodes its guide at
half the stage-1 size, so the stage-1 size must be a multiple of 64 (an odd stage-1 latent
crashes the guide encode). Render sizes are therefore a touch wider than the exact aspect;
the assembler centre-crops to the delivery aspect before scaling.
Sizes are given as the final (last-stage) render size."""
from __future__ import annotations

import math

FORMATS = {
    # name: (render w, h at quality), (render w, h at fast), delivery (w, h). Quality renders at
    # 1080p and above (stage 1 at half that), so a 1080p cut is made from real detail.
    "16:9": {"quality": (1920, 1152), "fast": (896, 512), "deliver": (1920, 1080)},
    "9:16": {"quality": (1152, 1920), "fast": (512, 896), "deliver": (1080, 1920)},
    "1:1": {"quality": (1536, 1536), "fast": (768, 768), "deliver": (1080, 1080)},
}
# MiniMax H3 renders on its own 32-px canvas, 768 on the short side (its native size), or 576
# at fast. Quality finishes at 1088 on the short side: the first steps run at the native size,
# then H3's latent upscaler lifts the clip and the last steps add the detail (SelfLift, see
# graphs.h3_render). H3_NATIVE is the size those first steps run at.
_H3 = {"16:9": ((1920, 1088), (1024, 576)), "9:16": ((1088, 1920), (576, 1024)), "1:1": ((1088, 1088), (640, 640))}
H3_NATIVE = {"16:9": (1344, 768), "9:16": (768, 1344), "1:1": (768, 768)}
for _k, _f in FORMATS.items():
    _f["fast_voice"] = _f["fast"]  # older runs' singing-at-fast preset
    _f["max"] = _f["quality"]  # Max: Quality's size on LTX-2.5's dev model
    _f["h3"], _f["h3_fast"] = _H3[_k]
# what Quality rendered at before 1080p: runs planned then (no "render_wh" in their plan) are
# cut and re-shot at these sizes
LEGACY_SIZES = {"16:9": {"quality": (1280, 768), "max": (1280, 768), "h3": (1344, 768)},
                "9:16": {"quality": (768, 1280), "max": (768, 1280), "h3": (768, 1344)},
                "1:1": {"quality": (1024, 1024), "max": (1024, 1024), "h3": (768, 768)}}
HIRES_MIN_GB = 40  # 1080p Quality needs this much GPU memory; smaller cards render at LEGACY_SIZES
FORMAT_NAMES = list(FORMATS)  # 16:9 first: the default

PRESETS = {
    # LTX-2.5 22B distilled, two stages (1/2 -> full), 25 fps: quality at the full size, fast at
    # the smaller one. LTX-2.5 lips lead their audio by only ~0.12 s (peak over 11 clips), so a
    # small nudge; Take QC measures each take with SyncNet and uses its own offset where it can.
    "quality": {"stages": 2, "fps": 25, "engine": "ltx25_distilled", "sync_delay": 0.05},
    "fast": {"stages": 2, "fps": 25, "engine": "ltx25_distilled", "sync_delay": 0.05},
    # Max: stage 1 on LTX-2.5's full dev model (20 steps at cfg 4), the upscale stage on dev with
    # the distilled LoRA (LTX's own dev recipe). Cleaner faces and motion, about 4x Quality's time.
    "max": {"stages": 2, "fps": 25, "engine": "ltx25_dev", "sync_delay": 0.05},
    # MiniMax H3 (fv/h3.py): one pass of the 8-step reference model at 24 fps, the clip's vocal
    # copied 1:1 as its soundtrack. Its lip offset isn't known yet: SyncNet measures every take.
    "h3": {"stages": 1, "fps": 24, "engine": "h3", "sync_delay": 0.0},
    "h3_fast": {"stages": 1, "fps": 24, "engine": "h3", "sync_delay": 0.0},
    # older runs (LTX-2.3, retired): only their assembly and re-shoots read these
    "fast_voice": {"stages": 2, "fps": 25, "engine": "ltx25_distilled", "sync_delay": 0.16},
}
LEGACY_SYNC = {"quality": 0.16, "fast_voice": 0.16}  # LTX-2.3's lips led by 0.22-0.26 s
ENGINES = {"ltx25": "LTX-2.5", "h3": "MiniMax H3"}
PRESET_NAMES = ["quality", "fast", "max"]  # what users pick; with the engine they make the render preset


def effective_preset(mode: str, preset: str, engine: str = "ltx25") -> str:
    """The render preset for a speed (max / quality / fast) on an engine (ltx25 / h3). Motion runs
    (pose-guided dance clips) stay on LTX-2.5. H3 has no Max: its full model looked no better."""
    if engine == "h3" and mode == "voice":
        return "h3_fast" if preset == "fast" else "h3"
    return preset if preset in ("quality", "fast", "max") else "quality"


def engine_of(preset: str) -> str:
    return "h3" if PRESETS.get(preset, {}).get("engine") == "h3" else "ltx25"


def base_sync_delay(plan: dict) -> float:
    """The cut's lip-sync shift for clips without a measured one (runs made before the engine
    choice rendered on LTX-2.3, whose lips led by more)."""
    if "sync_delay" in plan:
        return float(plan["sync_delay"])
    if "engine" not in plan and plan.get("preset") in LEGACY_SYNC:
        return LEGACY_SYNC[plan["preset"]]
    return float(PRESETS.get(plan.get("preset"), {}).get("sync_delay", 0.0))


def render_size(fmt: str, preset: str, plan: dict | None = None) -> tuple[int, int]:
    """The clips' size: a run's own (its plan's "render_wh"), else today's for the preset. A
    plan from before 1080p Quality gets the size its clips were rendered at."""
    if plan is not None:
        if plan.get("render_wh"):
            return tuple(plan["render_wh"])
        return LEGACY_SIZES.get(fmt, {}).get(preset) or FORMATS[fmt][preset]
    return FORMATS[fmt][preset]


def stage_sizes(fmt: str, preset: str, size: tuple[int, int] | None = None) -> list[tuple[int, int]]:
    """Sizes of every stage, smallest first (`size`: the last stage's, else the preset's)."""
    w, h = size or render_size(fmt, preset)
    n = PRESETS[preset]["stages"]
    out = [(w >> (n - 1 - i), h >> (n - 1 - i)) for i in range(n)]
    for (sw, sh) in out[:1] if PRESETS[preset]["engine"] != "h3" else []:
        assert sw % 64 == 0 and sh % 64 == 0, f"stage-1 size {sw}x{sh} not a multiple of 64"
    return out


RESOLUTIONS = (720, 1080, 1440, 2160)  # the delivery's short side
FRAME_RATES = (24, 25, 30, 50)  # 25 and 50 divide the 50 fps clips evenly; 24 and 30 now and then repeat a frame


def deliver_size(fmt: str, resolution: int | None = None) -> tuple[int, int]:
    """The delivery size: 1080 on the short side, or `resolution` (even sizes, same aspect)."""
    w, h = FORMATS[fmt]["deliver"]
    if not resolution or resolution == min(w, h):
        return w, h
    k = resolution / min(w, h)
    return int(round(w * k / 2)) * 2, int(round(h * k / 2)) * 2


def deliver_crop(fmt: str, preset: str, plan: dict | None = None) -> tuple[int, int, int, int]:
    """(x, y, w, h) centre crop of a render frame with the delivery aspect."""
    rw, rh = render_size(fmt, preset, plan)
    dw, dh = deliver_size(fmt)
    if rw * dh > dw * rh:  # render is wider than delivery
        w = int(round(rh * dw / dh / 2)) * 2
        return (rw - w) // 2, 0, w, rh
    h = int(round(rw * dh / dw / 2)) * 2
    return 0, (rh - h) // 2, rw, h


H3_FPS = 24
H3_MIN_FRAMES = 107  # 17k+5 frames at 24 fps: 4.46 s, H3's shortest reliable clip (4-15 s)


def ltx_frames(seconds: float, fps: float) -> int:
    """Smallest valid frame count covering `seconds`: 8k+1 for LTX; for MiniMax H3 (the only
    engine at 24 fps) 17k+5 and at least H3_MIN_FRAMES (a shorter shot uses the clip's start)."""
    n = max(1, int(math.ceil(seconds * fps - 1e-6)))
    if int(round(fps)) == H3_FPS:
        n = max(n, H3_MIN_FRAMES)
        return n + (5 - n % 17) % 17
    return 8 * max(1, math.ceil((n - 1) / 8)) + 1


def portrait_size(fmt: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """Krea 2 pass-1 size (about 1 MP) and redetail size for a format."""
    return {"9:16": ((768, 1344), (1152, 2016)),
            "1:1": ((1024, 1024), (1536, 1536)),
            "16:9": ((1344, 768), (2016, 1152))}[fmt]
