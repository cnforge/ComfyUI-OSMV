"""A 9:16 version of a finished 16:9 video, cropped around the singer.

Each shot (the stretch between two cuts of the run's cut list) gets its own crop: held still on
the main face when it stays put, panned smoothly after it when it moves, centred when there is
no face. The crop never moves across a cut by itself, so a cut stays a cut. The original audio
is copied. The crop is 9/16 of the frame's height wide (608 x 1080 from 1920 x 1080), scaled to
1080 x 1920, so it is softer than a video rendered at 9:16.
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from . import faces
from .assemble import ffmpeg_exe

OUT = (1080, 1920)
EVERY = 2          # look for faces on every other frame
HOLD = 0.12        # a face that stays within this share of the crop's width gets a still crop
MIN_HITS = 0.3     # a shot's faces count when found on this share of its sampled frames (2 at least)
SMOOTH_S = 0.5     # the moving crop's smoothing window, in seconds


def out_path(video: str) -> str:
    return os.path.splitext(video)[0] + "_to_9x16.mp4"


def _cuts(plan: dict, n: int, fps: float) -> list[int]:
    """Frame numbers where a shot starts, 0 and n included (the plan counts at its out_fps)."""
    r = fps / float(plan.get("out_fps") or fps)
    b = {0, n}
    for e in plan.get("edl") or []:
        b.update((int(round(e["f0"] * r)), int(round(e["f1"] * r))))
    return sorted(f for f in b if 0 <= f <= n)


def _lead(found: list, prev: float | None) -> float | None:
    """The face to frame: the one nearest the last pick when there is one, else the biggest."""
    if not found:
        return None
    if prev is not None:
        return min(found, key=lambda q: abs(q[0] - prev) - 0.5 * q[2])[0]
    return found[0][0]


def _track(path: str) -> tuple[list, float, int, int, int]:
    """The lead face's x for every frame (None where none was found)."""
    import cv2
    cap = cv2.VideoCapture(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    xs, k, prev = [], 0, None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if k % EVERY == 0:
            x = _lead(faces.detect_all(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), size=640), prev)
            prev = x if x is not None else prev
        else:
            x = xs[-1] if xs else None
        xs.append(x)
        k += 1
    cap.release()
    return xs, fps, w, h, k


def _path(xs: list, cuts: list[int], fps: float, half: float) -> np.ndarray:
    """The crop's centre (0..1 of the frame's width) for every frame."""
    out = np.full(len(xs), 0.5)
    win = max(1, int(SMOOTH_S * fps))
    for a, b in zip(cuts, cuts[1:]):
        seg = [xs[i] for i in range(a, min(b, len(xs)))]
        if not seg:
            continue
        idx = [i for i, x in enumerate(seg) if x is not None]
        sampled = [i for i in range(len(seg)) if (a + i) % EVERY == 0]
        hits = sum(seg[i] is not None for i in sampled)
        if hits < max(2, MIN_HITS * len(sampled)):  # no face to trust: the middle, where singers are framed
            c = np.full(len(seg), 0.5)
        else:
            c = np.interp(np.arange(len(seg)), idx, [seg[i] for i in idx])
            med = float(np.median(c))
            if np.max(np.abs(c - med)) <= HOLD * 2 * half:
                c = np.full(len(seg), med)  # the face stays put: a locked-off crop
            else:  # it moves: follow it, smoothed (edges padded so the start and end don't drift)
                k = np.ones(win) / win
                c = np.convolve(np.pad(c, (win // 2, win - 1 - win // 2), mode="edge"), k, mode="valid")
        out[a:a + len(seg)] = np.clip(c, half, 1 - half)
    return out


def reframe(run_dir: str, log=print) -> str:
    """Writes <final>_to_9x16.mp4 beside the run's cut and returns its path (reused while the cut
    hasn't changed)."""
    import cv2
    done = json.load(open(os.path.join(run_dir, "done.json"), encoding="utf-8"))
    src = done["video"]
    dst = out_path(src)
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return dst
    plan = json.load(open(os.path.join(run_dir, "plan.json"), encoding="utf-8"))
    xs, fps, w, h, n = _track(src)
    cw = int(round(h * 9 / 16)) // 2 * 2
    if cw >= w:
        raise ValueError("this video is already 9:16 or narrower")
    centre = _path(xs, _cuts(plan, n, fps), fps, cw / 2 / w)
    log(f"[OneShotMV] 9:16 crop: {sum(x is not None for x in xs[::EVERY])} of {len(xs[::EVERY])} sampled frames have a face")
    tmp = dst + ".part.mp4"
    cmd = [ffmpeg_exe(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT[0]}x{OUT[1]}",
           "-r", f"{fps}", "-i", "-", "-i", src, "-map", "0:v", "-map", "1:a?", "-c:a", "copy",
           "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
           "-shortest", tmp]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(src)
    for i in range(n):
        ok, f = cap.read()
        if not ok:
            break
        x0 = int(round(centre[i] * w - cw / 2))
        x0 = min(max(x0, 0), w - cw)
        enc.stdin.write(cv2.resize(f[:, x0:x0 + cw], OUT, interpolation=cv2.INTER_LANCZOS4).tobytes())
    cap.release()
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError("ffmpeg couldn't write the 9:16 version")
    os.replace(tmp, dst)
    return dst
