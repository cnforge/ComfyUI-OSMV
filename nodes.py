"""OneShot MV ComfyUI nodes."""
from __future__ import annotations

import importlib
import json
import os
import random

import numpy as np
import torch

import folder_paths

from . import fv as _fvpkg  # noqa: F401
from .fv import assemble as fv_assemble
from .fv import audiofx, beat, casting, director, export, faces, finish, formats, fx, graphs, lipsync, plan, pose, qc, stems, dsplan
from .fv import theme as fv_theme
from .fv import broll as fv_broll
from .fv import identity as fv_identity
from .fv import critic as fv_critic
from .fv import direction as fv_direction
from .fv import song as fv_song
from .fv import film as fv_film

CATEGORY = "OneShot MV"
_MODULES = [beat, dsplan, formats, casting, faces, plan, pose, graphs, fv_theme, fv_direction, fv_broll, fv_identity, fv_critic, fv_song, fv_film, stems, audiofx, fx, finish, export, lipsync, qc, fv_assemble, director]


def _reload():
    """Pick up edits to the core without restarting ComfyUI (cheap; modules are small)."""
    if os.environ.get("FV_NO_RELOAD"):
        return
    for m in _MODULES:
        try:
            importlib.reload(m)
        except Exception as e:  # keep the old module if an edit is broken
            print(f"[OneShotMV] reload {m.__name__} failed: {e}")


def _resolve(path: str) -> str:
    path = (path or "").strip().strip('"')
    if not path:
        return path
    if os.path.isabs(path):
        return path
    for base in (folder_paths.get_input_directory(), folder_paths.get_output_directory()):
        p = os.path.join(base, path)
        if os.path.exists(p):
            return p
    return os.path.join(folder_paths.get_input_directory(), path)


def _file_ref(path: str) -> dict:
    """A file under the output folder as a ComfyUI ui entry (so API / SDK clients can fetch it)."""
    rel = os.path.relpath(path, folder_paths.get_output_directory())
    sub, name = os.path.split(rel)
    return {"filename": name, "subfolder": sub.replace(os.sep, "/"), "type": "output"}


def _inputs_dir() -> str:
    d = os.path.join(folder_paths.get_output_directory(), "oneshot-mv", "inputs")
    os.makedirs(d, exist_ok=True)
    return d


def _save_image_input(image) -> str:
    """IMAGE tensor -> PNG under output/oneshot-mv/inputs (named by content, so reruns reuse it)."""
    import hashlib
    from PIL import Image
    arr = (image[0].detach().cpu().float().clamp(0, 1).numpy() * 255 + 0.5).astype(np.uint8)
    path = os.path.join(_inputs_dir(), f"artist_{hashlib.sha1(arr.tobytes()).hexdigest()[:12]}.png")
    if not os.path.exists(path):
        Image.fromarray(arr).save(path)
    return path


def _video_input_path(video) -> str:
    """VIDEO input -> a file path the render graphs can read."""
    try:
        src = video.get_stream_source()
        if isinstance(src, str) and os.path.exists(src):
            return src
    except Exception:
        pass
    import time
    path = os.path.join(_inputs_dir(), f"dance_{int(time.time())}.mp4")
    video.save_to(path)
    return path


# ------------------------------------------------------------------ loaders
class FVLoadImagePath:
    """Load an image from a path. Unlike Load Image it does not check the file when the job is
    queued, so it can read a file an earlier job in the same queue will write."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load"
    CATEGORY = CATEGORY + "/io"

    def load(self, path):
        from PIL import Image, ImageOps
        im = ImageOps.exif_transpose(Image.open(_resolve(path))).convert("RGB")
        return (torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)[None],)

    @classmethod
    def IS_CHANGED(cls, path):
        p = _resolve(path)
        return os.path.getmtime(p) if os.path.exists(p) else random.random()


class FVLoadVideoPath:
    """Frames of a video from a path (no check at queue time). Center-cropped to width x height
    (0 = keep), resampled to force_fps (0 = keep), starting at start_seconds."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {"default": ""}),
                             "width": ("INT", {"default": 0, "min": 0, "max": 8192}),
                             "height": ("INT", {"default": 0, "min": 0, "max": 8192}),
                             "frame_cap": ("INT", {"default": 0, "min": 0, "max": 100000}),
                             "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.001}),
                             "force_fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01})}}

    RETURN_TYPES = ("IMAGE", "FLOAT", "INT")
    RETURN_NAMES = ("images", "fps", "frame_count")
    FUNCTION = "load"
    CATEGORY = CATEGORY + "/io"

    def load(self, path, width, height, frame_cap, start_seconds, force_fps):
        import cv2
        p = _resolve(path)
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise FileNotFoundError(p)
        sfps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ofps = force_fps or sfps
        frames, want = [], frame_cap or 10 ** 9
        src = []
        first = int(round(start_seconds * sfps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        idx = first
        while True:
            ok, f = cap.read()
            if not ok:
                break
            src.append(f)
            idx += 1
            if len(src) / sfps > (want + 2) / ofps + 0.5:
                break
        cap.release()
        if not src:
            raise ValueError(f"no frames read from {p} at {start_seconds}s")
        k = 0
        while len(frames) < want:
            si = int(round(k * sfps / ofps))
            if si >= len(src):
                if frame_cap and len(frames) < frame_cap:
                    si = len(src) - 1  # hold the last frame to reach the requested length
                else:
                    break
            f = src[si]
            if width and height:
                f = fv_assemble.cover(f, (width, height), cv2)
            frames.append(f[:, :, ::-1])
            k += 1
        arr = np.stack(frames).astype(np.float32) / 255.0
        return (torch.from_numpy(arr), float(ofps), len(frames))

    @classmethod
    def IS_CHANGED(cls, path, **kw):
        p = _resolve(path)
        return os.path.getmtime(p) if os.path.exists(p) else random.random()


class FVLoadAudioPath:
    """Audio from a WAV path (seek + duration; silence past the end)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {"default": ""}),
                             "seek_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.001}),
                             "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.001})}}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "load"
    CATEGORY = CATEGORY + "/io"

    def load(self, path, seek_seconds, duration):
        p = _resolve(path)
        if p.lower().endswith(".wav"):
            x, sr = stems.read_wav(p, seek_seconds, duration)
        else:
            import torchaudio
            w, sr = torchaudio.load(p)
            a = int(seek_seconds * sr)
            b = a + int(duration * sr) if duration > 0 else w.shape[1]
            x = w[:, a:b].numpy()
        return ({"waveform": torch.from_numpy(np.ascontiguousarray(x, np.float32))[None], "sample_rate": int(sr)},)


# ------------------------------------------------------------------ the app
class FVDirector:
    """OneShot MV in one press: analyses the song, plans the cut in the chosen style pack,
    casts the performers and queues every render (grouped by model) plus the final assembly."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "mode": (["voice", "motion"], {"default": "voice"}),
            "pack": (casting.list_packs(), {"default": "crowd"}),
            "format": (formats.FORMAT_NAMES, {"default": "16:9"}),
            "preset": (formats.PRESET_NAMES, {"default": "quality"}),
            "seconds": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 600.0, "step": 0.5,
                                  "tooltip": "Length of the edit; 0 = the whole song"}),
            "start": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 3600.0, "step": 0.01,
                                "tooltip": "Song time to start at; -1 = pick the strongest stretch"}),
            "seed": ("INT", {"default": 1, "min": 0, "max": 2 ** 31 - 1}),
            "twins": (["auto", "loud", "all", "off"], {"default": "auto",
                      "tooltip": "Pose-matched twins for flash inserts (voice mode)"}),
            "queue_renders": ("BOOLEAN", {"default": True,
                              "tooltip": "Off = preview: plans the video and lists every shot (time, energy, who, "
                                         "wearing what, where) in the summary, rendering nothing. Check it, edit "
                                         "the World line if needed, then turn this on and press Queue again (the "
                                         "song analysis and the World are cached, so it starts faster)"}),
        }, "optional": {
            "theme": ("STRING", {"default": "", "multiline": True,
                      "tooltip": "The world of the whole video in a line or two, e.g. 'astronauts and ground crew on "
                                 "a 1970s space station, warm orange and teal light'. It sets who the characters "
                                 "are, what they wear, the places, the light and the colour grade; the pack still "
                                 "sets the edit. Empty = the pack's own looks. Expanded once by Gemma 3 (the "
                                 "LTX-2.3 text encoder) and cached."}),
            "direction": ("STRING", {"default": "", "multiline": True,
                          "tooltip": "How the video changes over the song, a line per part: 'Opening: quiet close-ups', "
                                     "'Verses: ...', 'Pre-choruses: ...', 'Choruses: ...', 'Bridge: ...', 'Final chorus "
                                     "and outro: ...', or a time range like '1:12-1:30: ...'. A line with no part goes "
                                     "on every shot. Each shot's notes shape its portrait; sung clips keep the "
                                     "performance alone (better lip sync), cutaways get the gist."}),
            "refine_details": ("BOOLEAN", {"default": False,
                               "tooltip": "LTX-2.5 at Quality, Max or Ultra: after the take checks, every clip in the cut is "
                                          "re-rendered with Lightricks' Refine-Details IC-LoRA (tiled, 8 steps) for "
                                          "sharper texture, edges and grain. Adds time to every clip."}),
            "draft": ("BOOLEAN", {"default": False,
                      "tooltip": "A draft: the chosen model and tier at a small size (960x544 on MiniMax H3, Fast's size "
                                 "on LTX-2.5), no RelayVSR or refine, to check the cast, shots and motion before the full "
                                 "render. The same seed gives the same stills and prompts."}),
            "broll_prompts": ("STRING", {"default": "", "multiline": True,
                              "tooltip": "Your own b-roll cutaways, one per line (a product shot, a detail, a place): used "
                                         "in order for every cutaway slot (again from the top when there are more slots "
                                         "than lines), as written; brands you name are kept. Empty: the cutaways come "
                                         "from the World (broll_style). broll_share sets how many."}),
            "h3_upscale": (["auto", "relayvsr"], {"default": "auto",
                           "tooltip": "MiniMax H3 above its native size: auto lifts the clip with SelfLift (or scales it "
                                      "up); relayvsr renders at the native size and upscales every clip 2x with RelayVSR."}),
            "image_model": (["krea", "h3_image"], {"default": "krea",
                            "tooltip": "The stills (singer portraits, b-roll stills) and the artist photo edits: Krea 2 "
                                       "(and Qwen-Image 2.1 for the photo edits), or MiniMax H3 Image for both."}),
            "dance_video_path": ("STRING", {"default": "", "tooltip": "Motion mode: path to the dance clip"}),
            "run_name": ("STRING", {"default": ""}),
            "twin_editor": (["klein", "qwen"], {"default": "klein"}),
            "guide_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            "queue_assemble": ("BOOLEAN", {"default": True,
                               "tooltip": "Also queue the final Assemble (off when a script assembles itself)"}),
            "framing": (["mixed", "close"], {"default": "mixed",
                        "tooltip": "mixed = wide, medium and close shots in quiet stretches, tighter in the "
                                   "build, close-ups on the drop; close = close-ups only"}),
            "artist_image_path": ("STRING", {"default": "",
                                  "tooltip": "One-artist mode: a photo of the singer (uses the artist pack). "
                                             "Several photos joined with | cast the singers in turn"}),
            "artist_image": ("IMAGE", {"tooltip": "One-artist mode: connect a Load Image with the singer's photo "
                                                  "(instead of artist_image_path)"}),
            "dance_video": ("VIDEO", {"tooltip": "Motion mode: connect a Load Video with the dance clip "
                                                 "(instead of dance_video_path)"}),
            "broll_share": ("FLOAT", {"default": 0.125, "min": 0.0, "max": 0.5, "step": 0.025,
                            "tooltip": "Voice mode: share of shots replaced by b-roll cutaways (nobody singing): "
                                       "first where lip sync stayed weak, then in instrumental gaps; never on the "
                                       "drop. 0 = off"}),
            "broll_style": (["mix", "world", "dance"], {"default": "mix",
                            "tooltip": "mix: a World character in a candid moment over weak lip sync, performers "
                                       "dancing (medium shots, not singing) in instrumental gaps; world or dance: "
                                       "one kind everywhere"}),
            "fps": (["auto", "24", "25", "30", "50"], {"default": "auto",
                    "tooltip": "The video's frame rate. auto = 25 (the clips render at 50 fps at quality, "
                               "25 at fast); 25 and 50 are the smoothest, 24 and 30 now and then repeat a frame"}),
            "resolution": (["1080", "720", "1440", "2160"], {"default": "1080",
                           "tooltip": "The video's short side in pixels (1080 = 1920x1080 at 16:9). The clips are "
                                      "scaled up to it, so 1440 and 2160 are bigger files, not more detail"}),
            "qc_retries": ("INT", {"default": 2, "min": 0, "max": 3,
                           "tooltip": "Check every take (missing, frozen, blown out, dead mouth, a cut or drift away "
                                      "from the shot, and lip sync with SyncNet if its weights are installed) and "
                                      "re-roll bad ones, up to this many rounds and at most doubling the render time; "
                                      "0 = off"}),
            "staged": ("BOOLEAN", {"default": True,
                       "tooltip": "Voice runs queued here: load each model as few times as possible (Krea for "
                                  "every image, LTX for every singer clip and its QC, Klein for every look, LTX "
                                  "for every look clip) and make the looks only from the final singer takes"}),
            "cuts_low": (plan.CUT_CHOICES, {"default": plan.CUT_DEFAULTS["low"],
                         "tooltip": "How low-energy sections (intro, breaks, a quiet outro) are cut"}),
            "cuts_medium": (plan.CUT_CHOICES, {"default": plan.CUT_DEFAULTS["medium"],
                            "tooltip": "How medium-energy sections (verses, the build into a drop) are cut"}),
            "cuts_high": (plan.CUT_CHOICES, {"default": plan.CUT_DEFAULTS["high"],
                          "tooltip": "How high-energy sections (choruses, drops) are cut. 'To the music' cuts "
                                     "on the beat to a new face in the same spot: subtle about every half bar, "
                                     "intense about every beat"}),
            "audio_fx": (["none", "subtle", "intense"], {"default": "subtle",
                         "tooltip": "Effects that hit with the drums: flash, zoom punch, glow, colour split. "
                                    "subtle tones them down; intense is stronger and adds a twitch (jumps of up "
                                    "to 8 px with motion blur) on the big hits; none leaves the picture alone "
                                    "(the colour grade and film grain still apply)"}),
            "look_groups": ("BOOLEAN", {"default": False,
                            "tooltip": "Experimental, with staged: each drop group's looks come from one LTX clip "
                                       "that switches person on the beat. Faster when it works; looks it doesn't "
                                       "deliver cleanly get their own clip"}),
            "cast_order": (["as written", "mixed"], {"default": "as written",
                           "tooltip": "as written: the characters the World line names are cast first, in the "
                                      "order it names them (write who leads first); mixed: shuffled, women and "
                                      "men alternating"}),
            "crowd": (["auto", "on", "off"], {"default": "auto",
                      "tooltip": "People behind the singers. auto: a soft-focus crowd when the World is a packed "
                                 "spectacle (arena, stadium, crowded stands), otherwise nobody else in frame; "
                                 "on / off: always / never"}),
            "models": (["auto", "24gb", "full"], {"default": "auto",
                       "tooltip": "Which model files to use. 24gb: the fp8 set (models.json), made for 24 GB GPUs. "
                                  "full: the bf16 set (models_full.json), for 48 GB or more. auto: the full set on "
                                  "a 48 GB+ GPU when its files are installed, otherwise the 24 GB set."}),
            "engine": (["ltx-2.5", "minimax-h3"], {"default": "ltx-2.5",
                       "tooltip": "The video model of the singer clips. ltx-2.5: LTX-2.5 distilled, lip sync from "
                                  "the vocal. minimax-h3: MiniMax H3 in reference mode (the portrait, the vocal "
                                  "copied 1:1, the lyrics transcribed with Whisper), 24 fps, at least 4.5 s a clip; "
                                  "slower. Each engine gets prompts in its own structure."}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("summary", "run_dir")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kw):
        return random.random()

    def run(self, audio, mode, pack, format, preset, seconds, start, seed, twins, queue_renders,
            dance_video_path="", run_name="", twin_editor="klein", guide_strength=1.0, queue_assemble=True,
            framing="mixed", artist_image_path="", qc_retries=2, artist_image=None, dance_video=None, theme="", broll_share=0.125, broll_style="mix",
            staged=True, look_groups=False, cuts_low=None, cuts_medium=None, cuts_high=None, audio_fx="subtle",
            cast_order="as written", crowd="auto", models="auto", fps="auto", resolution="1080", engine="ltx-2.5",
            direction="", refine_details=False, h3_upscale="auto", image_model="krea", broll_prompts="", draft=False):
        _reload()
        if artist_image is not None:
            artist_image_path = _save_image_input(artist_image)
        if dance_video is not None:
            dance_video_path = _video_input_path(dance_video)
        wf = audio["waveform"][0].detach().cpu().float().numpy()
        res = director.direct(audio=wf, sr=int(audio["sample_rate"]), out_root=folder_paths.get_output_directory(),
                              mode=mode, pack=pack, fmt=format, preset=preset, seconds=seconds, start=start,
                              seed=seed, twins=twins, dance_path=_resolve(dance_video_path), run_name=run_name,
                              queue=queue_renders, editor=twin_editor, guide_strength=guide_strength,
                              queue_assemble=queue_assemble, framing=framing,
                              artist_path="|".join(_resolve(x) for x in artist_image_path.split("|") if x.strip()),
                              qc_retries=qc_retries, theme=theme or "", broll_share=broll_share,
                              broll_style=broll_style, staged=staged,
                              look_groups=look_groups,
                              cuts={"low": cuts_low, "medium": cuts_medium, "high": cuts_high},
                              audio_fx=audio_fx or "", cast_order=cast_order or "as written",
                              crowd=crowd or "auto", models=models or "auto",
                              fps=0.0 if fps in (None, "", "auto") else float(fps), resolution=int(resolution or 1080),
                              engine="h3" if engine == "minimax-h3" else "ltx25", direction=direction or "",
                              refine=bool(refine_details), h3_upscale=h3_upscale or "auto",
                              image_model=image_model or "krea", broll_prompts=broll_prompts or "", draft=bool(draft))
        ui = {"text": [res["summary"]]}
        ui["text"].append(_file_ref(res["bundle"]))  # jobs.json as a job output (SDK clients)
        return {"ui": ui, "result": (res["summary"], res["run_dir"])}


class FVAssemble:
    """Cuts a OneShot MV run together on the beat, with audio-reactive FX and the finish.
    Queued automatically as the last job of a run; queue it again to re-cut."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"run_dir": ("STRING", {"default": ""}),
                             "filename_prefix": ("STRING", {"default": "oneshot-mv/final"})},
                "optional": {"fx_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                             "audio_fx": (["from the run", "none", "subtle", "intense"], {"default": "from the run",
                                          "tooltip": "Re-assemble with another audio-reactive FX level"}),
                             "finish": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview", "video_path")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kw):
        return random.random()

    def run(self, run_dir, filename_prefix, fx_intensity=1.0, finish=True, audio_fx="from the run"):
        _reload()
        import comfy.utils
        pbar = comfy.utils.ProgressBar(100)
        try:  # re-shot takes get their take check, so the Studio shows their own flags
            from .fv import director as _fv_director
            _fv_director.check_new_takes(_resolve(run_dir))
        except Exception as e:
            print(f"[OneShotMV] take check of re-shoots skipped ({e})")
        res = fv_assemble.assemble(_resolve(run_dir), fx_intensity=fx_intensity, do_finish=finish,
                                   fx_level=None if audio_fx == "from the run" else audio_fx,
                                   progress=lambda i, n: pbar.update_absolute(int(100 * i / max(1, n)), 100))
        prev = torch.from_numpy(np.stack(res["previews"]).astype(np.float32) / 255.0) if res["previews"] \
            else torch.zeros(1, 64, 64, 3)
        msg = f"{res['video']}  ({res['frames']} frames)"
        if res["missing_renders"]:
            msg += f"  missing renders: {res['missing_renders']}"
        ui = {"text": [msg]}
        out_dir = folder_paths.get_output_directory()
        try:  # expose the MP4 as a job output (UI preview, API / SDK download)
            rel = os.path.relpath(res["video"], out_dir)
            if not rel.startswith(".."):
                sub, name = os.path.split(rel)
                ui["gifs"] = [{"filename": name, "subfolder": sub, "type": "output", "format": "video/h264-mp4"}]
        except ValueError:
            pass
        return {"ui": ui, "result": (prev, res["video"])}


class FVQC:
    """Checks every take of a run (missing, frozen, blown exposure, dead mouth), re-rolls the bad
    ones with new seeds and queues itself again; when everything passes (or retries run out) it
    keeps the best take of each shot and queues the final Assemble."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"run_dir": ("STRING", {"default": ""}),
                             "pass_index": ("INT", {"default": 0, "min": 0, "max": 9}),
                             "max_passes": ("INT", {"default": 1, "min": 0, "max": 9})},
                "optional": {"then_assemble": ("BOOLEAN", {"default": True}),
                             "queue_rerolls": ("BOOLEAN", {"default": True,
                                               "tooltip": "off = return the re-roll jobs (qcN.json) for an "
                                                          "SDK client to submit"}),
                             "stage": (["all", "leads", "looks"], {"default": "all",
                                       "tooltip": "all = every take at once; leads / looks = the Director's "
                                                  "staged order: singer shots first, then the looks made from "
                                                  "the final singer takes"})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kw):
        return random.random()

    def run(self, run_dir, pass_index, max_passes, then_assemble=True, queue_rerolls=True, stage="all"):
        _reload()
        res = director.qc_pass(_resolve(run_dir), pass_idx=pass_index, max_passes=max_passes,
                               then_assemble=then_assemble, queue_rerolls=queue_rerolls, stage=stage)
        return {"ui": {"text": [res["message"], _file_ref(res["bundle"])]}, "result": (res["message"],)}


def _video_ref(path: str) -> list:
    """An MP4 under the output folder as a ComfyUI 'gifs' ui entry (UI preview, API / SDK download)."""
    try:
        rel = os.path.relpath(path, folder_paths.get_output_directory())
    except ValueError:
        return []
    if rel.startswith(".."):
        return []
    sub, name = os.path.split(rel)
    return [{"filename": name, "subfolder": sub.replace(os.sep, "/"), "type": "output", "format": "video/h264-mp4"}]


class FVRunResult:
    """Where a run is, and its final video once it is finished, as job outputs. For API and
    Comfy SDK clients (sdk_run.py): queued after a Director, it runs once the run's own stages
    (which jump the queue) are done."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"run_dir": ("STRING", {"default": "",
                                         "tooltip": "The run folder (the Director's run_dir output)"})}}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("status", "video_path")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kw):
        return random.random()

    def run(self, run_dir):
        _reload()
        import json
        st = director.run_status(_resolve(run_dir))
        if st["state"] == "rendering":
            try:  # nothing else queued and no video: one of the run's stages failed
                from server import PromptServer
                _running, pending = PromptServer.instance.prompt_queue.get_current_queue()
                if not pending:
                    st["state"] = "stopped"
                    st["message"] += ("; nothing is left in the queue, so the run stopped early "
                                      "(the ComfyUI console shows the error)")
            except Exception:
                pass
        ui = {"text": [st["message"]]}
        if st.get("run_dir"):
            sp = os.path.join(st["run_dir"], "status.json")
            json.dump(st, open(sp, "w", encoding="utf-8"), indent=1)
            ui["text"].append(_file_ref(sp))
        if st["state"] == "done":
            ui["gifs"] = _video_ref(st["video"])
        return {"ui": ui, "result": (st["message"], st.get("video", ""))}


class FVCritic:
    """The vision check (fv/critic.py): Gemma 3 looks at every portrait and b-roll still of a run
    against what its character should be (gender, facial hair, glasses, other people, the head in
    frame, writing, anything from after the World's period, the outfit). Bad ones are re-made with
    new seeds and a line that says what to fix, and checked again; then the clips are queued."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"run_dir": ("STRING", {"default": ""}),
                             "pass_index": ("INT", {"default": 0, "min": 0, "max": 9}),
                             "max_passes": ("INT", {"default": 2, "min": 0, "max": 9})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kw):
        return random.random()

    def run(self, run_dir, pass_index, max_passes):
        _reload()
        res = director.critic_pass(_resolve(run_dir), pass_idx=pass_index, max_passes=max_passes)
        return {"ui": {"text": [res["message"]]}, "result": (res["message"],)}


class FVLipSyncScore:
    """Scores a run's lip sync with SyncNet: per take, the confidence (higher = the mouth follows
    the words) and the offset (25 fps frames; negative = the mouth leads the voice). Writes
    <run>/lipsync.json; the run itself is not changed. Needs the SyncNet weights (see README)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"run_dir": ("STRING", {"default": ""}),
                             "which": (["lead clips", "all clips"], {"default": "lead clips"}),
                             "every_take": ("BOOLEAN", {"default": False,
                                            "tooltip": "also score the re-rolled takes QC kept aside"})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kw):
        return random.random()

    def run(self, run_dir, which, every_take):
        _reload()
        if not lipsync.available():
            msg = ("SyncNet weights not found: put syncnet_v2.model and sfd_face.pth in "
                   "ComfyUI/models/syncnet/ (see the OneShot MV README)")
            return {"ui": {"text": [msg]}, "result": (msg,)}
        res = lipsync.score_run(_resolve(run_dir), leads_only=(which == "lead clips"), every_take=every_take)
        return {"ui": {"text": [res["message"]]}, "result": (res["message"],)}


class FVFaceMatch:
    """Moves and scales an edited keyframe so its face sits where the reference frame's face is.
    Image edits asked for a new outfit tend to pull the camera back; a pose-matched look made
    from such a keyframe stays smaller and higher than the lead, so the cut doesn't match."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "reference": ("IMAGE",),
                             "max_scale": ("FLOAT", {"default": 2.2, "min": 1.0, "max": 4.0, "step": 0.05})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, reference, max_scale=2.2):
        _reload()
        return (faces.face_match(image, reference, max_scale),)


class FVFinish:
    """The OneShot MV finish on any frames: learned grade, diffusion, halation, contrast,
    gate vignette, grain (settings from a style pack)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "pack": (casting.list_packs(), {"default": "crowd"}),
                             "amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, images, pack, amount):
        _reload()
        pk = casting.load_pack(pack)
        dev = images.device if images.is_cuda else ("cuda" if torch.cuda.is_available() else "cpu")
        frames = [images[i].to(dev) for i in range(images.shape[0])]
        fin = finish.Finisher(pk["finish"], finish.lab_stats(frames[:: max(1, len(frames) // 16)]))
        out = [(fin(f, i) * amount + f * (1 - amount)).cpu() for i, f in enumerate(frames)]
        return (torch.stack(out),)


class FVOffloadToCPU:
    """Keeps a model's weight backups (made when LoRAs are patched in) in system RAM instead of
    VRAM. With --highvram ComfyUI parks them on the GPU, which doubles a 22B LTX model's
    footprint once LoRAs are applied; on a 96 GB card that spills into shared memory and
    sampling runs ~30x slower."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "run"
    CATEGORY = CATEGORY + "/io"

    def run(self, model):
        m = model.clone()
        m.offload_device = torch.device("cpu")
        return (m,)


class FVLeadPose:
    """Keeps only the lead dancer in a DWPose guide. DWPose draws everyone it finds (backup
    dancers, people at the back), and LTX's pose guide turns every skeleton into a person. This
    follows the tallest, most central person and redraws that one pose in DWPose's own style.
    Frames with a single person pass through unchanged."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pose_keypoint": ("POSE_KEYPOINT",), "image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY + "/io"

    def run(self, pose_keypoint, image):
        import cv2
        _reload()
        frames = pose_keypoint if isinstance(pose_keypoint, list) else [pose_keypoint]
        if len(frames) != image.shape[0] or not any(len(f.get("people") or []) > 1 for f in frames
                                                     if isinstance(f, dict)):
            return (image,)
        try:
            from custom_controlnet_aux.dwpose import decode_json_as_poses, draw_poses
        except Exception as e:  # controlnet_aux layout changed: leave the guide as it is
            print(f"[OneShotMV] lead pose unavailable ({e}); guide unchanged")
            return (image,)
        src = pose.pick_lead(frames)
        H_out, W_out = int(image.shape[1]), int(image.shape[2])
        out = []
        for k, (fr, s) in enumerate(zip(frames, src)):
            if not isinstance(fr, dict) or len(fr.get("people") or []) <= 1 and s == (k, 0):
                out.append(image[k])
                continue
            poses, _, H, W = decode_json_as_poses(pose.only(frames, s))
            canvas = draw_poses(poses, int(H), int(W), draw_body=True, draw_hand=True, draw_face=True)
            interp = cv2.INTER_CUBIC if W_out > W else cv2.INTER_AREA
            canvas = cv2.resize(canvas, (W_out, H_out), interpolation=interp)
            out.append(torch.from_numpy(canvas.astype(np.float32) / 255.0).to(image.device))
        n_multi = sum(1 for f in frames if isinstance(f, dict) and len(f.get("people") or []) > 1)
        print(f"[OneShotMV] lead pose: {n_multi}/{len(frames)} frames had extra people")
        return (torch.stack(out).to(image.dtype),)


class FVLeadMask:
    """Greys out everything but the lead dancer in source frames, so an image edit (the motion
    keyframes) sees one person: backup dancers otherwise come along into the new scene."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pose_keypoint": ("POSE_KEYPOINT",), "image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY + "/io"

    def run(self, pose_keypoint, image):
        import cv2
        _reload()
        frames = pose_keypoint if isinstance(pose_keypoint, list) else [pose_keypoint]
        if len(frames) != image.shape[0] or not any(len(f.get("people") or []) > 1 for f in frames
                                                     if isinstance(f, dict)):
            return (image,)
        src = pose.pick_lead(frames)
        B, H, W = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
        out = image.clone()
        for k, s in enumerate(src):
            if s is None:
                continue
            fr = pose.only(frames, s)
            box = pose.person_box(fr["people"][0], float(fr.get("canvas_width") or W),
                                  float(fr.get("canvas_height") or H)) if fr["people"] else None
            if box is None:
                continue
            m = np.zeros((H, W), np.float32)
            x0, y0, x1, y1 = int(box[0] * W), int(box[1] * H), int(np.ceil(box[2] * W)), int(np.ceil(box[3] * H))
            m[y0:y1, x0:x1] = 1.0
            r = max(3, int(0.03 * max(H, W))) | 1
            m = cv2.GaussianBlur(m, (r * 2 + 1, r * 2 + 1), 0)
            mt = torch.from_numpy(m).to(image.device, image.dtype)[..., None]
            out[k] = image[k] * mt + 0.5 * (1 - mt)
        return (out,)


class FVFreeMemory:
    """Unloads every model and clears ComfyUI's cache after this job (asks the server through its
    own /free endpoint). The Director queues one between model groups so RAM stays bounded."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"note": ("STRING", {"default": "free models between groups"})}}

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY + "/io"

    @classmethod
    def IS_CHANGED(cls, **kw):
        return random.random()

    def run(self, note):
        import gc
        import json as _json
        import urllib.request
        try:  # the most GPU memory the stage before this used (to check a run fits a smaller card)
            import torch
            if torch.cuda.is_available():
                print(f"[OneShotMV] peak GPU memory before '{note}': "
                      f"{torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB allocated, "
                      f"{torch.cuda.max_memory_reserved() / 2 ** 30:.1f} GB reserved")
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        try:
            req = urllib.request.Request(director.server_url() + "/free",
                                         data=_json.dumps({"unload_models": True, "free_memory": True}).encode(),
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10).read()
        except Exception as e:
            print(f"[OneShotMV] /free failed: {e}")
        try:
            import comfy.model_management as mm
            mm.unload_all_models()
            mm.soft_empty_cache()
        except Exception:
            pass
        gc.collect()
        return {"ui": {"text": ["models unloaded"]}}


class FVKeep:
    """Keeps models loaded across jobs that don't use them. ComfyUI drops a cached model when a
    job without its loader runs (a QC pass, a Klein edit), and LTX-2.3 then reloads from disk
    (5-9 min on a slow drive). The Director wires LTX's loaders in here in those jobs: they
    come from the cache instantly, and the model stays loaded for the next video job."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {},
                "optional": {"model": ("MODEL",), "clip": ("CLIP",), "vae": ("VAE",), "audio_vae": ("VAE",),
                             "upscaler": ("LATENT_UPSCALE_MODEL",)}}

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY + "/io"

    def run(self, **kw):
        return {"ui": {"text": [f"kept {len(kw)} model(s) loaded"]}}


class FVTheme:
    """Preview what a theme turns into before a run: the characters, outfits, places, light,
    in-scene actions and colour grade the Director will cast from (Gemma 3, the LTX-2.3 text
    encoder). The result is cached, so a Director run with the same theme and seed reuses it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"theme": ("STRING", {"default": "", "multiline": True}),
                             "seed": ("INT", {"default": 1, "min": 0, "max": 2 ** 31 - 1})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("world",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def run(self, theme, seed):
        _reload()
        b = fv_theme.expand(theme, seed=seed, out_root=folder_paths.get_output_directory())
        if not b:
            return {"ui": {"text": ["(no theme)"]}, "result": ("",)}
        lines = [f"Theme: {b['theme']}  ({b.get('source')}, grade: {b.get('grade')})"]
        if b.get("camera"):
            lines.append(f"camera: {b['camera']}")
        chars = b.get("characters") or []
        if chars:  # whole characters, as the cast uses them
            lines.append("\ncharacters:")
            for c in chars:
                lines.append(f"  - {c['who']}")
                lines.append(f"      wearing {c['wear']}")
                for k, lab in (("place", ""), ("light", "light: "), ("doing", "wide shot: "), ("face", "close-up: ")):
                    if c.get(k):
                        lines.append(f"      {lab}{c[k]}")
        for k in (("places", "light", "actions") if chars else ("who", "wear", "places", "light", "actions")):
            lines.append(f"\n{k}:")
            lines += [f"  - {x}" for x in b.get(k, [])]
        text = "\n".join(lines)
        return {"ui": {"text": [text]}, "result": (text,)}


class FVSong:
    """No song? Make one: Gemma 3 writes lyrics and a style for the World, then ACE-Step 1.5 XL
    turbo sings it. The WAV goes to the input folder (oneshot-mv-songs/) next to its lyrics."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"world": ("STRING", {"default": "", "multiline": True}),
                             "style": ("STRING", {"default": "", "multiline": False,
                                                  "tooltip": "Genre, mood, voice. Empty: picked to suit the World."}),
                             "seconds": ("FLOAT", {"default": 120.0, "min": 10.0, "max": 300.0, "step": 1.0}),
                             "seed": ("INT", {"default": 1, "min": 0, "max": 2 ** 31 - 1}),
                             "instrumental": ("BOOLEAN", {"default": False})},
                "optional": {"lyrics": ("STRING", {"default": "", "multiline": True,
                                                   "tooltip": "Your own lyrics, [Verse]/[Chorus] tags on their own lines. Empty: written for you."})}}

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "song")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def run(self, world, style, seconds, seed, instrumental, lyrics=""):
        _reload()
        r = fv_song.make(world, style, seconds, seed, instrumental, lyrics or "",
                         input_dir=folder_paths.get_input_directory())
        x, sr = stems.read_wav(os.path.join(folder_paths.get_input_directory(), r["song"]))
        audio = {"waveform": torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0), "sample_rate": sr}
        return {"ui": {"text": [json.dumps(r)]}, "result": (audio, r["song"])}


class FVFilm:
    """Film mode: a commercial or short film from a script. Gemma writes the cast, the look and the
    shot list; every shot is queued as text to video (LTX-2.5 or MiniMax H3), then the cut."""

    @classmethod
    def INPUT_TYPES(cls):
        presets = [k for k in formats.PRESETS if k not in ("fast_voice",)]
        return {"required": {"script": ("STRING", {"default": "", "multiline": True}),
                             "kind": (["commercial", "film", "music video", "trailer", "documentary"], {"default": "commercial"}),
                             "seconds": ("FLOAT", {"default": 30.0, "min": 5.0, "max": 300.0, "step": 1.0}),
                             "format": (list(formats.FORMATS.keys()), {"default": "16:9"}),
                             "preset": (presets, {"default": "quality"}),
                             "seed": ("INT", {"default": 1, "min": 0, "max": 2 ** 31 - 1}),
                             "queue_renders": ("BOOLEAN", {"default": True})},
                "optional": {"style": ("STRING", {"default": ""}),
                             "song": ("STRING", {"default": "", "tooltip": "A song in the input folder: the music, cut on its beats. Empty: the shots' own sound."}),
                             "song_start": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 3600.0, "tooltip": "-1: the strongest stretch"}),
                             "look": (["none"] + [k for k in finish.LOOKS if k != "none"], {"default": "none"}),
                             "h3_upscale": (["auto", "relayvsr"], {"default": "auto"}),
                             "sound": ("BOOLEAN", {"default": True, "tooltip": "Without a song: keep the sound each shot was made with"}),
                             "resolution": (["720", "1080", "1440", "2160"], {"default": "1080"}),
                             "run_name": ("STRING", {"default": ""}),
                             "shots": (["cuts", "one shot"], {"default": "cuts", "tooltip": "one shot: a single continuous take, no edit (at most 15 s on H3, 20 s on LTX)"}),
                             "draft": ("BOOLEAN", {"default": False, "tooltip": "A draft at a small size (960x544 on MiniMax H3, Fast's size on LTX-2.5) with the same model, to check the shots before the full render"})}}

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("nan")  # reads and writes files: never served from ComfyUI's cache

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("film_dir",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def run(self, script, kind, seconds, format, preset, seed, queue_renders, style="", song="", song_start=-1.0,
            look="none", h3_upscale="auto", sound=True, resolution="1080", run_name="", shots="cuts", draft=False):
        _reload()
        song_path = _resolve(os.path.join(folder_paths.get_input_directory(), song)) if song else ""
        film = fv_film.plan(script=script, seconds=seconds, fmt=format, preset=preset, seed=seed,
                            out_root=folder_paths.get_output_directory(), kind=kind, style=style, song_path=song_path,
                            song_start=song_start, look=look, run_name=run_name, h3_upscale=h3_upscale, sound=sound,
                            one_shot=shots == "one shot", draft=bool(draft))
        film["resolution"] = int(resolution or 1080)
        if queue_renders:
            ids = [director.queue_prompt(j["graph"]) for j in film["jobs"]]
            ids.append(director.queue_prompt(fv_film.cut_graph(film["dir"])))
            film["queued"] = ids
        fv_film.save(film)
        info = {"film": film["run"], "title": film["title"], "shots": len(film["shots"]), "dir": film["dir"]}
        return {"ui": {"text": [json.dumps(info)]}, "result": (film["dir"],)}


class FVFilmCut:
    """Film mode's cut: the shots' newest takes in order, the music or their sound, the finish.
    Run it again after a re-shoot or a new look; nothing re-renders."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"film_dir": ("STRING", {"default": ""})}}

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("nan")  # reads and writes files: never served from ComfyUI's cache

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def run(self, film_dir):
        _reload()
        out = fv_film.cut(film_dir)
        return {"ui": {"text": [json.dumps({"film_cut": os.path.basename(film_dir.rstrip("/")), "video": out})]},
                "result": (out,)}


NODE_CLASS_MAPPINGS = {
    "FVFreeMemory": FVFreeMemory,
    "FVKeep": FVKeep,
    "FVOffloadToCPU": FVOffloadToCPU,
    "FVLeadPose": FVLeadPose,
    "FVLeadMask": FVLeadMask,
    "FVDirector": FVDirector,
    "FVAssemble": FVAssemble,
    "FVRunResult": FVRunResult,
    "FVFinish": FVFinish,
    "FVQC": FVQC,
    "FVCritic": FVCritic,
    "FVLipSyncScore": FVLipSyncScore,
    "FVFaceMatch": FVFaceMatch,
    "FVTheme": FVTheme,
    "FVSong": FVSong,
    "FVFilm": FVFilm,
    "FVFilmCut": FVFilmCut,
    "FVLoadImagePath": FVLoadImagePath,
    "FVLoadVideoPath": FVLoadVideoPath,
    "FVLoadAudioPath": FVLoadAudioPath,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FVDirector": "OneShot MV: Director",
    "FVFreeMemory": "OneShot MV: Free Memory",
    "FVKeep": "OneShot MV: Keep Models Loaded",
    "FVOffloadToCPU": "OneShot MV: Weight Backups to RAM",
    "FVLeadPose": "OneShot MV: Lead Dancer Pose",
    "FVLeadMask": "OneShot MV: Lead Dancer Only",
    "FVAssemble": "OneShot MV: Assemble",
    "FVRunResult": "OneShot MV: Run Result",
    "FVFinish": "OneShot MV: Finish",
    "FVQC": "OneShot MV: Take QC",
    "FVCritic": "OneShot MV: Vision Check",
    "FVLoadImagePath": "OneShot MV: Load Image (path)",
    "FVLoadVideoPath": "OneShot MV: Load Video (path)",
    "FVLoadAudioPath": "OneShot MV: Load Audio (path)",
    "FVLipSyncScore": "OneShot MV: Lip Sync Score",
    "FVFaceMatch": "OneShot MV: Face Match",
    "FVTheme": "OneShot MV: Theme Preview",
    "FVSong": "OneShot MV: Make a Song",
    "FVFilm": "OneShot MV: Film (script to video)",
    "FVFilmCut": "OneShot MV: Film Cut",
}
