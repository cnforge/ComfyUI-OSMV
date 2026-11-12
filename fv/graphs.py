"""ComfyUI API-format graph builders for every OneShot MV job.

All file inputs go through the pack's path loaders (FVLoadImagePath / FVLoadVideoPath /
FVLoadAudioPath), which do not check that the file exists when the job is queued, so the
Director can queue a whole run up front (later jobs read files earlier jobs will write)."""
from __future__ import annotations

import json
import math
import os

from . import formats

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDERS = ("checkpoints", "diffusion_models", "unet", "loras", "text_encoders", "clip", "vae",
                 "latent_upscale_models")
MODEL_EXT = (".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin", ".gguf")


def _model_index():
    """(names ComfyUI lists, {file name: listed name}) across the model folders; None outside ComfyUI."""
    try:
        import folder_paths  # type: ignore
    except Exception:
        return None
    have, by_name = set(), {}
    for kind in MODEL_FOLDERS:
        if kind not in getattr(folder_paths, "folder_names_and_paths", {}):
            continue
        try:
            names = folder_paths.get_filename_list(kind)
        except Exception:
            continue
        for n in names:
            have.add(n)
            by_name.setdefault(os.path.basename(n.replace("\\", "/")).lower(), n)
    return have, by_name


def _resolve_models(m: dict, idx=None) -> tuple[dict, list]:
    """Each model file as this ComfyUI lists it, and the files it can't find. A name that isn't
    found as written is looked up by its file name in the model folders, so the files can sit in
    any sub-folder and the path separator doesn't matter. Names that aren't found are left as
    written (the job then says which file is missing)."""
    idx = idx if idx is not None else _model_index()
    if idx is None:
        return dict(m), []
    have, by_name = idx
    missing = []

    def fix(k, v):
        if not isinstance(v, str) or not v.lower().endswith(MODEL_EXT) or v in have:
            return v
        alt = v.replace("\\", os.sep).replace("/", os.sep)
        if alt in have:
            return alt
        hit = by_name.get(os.path.basename(v.replace("\\", "/")).lower())
        if hit is None:
            missing.append((k, os.path.basename(v.replace("\\", "/"))))
        return hit or v

    out = {k: [[fix(k, x[0]), *x[1:]] if isinstance(x, list) else fix(k, x) for x in v] if isinstance(v, list)
           else fix(k, v) for k, v in m.items()}
    return out, missing


# ---- model sets: "24gb" (models.json, fp8, the default) and "full" (models_full.json on top, bf16)
MODEL_SETS = ("24gb", "full")
FULL_MIN_GB = 46          # the full set wants a 48 GB card or bigger
RUN_KEYS = ("krea_turbo", "krea_raw", "krea_turbo_lora", "krea_te", "krea_vae", "ltx25_unet", "ltx25_vae",
            "ltx25_audio_vae", "ltx25_te", "ltx25_upscaler", "ltx_union_lora",  # what a default voice run loads (looks too)
            "ltx23_ckpt", "ltx23_te")  # LTX-2.3's Gemma 3: the language model of the vision check and the World
H3_KEYS = ("h3_unet", "h3_te", "h3_vae", "h3_audio_vae")  # the singer clips on MiniMax H3
DYNAMIC_KEYS = ("h3_full_unet",)  # More motion: H3's whole reference model (ref2va; bf16 unpruned in the full set)
SINGULARITY_KEYS = ("h3_singularity_unet",)  # Singularity: More motion on the Singularity ref2va fine-tune
PIKA_KEYS = ("h3_pika_unet", "h3_pika_lora")  # H3 b-roll on Pika-H3-Flex (optional: else on h3_unet); the Pika preset
MAX_KEYS = ("ltx25_dev_unet", "ltx25_distill_lora")  # the Max preset (LTX-2.5 dev)
H3IMG_KEYS = ("h3img_unet", "h3_te", "h3img_vae")  # MiniMax H3 Image (the image model choice): H3's text encoder
IMAGE_MODELS = ("krea", "h3_image")
ARTIST_KEYS = ("qwen21_unet", "qwen21_te", "qwen21_vae",   # a character photo's portraits
               "klein_unet", "klein_te", "klein_vae")      # and its looks


def _set_files(name: str) -> dict:
    m = json.load(open(os.path.join(HERE, "models.json"), encoding="utf-8"))
    if name == "full":
        m.update({k: v for k, v in json.load(open(os.path.join(HERE, "models_full.json"), encoding="utf-8")).items()
                  if not k.startswith("_")})
    return m


def reserved_gb() -> float:
    """GPU memory ComfyUI was told to leave alone with --reserve-vram (0 if not set)."""
    try:
        from comfy.cli_args import args  # type: ignore
        return float(args.reserve_vram or 0)
    except Exception:
        return 0.0


def vram_gb() -> float:
    """The GPU memory ComfyUI may use in GB: the card's total minus any --reserve-vram (0 without CUDA)."""
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 2 ** 30
            return max(total - reserved_gb(), 0.0)
    except Exception:
        pass
    return 0.0


def missing_files(name: str, idx=None, keys=RUN_KEYS) -> list:
    """The files of a model set that a default run (or the given keys) needs and this ComfyUI doesn't have."""
    m = {k: v for k, v in _set_files(name).items() if k in keys}
    if m.get("krea_raw") == m.get("krea_turbo"):
        m.pop("krea_turbo_lora", None)  # one Krea file for both passes: the Turbo LoRA isn't used
    return sorted(set(f for _, f in _resolve_models(m, idx)[1]))


def pick_models(requested: str | None = "auto") -> str:
    """The model set for a run: the requested one, or with "auto" the full set on cards with 48 GB or
    more when its files are installed, otherwise the 24 GB set."""
    if requested in MODEL_SETS:
        return requested
    idx = _model_index()
    big = vram_gb() >= FULL_MIN_GB
    if big and not missing_files("full", idx):
        return "full"
    if missing_files("24gb", idx) and big:
        return "full"  # neither set complete: the one this card can run
    return "24gb"


RUN_NODES = {"ClownsharKSampler_Beta": "RES4LYF", "DWPreprocessor": "comfyui_controlnet_aux"}  # other packs' nodes a default run needs


def missing_nodes() -> list:
    """The packs a default run needs that didn't load in this ComfyUI ([] outside ComfyUI)."""
    try:
        import nodes as comfy_nodes  # type: ignore
    except Exception:
        return []
    return sorted({p for n, p in RUN_NODES.items() if n not in comfy_nodes.NODE_CLASS_MAPPINGS})


def not_ready(requested: str | None = "auto", artist: bool = False, engine: str = "ltx25",
              preset: str = "quality") -> str:
    """Why a render with this model set (and a character photo, an engine and a preset) can't run
    here ('' when it can)."""
    name = pick_models(requested)
    files, packs = missing_files(name), missing_nodes()
    if artist:
        files += missing_files(name, keys=ARTIST_KEYS)
    if engine == "h3":
        files += missing_files(name, keys=H3_KEYS)
        if not _has_node("MiniMaxH3ReferenceToVideo"):
            packs = packs + ["ComfyUI's MiniMax H3 nodes (update ComfyUI)"]
        if preset == "pika":
            files += missing_files(name, keys=PIKA_KEYS)
        elif preset == "dynamic":
            files += missing_files(name, keys=DYNAMIC_KEYS)
        elif preset == "singularity":
            files += missing_files(name, keys=SINGULARITY_KEYS)
    elif preset in ("max", "ultra"):
        files += missing_files(name, keys=MAX_KEYS)
    why = []
    if files:
        why.append(f"{'a character run' if artist else 'the ' + ('full' if name == 'full' else '24 GB') + ' model set'}"
                   f" is missing {len(files)} file{'s' if len(files) > 1 else ''} ({', '.join(files[:3])}"
                   f"{f' and {len(files) - 3} more' if len(files) > 3 else ''}; the README lists the downloads)")
    if packs:
        why.append(f"{', '.join(packs)} didn't load (look for 'IMPORT FAILED' in the ComfyUI console)")
    return "; ".join(why)


def models_report() -> dict:
    """For the Studio: the GPU's memory, the set "auto" picks, the missing files of each set and
    the node packs that didn't load."""
    idx = _model_index()
    return {"vram_gb": round(vram_gb(), 1), "reserved_gb": round(reserved_gb(), 1),
            "auto": pick_models("auto"), "full_min_gb": FULL_MIN_GB,
            "missing": {n: missing_files(n, idx) for n in MODEL_SETS}, "missing_packs": missing_nodes(),
            "missing_character": missing_files("24gb", idx, ARTIST_KEYS),  # the same files in both sets
            "missing_h3": {n: missing_files(n, idx, H3_KEYS) for n in MODEL_SETS},
            "missing_max": {n: missing_files(n, idx, MAX_KEYS) for n in MODEL_SETS},
            "missing_pika": {n: missing_files(n, idx, PIKA_KEYS) for n in MODEL_SETS},
            "missing_dynamic": {n: missing_files(n, idx, DYNAMIC_KEYS) for n in MODEL_SETS},
            "missing_singularity": {n: missing_files(n, idx, SINGULARITY_KEYS) for n in MODEL_SETS},
            "missing_h3img": {n: missing_files(n, idx, H3IMG_KEYS) for n in MODEL_SETS},
            "missing_checks": missing_checks()}


def missing_checks() -> list:
    """Optional QC models a run works without, but worse: SyncNet (each take's lip-sync offset and
    its re-rolls) and YuNet (faces on OpenCV 5, which has no Haar cascades)."""
    from . import faces, lipsync
    out = [] if lipsync.available() else ["SyncNet (syncnet_v2.model and sfd_face.pth in models/syncnet)"]
    if faces._cascade() is None:
        out.append(f"YuNet ({faces.YUNET} in models/facedetection)")
    return out


def use_models(name: str | None = "auto") -> str:
    """Point M at a model set (in place, so every module that holds graphs.M sees it)."""
    name = pick_models(name)
    resolved, _ = _resolve_models(_set_files(name))
    M.clear()
    M.update(resolved)
    M["_set"] = name
    return name


SAGE_FUNCS = {"auto": "sageattn", "sageattn_qk_int8_pv_fp8_cuda++": "sageattn_qk_int8_pv_fp8_cuda"}


def _sage_mode(want: str) -> str:
    """The SageAttention kernel the LTX jobs can use: the one asked for if this install has it, else
    'auto', else '' (no patch; also when none is asked for). SageAttention is optional, and 1.x only
    has 'auto': asking it for a 2.x kernel failed every clip with an ImportError."""
    if not want or want == "disabled":
        return ""
    try:
        import sageattention  # type: ignore
    except Exception:
        return ""
    for mode in (want, "auto"):
        if mode and hasattr(sageattention, SAGE_FUNCS.get(mode, mode)):
            return mode
    return ""


M: dict = {}
use_models("auto")

R8_S2 = "0.909, 0.880, 0.840, 0.780, 0.700, 0.600, 0.500, 0.400, 0.250, 0.100, 0.0"
R8_S3 = "0.909375, 0.725, 0.421875, 0.0"
D25_S1 = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
D25_S2 = "0.909375, 0.725, 0.421875, 0.0"
# Max (LTX-2.5 dev): stage-1 steps and CFG, and the distilled LoRA's strength on the upscale stage
DEV_STEPS, DEV_CFG, DEV_LORA = 20, 4.0, 0.6
# Ultra: the upscale stage in 8 steps (from the same 0.909 start), and stage 1's guidance as in
# LTX-2's dev workflows: per modality CFG, STG (block 29 perturbed) and cross-modal guidance
D25_S2_8 = "0.909375, 0.85, 0.78, 0.725, 0.6, 0.421875, 0.25, 0.1, 0.0"
STG_VIDEO = dict(cfg=3.0, stg=1.0, rescale=0.7, modality_scale=3.0)
STG_AUDIO = dict(cfg=7.0, stg=1.0, rescale=0.7, modality_scale=3.0)
STG_BLOCKS = "29"


class Graph:
    def __init__(self):
        self.nodes, self._n = {}, 0

    def add(self, class_type, **inputs):
        self._n += 1
        self.nodes[str(self._n)] = {"class_type": class_type, "inputs": inputs}
        return str(self._n)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.nodes, f, indent=1)
        return path


def O(nid, slot=0):
    return [nid, slot]


def _save_video(g, images, fps, prefix, audio=None):
    kw = {"images": images, "fps": float(fps)}
    if audio is not None:
        kw["audio"] = audio
    v = g.add("CreateVideo", **kw)
    # near-lossless (crf 10): the cut is the only real compression a clip goes through
    g.add("SaveVideo", video=O(v), filename_prefix=prefix,
          **{"format": "mp4", "format.codec": "h264", "format.codec.encoding": "re-encode",
             "format.codec.encoding.crf": 10.0})


# ------------------------------------------------------------------ Krea 2 W5
def krea2_portrait(*, prompt, seed, prefix, size1, size2, denoise=0.25, turbo_only=False, image_model="krea"):
    """Krea 2 W5 hybrid (Turbo composes, RAW + Turbo LoRA redetails). turbo_only=True uses the
    Turbo model for both passes: one model to load instead of two (the fast preset).
    denoise: how hard the RAW pass redetails the upscaled image; 0.40 looks too gritty, so 0.25
    keeps Turbo's cleaner composition and adds less texture. A set that names the same file for
    Turbo and RAW also runs Turbo for both passes (the Turbo LoRA is only for the RAW model).
    image_model "h3_image": MiniMax H3 Image makes the still instead (h3img_still)."""
    if image_model == "h3_image":
        return h3img_still(prompt=prompt, seed=seed, prefix=prefix, size2=size2)
    g = Graph()
    turbo = g.add("UNETLoader", unet_name=M["krea_turbo"], weight_dtype="default")
    if turbo_only or M["krea_raw"] == M["krea_turbo"]:
        lora = turbo
    else:
        raw = g.add("UNETLoader", unet_name=M["krea_raw"], weight_dtype="default")
        lora = g.add("LoraLoaderModelOnly", model=O(raw), lora_name=M["krea_turbo_lora"], strength_model=0.6)
    clip = g.add("CLIPLoader", clip_name=M["krea_te"], type="krea2", device="default")
    vae = g.add("VAELoader", vae_name=M["krea_vae"])
    pos = g.add("CLIPTextEncode", clip=O(clip), text=prompt)
    neg = g.add("ConditioningZeroOut", conditioning=O(pos))
    lat = g.add("EmptyLatentImage", width=size1[0], height=size1[1], batch_size=1)
    ks = g.add("KSampler", model=O(turbo), seed=seed, steps=8, cfg=1.0, sampler_name="euler",
               scheduler="simple", positive=O(pos), negative=O(neg), latent_image=O(lat), denoise=1.0)
    dec = g.add("VAEDecode", samples=O(ks), vae=O(vae))
    up = g.add("ImageScale", image=O(dec), upscale_method="lanczos", width=size2[0], height=size2[1],
               crop="disabled")
    enc = g.add("VAEEncode", pixels=O(up), vae=O(vae))
    cs = g.add("ClownsharKSampler_Beta", eta=0.5, sampler_name="multistep/deis_2m", scheduler="beta",
               steps=8, steps_to_run=-1, denoise=denoise, cfg=1.0, seed=seed, sampler_mode="standard",
               bongmath=True, model=O(lora), positive=O(pos), negative=O(neg), latent_image=O(enc))
    dec2 = g.add("VAEDecode", samples=O(cs, 0), vae=O(vae))
    g.add("SaveImage", images=O(dec2), filename_prefix=prefix)
    return g


def artist_portrait(*, artist_path, prompt, seed, prefix, size2, resolution=1024, image_model="krea"):
    """One-artist mode: the artist photo re-staged by Qwen-Image 2.1 (identity kept), or by
    MiniMax H3 Image's edit with image_model "h3_image"."""
    g = Graph()
    ref = g.add("FVLoadImagePath", path=artist_path)
    if image_model == "h3_image":
        out = _h3img(g, f"Edit <Picture 1>, keeping this exact person: {prompt}", seed, images=[O(ref)])
    else:
        out = _qwen21(g, O(ref), prompt, seed=seed, resolution=resolution)
    out = O(g.add("ImageScale", image=out, upscale_method="lanczos", width=size2[0], height=size2[1], crop="center"))
    g.add("SaveImage", images=out, filename_prefix=prefix)
    return g


# ------------------------------------------------------------------ MiniMax H3 Image
H3IMG_MP = 4.2  # megapixels: H3 Image works from about 4 MP up (16 MP is its best, and far slower)
H3IMG_STEPS = 6


def h3img_ready() -> bool:
    return not missing_files(M.get("_set", "24gb"), keys=H3IMG_KEYS)


def h3img_size(size) -> tuple[int, int]:
    """About H3IMG_MP megapixels in the shape of `size`, in multiples of 32."""
    w, h = size
    k = math.sqrt(H3IMG_MP * 1e6 / (w * h))
    return max(32, round(w * k / 32) * 32), max(32, round(h * k / 32) * 32)


def _h3img(g, prompt, seed, size=None, images=(), steps=H3IMG_STEPS):
    """MiniMax H3 Image (ComfyUI's template: 6 steps, cfg 1, euler/simple). Text to image at `size`,
    or with images an edit of image_1 (<Picture 1>; the others are <Picture 2>...) at its size, which
    is first brought to about H3IMG_MP megapixels: the model draws poorly below 4 MP."""
    unet = g.add("UNETLoader", unet_name=M["h3img_unet"], weight_dtype="default")
    clip = g.add("CLIPLoader", clip_name=M["h3_te"], type="minimax", device="default")
    vae = g.add("VAELoader", vae_name=M["h3img_vae"])
    kw = {}
    for i, im in enumerate(images):
        if i == 0:
            im = O(g.add("ImageScaleToTotalPixels", image=im, upscale_method="lanczos", megapixels=H3IMG_MP,
                         resolution_steps=32))
            dims = g.add("GetImageSize", image=im)
            size = (O(dims, 0), O(dims, 1))
        kw[f"images.image_{i + 1}"] = im
    w, h = size or (2048, 2048)
    cond = g.add("MiniMaxH3ImageConditioning", clip=O(clip), vae=O(vae), prompt=prompt, width=w, height=h, **kw)
    neg = g.add("ConditioningZeroOut", conditioning=O(cond, 0))
    ks = g.add("KSampler", model=O(unet), seed=seed, steps=steps, cfg=1.0, sampler_name="euler", scheduler="simple",
               positive=O(cond, 0), negative=O(neg), latent_image=O(cond, 1), denoise=1.0)
    return O(g.add("VAEDecode", samples=O(ks), vae=O(vae)))


def h3img_still(*, prompt, seed, prefix, size2):
    """A still from MiniMax H3 Image: made at about 4 MP in size2's shape, then scaled to size2."""
    g = Graph()
    out = _h3img(g, prompt, seed, size=h3img_size(size2))
    out = O(g.add("ImageScale", image=out, upscale_method="lanczos", width=size2[0], height=size2[1], crop="center"))
    g.add("SaveImage", images=out, filename_prefix=prefix)
    return g


# ------------------------------------------------------------------ LTX models
def _has_node(name: str) -> bool:
    """Whether the server this graph is built on has a node class (True when not inside ComfyUI)."""
    try:
        import nodes as _comfy_nodes
        return name in _comfy_nodes.NODE_CLASS_MAPPINGS
    except Exception:
        return True


def _ram_backups(g, model):
    """LoRA weight backups to system RAM (see FVOffloadToCPU); skipped on servers without it."""
    if not _has_node("FVOffloadToCPU"):
        return model
    return O(g.add("FVOffloadToCPU", model=model))


def _ltx23(g, union=False):
    ck = g.add("CheckpointLoaderSimple", ckpt_name=M["ltx23_ckpt"])
    model = _ram_backups(g, O(ck, 0))
    sage = _sage_mode(M.get("ltx23_sage") or "")
    if sage:
        model = O(g.add("PathchSageAttentionKJ", model=model, sage_attention=sage,
                        allow_compile=False))
    for name, st in M["ltx23_loras"]:
        model = O(g.add("LoraLoaderModelOnly", model=model, lora_name=name, strength_model=st))
    if M.get("ltx23_av_tuner"):
        vs, as_, a2v, v2a = M["ltx23_av_tuner"]
        model = O(g.add("LTX2AttentionTunerPatch", model=model, blocks="", video_scale=vs,
                        audio_scale=as_, audio_to_video_scale=a2v, video_to_audio_scale=v2a,
                        triton_kernels=False))
    dsf = 1.0
    if union:
        lo = g.add("LTXICLoRALoaderModelOnly", model=model, lora_name=M["ltx_union_lora"], strength_model=1.0)
        model, dsf = O(lo), O(lo, 1)
    avae = g.add("LTXVAudioVAELoader", ckpt_name=M["ltx23_ckpt"])
    clip = g.add("LTXAVTextEncoderLoader", text_encoder=M["ltx23_te"], ckpt_name=M["ltx23_ckpt"], device="default")
    ups = g.add("LatentUpscaleModelLoader", model_name=M["ltx23_upscaler"])
    return dict(model=model, dsf=dsf, vae=O(ck, 2), avae=O(avae), clip=O(clip), ups=O(ups))


def _ltx25(g, union=False, dev=False):
    """dev: the full dev model (the Max preset) in place of the distilled one."""
    unet = g.add("UNETLoader", unet_name=M["ltx25_dev_unet" if dev else "ltx25_unet"], weight_dtype="default")
    model, dsf = _ram_backups(g, O(unet)), 1.0
    if union:
        lo = g.add("LTXICLoRALoaderModelOnly", model=model, lora_name=M["ltx_union_lora"], strength_model=1.0)
        model, dsf = O(lo), O(lo, 1)
    vae = g.add("VAELoader", vae_name=M["ltx25_vae"])
    avae = g.add("VAELoader", vae_name=M["ltx25_audio_vae"])
    clip = g.add("CLIPLoader", clip_name=M["ltx25_te"], type="ltxv", device="default")
    ups = g.add("LatentUpscaleModelLoader", model_name=M["ltx25_upscaler"])
    return dict(model=model, dsf=dsf, vae=O(vae), avae=O(avae), clip=O(clip), ups=O(ups))


def _pose(g, video_path, w, h, frames, fps, start_s=0.0, face=True, lead_only=False):
    """DWPose guide frames. lead_only: keep just the lead dancer (FVLeadPose) so backup dancers
    in the source clip don't turn into extra people."""
    src = g.add("FVLoadVideoPath", path=video_path, width=w, height=h, frame_cap=frames,
                start_seconds=float(start_s), force_fps=float(fps))
    dw = g.add("DWPreprocessor", image=O(src), detect_hand="enable", detect_body="enable",
               detect_face="enable" if face else "disable", resolution=512,
               bbox_detector=M["dwpose_bbox"], pose_estimator=M["dwpose_pose"],
               scale_stick_for_xinsr_cn="disable")
    if lead_only and _has_node("FVLeadPose"):
        return O(g.add("FVLeadPose", pose_keypoint=O(dw, 1), image=O(dw, 0)))
    return O(dw)


# LTX (the fast LTX-2.5 distilled model especially) sometimes cuts away mid-clip to another
# person or place, as if editing a music video. At cfg 1 a negative prompt can't stop it, so every
# clip says, in positive terms, that it is one unbroken take. QC's cut check catches the rest.
CONTINUITY = "One continuous take with no cuts: the same person in the same place, filmed by the same camera from the first frame to the last."


def ltx_render(*, preset, fmt, key_path, prompt, negative, frames, seed, prefix,
               audio_path=None, audio_seek=0.0, guide_video=None, guide_start=0.0,
               guide_strength=1.0, guide_face=True, save_pose=False, key_strengths=None,
               keyframes=None, continuity=True, size=None, broll=False, vsr=False, sound=False):
    """One LTX clip. audio_path -> lip sync on a frozen vocal latent (voice);
    guide_video -> DWPose union-control guide (twins, motion).
    keyframes: [(image_path, frame_idx, strength)] pinned at later frames of the clip, in every
    stage (a look group: one clip that switches person on the beat, each look's keyframe pinned
    at its cut frame; continuity=False, since the clip is meant to change person).
    size: the clip's (last stage's) size, the run's own (formats.render_size with its plan);
    None for the preset's current size. broll: a cutaway (on MiniMax H3: see h3_render)."""
    pr = formats.PRESETS[preset]
    if pr["engine"] == "h3":  # the same call renders on MiniMax H3 (its prompt written for it)
        return h3_render(preset=preset, fmt=fmt, key_path=key_path, prompt=prompt, frames=frames, seed=seed,
                         prefix=prefix, audio_path=audio_path, audio_seek=audio_seek, guide_video=guide_video,
                         guide_start=guide_start, size=size, broll=broll, vsr=vsr)
    fps = pr["fps"]
    sizes = formats.stage_sizes(fmt, preset, size)
    g = Graph()
    union = guide_video is not None
    dev = pr["engine"] == "ltx25_dev"
    m = _ltx23(g, union) if pr["engine"] == "ltx23_r8" else _ltx25(g, union, dev)
    if continuity and CONTINUITY not in prompt:
        prompt = f"{prompt.rstrip()} {CONTINUITY}"

    def pin(stage, pos_, neg_, lat_):
        for per, idx, st in extra:
            ag = g.add("LTXVAddGuide", positive=pos_, negative=neg_, vae=m["vae"], latent=lat_,
                       image=per[stage], frame_idx=idx, strength=st)
            pos_, neg_, lat_ = O(ag, 0), O(ag, 1), O(ag, 2)
        return pos_, neg_, lat_
    pos = g.add("CLIPTextEncode", clip=m["clip"], text=prompt)
    neg = g.add("CLIPTextEncode", clip=m["clip"], text=negative)
    cond = g.add("LTXVConditioning", positive=O(pos), negative=O(neg), frame_rate=float(fps))
    p_ref, n_ref = O(cond, 0), O(cond, 1)
    keys = []
    if key_path:  # image to video; without a key it is text to video (b-roll)
        key = g.add("FVLoadImagePath", path=key_path)
        for (w, h) in sizes:
            k = g.add("ImageScale", image=O(key), upscale_method="lanczos", width=w, height=h, crop="center")
            keys.append(O(g.add("LTXVPreprocess", image=O(k), img_compression=33)))
    extra = []  # per later keyframe: its image at each stage size (after the main key: the
    for path, idx, st in (keyframes or []):  # first image a job loads is its keyframe)
        im = g.add("FVLoadImagePath", path=path)
        per = []
        for (w, h) in sizes:
            k = g.add("ImageScale", image=O(im), upscale_method="lanczos", width=w, height=h, crop="center")
            per.append(O(g.add("LTXVPreprocess", image=O(k), img_compression=33)))
        extra.append((per, int(idx), float(st)))
    ks = key_strengths or pr.get("key_strengths") or ((0.7, 0.75, 0.8) if pr["stages"] == 3 else (0.9, 0.9))
    w1, h1 = sizes[0]
    lat = g.add("EmptyLTXVLatentVideo", width=w1, height=h1, length=frames, batch_size=1)
    lat_ref = O(g.add("LTXVImgToVideoInplace", vae=m["vae"], image=keys[0], latent=O(lat),
                      strength=ks[0], bypass=False)) if keys else O(lat)
    if union:
        ctl = _pose(g, guide_video, w1, h1, frames, fps, guide_start, guide_face, lead_only=not guide_face)
        gd = g.add("LTXAddVideoICLoRAGuide", positive=p_ref, negative=n_ref, vae=m["vae"], latent=lat_ref,
                   image=ctl, frame_idx=0, strength=guide_strength, latent_downscale_factor=m["dsf"],
                   crop="disabled", use_tiled_encode=False, tile_size=256, tile_overlap=64)
        p_ref, n_ref, lat_ref = O(gd, 0), O(gd, 1), O(gd, 2)
        if save_pose:
            _save_video(g, ctl, fps, prefix + "_pose")
    if extra:
        p_ref, n_ref, lat_ref = pin(0, p_ref, n_ref, lat_ref)
    au = None
    if audio_path:
        au = O(g.add("FVLoadAudioPath", path=audio_path, seek_seconds=float(audio_seek),
                     duration=round(frames / float(fps), 4)))
        alat = g.add("LTXVAudioVAEEncode", audio=au, audio_vae=m["avae"])
        mask = g.add("SolidMask", value=0.0, width=512, height=512)
        a_ref = O(g.add("SetLatentNoiseMask", samples=O(alat), mask=O(mask)))
    else:
        a_ref = O(g.add("LTXVEmptyLatentAudio", frames_number=frames, frame_rate=int(fps), batch_size=1,
                        audio_vae=m["avae"]))
    # stage 1
    av = g.add("LTXVConcatAVLatent", video_latent=lat_ref, audio_latent=a_ref)
    if dev and pr.get("stg") and _has_node("MultimodalGuider"):
        gp = None
        for mod, kw in (("VIDEO", STG_VIDEO), ("AUDIO", STG_AUDIO)):
            gp = g.add("GuiderParameters", modality=mod, perturb_attn=True, skip_step=0, cross_attn=True,
                       **kw, **({"parameters": O(gp)} if gp else {}))
        guider = g.add("MultimodalGuider", model=m["model"], positive=p_ref, negative=n_ref, parameters=O(gp),
                       skip_blocks=STG_BLOCKS)
    else:
        guider = g.add("CFGGuider", model=m["model"], positive=p_ref, negative=n_ref, cfg=DEV_CFG if dev else 1.0)
    if dev:  # LTX's dev recipe: its own scheduler, real CFG (so the negative prompt counts), res_2s
        sig = g.add("LTXVScheduler", steps=int(pr.get("steps", DEV_STEPS)), max_shift=2.05, base_shift=0.95, stretch=True,
                    terminal=0.1, latent=O(av))
        smp = g.add("KSamplerSelect", sampler_name="res_2s")
    elif pr["engine"] == "ltx23_r8":
        sig = g.add("LTXVScheduler", steps=int(pr.get("steps1", 30)), max_shift=2.05, base_shift=0.95, stretch=True,
                    terminal=0.1, latent=O(av))
        smp = g.add("KSamplerSelect", sampler_name="res_2s")
    else:
        sig = g.add("ManualSigmas", sigmas=D25_S1)
        smp = g.add("KSamplerSelect", sampler_name="euler_ancestral")
    s = g.add("SamplerCustomAdvanced", noise=O(g.add("RandomNoise", noise_seed=seed)), guider=O(guider),
              sampler=O(smp), sigmas=O(sig), latent_image=O(av))
    sep = g.add("LTXVSeparateAVLatent", av_latent=O(s))
    crop = g.add("LTXVCropGuides", positive=p_ref, negative=n_ref, latent=O(sep, 0))
    vlat, cpos, cneg = O(crop, 2), O(crop, 0), O(crop, 1)
    # the upscale stages: LTX-2.3 runs the quality schedule's last (stages - 1) of them
    plan = ([("dpmpp_sde", R8_S2), ("dpmpp_2m_sde", R8_S3)][-(pr["stages"] - 1):] if pr["engine"] == "ltx23_r8"
            else [("res_2s" if dev else "euler_ancestral", D25_S2_8 if pr.get("up_steps") == 8 else D25_S2)])
    smodel = m["model"]
    if dev:  # the upscale stage: dev with the distilled LoRA, at cfg 1
        smodel = O(g.add("LoraLoaderModelOnly", model=smodel, lora_name=M["ltx25_distill_lora"],
                         strength_model=DEV_LORA))
    for i, (sname, sigs) in enumerate(plan, start=1):
        up = g.add("LTXVLatentUpsampler", samples=vlat, upscale_model=m["ups"], vae=m["vae"])
        li = O(g.add("LTXVImgToVideoInplace", vae=m["vae"], image=keys[i], latent=O(up), strength=ks[i],
                     bypass=False)) if keys else O(up)
        spos, sneg = cpos, cneg
        if extra:  # pin the later keyframes again at this stage's size, crop them after
            spos, sneg, li = pin(i, cpos, cneg, li)
        av = g.add("LTXVConcatAVLatent", video_latent=li, audio_latent=a_ref if audio_path else O(sep, 1))
        gd2 = g.add("CFGGuider", model=smodel, positive=spos, negative=sneg, cfg=1.0)
        s = g.add("SamplerCustomAdvanced", noise=O(g.add("RandomNoise", noise_seed=12345 + seed % 1000)),
                  guider=O(gd2), sampler=O(g.add("KSamplerSelect", sampler_name=sname)),
                  sigmas=O(g.add("ManualSigmas", sigmas=sigs)), latent_image=O(av))
        sep = g.add("LTXVSeparateAVLatent", av_latent=O(s))
        vlat = O(sep, 0)
        if extra:
            cr = g.add("LTXVCropGuides", positive=spos, negative=sneg, latent=vlat)
            vlat, cpos, cneg = O(cr, 2), O(cr, 0), O(cr, 1)
    # the whole clip in one temporal tile: 64-frame tiles leave a soft frame at every seam
    # (frames 56, 112... of a 50 fps clip); spatial tiles keep the memory in check
    t_size = min(512, (frames + 7) // 8 * 8 + 8)
    dec = g.add("VAEDecodeTiled", samples=vlat, vae=m["vae"], tile_size=512, overlap=64,
                temporal_size=t_size, temporal_overlap=8)
    if au is None and sound:  # text to video (film mode): keep the sound LTX made with the picture
        au = O(g.add("LTXVAudioVAEDecode", samples=O(sep, 1), audio_vae=m["avae"]))
    _save_video(g, O(dec), fps, prefix, audio=au)
    return g


# ------------------------------------------------------------------ detail refine (LTX-2.5 IC-LoRA)
# Lightricks' Refine-Details IC-LoRA: the finished clip as a frame-aligned guide, re-rendered on the
# distilled model in overlapping 1024x576 tiles fused at every step (LTXVTiledFusionSampler), as in
# ComfyUI-LTXVideo's LTX-2.5_V2V_TiledFusion_Upscale workflow. Framing, colour and motion stay; the
# texture, edges and grain come back that the latent upscale softened. The prompt is about the
# rendering only: every tile gets all of it, so a named subject would turn up in every tile.
REFINE_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
REFINE_PROMPT = ("sharp photographic detail, crisp natural texture, fine surface detail, clean edges, "
                 "natural film grain, high resolution footage")
REFINE_NEG = "blurry, soft, plastic, smeared detail, oversharpened halos, warped faces"
REFINE_WINDOW = 97  # frames per temporal window (the LoRA's training length; guide and sampler agree)


def refine_ready() -> bool:
    return bool(M.get("ltx25_refine_lora")) and _has_node("LTXVTiledFusionSampler") and \
        _has_node("LTXAddVideoICLoRAGuide")


def refine_clip(*, video_path, size, frames, fps, seed, prefix):
    """A clip re-rendered with the Refine-Details IC-LoRA at its own size (size: (w, h), multiples of
    32; frames: 8n+1, as LTX renders them)."""
    w, h = size
    g = Graph()
    m = _ltx25(g)
    lo = g.add("LTXICLoRALoaderModelOnly", model=m["model"], lora_name=M["ltx25_refine_lora"], strength_model=1.0)
    src = g.add("FVLoadVideoPath", path=video_path, width=w, height=h, frame_cap=frames, start_seconds=0.0,
                force_fps=0.0)
    pos = g.add("CLIPTextEncode", clip=m["clip"], text=REFINE_PROMPT)
    neg = g.add("CLIPTextEncode", clip=m["clip"], text=REFINE_NEG)
    cond = g.add("LTXVConditioning", positive=O(pos), negative=O(neg), frame_rate=float(fps))
    lat = g.add("EmptyLTXVLatentVideo", width=w, height=h, length=frames, batch_size=1)
    gd = g.add("LTXAddVideoICLoRAGuide", positive=O(cond, 0), negative=O(cond, 1), vae=m["vae"], latent=O(lat),
               image=O(src), frame_idx=0, strength=1.0, latent_downscale_factor=1.0, crop="disabled",
               use_tiled_encode=False, tile_size=256, tile_overlap=64, use_streaming=frames > REFINE_WINDOW,
               tile_frames=REFINE_WINDOW)
    tw, th = (1024, 576) if w >= h else (576, 1024)
    s = g.add("LTXVTiledFusionSampler", model=O(lo), positive=O(gd, 0), negative=O(gd, 1), latents=O(gd, 2),
              sigmas=O(g.add("ManualSigmas", sigmas=REFINE_SIGMAS)),
              sampler=O(g.add("KSamplerSelect", sampler_name="euler")), seed=int(seed), cfg=1.0,
              tile_width=min(tw, w), tile_height=min(th, h), overlap_frac=0.5, blend_var=0.05, grid_cycle=1,
              tile_frames=REFINE_WINDOW if frames > REFINE_WINDOW else 0, vae=m["vae"])
    dec = g.add("VAEDecodeTiled", samples=O(s), vae=m["vae"], tile_size=1024, overlap=128,
                temporal_size=256, temporal_overlap=32)
    _save_video(g, O(dec), fps, prefix)
    return g


# ------------------------------------------------------------------ MiniMax H3
H3_STEPS = 8          # the fast omni-reference model's schedule
H3_LIFT_AT = 6        # SelfLift: steps at the native size before the lift (the last 2 at full size)
H3_SHIFT = (12.0, 3.0)  # ModelSamplingMiniMaxH3: video, audio
# Pika-H3-Flex (a LoRA on H3's ref2va model): its six knots 1000/750/500/250/100/40 at shift 12, with
# fresh noise at every step (lcm), as in ComfyUI's Pika template; under SelfLift (euler only) the
# lift comes after 4 of them
PIKA_SIGMAS = "1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0"
PIKA_LIFT_AT = 4
# More motion: the full reference model's own schedule (as H3 Studio renders it), lifted after 15 of 20
H3_DYNAMIC_STEPS = 20
H3_DYNAMIC_LIFT_AT = 15


def pika_ready() -> bool:
    """Whether H3 b-roll can render on Pika-H3-Flex here (its LoRA and base model installed)."""
    return not missing_files(M.get("_set", "24gb"), keys=PIKA_KEYS)


def _dynamic_unet(pr: dict) -> str | None:
    """The models.json key of a More-motion preset's model (None for the other presets)."""
    return pr.get("unet", "h3_full_unet") if pr.get("dynamic") else None


def _h3(g, pika=False, full=None):
    """full: a models.json key (More motion's model) in place of fasth3."""
    unet = g.add("UNETLoader", unet_name=M[full or ("h3_pika_unet" if pika else "h3_unet")], weight_dtype="default")
    model = _ram_backups(g, O(unet))
    if pika:
        model = O(g.add("LoraLoaderModelOnly", model=model, lora_name=M["h3_pika_lora"], strength_model=1.0))
    model = O(g.add("MiniMaxH3SigmaShift", model=model, shift_video=H3_SHIFT[0], shift_audio=H3_SHIFT[1]))
    clip = g.add("CLIPLoader", clip_name=M["h3_te"], type="minimax", device="default")
    vae = g.add("VAELoader", vae_name=M["h3_vae"])
    avae = g.add("VAELoader", vae_name=M["h3_audio_vae"])
    return dict(model=model, clip=O(clip), vae=O(vae), avae=O(avae))


RELAYVSR = {"unet": "relayvsr_x2_lightx2v_int8_convrot.safetensors", "vae": "relayvsr_flash_vae_x2_fp32.safetensors"}


def relayvsr_ready() -> bool:
    """RelayVSR x2 (ComfyUI's RelayVSRConditioning and VSR Decode, Comfy-Org/RelayVSR's files)."""
    if not (_has_node("RelayVSRConditioning") and _has_node("VSRDecode")):
        return False
    try:
        import folder_paths  # type: ignore
        return (RELAYVSR["unet"] in folder_paths.get_filename_list("diffusion_models")
                and RELAYVSR["vae"] in folder_paths.get_filename_list("vae"))
    except Exception:
        return False


def _relayvsr(g, images, seed):
    """images 2x with RelayVSR: every 15th frame through the model (4 steps, cfg 1), the Flash
    decoder restores every frame from the low-quality ones."""
    model = g.add("UNETLoader", unet_name=RELAYVSR["unet"], weight_dtype="default")
    try:  # comfy kitchen's int8 attention, when this GPU and ComfyUI have it (as the RelayVSR workflow)
        import comfy.ldm.modules.attention as _att  # type: ignore
        kitchen = bool(getattr(_att, "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE", False))
    except Exception:
        kitchen = False
    if kitchen and _has_node("ModelAttentionBackend"):
        model = g.add("ModelAttentionBackend", model=O(model), attention="comfy kitchen attention")
    cond = g.add("RelayVSRConditioning", model=O(model), images=images)
    guider = g.add("CFGGuider", model=O(model), positive=O(cond, 0), negative=O(cond, 0), cfg=1.0)
    s = g.add("SamplerCustomAdvanced", noise=O(g.add("RandomNoise", noise_seed=seed)), guider=O(guider),
              sampler=O(g.add("KSamplerSelect", sampler_name="euler")),
              sigmas=O(g.add("BasicScheduler", model=O(model), scheduler="simple", steps=4, denoise=1.0)),
              latent_image=O(cond, 1))
    vae = g.add("VAELoader", vae_name=RELAYVSR["vae"])
    return O(g.add("VSRDecode", vae=O(vae), samples=O(s), images=images, detail=1.0))


def h3_upscaler() -> str | None:
    """The H3 latent upscaler SelfLift lifts Quality clips with, or None (no SelfLift node or
    no upscaler file: Quality then renders at H3's native size and is scaled up)."""
    if not _has_node("SelfLiftH3Sampler"):
        return None
    try:
        import folder_paths  # type: ignore
        names = folder_paths.get_filename_list("latent_upscale_models")
    except Exception:
        return "minimax_h3_latent_upscaler_3d_fp32.pth"  # outside ComfyUI: the tested one
    h3 = sorted((n for n in names if "minimax_h3_latent_upscaler" in n.lower()),
                key=lambda n: ("3d_fp32" not in n, n))  # 3d_fp32: the one tested on OneShot MV clips
    return h3[0] if h3 else None


def h3_render(*, preset, fmt, key_path, prompt, frames, seed, prefix, audio_path=None, audio_seek=0.0,
              guide_video=None, guide_start=0.0, size=None, broll=False, vsr=False):
    """One MiniMax H3 clip in reference mode (fv/h3.py writes its prompt): <Picture 1> the
    keyframe, also pinned as the first frame; <Audio 1> the clip's vocal, copied 1:1 and pinned
    at frame 0 so the mouth lands on it; <Video 1> (a look) the lead's clip, for the pose. The
    take is saved with the original vocal, like an LTX clip.
    Above H3's native size (Quality), the first H3_LIFT_AT steps run at the native size, H3's
    latent upscaler lifts the clip, and the last steps run at full size (SelfLift); without
    SelfLift the clip renders at the native size and is scaled up.
    broll: a cutaway, rendered on Pika-H3-Flex when it's installed (on B-roll it kept to the
    keyframe and never letterboxed, in about 80% of fasth3's time; on singers it was softer);
    on More motion it renders on the full model like the singers.
    The Pika preset renders every clip on it."""
    fps = formats.H3_FPS
    dynamic = formats.PRESETS[preset].get("dynamic", False)  # More motion: singers and b-roll on the full model
    pika = ((broll and not dynamic) or formats.PRESETS[preset].get("pika", False)) and pika_ready()
    w, h = size or formats.render_size(fmt, preset)
    nw, nh = formats.H3_NATIVE[fmt]
    lift = w * h > nw * nh
    vsr = bool(vsr) and lift and relayvsr_ready()  # RelayVSR: sample at the native size, then 2x
    ups = h3_upscaler() if lift and not vsr else None
    rw, rh = (w, h) if ups or not lift else (nw, nh)  # the size H3 samples at
    g = Graph()
    m = _h3(g, pika, full=_dynamic_unet(formats.PRESETS[preset]))
    refs = {}
    key = None
    if key_path:
        key = g.add("FVLoadImagePath", path=key_path)
        refs["ref_images.ref_image_0"] = O(key)
    au = None
    if audio_path:
        au = O(g.add("FVLoadAudioPath", path=audio_path, seek_seconds=max(0.0, float(audio_seek)),
                     duration=round(frames / float(fps), 4)))
        refs["ref_audios.ref_audio_0"] = au
    if guide_video:
        vid = g.add("FVLoadVideoPath", path=guide_video, width=nw if lift else w, height=nh if lift else h, frame_cap=frames,
                    start_seconds=float(guide_start), force_fps=float(fps))
        refs["ref_videos.ref_video_0"] = O(vid)
    r2v = g.add("MiniMaxH3ReferenceToVideo", clip=m["clip"], vae=m["vae"], audio_vae=m["avae"], prompt=prompt,
                width=rw, height=rh, length=int(frames), ref_image_size="match", **refs)
    pos, lat = O(r2v, 0), O(r2v, 1)
    pin = key is not None and not dynamic  # More motion: the keyframe guides the face, not frame 0
    if pin or au is not None:
        kw = {"image": O(key), "vae": m["vae"]} if pin else {}
        if au is not None:
            kw.update(audio=au, audio_vae=m["avae"])
        pos = O(g.add("MiniMaxH3AddGuide", positive=pos, latent=lat, frame_idx=0, **kw))
    guider = g.add("BasicGuider", model=m["model"], conditioning=pos)
    sig = (g.add("ManualSigmas", sigmas=PIKA_SIGMAS) if pika else
           g.add("BasicScheduler", model=m["model"], scheduler="simple", steps=H3_DYNAMIC_STEPS, denoise=1.0) if dynamic else
           g.add("BasicScheduler", model=m["model"], scheduler="beta", steps=H3_STEPS, denoise=1.0))
    euler = g.add("KSamplerSelect", sampler_name="lcm" if pika and not ups else
                  "res_multistep" if dynamic and not ups else "euler")
    if ups:
        s = g.add("SelfLiftH3Sampler", model=m["model"], positive=pos, negative=pos, vae=m["vae"], latent_image=lat,
                  sampler=O(euler), sigmas=O(sig), seed=seed, cfg=1.0,
                  transition_step=PIKA_LIFT_AT if pika else H3_DYNAMIC_LIFT_AT if dynamic else H3_LIFT_AT,
                  lowres_scale=round(nh / h, 4), rho=0.0, w_min=0.5, w_max=1.0, upscaler_model=ups,
                  upscaler_unload=True)
    else:
        s = g.add("SamplerCustomAdvanced", noise=O(g.add("RandomNoise", noise_seed=seed)), guider=O(guider),
                  sampler=O(euler), sigmas=O(sig), latent_image=lat)
    dec = O(g.add("VAEDecode", samples=O(s), vae=m["vae"]))
    if vsr:
        dec = _relayvsr(g, dec, seed)
    if (rw, rh) != (w, h):
        dec = O(g.add("ImageScale", image=dec, upscale_method="lanczos", width=w, height=h, crop="disabled"))
    if au is None:  # no vocal (b-roll): keep H3's own sound
        au = O(g.add("VAEDecodeAudio", samples=O(s), vae=m["avae"]))
    _save_video(g, dec, fps, prefix, audio=au)
    return g


# ------------------------------------------------------------------ image edits
def _klein(g, image_ref, prompt, refs=(), seed=7, steps=4):
    unet = g.add("UNETLoader", unet_name=M["klein_unet"], weight_dtype="default")
    clip = g.add("CLIPLoader", clip_name=M["klein_te"], type="flux2", device="default")
    vae = g.add("VAELoader", vae_name=M["klein_vae"])
    size = g.add("GetImageSize", image=image_ref)
    pos = g.add("CLIPTextEncode", clip=O(clip), text=prompt)
    neg = g.add("ConditioningZeroOut", conditioning=O(pos))
    p_ref, n_ref = O(pos), O(neg)
    for im in [image_ref] + list(refs):
        enc = g.add("VAEEncode", pixels=im, vae=O(vae))
        p_ref = O(g.add("ReferenceLatent", conditioning=p_ref, latent=O(enc)))
        n_ref = O(g.add("ReferenceLatent", conditioning=n_ref, latent=O(enc)))
    guider = g.add("CFGGuider", model=O(unet), positive=p_ref, negative=n_ref, cfg=1.0)
    sched = g.add("Flux2Scheduler", steps=steps, width=O(size, 0), height=O(size, 1))
    lat = g.add("EmptyFlux2LatentImage", width=O(size, 0), height=O(size, 1), batch_size=1)
    smp = g.add("SamplerCustomAdvanced", noise=O(g.add("RandomNoise", noise_seed=seed)), guider=O(guider),
                sampler=O(g.add("KSamplerSelect", sampler_name="euler")), sigmas=O(sched), latent_image=O(lat))
    return O(g.add("VAEDecode", samples=O(smp), vae=O(vae)))


def _qwen21(g, image_ref, prompt, refs=(), seed=7, steps=25, resolution=1024):
    unet = g.add("UNETLoader", unet_name=M["qwen21_unet"], weight_dtype="default")
    cache = g.add("QwenImage21Cache", model=O(unet), device="auto", dtype="default")
    clip = g.add("CLIPLoader", clip_name=M["qwen21_te"], type="qwen_image", device="default")
    vae = g.add("VAELoader", vae_name=M["qwen21_vae"])
    imgs = {"images.image_1": image_ref}
    for i, r in enumerate(refs):
        imgs[f"images.image_{i + 2}"] = r
    enc = g.add("TextEncodeQwenImage21", clip=O(clip), prompt=prompt, negative_prompt="",
                resolution=resolution, vae=O(vae), **imgs)
    ks = g.add("KSampler", model=O(cache), seed=seed, steps=steps, cfg=1.0, sampler_name="euler",
               scheduler="simple", positive=O(enc, 0), negative=O(enc, 1), latent_image=O(enc, 2), denoise=1.0)
    return O(g.add("VAEDecode", samples=O(ks), vae=O(vae)))


TWIN_PROMPT = ("Recreate image 1 with the person from image 2. Keep image 1's exact camera angle, framing, "
               "head position, head tilt, mouth shape and facial expression. Use image 2's identity, face, "
               "hair, skin, makeup, jewelry, clothing, lighting colour and background. Photorealistic.")


TWIN_IDENTITY = ("This is the same person as in image 2: keep their exact face, eyes, nose, lips, jawline, skin "
                 "tone, hair colour and age. Only the hairstyle, outfit and place change.")


TWIN_TEXT = ("Change only the person into {who}. {detail} {wear} Light: {light}. Background: {background}. "
             "Keep the camera exactly as in the image: the same distance, lens, angle and framing, the same space "
             "above the head, the same amount of shoulders and body in frame, the face the same size and in the "
             "same place. Keep the head position, head tilt and the open, mid-word mouth exactly as in the image. "
             "A completely different person. No text or lettering anywhere.")  # + casting.edit_realism()
# no "the same tight crop" and no close-up skin brief: with them Klein redraws every look as a
# tight, mouth-shut face shot (in one test all 24 looks came out tighter than their lead's frame)


def twin_key(*, video_path, prefix, size, seed=7, editor="klein", prompt=None, portrait_path=None,
             start_s=0.0, identity_path=None, keep=None):
    """The frame of performer A's clip at `start_s` re-cast as performer B (same pose and mouth).
    editor 'klein' (default): text-described B, pose kept by construction (fast, no portrait).
    editor 'qwen': B's Krea 2 portrait as the identity reference (tends to copy the portrait's pose).
    identity_path (artist runs, Klein): the artist photo as a second reference image, so a new
    outfit, hairstyle and place keep the artist's face (without it the looks drift into other
    people: ArcFace similarity to the artist 0.20-0.37 without, 0.33-0.63 with)."""
    g = Graph()
    fr = g.add("FVLoadVideoPath", path=video_path, width=size[0], height=size[1], frame_cap=1,
               start_seconds=float(start_s), force_fps=0.0)
    if editor == "qwen":
        ref = g.add("FVLoadImagePath", path=portrait_path)
        out = _qwen21(g, O(fr), prompt or TWIN_PROMPT, [O(ref)], seed=seed)
    elif identity_path:
        ref = g.add("FVLoadImagePath", path=identity_path)
        out = _klein(g, O(fr), f"{prompt} {TWIN_IDENTITY}", refs=[O(ref)], seed=seed)
    else:
        out = _klein(g, O(fr), prompt, seed=seed)
    out = O(g.add("ImageScale", image=out, upscale_method="lanczos", width=size[0], height=size[1], crop="center"))
    if _has_node("FVFaceMatch"):  # the edit pulls the camera back: put the face back where the lead's is
        out = O(g.add("FVFaceMatch", image=out, reference=O(fr), max_scale=2.2))
    g.add("SaveImage", images=out, filename_prefix=prefix)
    if keep:  # staged runs: LTX stays loaded between the singer clips and the look clips
        keep_ltx(g, keep)
    return g


def motion_key(*, video_path, start_s, prompt, prefix, size, seed=7):
    """The dance frame at `start_s` re-cast as a character (Klein edit, keeps the pose)."""
    g = Graph()
    fr = g.add("FVLoadVideoPath", path=video_path, width=size[0], height=size[1], frame_cap=1,
               start_seconds=float(start_s), force_fps=0.0)
    img = O(fr)
    if _has_node("FVLeadMask"):  # grey out backup dancers so the edit keeps one person
        dw = g.add("DWPreprocessor", image=img, detect_hand="enable", detect_body="enable",
                   detect_face="disable", resolution=512, bbox_detector=M["dwpose_bbox"],
                   pose_estimator=M["dwpose_pose"], scale_stick_for_xinsr_cn="disable")
        img = O(g.add("FVLeadMask", pose_keypoint=O(dw, 1), image=img))
    out = _klein(g, img, prompt, seed=seed)
    g.add("SaveImage", images=out, filename_prefix=prefix)
    return g


def free_memory(note="free"):
    g = Graph()
    g.add("FVFreeMemory", note=note)
    return g


def assemble(*, run_dir, prefix):
    g = Graph()
    g.add("FVAssemble", run_dir=run_dir, filename_prefix=prefix)
    return g


def keep_ltx(g, preset="quality"):
    """Wire the video model's loaders into an FVKeep node of graph g, so a job that doesn't use it
    (a QC pass, a Klein edit) keeps it loaded for the next video job instead of a reload from disk."""
    if not _has_node("FVKeep"):
        return g
    pr = formats.PRESETS.get(preset, formats.PRESETS["quality"])
    if pr["engine"] == "h3":
        m = _h3(g, pr.get("pika", False) and pika_ready(), full=_dynamic_unet(pr))
        g.add("FVKeep", model=m["model"], clip=m["clip"], vae=m["vae"], audio_vae=m["avae"])
        return g
    m = _ltx23(g) if pr["engine"] == "ltx23_r8" else _ltx25(g, dev=pr["engine"] == "ltx25_dev")
    g.add("FVKeep", model=m["model"], clip=m["clip"], vae=m["vae"], audio_vae=m["avae"], upscaler=m["ups"])
    return g


def qc(*, run_dir, pass_index=0, max_passes=1, then_assemble=True, stage="all", keep=None):
    """keep: the preset whose LTX stays loaded through this pass (staged runs)."""
    g = Graph()
    kw = {"stage": stage} if stage != "all" else {}
    g.add("FVQC", run_dir=run_dir, pass_index=pass_index, max_passes=max_passes, then_assemble=then_assemble, **kw)
    if keep:
        keep_ltx(g, keep)
    return g


def critic(*, run_dir, pass_index=0, max_passes=2):
    """The vision check of a run's stills (director.critic_pass)."""
    g = Graph()
    g.add("FVCritic", run_dir=run_dir, pass_index=pass_index, max_passes=max_passes)
    return g
