"""Make the song when there isn't one: Gemma 3 (the LTX-2.3 text encoder) writes the lyrics and the
style from the World line, then ACE-Step 1.5 sings it. The song lands in ComfyUI's input folder
(oneshot-mv-songs/), so the rest of the run treats it exactly like an uploaded one."""
from __future__ import annotations

import json
import os
import re
import time

import numpy as np

SUBDIR = "oneshot-mv-songs"
# ACE-Step 1.5 XL turbo, ComfyUI's split files (huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files)
M = {"unet": "acestep_v1.5_xl_turbo_bf16.safetensors", "te1": "qwen_0.6b_ace15.safetensors",
     "te2": "qwen_4b_ace15.safetensors", "vae": "ace_1.5_vae.safetensors"}
_FOLDERS = {"unet": "diffusion_models", "te1": "text_encoders", "te2": "text_encoders", "vae": "vae"}
KEYS = [f"{r} {q}" for q in ("major", "minor") for r in ("C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B")]


def missing() -> list[str]:
    """The ACE-Step files ComfyUI can't find."""
    try:
        import folder_paths  # type: ignore
    except Exception:
        return list(M.values())
    out = []
    for k, f in M.items():
        if f not in (folder_paths.get_filename_list(_FOLDERS[k]) or []):
            out.append(f"{_FOLDERS[k]}/{f}")
    return out


def _structure(seconds: float) -> str:
    if seconds <= 75:
        return "[Verse], [Chorus], [Verse], [Chorus]"
    if seconds <= 150:
        return "[Intro], [Verse], [Pre-Chorus], [Chorus], [Verse], [Chorus], [Bridge], [Chorus], [Outro]"
    return ("[Intro], [Verse], [Pre-Chorus], [Chorus], [Verse], [Pre-Chorus], [Chorus], [Bridge], "
            "[Chorus], [Chorus], [Outro]")


_SECTION = re.compile(r"^\W*(intro|verse|pre-?chorus|chorus|bridge|hook|outro|breakdown|instrumental|interlude)"
                      r"(\s*\d+)?\W*$", re.I)


def tag_sections(lyrics: str) -> str:
    """Section names on their own line as ACE-Step's tags ("Verse 1:" -> "[Verse 1]"); blank lines go."""
    out = []
    for ln in lyrics.splitlines():
        t = ln.strip()
        if not t:
            continue
        m = _SECTION.match(t)
        out.append(f"[{m.group(1).title()}{(m.group(2) or '').rstrip()}]" if m and not t.startswith("[") else t)
    return "\n".join(out)


def write(world: str, style: str = "", seconds: float = 120, seed: int = 1, instrumental: bool = False,
          log=print, have_lyrics: bool = False) -> dict:
    """{title, tags, bpm, key, lyrics} for a song that belongs in this World."""
    from . import theme
    lines = max(8, int(seconds / 4.5))
    want = (f"Write an original pop song for a music video set in this world: {world.strip() or 'anything you like'}.\n"
            + (f"The music should be: {style.strip()}.\n" if style.strip() else
               "Pick a music genre that suits this world and makes a catchy, danceable music video.\n")
            + ("It is an instrumental: no lyrics.\n" if instrumental else
               "The lyrics are written already: only the music.\n" if have_lyrics else
               f"Lyrics: about {lines} short singable lines in English, a big repeated hook in the chorus, "
               f"in this structure: {_structure(seconds)}. Each section starts with its tag on its own line. "
               "No artist or brand names, nothing copied from real songs.\n")
            + "Reply with only a JSON object with these keys: \"title\" (2 to 4 words), \"tags\" (one line: genre, "
              "mood, instruments, the lead voice (male or female vocal), production, comma separated, no BPM), \"bpm\" (a number), "
              "\"key\" (like \"A minor\")" + ("" if instrumental or have_lyrics else ", \"lyrics\" (the lyrics, with \\n between lines)") + ".")
    d = None
    for attempt in range(3):
        d = theme._parse(theme.llm(want, seed=seed + attempt, max_length=1400, log=log) or "")
        if d and d.get("tags") and (instrumental or have_lyrics or len(str(d.get("lyrics", "")).strip()) > 40):
            break
        d = None
    d = d or {}
    tags = str(d.get("tags") or style or "upbeat pop, catchy synth hook, punchy drums, clear lead vocal").strip()
    tags = ", ".join(t.strip() for t in tags.split(",")  # the prompt's own words echoed back ("comma separated")
                     if t.strip() and not re.search(r"comma|separated|\bbpm\b", t, re.I))
    if style.strip() and style.strip().lower() not in tags.lower():
        tags = f"{style.strip()}, {tags}"
    try:
        bpm = int(round(float(re.sub(r"[^0-9.]", "", str(d.get("bpm", "")) or "0"))))
    except ValueError:
        bpm = 0
    bpm = bpm if 60 <= bpm <= 200 else 118
    key = str(d.get("key") or "").strip()
    key = next((k for k in KEYS if k.lower() == key.lower()), "A minor")
    lyrics = "[Instrumental]" if instrumental else str(d.get("lyrics") or "").replace("\\n", "\n").strip()
    lyrics = tag_sections(lyrics)
    if not lyrics:
        lyrics = ("[Verse]\nLights come up and the night is new\nEverybody here is waiting on you\n"
                  "[Chorus]\nOne more time, turn it up loud\nOne more time, lose it in the crowd\n") * 2
    title = re.sub(r"[^\w' -]+", "", str(d.get("title") or "Untitled")).strip()[:40] or "Untitled"
    return {"title": title, "tags": tags, "bpm": bpm, "key": key, "lyrics": lyrics}


def sing(spec: dict, seconds: float, seed: int, log=print) -> tuple[np.ndarray, int]:
    """ACE-Step 1.5 XL turbo: 8 steps, cfg 1 (ComfyUI's template). (channels, samples), sample rate."""
    import nodes as comfy_nodes  # type: ignore
    from .theme import _call
    N = comfy_nodes.NODE_CLASS_MAPPINGS
    model = _call(N["UNETLoader"], unet_name=M["unet"], weight_dtype="default")[0]
    model = _call(N["ModelSamplingAuraFlow"], model=model, shift=3.0)[0]
    clip = _call(N["DualCLIPLoader"], clip_name1=M["te1"], clip_name2=M["te2"], type="ace", device="default")[0]
    vae = _call(N["VAELoader"], vae_name=M["vae"])[0]
    pos = _call(N["TextEncodeAceStepAudio1.5"], clip=clip, tags=spec["tags"], lyrics=spec["lyrics"], seed=int(seed),
                bpm=int(spec["bpm"]), duration=float(seconds), timesignature="4", language="en", keyscale=spec["key"],
                generate_audio_codes=True, cfg_scale=2.0, temperature=0.85, top_p=0.9, top_k=0, min_p=0.0)[0]
    neg = _call(N["ConditioningZeroOut"], conditioning=pos)[0]
    lat = _call(N["EmptyAceStep1.5LatentAudio"], seconds=float(seconds), batch_size=1)[0]
    lat = _call(N["KSampler"], model=model, seed=int(seed), steps=8, cfg=1.0, sampler_name="euler", scheduler="simple",
                positive=pos, negative=neg, latent_image=lat, denoise=1.0)[0]
    audio = _call(N["VAEDecodeAudio"], vae=vae, samples=lat)[0]
    w = audio["waveform"][0].float().cpu().numpy()
    peak = float(np.abs(w).max() or 1.0)
    if peak > 0.98:
        w = w * (0.98 / peak)
    return w.astype(np.float32), int(audio["sample_rate"])


def make(world: str, style: str = "", seconds: float = 120, seed: int = 1, instrumental: bool = False,
         lyrics: str = "", input_dir: str = "", log=print) -> dict:
    """Write and sing a song into <input>/oneshot-mv-songs/. {song: its input-folder name, ...the spec}"""
    from . import stems
    miss = missing()
    if miss:
        raise RuntimeError("ACE-Step 1.5 isn't installed: " + ", ".join(miss))
    t = time.time()
    spec = write(world, style, seconds, seed, instrumental, log, have_lyrics=bool(lyrics.strip()))
    if lyrics.strip() and not instrumental:
        spec["lyrics"] = lyrics.strip()
    log(f"[OneShotMV] song: \"{spec['title']}\", {spec['bpm']} BPM, {spec['key']}: {spec['tags']} ({time.time() - t:.0f} s)")
    t = time.time()
    x, sr = sing(spec, seconds, seed, log)
    log(f"[OneShotMV] song: sung in {time.time() - t:.0f} s")
    d = os.path.join(input_dir, SUBDIR)
    os.makedirs(d, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", spec["title"].lower()).strip("-")[:40] or "song"
    stem = f"{slug}-{time.strftime('%Y%m%d-%H%M%S')}"
    stems.write_wav(os.path.join(d, stem + ".wav"), x, sr)
    meta = {**spec, "world": world, "style": style, "seconds": seconds, "seed": seed}
    with open(os.path.join(d, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return {"song": f"{SUBDIR}/{stem}.wav", **meta}
