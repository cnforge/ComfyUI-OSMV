"""The run's lyrics, word by word, from its vocal stem (OpenAI Whisper).

MiniMax H3 copies a clip's vocal as its soundtrack, but its prompt has to spell out the words
that are sung (`<d>[English] ...</d>`, see fv/h3.py), so a run on H3 transcribes its vocal once
and each clip takes the words that fall inside it. The model (large-v3-turbo, about 1.6 GB) is
downloaded on first use into ComfyUI/models/whisper.
"""
from __future__ import annotations

import json
import os

MODEL = "turbo"
SR = 16000


def _model_dir() -> str:
    try:
        import folder_paths  # type: ignore
        return os.path.join(folder_paths.models_dir, "whisper")
    except Exception:
        return os.path.join(os.path.expanduser("~"), ".cache", "whisper")


def transcribe(run_dir: str, vocal_path: str, t0: float = 0.0, log=print) -> dict:
    """{"language": "English", "words": [{"t0", "t1", "w"}]} with times on the song window's
    clock (the stem starts `t0` seconds before the window); cached in <run>/lyrics.json."""
    out = os.path.join(run_dir, "lyrics.json")
    if os.path.exists(out):
        return json.load(open(out, encoding="utf-8"))
    import numpy as np
    import whisper  # type: ignore
    from whisper.tokenizer import LANGUAGES  # type: ignore
    from . import stems
    x, sr = stems.read_wav(vocal_path)
    x = x.mean(0) if x.ndim == 2 else x
    if sr != SR:
        import torch
        import torchaudio.functional as AF
        x = AF.resample(torch.from_numpy(np.ascontiguousarray(x, np.float32)), sr, SR).numpy()
    model = whisper.load_model(MODEL, download_root=_model_dir())
    try:
        res = model.transcribe(x.astype(np.float32), word_timestamps=True, condition_on_previous_text=False,
                               no_speech_threshold=0.5, hallucination_silence_threshold=1.0, verbose=None)
    finally:
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    words = [{"t0": round(w["start"] - t0, 3), "t1": round(w["end"] - t0, 3), "w": w["word"].strip()}
             for seg in res.get("segments", []) for w in seg.get("words", []) if w.get("word", "").strip()]
    lang = LANGUAGES.get(res.get("language") or "en", "english").title()
    data = {"language": lang, "words": words}
    json.dump(data, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    log(f"[OneShotMV] lyrics: {len(words)} words ({lang})")
    return data


def between(lyr: dict | None, t0: float, t1: float) -> str:
    """The words sung between t0 and t1 (window seconds): a word counts when its middle is inside."""
    if not lyr:
        return ""
    ws = [w["w"] for w in lyr.get("words", []) if t0 <= (w["t0"] + w["t1"]) / 2 < t1]
    return " ".join(ws).strip()
