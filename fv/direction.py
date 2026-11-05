"""The Direction: how the video should change over the song, in the director's own words.

One note per part of the song, a line each, e.g.

    Opening: quiet, intimate close-ups, almost no magic.
    Verses: playful performance, miniature cities.
    Pre-choruses: wings growing, bigger camera moves.
    Choruses: explosive dancing, spectacular effects.
    Bridge: emotional, stripped back.
    Final chorus and outro: maximum spectacle.
    1:12-1:30: the wings fold away.

A line names one or more parts (opening / intro, verse, pre-chorus / build, chorus / hook / drop,
bridge / break, final chorus, outro / ending), or a time range in the song (m:ss-m:ss). A line
with no part ("Everything in warm light.", or "All: ...") goes on every shot. Each shot gets the
notes for its part of the song (beat.label_sections and the planner's energy phases say which
part that is), added to its singer's portrait and its cutaways' stills, and a short gist of its
own part's notes to the cutaways' motion. Singer clip prompts get none: a long note buries LTX's
lip-sync wording (lead clips scored SyncNet ~6 with ~100-word prompts, under 2 with 300 words of
Direction). Notes about the edit ("Editing rules: ...") stay out of the pictures.
"""
from __future__ import annotations

import re

# a part's names in a note -> the part
_PARTS = [
    ("final", r"(?:final|last|closing)\s+(?:chorus(?:es)?|hooks?|drops?)"),
    ("build", r"pre[\s-]?chorus(?:es)?|builds?|build[\s-]?ups?|risers?|lift"),
    ("intro", r"opening|intros?|introduction|the\s+start|beginning|cold\s+open"),
    ("verse", r"verses?"),
    ("chorus", r"chorus(?:es)?|hooks?|refrains?|drops?"),
    ("break", r"bridges?|breaks?|breakdowns?|middle\s+(?:eight|8)"),
    ("outro", r"outros?|ending|the\s+end|coda|fade[\s-]?out"),
    ("all", r"all|everything|everywhere|throughout|overall|general|always|whole\s+(?:video|song)"),
]
_PART_RE = [(k, re.compile(rf"\b(?:{rx})\b", re.I)) for k, rx in _PARTS]
_TIME = re.compile(r"^\s*(\d{1,2}):(\d{2}(?:\.\d+)?)\s*[-–—to]+\s*(\d{1,2}):(\d{2}(?:\.\d+)?)\s*[:.)-]?\s*(.*)$", re.I)
_BULLET = re.compile(r"^\s*(?:[-*•·]|\d+[.)])\s+")
_EDIT = re.compile(r"^(?:editing|edit|cutting|cuts|pacing|transitions?)\b[^:]{0,30}:", re.I)
CLIP_WORDS, STILL_WORDS = 30, 90
LABELS = {"intro": "opening", "verse": "verses", "build": "pre-choruses", "chorus": "choruses", "break": "bridge",
          "final": "final chorus", "outro": "outro", "all": "every shot"}


def parse(text: str) -> list[dict]:
    """The notes in a Direction: [{"parts": [...], "text": ...} or {"t0", "t1", "text"}]."""
    out = []
    for raw in (text or "").splitlines():
        line = _BULLET.sub("", raw).strip()
        if not line or _EDIT.match(line):  # how to cut it, not what is in the picture
            continue
        m = _TIME.match(line)
        if m and m.group(5).strip():
            a = int(m.group(1)) * 60 + float(m.group(2))
            b = int(m.group(3)) * 60 + float(m.group(4))
            if b > a:
                out.append({"t0": a, "t1": b, "text": m.group(5).strip()})
                continue
        head, sep, body = line.partition(":")
        parts = []
        if sep and body.strip() and len(head) <= 60:
            rest = head
            for k, rx in _PART_RE:  # the longest names first: "final chorus" before "chorus"
                if rx.search(rest):
                    parts.append(k)
                    rest = rx.sub(" ", rest)
        if parts:
            out.append({"parts": parts, "text": body.strip()})
        elif out and "parts" in out[-1] and not sep and raw[:1].isspace():
            out[-1]["text"] += " " + line  # a wrapped note
        else:
            out.append({"parts": ["all"], "text": line})
    for n in out:
        n["text"] = " ".join(n["text"].split())
    return [n for n in out if n["text"]]


def _part_of(p: dict) -> dict:
    """shot index -> its part of the song (intro / verse / build / chorus / break / outro, and
    "final" on the last run of chorus shots)."""
    from .plan import _intense
    secs = p.get("sections") or []
    shots = [s for s in p.get("shots", []) if "index" in s]
    bar = 240.0 / max(1.0, float(p.get("tempo") or 120.0))
    hot = [s for s in secs if _intense(s)]

    def sec_at(t):
        for s in secs:
            if s["start"] - 1e-3 <= t < s["end"]:
                return s
        if secs and t < secs[0]["start"]:  # a pickup before the first section
            return secs[0]
        return secs[-1] if secs else {}
    part = {}
    for sh in shots:
        mid = 0.5 * (sh["t0"] + sh["t1"])
        s = sec_at(mid)
        lab = s.get("label", "")
        if sh.get("phase") == "drop" or (s and _intense(s)):
            part[sh["index"]] = "chorus"
        elif lab == "build" or any(0 <= h["start"] - sh["t1"] <= 2 * bar + 0.05 or
                                   (h["start"] > sh["t0"] and h["start"] - sh["t0"] <= 2 * bar)
                                   for h in hot):
            part[sh["index"]] = "build"
        elif lab in ("intro", "break", "outro", "verse"):
            part[sh["index"]] = lab
        else:
            part[sh["index"]] = "verse"
    # a song that starts straight in a verse still has an opening: its first four bars
    if shots and "intro" not in part.values() and float(p.get("song_t0", 0.0)) < 1.0:
        first = sec_at(0.0)
        for sh in shots:
            if sh["t0"] < min(first.get("end", 0.0), first.get("start", 0.0) + 4 * bar) - 1e-3 \
                    and part[sh["index"]] == "verse":
                part[sh["index"]] = "intro"
    # the final chorus: the last unbroken run of chorus shots (a loud outro included)
    idx = [sh["index"] for sh in shots]
    run = []
    for i in reversed(idx):
        if part[i] == "chorus":
            run.append(i)
        elif run:
            break
    if run and any(part[i] == "chorus" for i in idx if i not in run):  # one chorus is just "the chorus"
        for i in run:
            part[i] = "final"
    return part


def assign(p: dict, text: str) -> dict:
    """Puts each shot's notes on it (shot["direction"]: its part's notes (a final chorus also takes
    the chorus notes when there are no final-chorus ones), time-range ones, then every-shot ones;
    shot["direction_clip"]: a short gist of its own part's) and returns {shot index: text}."""
    notes = parse(text)
    if not notes:
        return {}
    part = _part_of(p)
    t0 = float(p.get("song_t0", 0.0))
    have = {k for n in notes for k in n.get("parts", [])}
    out = {}
    for sh in p.get("shots", []):
        if "index" not in sh:
            continue
        pt = part.get(sh["index"], "verse")
        want = {"all", pt} | ({"chorus"} if pt == "final" and "final" not in have else set())
        a, b = t0 + sh["t0"], t0 + sh["t1"]
        own = [n["text"] for n in notes if "parts" in n and (want - {"all"}) & set(n["parts"])]
        own += [n["text"] for n in notes if "t0" in n and n["t0"] < b and n["t1"] > a]
        got = own + [n["text"] for n in notes if "all" in n.get("parts", ())]
        sh["part"] = pt
        if got:
            sh["direction"] = " ".join(x if x[-1] in ".!?" else x + "." for x in got)
            sh["direction_clip"] = gist(own, CLIP_WORDS)
            out[sh["index"]] = sh["direction"]
    p["direction"] = text.strip()
    return out


def summary(p: dict) -> list[str]:
    """'0:00-0:14 opening (4 shots): quiet, intimate close-ups.' per run of shots with the same note."""
    t0 = float(p.get("song_t0", 0.0))
    runs = []
    for sh in p.get("shots", []):
        if "index" not in sh:
            continue
        key = (sh.get("part", ""), sh.get("direction", ""))
        if runs and runs[-1][0] == key:
            runs[-1][2], runs[-1][3] = sh["t1"], runs[-1][3] + 1
        else:
            runs.append([key, sh["t0"], sh["t1"], 1])

    def mmss(t):
        return f"{int(t // 60)}:{int(t % 60):02d}"
    return [f"{mmss(t0 + a)}-{mmss(t0 + b)} {LABELS.get(pt, pt)} ({n} shot{'s' if n > 1 else ''}): {d or '(no note)'}"
            for (pt, d), a, b, n in runs]


def gist(notes: list[str], words: int) -> str:
    """The notes' first sentences, up to `words` words (the first sentence cut short if it alone
    is longer)."""
    out, n = [], 0
    for x in notes:
        for sent in re.split(r"(?<=[.!?])\s+", x.strip()):
            w = sent.split()
            if out and n + len(w) > words:
                return " ".join(out)
            out.append(" ".join(w[:words]) + ("." if len(w) > words else ""))
            n += len(w)
            if n >= words:
                return " ".join(out)
    return " ".join(out)


def still(prompt: str, note: str) -> str:
    """A portrait or still prompt with its shot's note, cut to STILL_WORDS."""
    return f"{prompt.rstrip()} In this part of the video: {gist([note], STILL_WORDS)}" if note else prompt


def clip(prompt: str, note: str) -> str:
    """A cutaway's motion prompt with its shot's direction_clip gist."""
    return f"{prompt.rstrip()} In this part of the video: {note}" if note else prompt
