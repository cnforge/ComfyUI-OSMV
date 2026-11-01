"""Clip prompts in each video engine's own structure.

The planner and casting write one plain description per clip (who, the shot, the singing or
not, the camera); this turns it into what the engine was trained on:

- LTX-2.5 (the LTX-2.5 prompting guide, github.com/benjiyaya/LTX-2.5-Agent-Prompt-Skill): one
  flowing present-tense paragraph, shot first, then the action, the camera, and last the sound.
  From a first frame it describes the motion, not what the frame already shows.
- MiniMax H3 (H3's prompt-writing skill, github.com/MiniMax-AI/MiniMax-H3, skills/
  h3-prompt-writing, full-reference mode): six labelled sections. `subject_definitions` names
  every reference (<Picture 1> the keyframe, <Audio 1> the vocal, <Video 1> a look's lead clip),
  `retention_analysis` says how each is kept (the vocal is `fully_copy`: H3 keeps it 1:1 as the
  soundtrack), `detailed_description` runs the shot in playback order with the sung words in
  `<d>[Language] ...</d>` against a speaker ID (S1), then `overall_soundscape` and
  `non_diegetic_music`.
"""
from __future__ import annotations

import re

_PERSON = re.compile(r"\b(?:the person|the singer)\b")


def ltx25(text: str, *, sung: bool = True) -> str:
    """An LTX-2.5 single-shot prompt: the clip's description, then its sound (LTX-2.5 makes the
    audio with the video, so the guide ends every prompt on it)."""
    text = " ".join(text.split())
    if not text.endswith("."):
        text += "."
    sound = (" The sound is the singer's own voice, close and clear, every word landing on the lips."
             if sung else " Only the music is heard; nobody sings.")
    return text + sound


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", " ".join(text.split())) if s.strip()]


def h3(text: str, *, seconds: float, words: str = "", language: str = "English", sung: bool = True,
       look: bool = False, broll: bool = False, audio: str = "vocal") -> str:
    """A MiniMax H3 full-reference prompt for one clip.
    text: the clip's plain description; seconds: its length; words: the lyrics sung in it;
    look: the clip follows the lead's clip (<Video 1>) for pose and timing; broll: a cutaway
    (nobody sings); audio: what <Audio 1> is ("vocal", "drums" or "" for none)."""
    sub = "<Subject 1>"
    body = _PERSON.sub(sub, " ".join(text.split()))
    sung = sung and not broll
    words = " ".join(words.split())
    if words and words[-1] not in ".!?":
        words += "."  # the guide ends every <d> line on its punctuation
    # subject_definitions
    defs = [f"{sub} is what <Picture 1> shows: its people, place and light." if broll else
            f"{sub} is the person in <Picture 1>: their face, hair, clothes and surroundings.",
            "<Picture 1> is the first frame of the target video."]
    if look:
        defs.append(f"<Video 1> is a clip of another singer; {sub} follows its head movement, pose and timing.")
    if audio == "vocal":
        defs.append(f"<Audio 1> is the isolated lead vocal of the song for this clip" +
                    (f", sung by {sub}." if sung else ", nearly silent here."))
    elif audio == "drums":
        defs.append("<Audio 1> is the drum track of the song for this clip.")
    # summary
    kinds = ["image reference"] + (["video reference"] if look else []) + (["audio reuse"] if audio else [])
    if broll:
        act = f"{sub} is caught in a candid moment, not singing"
    elif sung:
        act = f"{sub} sings the words of <Audio 1> in perfect lip sync"
    else:
        act = f"{sub} listens to the music with the lips closed, moving to the beat, not singing"
    summary = (f"[{' + '.join(kinds)}] A {seconds:.2f}-second music-video shot that starts exactly on <Picture 1>: "
               f"{act}." + (" <Audio 1> is reused 1:1 as the complete soundtrack." if audio else ""))
    # retention_analysis
    ret = [f"{sub} (appears in [Shot 1]): fully_preserved - the face, identity, hair, clothing, lighting and "
           f"setting of <Picture 1> stay the same throughout.",
           "<Picture 1>: fully_preserved - it is the frame at 0.00 seconds."]
    if look:
        ret.append(f"<Video 1>: reference - only its head movement, pose and timing are followed; the person, "
                   f"clothes and place come from <Picture 1>.")
    if audio:
        ret.append("<Audio 1>: fully_copy - <Audio 1> is reused 1:1 as the target video's complete final audio track.")
    # detailed_description
    lines = ["The target video is in realistic photographic style.",
             f"[Shot 1] At 0.00 seconds the shot is exactly <Picture 1>. " + " ".join(_sentences(body))]
    if sung and words:
        lines.append(f"{sub} (S1), singing in the voice of <Audio 1>, sings <d>[{language}] {words}</d> "
                     f"The lips shape every syllable exactly when it is heard in <Audio 1>, and close between lines.")
    elif sung:  # the transcription found no words: still in time with the vocal
        lines.append(f"{sub} (S1) sings along with <Audio 1>, the lips shaping every syllable exactly when it is "
                     f"heard, and closing between lines.")
    elif not broll:
        lines.append(f"{sub} does not sing: the lips stay closed for the whole {seconds:.2f} seconds while the head "
                     f"nods with the beat.")
    lines.append("One continuous shot from the first frame to the last, with no cuts: the same person in the same "
                 "place, filmed by the same camera.")
    detail = " ".join(lines)
    if audio == "vocal":
        sound = "The soundtrack is <Audio 1> copied unchanged: the lead vocal on its own, with no added sound effects."
    elif audio == "drums":
        sound = "The soundtrack is <Audio 1> copied unchanged: the song's drums, with no added sound effects."
    else:
        sound = "Quiet room tone, with soft movement sounds."
    return "\n\n".join([
        "subject_definitions:\n" + "\n".join(defs),
        "summary:\n" + summary,
        "retention_analysis:\n" + "\n".join(ret),
        "detailed_description:\n" + detail,
        "overall_soundscape:\n" + sound,
        "non_diegetic_music:\nN/A",
    ])
