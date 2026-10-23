"""Write a Transcript as txt, md, srt, vtt or json."""

from __future__ import annotations

import json
from pathlib import Path

from .assemble import build_cues
from .types import Transcript

FORMATS = ("txt", "md", "srt", "vtt", "json")


def stamp(seconds: float, sep: str = ".", hours: bool = True) -> str:
    ms = int(round(max(seconds, 0.0) * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    base = f"{h:02d}:{m:02d}:{s:02d}" if hours else f"{h * 60 + m:02d}:{s:02d}"
    return f"{base}{sep}{ms:03d}" if sep else base


def _clock(seconds: float, total: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if total >= 3600 else f"{m:02d}:{s:02d}"


def to_txt(t: Transcript, timestamps: bool = True) -> str:
    multi = t.speaker_count > 1
    blocks = []
    for seg in t.segments:
        head = []
        if timestamps:
            head.append(f"[{_clock(seg.start, t.duration)}]")
        if multi:
            head.append(f"{t.speaker_name(seg.speaker)}:")
        blocks.append((" ".join(head) + " " if head else "") + seg.text)
    return "\n\n".join(blocks) + "\n"


def to_md(t: Transcript) -> str:
    multi = t.speaker_count > 1
    name = Path(t.source).name or "Transcript"
    lines = [f"# {name}", "", f"*Duration {_clock(t.duration, 3600)} · {t.speaker_count} speaker(s) · engine {t.engine}*", ""]
    for seg in t.segments:
        who = f"**{t.speaker_name(seg.speaker)}** " if multi else ""
        lines += [f"{who}`{_clock(seg.start, t.duration)}`", "", seg.text, ""]
    return "\n".join(lines)


def _cue_text(t: Transcript, cue, with_speaker: bool, prev_speaker) -> str:
    body = "\n".join(cue.lines)
    if with_speaker and cue.speaker != prev_speaker:
        return f"{t.speaker_name(cue.speaker)}: {body}"
    return body


def to_srt(t: Transcript) -> str:
    multi = t.speaker_count > 1
    out, prev = [], object()
    for i, cue in enumerate(build_cues(t.words), 1):
        out.append(f"{i}\n{stamp(cue.start, ',')} --> {stamp(cue.end, ',')}\n{_cue_text(t, cue, multi, prev)}\n")
        prev = cue.speaker
    return "\n".join(out)


def to_vtt(t: Transcript) -> str:
    multi = t.speaker_count > 1
    out, prev = ["WEBVTT\n"], object()
    for cue in build_cues(t.words):
        out.append(f"{stamp(cue.start)} --> {stamp(cue.end)}\n{_cue_text(t, cue, multi, prev)}\n")
        prev = cue.speaker
    return "\n".join(out)


def to_json(t: Transcript) -> str:
    data = {
        "source": t.source,
        "duration": round(t.duration, 3),
        "language": t.language,
        "engine": t.engine,
        "speaker_count": t.speaker_count,
        "timings": {k: round(v, 2) for k, v in t.timings.items()},
        "segments": [
            {
                "speaker": seg.speaker,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text,
                "words": [w.to_dict() for w in seg.words],
            }
            for seg in t.segments
        ],
    }
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


RENDERERS = {"txt": to_txt, "md": to_md, "srt": to_srt, "vtt": to_vtt, "json": to_json}


def write_all(t: Transcript, out_dir: Path, stem: str, formats: list[str]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        if fmt not in RENDERERS:
            raise ValueError(f"Unknown format '{fmt}'. Choose from {', '.join(FORMATS)}")
        path = out_dir / f"{stem}.{fmt}"
        path.write_text(RENDERERS[fmt](t), encoding="utf-8")
        written.append(path)
    return written
