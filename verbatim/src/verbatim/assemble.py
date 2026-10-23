"""Words into paragraphs and caption cues."""

from __future__ import annotations

from dataclasses import dataclass

from .types import Segment, Word

_SENTENCE_END = (".", "?", "!", "…")
_CLAUSE_END = (",", ";", ":")


def ends_sentence(word: Word) -> bool:
    return word.text.rstrip("\"')”’»").endswith(_SENTENCE_END)


def build_segments(
    words: list[Word],
    pause: float = 1.5,
    long_pause: float = 3.0,
    max_chars: int = 700,
) -> list[Segment]:
    """Group words into paragraphs.

    A new paragraph starts on a speaker change, after a pause that follows a finished
    sentence, after a very long pause anywhere, or when a paragraph gets too long and
    a sentence just ended.
    """
    segments: list[Segment] = []
    current: list[Word] = []
    size = 0
    for w in words:
        if current:
            prev = current[-1]
            gap = w.start - prev.end
            brk = (
                w.speaker != prev.speaker
                or (gap >= pause and ends_sentence(prev))
                or gap >= long_pause
                or (size >= max_chars and ends_sentence(prev))
            )
            if brk:
                segments.append(Segment(current[0].speaker, current))
                current, size = [], 0
        current.append(w)
        size += len(w.text) + 1
    if current:
        segments.append(Segment(current[0].speaker, current))
    return segments


@dataclass
class Cue:
    start: float
    end: float
    lines: list[str]
    speaker: int | None


def _wrap(words: list[Word], max_chars: int) -> list[str]:
    """Balance text over at most two lines, preferring to break after punctuation."""
    text = " ".join(w.text for w in words)
    if len(text) <= max_chars:
        return [text]
    best, best_score = None, None
    acc = 0
    for i, w in enumerate(words[:-1]):
        acc += len(w.text) + (1 if i else 0)
        left, right = acc, len(text) - acc - 1
        if left > max_chars or right > max_chars:
            continue
        score = abs(left - right) - (8 if w.text.endswith(_CLAUSE_END + _SENTENCE_END) else 0)
        if best_score is None or score < best_score:
            best, best_score = i, score
    if best is None:
        return [text]
    return [" ".join(w.text for w in words[: best + 1]), " ".join(w.text for w in words[best + 1 :])]


def build_cues(
    words: list[Word],
    max_chars_per_line: int = 42,
    max_lines: int = 2,
    max_duration: float = 6.0,
    min_duration: float = 1.0,
    pause_break: float = 0.8,
) -> list[Cue]:
    """Subtitle cues following broadcast conventions: at most two short lines, a few
    seconds on screen, broken at sentence ends, pauses and speaker changes."""
    budget = max_chars_per_line * max_lines
    groups: list[list[Word]] = []
    cur: list[Word] = []
    chars = 0
    for w in words:
        if cur:
            prev = cur[-1]
            gap = w.start - prev.end
            too_long = chars + 1 + len(w.text) > budget or (w.end - cur[0].start) > max_duration
            # Close the cue at a sentence end once it has some substance (avoids one word
            # cues), and always on a speaker change or a real pause.
            sentence_done = ends_sentence(prev) and chars >= 20
            if too_long or w.speaker != prev.speaker or gap >= pause_break or sentence_done:
                groups.append(cur)
                cur, chars = [], 0
        cur.append(w)
        chars += len(w.text) + (1 if len(cur) > 1 else 0)
    if cur:
        groups.append(cur)

    cues: list[Cue] = []
    for g in groups:
        cues.append(Cue(g[0].start, g[-1].end, _wrap(g, max_chars_per_line), g[0].speaker))
    # Give very short cues a little more screen time without overlapping the next one.
    for i, c in enumerate(cues):
        limit = cues[i + 1].start - 0.04 if i + 1 < len(cues) else c.end + min_duration
        if c.end - c.start < min_duration:
            c.end = max(c.end, min(c.start + min_duration, limit))
    return cues
