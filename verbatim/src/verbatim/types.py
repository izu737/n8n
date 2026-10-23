"""Plain data types shared by every stage of the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Word:
    """One spoken word. Punctuation stays attached to the word it follows."""

    text: str
    start: float
    end: float
    confidence: float = 1.0
    speaker: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "confidence": round(self.confidence, 3),
            "speaker": self.speaker,
        }


@dataclass
class Segment:
    """A paragraph: consecutive words from one speaker."""

    speaker: Optional[int]
    words: list[Word]

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


@dataclass
class Turn:
    """A stretch of time attributed to one speaker by diarization."""

    start: float
    end: float
    speaker: int


@dataclass
class Transcript:
    words: list[Word]
    segments: list[Segment]
    duration: float
    language: str
    engine: str
    speaker_count: int = 0
    timings: dict[str, float] = field(default_factory=dict)
    source: str = ""

    def speaker_name(self, speaker: Optional[int]) -> str:
        return "Speaker ?" if speaker is None else f"Speaker {speaker + 1}"

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)
