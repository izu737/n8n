"""Speech recognition engines. Each turns audio chunks into timed, scored words."""

from __future__ import annotations

import math
import os
from typing import Callable, Optional, Protocol

import numpy as np

from . import models
from .audio import SAMPLE_RATE
from .types import Word
from .vad import Chunk

ProgressFn = Callable[[float], None]

# Punctuation that belongs to the word before it, even if the tokeniser put a space first.
_CLOSING = set(".,!?;:%)]}…»”’")
_PUNCT = set(".,!?;:%()[]{}…\"'«»“”‘’-–—")


def _is_punct(token: str) -> bool:
    stripped = token.strip()
    return bool(stripped) and all(c in _PUNCT for c in stripped)


class Engine(Protocol):
    name: str
    language: str

    def transcribe(
        self, audio: np.ndarray, chunks: list[Chunk], progress: Optional[ProgressFn] = None
    ) -> list[Word]: ...


def tokens_to_words(
    tokens: list[str],
    starts: list[float],
    durations: list[float],
    log_probs: list[float],
    offset: float = 0.0,
) -> list[Word]:
    """Group subword tokens into words.

    A token that starts with a space begins a new word. Punctuation tokens stay on the word
    they follow but do not stretch its timing or drag down its confidence, because a shaky
    comma is not a misheard word.
    """
    words: list[Word] = []
    # Running state for the word being built.
    text = ""
    w_start = w_end = 0.0
    probs: list[float] = []
    open_word = False

    def close() -> None:
        nonlocal text, probs, open_word
        if not open_word:
            return
        conf = math.exp(sum(probs) / len(probs)) if probs else 1.0
        words.append(Word(text=text, start=offset + w_start, end=offset + max(w_end, w_start + 0.04),
                          confidence=min(1.0, conf)))
        text, probs, open_word = "", [], False

    for i, tok in enumerate(tokens):
        t_start = float(starts[i])
        t_end = t_start + float(durations[i]) if i < len(durations) else t_start
        lp = float(log_probs[i]) if i < len(log_probs) else 0.0
        punct = _is_punct(tok)
        begins = tok.startswith(" ") or not open_word
        if begins and punct and tok.strip()[0] in _CLOSING and open_word:
            begins = False  # "word ," -> "word,"

        if begins:
            close()
            text = tok.strip()
            w_start, w_end = t_start, (t_start if punct else t_end)
            probs = [] if punct else [lp]
            open_word = True
        else:
            text += tok.strip() if tok.startswith(" ") else tok
            if not punct:
                w_end = max(w_end, t_end)
                probs.append(lp)
    close()

    # Words must not overlap or run backwards; clamp to the next word's start.
    for a, b in zip(words, words[1:]):
        if a.end > b.start:
            a.end = max(a.start + 0.02, b.start)
    return [w for w in words if w.text]


class ParakeetEngine:
    """NVIDIA Parakeet TDT via sherpa-onnx.

    A transducer that emits a start time and duration for every token it writes, so word
    timing comes straight out of recognition with no separate alignment model. It also writes
    punctuation and capitalisation itself.
    """

    def __init__(self, variant: str = "en", threads: Optional[int] = None, batch_size: int = 4,
                 model_root=None, quiet: bool = False):
        import sherpa_onnx

        key = models.ASR_MODELS[variant]
        folder = models.ensure(key, model_root, quiet)
        self.name = key
        self.language = "en" if variant == "en" else "auto"
        self.batch_size = max(1, batch_size)
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(folder / "encoder.int8.onnx"),
            decoder=str(folder / "decoder.int8.onnx"),
            joiner=str(folder / "joiner.int8.onnx"),
            tokens=str(folder / "tokens.txt"),
            num_threads=threads or max(1, min(os.cpu_count() or 4, 8)),
            sample_rate=SAMPLE_RATE,
            model_type="nemo_transducer",
        )

    def transcribe(self, audio: np.ndarray, chunks: list[Chunk], progress: Optional[ProgressFn] = None) -> list[Word]:
        words: list[Word] = []
        # Longest first inside each batch would help padding, but order must be kept for
        # timestamps, so batch neighbours (similar length in practice) and keep order.
        for b in range(0, len(chunks), self.batch_size):
            batch = chunks[b : b + self.batch_size]
            streams, kept = [], []
            for chunk in batch:
                samples = audio[int(chunk.start * SAMPLE_RATE) : int(chunk.end * SAMPLE_RATE)]
                if samples.size < SAMPLE_RATE * 0.1 or float(np.abs(samples).max()) < 1e-4:
                    continue  # empty or digital silence: nothing to hallucinate on
                stream = self._recognizer.create_stream()
                stream.accept_waveform(SAMPLE_RATE, samples)
                streams.append(stream)
                kept.append(chunk)
            if streams:
                self._recognizer.decode_streams(streams)
                for stream, chunk in zip(streams, kept):
                    r = stream.result
                    words.extend(tokens_to_words(list(r.tokens), list(r.timestamps), list(r.durations),
                                                 list(r.ys_log_probs), offset=chunk.start))
            if progress:
                progress(min(1.0, (b + len(batch)) / len(chunks)))
        return words
