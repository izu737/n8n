"""Find speech, then cut it into chunks the recogniser handles well.

Why this exists: speech models lose accuracy and speed on long inputs, and cutting
blindly every N seconds slices words in half. Every serious pipeline (WhisperX,
Descript's backend, faster-whisper's VAD filter) therefore detects speech first and
cuts only inside silences. This module does the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .audio import SAMPLE_RATE


vad_window_size = 512


@dataclass(frozen=True)
class Chunk:
    start: float  # seconds into the file
    end: float


def detect_speech(
    audio: np.ndarray,
    model: Path,
    threshold: float = 0.45,
    min_silence: float = 0.2,
    min_speech: float = 0.15,
    progress: Optional[Callable[[float], None]] = None,
) -> list[tuple[float, float]]:
    """Return (start, end) seconds of every stretch of speech, using Silero VAD."""
    import sherpa_onnx

    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(model)
    config.silero_vad.threshold = threshold
    config.silero_vad.min_silence_duration = min_silence
    config.silero_vad.min_speech_duration = min_speech
    config.silero_vad.max_speech_duration = 120.0  # we do our own, smarter splitting below
    config.sample_rate = SAMPLE_RATE
    vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=180)

    regions: list[tuple[float, float]] = []

    def drain() -> None:
        while not vad.empty():
            seg = vad.front
            start = seg.start / SAMPLE_RATE
            regions.append((start, start + len(seg.samples) / SAMPLE_RATE))
            vad.pop()

    # The Silero model consumes exactly 512 samples per step. sherpa-onnx's detector gives
    # silently wrong segments when handed bigger blocks (it dropped 1.3 s of speech at the
    # start of a test file), so feed it one window at a time.
    window = vad_window_size
    next_report = 0
    for i in range(0, len(audio), window):
        vad.accept_waveform(audio[i : i + window])
        if not vad.empty():
            drain()
        if progress and i >= next_report:
            progress(min(1.0, i / len(audio)))
            next_report = i + SAMPLE_RATE * 30
    vad.flush()
    drain()
    return regions


def _quietest_point(audio: np.ndarray, lo: float, hi: float) -> float:
    """Time of the lowest energy 30 ms window between lo and hi seconds."""
    a, b = int(lo * SAMPLE_RATE), int(hi * SAMPLE_RATE)
    seg = audio[a:b]
    win = int(0.03 * SAMPLE_RATE)
    if len(seg) < win * 2:
        return (lo + hi) / 2
    frames = len(seg) // win
    energy = np.sqrt(np.mean(np.square(seg[: frames * win].reshape(frames, win), dtype=np.float64), axis=1))
    # Smooth so one dip inside a plosive does not win over a true pause.
    energy = np.convolve(energy, np.ones(3) / 3, mode="same")
    return lo + (int(np.argmin(energy)) + 0.5) * win / SAMPLE_RATE


def _split_long_region(audio: np.ndarray, start: float, end: float, max_len: float) -> list[tuple[float, float]]:
    """A region with no pause longer than the VAD threshold: cut at the quietest moments."""
    pieces: list[tuple[float, float]] = []
    cur = start
    while end - cur > max_len:
        cut = _quietest_point(audio, cur + max_len * 0.6, cur + max_len)
        pieces.append((cur, cut))
        cur = cut
    pieces.append((cur, end))
    return pieces


def plan_chunks(
    audio: np.ndarray,
    regions: list[tuple[float, float]],
    max_len: float = 30.0,
    pad: float = 0.15,
) -> list[Chunk]:
    """Pack speech regions into chunks of at most max_len seconds, cutting at the longest pauses."""
    if not regions:
        return []
    total = len(audio) / SAMPLE_RATE

    flat: list[tuple[float, float]] = []
    for s, e in regions:
        if e - s > max_len:
            flat.extend(_split_long_region(audio, s, e, max_len - 2 * pad))
        else:
            flat.append((s, e))

    min_fill = max_len * 0.35
    chunks_regions: list[list[tuple[float, float]]] = []
    i = 0
    while i < len(flat):
        j = i + 1
        while j < len(flat) and flat[j][1] - flat[i][0] <= max_len - 2 * pad:
            j += 1
        # flat[i:j] fits. If more speech follows, end the chunk at the best pause instead of
        # the last place that happens to fit, so we split between sentences rather than inside one.
        if j < len(flat):
            best_k, best_gap = j, -1.0
            for k in range(i + 1, j + 1):
                left_len = flat[k - 1][1] - flat[i][0]
                gap = flat[k][0] - flat[k - 1][1]
                if left_len >= min_fill and gap > best_gap:
                    best_k, best_gap = k, gap
            j = best_k
        chunks_regions.append(flat[i:j])
        i = j

    chunks: list[Chunk] = []
    for idx, group in enumerate(chunks_regions):
        prev_end = chunks_regions[idx - 1][-1][1] if idx > 0 else 0.0
        next_start = chunks_regions[idx + 1][0][0] if idx + 1 < len(chunks_regions) else total
        s = max(group[0][0] - pad, (prev_end + group[0][0]) / 2, 0.0)
        e = min(group[-1][1] + pad, (group[-1][1] + next_start) / 2, total)
        chunks.append(Chunk(s, e))
    return chunks
