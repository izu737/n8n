"""Turn any video or audio file into clean 16 kHz mono samples."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


class AudioError(RuntimeError):
    pass


@dataclass
class MediaInfo:
    duration: float | None
    audio_streams: int
    channels: int
    has_video: bool


def _require_tool(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise AudioError(
            f"{name} was not found on PATH. Install FFmpeg (https://ffmpeg.org) and try again."
        )
    return found


def probe(path: str | Path) -> MediaInfo:
    ffprobe = _require_tool("ffprobe")
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration:stream=codec_type,channels",
         "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise AudioError(f"Could not read '{path}': {proc.stderr.strip() or 'unknown ffprobe error'}")
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    duration = None
    try:
        duration = float(data.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        pass
    return MediaInfo(
        duration=duration,
        audio_streams=len(audio),
        channels=int(audio[0].get("channels", 1)) if audio else 0,
        has_video=any(s.get("codec_type") == "video" for s in streams),
    )


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


def to_mono(stereo: np.ndarray) -> np.ndarray:
    """Mix interleaved stereo down to mono without letting phase problems erase speech.

    Plain averaging cancels the signal when the channels are out of phase. Descript's own
    docs list this as a cause of blank transcripts, so we detect it and compensate.
    """
    left, right = stereo[0::2], stereo[1::2]
    rl, rr = _rms(left), _rms(right)
    peak = max(rl, rr)
    if peak < 1e-6:
        return ((left + right) * 0.5).astype(np.float32)

    # Inverted polarity on one channel: flip it back, then average.
    n = min(len(left), len(right), SAMPLE_RATE * 600)
    corr = float(np.dot(left[:n], right[:n]) / (np.linalg.norm(left[:n]) * np.linalg.norm(right[:n]) + 1e-12))
    if corr < -0.3:
        right = -right

    mono = (left + right) * 0.5
    # Delay or partial cancellation: if the mix is much quieter than the best channel, use that channel.
    if _rms(mono) < 0.5 * peak and not (corr < -0.3):
        mono = left if rl >= rr else right
    return np.ascontiguousarray(mono, dtype=np.float32)


def load_audio(path: str | Path, stream: int = 0) -> tuple[np.ndarray, MediaInfo]:
    """Decode the chosen audio stream to float32 mono at 16 kHz."""
    ffmpeg = _require_tool("ffmpeg")
    info = probe(path)
    if info.audio_streams == 0:
        raise AudioError(f"'{path}' has no audio track to transcribe.")
    if stream >= info.audio_streams:
        raise AudioError(f"'{path}' has {info.audio_streams} audio stream(s); stream {stream} does not exist.")

    channels = 1 if info.channels <= 1 else 2
    proc = subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-i", str(path), "-map", f"0:a:{stream}", "-vn",
         "-ac", str(channels), "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg could not decode '{path}': {proc.stderr.decode(errors='replace').strip()}")

    samples = np.frombuffer(proc.stdout, dtype=np.float32)
    if channels == 2:
        samples = samples[: len(samples) // 2 * 2]
        samples = to_mono(samples)
    if samples.size == 0:
        raise AudioError(f"'{path}' decoded to zero samples.")
    info.duration = samples.size / SAMPLE_RATE
    return samples, info
