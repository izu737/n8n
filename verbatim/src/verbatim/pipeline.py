"""The whole job: media file in, Transcript out."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from . import models
from .assemble import build_segments
from .audio import load_audio
from .diarize import assign_speakers, diarize
from .glossary import Glossary
from .types import Transcript
from .vad import detect_speech, plan_chunks

# progress(stage, fraction 0..1, human readable message)
ProgressFn = Callable[[str, float, str], None]


@dataclass
class Options:
    language: str = "en"  # "en" uses the English model; anything else uses the multilingual one
    diarize: bool = True
    speakers: Optional[int] = None  # exact speaker count if known
    diarize_threshold: float = 0.75
    glossary: Optional[Path] = None
    max_chunk: float = 30.0
    vad_threshold: float = 0.45
    threads: Optional[int] = None
    audio_stream: int = 0
    quiet: bool = False


_engines: dict = {}


def _engine(variant: str, threads: Optional[int], quiet: bool):
    from .asr import ParakeetEngine

    key = (variant, threads)
    if key not in _engines:  # loading a 600M parameter model takes seconds; reuse it
        _engines[key] = ParakeetEngine(variant, threads=threads, quiet=quiet)
    return _engines[key]


def transcribe(path: str | Path, opts: Optional[Options] = None, progress: Optional[ProgressFn] = None) -> Transcript:
    opts = opts or Options()
    report = progress or (lambda stage, frac, msg: None)
    timings: dict[str, float] = {}
    t_all = time.time()

    def stage(name: str):
        class _S:
            def __enter__(s):
                s.t = time.time()

            def __exit__(s, *a):
                timings[name] = time.time() - s.t

        return _S()

    report("decode", 0.0, "Reading audio")
    with stage("decode"):
        audio, info = load_audio(path, opts.audio_stream)
    duration = len(audio) / 16000

    report("load", 0.0, "Loading speech model")
    variant = "en" if opts.language.lower() in ("en", "english") else "multi"
    with stage("load_models"):
        vad_model = models.ensure("silero_vad", quiet=opts.quiet) / "silero_vad.onnx"
        engine = _engine(variant, opts.threads, opts.quiet)

    report("vad", 0.0, "Finding speech")
    with stage("vad"):
        regions = detect_speech(audio, vad_model, threshold=opts.vad_threshold,
                                progress=lambda f: report("vad", f, "Finding speech"))
        chunks = plan_chunks(audio, regions, max_len=opts.max_chunk)

    report("asr", 0.0, f"Transcribing {len(chunks)} segments")
    with stage("asr"):
        words = engine.transcribe(audio, chunks,
                                  progress=lambda f: report("asr", f, f"Transcribing {len(chunks)} segments"))

    speaker_count = 0
    if opts.diarize and words:
        report("diarize", 0.0, "Working out who is speaking")
        with stage("diarize"):
            turns = diarize(audio, num_speakers=opts.speakers, threshold=opts.diarize_threshold,
                            threads=opts.threads or 4, quiet=opts.quiet,
                            progress=lambda f: report("diarize", f, "Working out who is speaking"))
            speaker_count = assign_speakers(words, turns)

    if opts.glossary and words:
        words, _ = Glossary.load(opts.glossary).apply(words)

    segments = build_segments(words)
    timings["total"] = time.time() - t_all
    report("done", 1.0, "Done")
    return Transcript(
        words=words,
        segments=segments,
        duration=duration,
        language=engine.language,
        engine=engine.name,
        speaker_count=speaker_count,
        timings=timings,
        source=str(path),
    )
