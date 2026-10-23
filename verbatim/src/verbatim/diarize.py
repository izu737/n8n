"""Who spoke when, and attaching that to words."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from . import models
from .audio import SAMPLE_RATE
from .types import Turn, Word


def diarize(
    audio: np.ndarray,
    num_speakers: Optional[int] = None,
    threshold: float = 0.75,
    threads: int = 4,
    model_root=None,
    quiet: bool = False,
    progress: Optional[Callable[[float], None]] = None,
) -> list[Turn]:
    """pyannote segmentation 3.0 finds speech and overlaps, an embedding model fingerprints
    each stretch, and clustering groups the fingerprints into speakers.

    Pass num_speakers when you know it: it is the single biggest accuracy lever. Otherwise
    the clustering threshold decides (lower splits into more speakers).
    """
    import sherpa_onnx

    seg_dir = models.ensure("segmentation", model_root, quiet)
    emb_dir = models.ensure("embedding", model_root, quiet)
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(seg_dir / "model.int8.onnx")),
            num_threads=threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(emb_dir / "nemo_en_titanet_small.onnx"), num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers if num_speakers and num_speakers > 0 else -1,
            threshold=threshold),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError("Invalid diarization configuration")
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)

    def cb(done: int, total: int) -> int:
        if progress and total:
            progress(done / total)
        return 0

    result = sd.process(audio, callback=cb).sort_by_start_time()
    return [Turn(r.start, r.end, int(r.speaker)) for r in result]


def assign_speakers(words: list[Word], turns: list[Turn], max_gap: float = 1.0) -> int:
    """Label each word with the speaker whose turn overlaps it most. Returns speaker count.

    Speakers are renumbered by order of first appearance so the transcript reads Speaker 1,
    Speaker 2, ... from the top.
    """
    if not turns or not words:
        return 0
    turns = sorted(turns, key=lambda t: t.start)
    starts = np.array([t.start for t in turns])
    ends = np.array([t.end for t in turns])
    ids = np.array([t.speaker for t in turns])

    raw: list[int | None] = []
    for w in words:
        overlap = np.minimum(ends, w.end) - np.maximum(starts, w.start)
        k = int(np.argmax(overlap))
        if overlap[k] > 0:
            # Sum overlap per speaker: overlapping turns of one person should not lose to a sliver.
            per: dict[int, float] = {}
            for i in np.nonzero(overlap > 0)[0]:
                per[int(ids[i])] = per.get(int(ids[i]), 0.0) + float(overlap[i])
            raw.append(max(per, key=per.get))
        else:
            # Word sits in a gap: take the nearest turn if it is close enough.
            mid = (w.start + w.end) / 2
            dist = np.where(mid < starts, starts - mid, np.where(mid > ends, mid - ends, 0.0))
            j = int(np.argmin(dist))
            raw.append(int(ids[j]) if dist[j] <= max_gap else None)

    # Fill unknowns from neighbours, then remove one word flickers (a lone word whose
    # neighbours agree with each other and which is short: almost always a boundary error).
    for i, s in enumerate(raw):
        if s is None:
            raw[i] = next((raw[j] for j in range(i - 1, -1, -1) if raw[j] is not None), None)
    for i, s in enumerate(raw):
        if s is None:
            raw[i] = next((raw[j] for j in range(i + 1, len(raw)) if raw[j] is not None), 0)
    for i in range(1, len(raw) - 1):
        w = words[i]
        if raw[i - 1] == raw[i + 1] != raw[i] and (w.end - w.start) < 0.5:
            raw[i] = raw[i - 1]

    order: dict[int, int] = {}
    for s in raw:
        order.setdefault(int(s), len(order))
    for w, s in zip(words, raw):
        w.speaker = order[int(s)]
    return len(order)
