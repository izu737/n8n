"""Command line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__


def _progress_printer():
    last = {"stage": None, "pct": -1}

    def report(stage: str, frac: float, msg: str) -> None:
        if not sys.stderr.isatty():
            if stage != last["stage"]:
                print(msg + "...", file=sys.stderr)
                last["stage"] = stage
            return
        pct = int(frac * 100)
        if stage != last["stage"] or pct != last["pct"]:
            sys.stderr.write(f"\r\033[K{msg}: {pct:3d}%")
            if stage == "done":
                sys.stderr.write("\n")
            sys.stderr.flush()
            last["stage"], last["pct"] = stage, pct

    return report


def _cmd_transcribe(a: argparse.Namespace) -> int:
    from .audio import AudioError
    from .formats import FORMATS, write_all
    from .models import ModelError
    from .pipeline import Options, transcribe

    formats = [f.strip() for f in a.format.split(",") if f.strip()]
    bad = [f for f in formats if f not in FORMATS]
    if bad:
        print(f"Unknown format(s): {', '.join(bad)}. Choose from {', '.join(FORMATS)}.", file=sys.stderr)
        return 2
    opts = Options(language=a.language, diarize=not a.no_diarize, speakers=a.speakers,
                   diarize_threshold=a.diarize_threshold, glossary=a.glossary, threads=a.threads,
                   audio_stream=a.audio_stream, max_chunk=a.max_chunk)
    status = 0
    for media in a.media:
        path = Path(media)
        if not path.is_file():
            print(f"Not a file: {path}", file=sys.stderr)
            status = 1
            continue
        try:
            t = transcribe(path, opts, _progress_printer())
        except (AudioError, ModelError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            status = 1
            continue
        out_dir = Path(a.output) if a.output else path.parent
        written = write_all(t, out_dir, path.stem, formats)
        rtf = t.duration / t.timings["total"] if t.timings["total"] else 0
        print(f"{path.name}: {t.duration:.0f}s of audio in {t.timings['total']:.1f}s ({rtf:.1f}x realtime), "
              f"{len(t.words)} words, {t.speaker_count} speaker(s)", file=sys.stderr)
        for w in written:
            print(w)
        if a.stdout:
            print()
            from .formats import to_txt
            print(to_txt(t))
    return status


def _cmd_models(a: argparse.Namespace) -> int:
    from . import models

    wanted = ["silero_vad", "parakeet_v2", "segmentation", "embedding"]
    if a.multilingual:
        wanted.append("parakeet_v3")
    print(f"Model folder: {models.models_dir()}")
    for key in models.MODELS:
        spec = models.MODELS[key]
        state = "installed" if models.is_installed(key) else "missing"
        print(f"  {key:14} {state:10} {spec.description}")
    if a.download:
        for key in wanted:
            models.ensure(key)
        print("All requested models are ready.")
    return 0


def _cmd_wer(a: argparse.Namespace) -> int:
    from .wer import wer

    print(wer(Path(a.reference).read_text(encoding="utf-8"), Path(a.hypothesis).read_text(encoding="utf-8")))
    return 0


def _cmd_serve(a: argparse.Namespace) -> int:
    from .server import serve

    serve(a.host, a.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="verbatim", description="Accurate video and audio transcription.")
    p.add_argument("--version", action="version", version=f"verbatim {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe", help="Transcribe one or more video or audio files")
    t.add_argument("media", nargs="+", help="video or audio file(s)")
    t.add_argument("-o", "--output", help="folder for results (default: next to each input)")
    t.add_argument("-f", "--format", default="txt,srt,json", help="comma list of: txt, md, srt, vtt, json")
    t.add_argument("-l", "--language", default="en", help="'en' (default, most accurate for English) or any other code for the multilingual model")
    t.add_argument("--speakers", type=int, help="exact number of speakers, if known (improves labels a lot)")
    t.add_argument("--no-diarize", action="store_true", help="skip speaker labels (faster)")
    t.add_argument("--diarize-threshold", type=float, default=0.75, help="higher merges speakers, lower splits them")
    t.add_argument("--glossary", type=Path, help="custom vocabulary file")
    t.add_argument("--threads", type=int, help="CPU threads (default: up to 8)")
    t.add_argument("--audio-stream", type=int, default=0, help="which audio track to use")
    t.add_argument("--max-chunk", type=float, default=30.0, help="longest piece sent to the model, seconds")
    t.add_argument("--stdout", action="store_true", help="also print the transcript")
    t.set_defaults(func=_cmd_transcribe)

    m = sub.add_parser("models", help="List or download the models")
    m.add_argument("--download", action="store_true", help="download everything needed now")
    m.add_argument("--multilingual", action="store_true", help="include the multilingual model")
    m.set_defaults(func=_cmd_models)

    w = sub.add_parser("wer", help="Word error rate between a reference transcript and a result")
    w.add_argument("reference")
    w.add_argument("hypothesis")
    w.set_defaults(func=_cmd_wer)

    s = sub.add_parser("serve", help="Start the local web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(func=_cmd_serve)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
