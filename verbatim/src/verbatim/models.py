"""Model registry and downloader.

Every model verbatim needs is published on GitHub releases by the sherpa-onnx
project, so nothing here requires a Hugging Face account, a gated licence
click-through or an API key. Models are fetched once into a cache directory.
"""

from __future__ import annotations

import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_GH = "https://github.com/k2-fsa/sherpa-onnx/releases/download"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    url: str
    description: str
    size_mb: int
    # Files that must exist (relative to the model's folder) for it to count as installed.
    files: tuple[str, ...]
    # Folder the archive unpacks to. None means the download is a single file.
    folder: Optional[str] = None


MODELS: dict[str, ModelSpec] = {
    "parakeet_v2": ModelSpec(
        key="parakeet_v2",
        url=f"{_GH}/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2",
        description="NVIDIA Parakeet TDT 0.6B v2, English, int8",
        size_mb=482,
        folder="sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
        files=("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"),
    ),
    "parakeet_v3": ModelSpec(
        key="parakeet_v3",
        url=f"{_GH}/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2",
        description="NVIDIA Parakeet TDT 0.6B v3, 25 European languages, int8",
        size_mb=487,
        folder="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
        files=("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"),
    ),
    "silero_vad": ModelSpec(
        key="silero_vad",
        url=f"{_GH}/asr-models/silero_vad.onnx",
        description="Silero voice activity detector",
        size_mb=1,
        files=("silero_vad.onnx",),
    ),
    "segmentation": ModelSpec(
        key="segmentation",
        url=f"{_GH}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        description="pyannote segmentation 3.0 (who speaks when)",
        size_mb=7,
        folder="sherpa-onnx-pyannote-segmentation-3-0",
        files=("model.int8.onnx",),
    ),
    "embedding": ModelSpec(
        key="embedding",
        url=f"{_GH}/speaker-recongition-models/nemo_en_titanet_small.onnx",
        description="NeMo TitaNet small speaker embedding",
        size_mb=40,
        files=("nemo_en_titanet_small.onnx",),
    ),
}

# What each pipeline configuration needs.
ASR_MODELS = {"en": "parakeet_v2", "multi": "parakeet_v3"}
DIARIZATION_MODELS = ("segmentation", "embedding")


class ModelError(RuntimeError):
    pass


def models_dir() -> Path:
    override = os.environ.get("VERBATIM_MODELS")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "verbatim" / "models"


def model_path(key: str, root: Optional[Path] = None) -> Path:
    """Folder (or file) holding an installed model. Does not download."""
    spec = MODELS[key]
    root = root or models_dir()
    return root / spec.folder if spec.folder else root


def is_installed(key: str, root: Optional[Path] = None) -> bool:
    spec = MODELS[key]
    base = model_path(key, root)
    return all((base / f).is_file() for f in spec.files)


def _safe_extract(archive: Path, dest: Path) -> None:
    dest_resolved = dest.resolve()
    with tarfile.open(archive, "r:bz2") as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if not str(target).startswith(str(dest_resolved) + os.sep) and target != dest_resolved:
                raise ModelError(f"Refusing to extract unsafe path from archive: {member.name}")
            if member.issym() or member.islnk() or member.isdev():
                raise ModelError(f"Refusing to extract link or device from archive: {member.name}")
        tar.extractall(dest)


def _download(url: str, out: Path, label: str, quiet: bool) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "verbatim"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        with open(out, "wb") as fh:
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                fh.write(block)
                done += len(block)
                if not quiet and sys.stderr.isatty() and total:
                    pct = done * 100 // total
                    sys.stderr.write(f"\r  downloading {label}: {pct:3d}% ({done >> 20}/{total >> 20} MB)")
                    sys.stderr.flush()
        if total and done != total:
            raise ModelError(f"Download of {label} was cut short ({done} of {total} bytes)")
    if not quiet and sys.stderr.isatty():
        sys.stderr.write("\n")


def ensure(key: str, root: Optional[Path] = None, quiet: bool = False) -> Path:
    """Return the path to a model, downloading it first if needed."""
    if key not in MODELS:
        raise ModelError(f"Unknown model '{key}'. Known: {', '.join(MODELS)}")
    spec = MODELS[key]
    root = root or models_dir()
    if is_installed(key, root):
        return model_path(key, root)

    root.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print(f"Fetching {spec.description} (~{spec.size_mb} MB), one time only...", file=sys.stderr)

    with tempfile.TemporaryDirectory(dir=root, prefix=".dl_") as tmp:
        tmp_dir = Path(tmp)
        filename = spec.url.rsplit("/", 1)[1]
        part = tmp_dir / filename
        try:
            _download(spec.url, part, spec.key, quiet)
        except OSError as exc:
            raise ModelError(
                f"Could not download {spec.key} from {spec.url}: {exc}\n"
                f"If you are offline or behind a firewall, download that file by hand and "
                f"place it (extracted) under {root}."
            ) from exc

        if spec.folder:
            staging = tmp_dir / "x"
            staging.mkdir()
            _safe_extract(part, staging)
            extracted = staging / spec.folder
            if not extracted.is_dir():
                raise ModelError(f"Archive for {spec.key} did not contain {spec.folder}/")
            final = root / spec.folder
            if final.exists():
                shutil.rmtree(final)
            shutil.move(str(extracted), str(final))
        else:
            shutil.move(str(part), str(root / filename))

    if not is_installed(key, root):
        raise ModelError(f"{spec.key} downloaded but expected files are missing: {spec.files}")
    return model_path(key, root)
