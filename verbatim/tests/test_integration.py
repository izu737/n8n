"""Tests that run the real models. Skipped automatically when the models are not installed.

Set VERBATIM_MODELS to the model folder to run them offline.
"""

import shutil
import subprocess

import numpy as np
import pytest

from verbatim import models
from verbatim.wer import wer

needs_models = pytest.mark.skipif(
    not (models.is_installed("parakeet_v2") and models.is_installed("silero_vad")),
    reason="speech models are not installed (run: verbatim models --download)",
)
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

SAMPLE = models.model_path("parakeet_v2") / "test_wavs" / "0.wav"
SAMPLE_TEXT = ("Well, I don't wish to see it any more, observed Phoebe, turning away her eyes. "
               "It is certainly very like the old portrait.")


def _sample_audio():
    import soundfile as sf

    x, sr = sf.read(str(SAMPLE), dtype="float32")
    assert sr == 16000
    return x


@needs_models
@pytest.mark.skipif(not SAMPLE.exists(), reason="sample wav not available")
def test_vad_does_not_clip_the_start_of_speech():
    """Regression: feeding sherpa's VAD large blocks dropped the first 1.3 s of speech."""
    from verbatim.vad import detect_speech

    regions = detect_speech(_sample_audio(), models.model_path("silero_vad") / "silero_vad.onnx")
    assert regions[0][0] < 0.7, regions  # speech truly begins at about 0.5 s


@needs_models
@pytest.mark.skipif(not SAMPLE.exists(), reason="sample wav not available")
def test_sample_transcribes_accurately_with_word_times():
    from verbatim.asr import ParakeetEngine
    from verbatim.vad import detect_speech, plan_chunks

    audio = _sample_audio()
    regions = detect_speech(audio, models.model_path("silero_vad") / "silero_vad.onnx")
    words = ParakeetEngine("en").transcribe(audio, plan_chunks(audio, regions))
    assert wer(SAMPLE_TEXT, " ".join(w.text for w in words)).wer < 0.1
    assert words[0].text.startswith("Well") and 0.4 < words[0].start < 0.7
    assert all(a.end <= b.start + 1e-6 for a, b in zip(words, words[1:]))


@needs_models
@needs_ffmpeg
@pytest.mark.skipif(not SAMPLE.exists(), reason="sample wav not available")
def test_video_file_end_to_end_and_phase_inverted_audio(tmp_path):
    """A real mp4, and the same speech with one channel inverted (naive downmix = silence)."""
    from verbatim.pipeline import Options, transcribe

    wav = tmp_path / "s.wav"
    shutil.copy(SAMPLE, wav)
    mp4 = tmp_path / "s.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=160x90:r=5", "-i", str(wav),
                    "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(mp4)], check=True)
    t = transcribe(mp4, Options(diarize=False, quiet=True))
    assert wer(SAMPLE_TEXT, t.text).wer < 0.12

    inv = tmp_path / "inv.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(wav), "-af",
                    "pan=stereo|c0=c0|c1=-1*c0", str(inv)], check=True)
    t2 = transcribe(inv, Options(diarize=False, quiet=True))
    assert wer(SAMPLE_TEXT, t2.text).wer < 0.12


@needs_models
def test_silence_and_noise_produce_no_words(tmp_path):
    import soundfile as sf

    from verbatim.pipeline import Options, transcribe

    sf.write(tmp_path / "quiet.wav", np.zeros(16000 * 5, dtype=np.float32), 16000)
    t = transcribe(tmp_path / "quiet.wav", Options(diarize=False, quiet=True))
    assert t.words == [] and t.segments == []
