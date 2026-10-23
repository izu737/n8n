# Verbatim

Video in, accurate speaker labelled transcript out. It does that one job and nothing else: no editor, no
accounts, no cloud. Everything runs on your own machine.

## Quick start

```bash
# needs Python 3.10+ and FFmpeg on your PATH
pip install -e .
verbatim models --download          # one time, about 530 MB, from GitHub releases
verbatim transcribe meeting.mp4     # writes meeting.txt, meeting.srt, meeting.json next to the video
verbatim serve                      # local web app at http://127.0.0.1:8765
```

Useful options:

```bash
verbatim transcribe talk.mp4 -f txt,md,srt,vtt,json -o results/
verbatim transcribe panel.mp4 --speakers 4          # tell it the speaker count if you know it
verbatim transcribe lecture.mp4 --no-diarize        # skip speaker labels, roughly twice as fast
verbatim transcribe call.mp4 --glossary terms.txt   # custom vocabulary
verbatim transcribe entrevista.mp4 -l es            # any non English European language
verbatim wer reference.txt result.txt               # measure accuracy on your own audio
```

The web app lets you drop a file in, then click any word to jump the video to that moment. The word being
spoken is highlighted as it plays, and words the model was unsure about get a dotted underline so you know
what to proof read.

## How it works

This follows the same architecture as Descript's transcription, WhisperX and similar tools.

| Stage | What happens | Why |
|---|---|---|
| 1. Decode | FFmpeg turns any video into 16 kHz mono. Stereo is checked for phase cancellation. | Out of phase channels average to silence. Descript documents this as a cause of blank transcripts. |
| 2. Find speech | Silero VAD marks where people are talking. | Models lose accuracy on long inputs, so audio must be cut. Cutting blindly splits words. |
| 3. Chunk | Speech is packed into pieces of up to 30 s, cut at the longest pauses. | Every cut lands in silence, never inside a word. |
| 4. Transcribe | NVIDIA Parakeet TDT 0.6B via sherpa onnx. | Writes punctuation and capitals itself and emits a start time and duration for every token, so word timestamps come straight out of recognition. No separate alignment model needed. Each word also gets a confidence score. |
| 5. Speakers | pyannote segmentation 3.0 finds who speaks when, TitaNet embeddings fingerprint voices, clustering groups them. Words are assigned by overlap. | Speaker labels without any account or token. |
| 6. Vocabulary | Optional glossary corrects names and jargon. | Same idea as Descript's transcription glossary. |
| 7. Output | Paragraphs, subtitles that follow broadcast rules, JSON with every word. | |

Every model is published on GitHub releases, so nothing needs a Hugging Face account, a gated licence
click through or an API key. Models are cached in `~/.cache/verbatim/models` (override with `VERBATIM_MODELS`).

Parakeet TDT 0.6B v2 scores 6.05% average WER on the Open ASR Leaderboard paper's English test sets
(arXiv 2510.06961), versus 7.44% for Whisper large v3, and is far faster. Whisper covers 99 languages against
Parakeet's 25, which is why it is the obvious next engine to add (see the limits below).

## Custom vocabulary

```
# terms.txt
PostgreSQL                 # a term: fixes spelling and case of close matches
Kubernetes
en eight en => n8n         # a rule: replace exactly this heard phrase
```

Matching is deliberately conservative. A bad correction is worse than a missed one, and the first version of
this feature had a bug that swallowed neighbouring words, which is now covered by a regression test.

## Measured results

Measured in the development sandbox on 4 CPU cores with no GPU. Treat the accuracy figures as a smoke test,
not a benchmark (see the limits below).

| Test | Result |
|---|---|
| 110 s, 3 speaker meeting (synthetic voices), video file | 301 of 301 words given the right speaker, with no speaker count supplied |
| Same file, real word errors | 4 in 321 words. Most remaining differences are number formatting (see below). |
| Real recorded 57 s, 4 speaker audio | Exactly 4 speakers found for clustering thresholds 0.7 to 0.9 |
| 30 minute video | 502 s total (3.6x realtime), peak memory 3.3 GB, 3 speakers, no ordering or timing errors |
| Time split on that run | transcription 223 s, speaker detection 251 s, VAD 17 s |

Speaker detection costs about as much as transcription. `--no-diarize` skips it.

## Limits you should know about

* **Accuracy numbers here come from clean synthetic speech and one short real clip.** I could not test accented,
  noisy or overlapping speech in the sandbox. Run `verbatim wer` on your own recordings with a hand typed
  reference before you rely on it.
* **Numbers are written as digits** ("45", "$400"). That is usually what you want in a transcript, but it
  counts as an error if your reference spells numbers out, so write references the same way.
* **Fillers are mostly dropped.** Parakeet tends to omit "um" and "uh". Good for reading, wrong if you need a
  strictly verbatim record.
* **Languages:** English (v2 model) and 25 European languages (v3, use `-l <anything but en>`). Not Yoruba, Igbo,
  Hausa or Nigerian Pidgin. For those a Whisper based engine would be needed. It is not built yet because the
  sandbox could not reach Hugging Face to test it, and untested code should not be shipped as a feature.
* **Overlapping speech** is not handled well by either recognition or speaker detection.
* **Speaker detection is weakest when two voices are similar.** Two TTS voices that scored 0.63 similarity
  against each other were merged even with the speaker count supplied. Give `--speakers N` when you know it.
* **Memory** grows with file length because the audio is decoded in one piece (about 3.3 GB for 30 minutes
  with speaker detection). Split very long recordings.
* **Music and song lyrics** are not transcribed as lyrics.
* The web app has no login. It binds to localhost by default. Do not expose it to a network.

## Development

```bash
pip install -e ".[dev]"
pytest                      # 43 tests; the model backed ones skip themselves if models are missing
```

Layout: `audio.py` decode, `vad.py` chunking, `asr.py` recognition, `diarize.py` speakers,
`glossary.py` vocabulary, `assemble.py` paragraphs and captions, `formats.py` outputs,
`pipeline.py` ties it together, `server.py` and `web/index.html` the app.

One trap worth knowing: sherpa onnx's VAD silently returns wrong segments if you feed it large blocks of
audio. It dropped 1.3 s of speech at the start of a test clip. `vad.py` feeds exact 512 sample windows, and
`tests/test_integration.py` has a regression test for it.
