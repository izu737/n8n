"""Fast tests that need no models and no FFmpeg audio fixtures."""

import numpy as np

from verbatim.assemble import build_cues, build_segments
from verbatim.asr import tokens_to_words
from verbatim.audio import SAMPLE_RATE, to_mono
from verbatim.diarize import assign_speakers
from verbatim.formats import stamp, to_srt, to_txt, to_vtt
from verbatim.glossary import Glossary
from verbatim.types import Transcript, Turn, Word
from verbatim.vad import plan_chunks
from verbatim.wer import wer


def W(text, start, end, speaker=None, conf=1.0):
    return Word(text, start, end, conf, speaker)


# ---------- tokens to words ----------

def test_tokens_group_into_words_with_timing_and_punctuation():
    tokens = [" Well", ",", " I", " don", "'", "t", " go", "."]
    starts = [0.4, 0.64, 0.72, 0.88, 1.04, 1.04, 1.2, 1.4]
    durs = [0.24, 0.08, 0.16, 0.16, 0.0, 0.0, 0.16, 0.08]
    logp = [0.0] * 8
    words = tokens_to_words(tokens, starts, durs, logp, offset=10.0)
    assert [w.text for w in words] == ["Well,", "I", "don't", "go."]
    assert words[0].start == 10.4 and abs(words[0].end - 10.64) < 1e-9  # comma does not stretch the word
    assert abs(words[2].start - 10.88) < 1e-9


def test_confidence_ignores_punctuation_tokens():
    import math

    words = tokens_to_words([" Hi", ","], [0, 0.2], [0.1, 0.1], [math.log(0.9), math.log(0.1)])
    assert abs(words[0].confidence - 0.9) < 1e-6


def test_words_never_overlap():
    words = tokens_to_words([" a", " b"], [0.0, 0.1], [0.5, 0.2], [0.0, 0.0])
    assert words[0].end <= words[1].start


# ---------- chunk planning ----------

def test_chunks_cut_inside_silence_not_inside_speech():
    audio = np.zeros(SAMPLE_RATE * 100, dtype=np.float32)
    # Speech 0-12, 12.5-25, 26-38, with gaps of 0.5s and 1s between them.
    regions = [(0.0, 12.0), (12.5, 25.0), (26.0, 38.0)]
    chunks = plan_chunks(audio, regions, max_len=30.0)
    assert len(chunks) == 2
    # The cut should land in the longer 1s pause (25-26), not the 0.5s one.
    assert 25.0 <= chunks[0].end <= 26.0 and 25.0 <= chunks[1].start <= 26.0
    for r_start, r_end in regions:  # every region fully covered by exactly the chunk it belongs to
        assert any(c.start <= r_start and r_end <= c.end for c in chunks)


def test_chunks_cover_all_speech_and_respect_max_len():
    audio = np.zeros(SAMPLE_RATE * 600, dtype=np.float32)
    rng = np.random.default_rng(1)
    regions, t = [], 0.0
    while t < 590:
        dur = float(rng.uniform(1, 9))
        regions.append((t, t + dur))
        t += dur + float(rng.uniform(0.2, 1.5))
    chunks = plan_chunks(audio, regions, max_len=30.0)
    assert all(c.end - c.start <= 30.0 + 1e-6 for c in chunks)
    for s, e in regions:
        assert any(c.start <= s + 1e-6 and e <= c.end + 1e-6 for c in chunks), (s, e)
    assert all(a.end <= b.start + 1e-6 for a, b in zip(chunks, chunks[1:]))


def test_continuous_speech_longer_than_max_is_split_at_quiet_point():
    sr = SAMPLE_RATE
    audio = (np.random.default_rng(0).normal(0, 0.1, sr * 70)).astype(np.float32)
    audio[int(22.0 * sr) : int(22.1 * sr)] = 0.0  # one quiet dip
    chunks = plan_chunks(audio, [(0.0, 70.0)], max_len=30.0)
    assert all(c.end - c.start <= 30.0 + 1e-6 for c in chunks)
    assert any(21.9 <= c.end <= 22.2 for c in chunks)


def test_no_speech_means_no_chunks():
    assert plan_chunks(np.zeros(SAMPLE_RATE, dtype=np.float32), []) == []


# ---------- stereo ----------

def test_out_of_phase_stereo_does_not_vanish():
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    sig = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    stereo = np.empty(sig.size * 2, dtype=np.float32)
    stereo[0::2], stereo[1::2] = sig, -sig  # a naive average would be silence
    mono = to_mono(stereo)
    assert np.sqrt(np.mean(mono**2)) > 0.15


def test_normal_stereo_is_averaged():
    sig = np.full(1000, 0.2, dtype=np.float32)
    stereo = np.empty(2000, dtype=np.float32)
    stereo[0::2], stereo[1::2] = sig, sig
    assert np.allclose(to_mono(stereo), 0.2)


def test_one_silent_channel_still_audible():
    sig = np.sin(np.linspace(0, 200, 16000)).astype(np.float32) * 0.3
    stereo = np.zeros(sig.size * 2, dtype=np.float32)
    stereo[0::2] = sig
    assert np.sqrt(np.mean(to_mono(stereo) ** 2)) > 0.05


# ---------- speakers ----------

def test_assign_speakers_by_overlap_and_order_of_appearance():
    words = [W("a", 0.0, 0.5), W("b", 0.6, 1.0), W("c", 5.0, 5.5), W("d", 5.6, 6.0)]
    turns = [Turn(0.0, 1.2, 7), Turn(4.9, 6.1, 3)]
    n = assign_speakers(words, turns)
    assert n == 2
    assert [w.speaker for w in words] == [0, 0, 1, 1]  # renumbered by first appearance


def test_word_in_gap_takes_nearest_turn_and_flicker_is_removed():
    words = [W("a", 0.0, 0.4), W("uh", 0.5, 0.7), W("b", 0.8, 1.2), W("far", 30.0, 30.3)]
    turns = [Turn(0.0, 0.45, 0), Turn(0.75, 1.3, 0)]
    assign_speakers(words, turns)
    assert [w.speaker for w in words[:3]] == [0, 0, 0]
    assert words[3].speaker == 0  # beyond max_gap: inherits the previous speaker, never None


def test_single_word_island_between_same_speaker_is_corrected():
    words = [W("x", 0, 0.3), W("y", 0.3, 0.5), W("z", 0.5, 0.8)]
    turns = [Turn(0, 0.3, 0), Turn(0.3, 0.5, 1), Turn(0.5, 0.8, 0)]
    assign_speakers(words, turns)
    assert [w.speaker for w in words] == [0, 0, 0]


# ---------- paragraphs and captions ----------

def test_segments_split_on_speaker_and_on_pause_after_sentence():
    words = [W("Hi.", 0, 0.5, 0), W("Yes", 0.6, 0.9, 1), W("ok.", 1.0, 1.3, 1), W("Later", 4.0, 4.4, 1)]
    segs = build_segments(words)
    assert [s.text for s in segs] == ["Hi.", "Yes ok.", "Later"]


def test_pause_without_sentence_end_keeps_paragraph_together():
    words = [W("so", 0, 0.3, 0), W("then", 2.0, 2.3, 0)]  # 1.7s gap but no full stop
    assert len(build_segments(words)) == 1


def test_cues_respect_line_length_and_speaker_changes():
    text = "this is a fairly long sentence that has to be wrapped across two lines for readability on screen".split()
    words = [W(t, i * 0.3, i * 0.3 + 0.25, 0) for i, t in enumerate(text)]
    words.append(W("Right.", 8.0, 8.4, 1))
    cues = build_cues(words)
    for c in cues:
        assert len(c.lines) <= 2 and all(len(line) <= 42 for line in c.lines)
    assert cues[-1].speaker == 1 and cues[-1].lines == ["Right."]
    assert all(a.end <= b.start for a, b in zip(cues, cues[1:]))


# ---------- formats ----------

def _transcript(words, speakers=2):
    return Transcript(words=words, segments=build_segments(words), duration=60, language="en",
                      engine="t", speaker_count=speakers, source="x.mp4")


def test_timestamp_formatting():
    assert stamp(3661.5, ",") == "01:01:01,500"
    assert stamp(0.0004) == "00:00:00.000"


def test_srt_and_vtt_structure():
    t = _transcript([W("Hello", 1.0, 1.4, 0), W("there.", 1.5, 2.0, 0), W("Hi.", 3.0, 3.3, 1)])
    srt, vtt = to_srt(t), to_vtt(t)
    assert srt.startswith("1\n00:00:01,000 --> ") and "Speaker 1: Hello there." in srt and "Speaker 2: Hi." in srt
    assert vtt.startswith("WEBVTT") and "00:00:01.000 --> " in vtt


def test_txt_hides_speaker_labels_when_only_one_speaker():
    t = _transcript([W("Hello.", 0, 1, 0)], speakers=1)
    assert "Speaker" not in to_txt(t)


# ---------- glossary ----------

def _g(rules, sentence):
    words = [W(w, i, i + 0.9) for i, w in enumerate(sentence.split())]
    out, n = Glossary.parse(rules).apply(words)
    return n, " ".join(w.text for w in out), out


def test_glossary_rule_keeps_punctuation_and_spans_timing():
    n, text, out = _g("Postgres Will => PostgreSQL", "we use Postgres Will, daily")
    assert text == "we use PostgreSQL, daily" and n == 1
    assert out[2].start == 2 and out[2].end == 3.9


def test_glossary_does_not_swallow_neighbouring_words():
    # Regression: "to" + "PostgreSQL" used to fuzzy match the term and eat "to".
    _, text, _ = _g("PostgreSQL", "moved the database to PostgreSQL last week")
    assert text == "moved the database to PostgreSQL last week"


def test_glossary_term_fixes_case_and_near_spelling():
    assert _g("Kubernetes", "we run kubernetes now")[1] == "we run Kubernetes now"
    assert _g("Grafana", "open the grafanna board")[1] == "open the Grafana board"


def test_glossary_leaves_unrelated_words_alone():
    assert _g("Sarah", "Sure I am sorry")[1] == "Sure I am sorry"


# ---------- WER ----------

def test_wer_ignores_case_and_punctuation():
    assert wer("Hello, World!", "hello world").wer == 0


def test_wer_counts_each_error_type():
    r = wer("the cat sat on the mat", "the dog sat the mat now")
    assert (r.substitutions, r.deletions, r.insertions) == (1, 1, 1)
    assert abs(r.wer - 3 / 6) < 1e-9
