"""Word error rate, so you can measure instead of guessing."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_CONTRACTIONS_JOIN = re.compile(r"(?<=\w)['’](?=\w)")


def normalise(text: str) -> list[str]:
    """Lowercase, strip accents' punctuation but keep letters/digits, so style differences
    (capitals, commas) are not counted as errors."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _CONTRACTIONS_JOIN.sub("", text)  # don't -> dont, same on both sides
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


@dataclass
class WerResult:
    wer: float
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    def __str__(self) -> str:
        return (f"WER {self.wer * 100:.2f}%  (sub {self.substitutions}, del {self.deletions}, "
                f"ins {self.insertions}, over {self.reference_words} reference words)")


def wer(reference: str, hypothesis: str) -> WerResult:
    ref, hyp = normalise(reference), normalise(hypothesis)
    n, m = len(ref), len(hyp)
    # dp[i][j] = (cost, sub, del, ins)
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                best = prev[j - 1]
            else:
                c, s, d, ins = prev[j - 1]
                best = (c + 1, s + 1, d, ins)
            c, s, d, ins = prev[j]
            if c + 1 < best[0]:
                best = (c + 1, s, d + 1, ins)
            c, s, d, ins = cur[j - 1]
            if c + 1 < best[0]:
                best = (c + 1, s, d, ins + 1)
            cur.append(best)
        prev = cur
    cost, s, d, ins = prev[m]
    return WerResult(cost / n if n else (0.0 if m == 0 else 1.0), s, d, ins, n)
