"""Custom vocabulary, the equivalent of Descript's transcription glossary.

File format, one entry per line (blank lines and lines starting with # are ignored):

    n8n                      a term: fix its spelling and case wherever it is heard approximately
    en eight en => n8n       a rule: replace exactly this heard phrase with the right text
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path

from .types import Word

_EDGE = re.compile(r"^([^\w]*)(.*?)([^\w]*)$", re.S)


def _norm(text: str) -> str:
    return re.sub(r"[^\w]", "", text.lower())


def _split_edges(text: str) -> tuple[str, str, str]:
    m = _EDGE.match(text)
    return m.group(1), m.group(2), m.group(3)


@dataclass
class Glossary:
    rules: dict[tuple[str, ...], str]  # normalised heard words -> replacement
    terms: list[str]

    @classmethod
    def parse(cls, text: str) -> "Glossary":
        rules: dict[tuple[str, ...], str] = {}
        terms: list[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=>" in line:
                heard, right = (p.strip() for p in line.split("=>", 1))
                key = tuple(_norm(p) for p in heard.split() if _norm(p))
                if key and right:
                    rules[key] = right
            else:
                terms.append(line)
        return cls(rules, terms)

    @classmethod
    def load(cls, path: str | Path) -> "Glossary":
        return cls.parse(Path(path).read_text(encoding="utf-8"))

    def apply(self, words: list[Word], fuzzy: float = 0.86) -> tuple[list[Word], int]:
        """Return corrected words and the number of corrections made."""
        out: list[Word] = []
        changes = 0
        max_n = max([len(k) for k in self.rules] + [1])
        term_keys = {t: _norm(t) for t in self.terms}
        i = 0
        while i < len(words):
            hit = False
            # Longest explicit rule first.
            for n in range(min(max_n, len(words) - i), 0, -1):
                key = tuple(_norm(w.text) for w in words[i : i + n])
                if key in self.rules:
                    out.append(self._merge(words[i : i + n], self.rules[key]))
                    i += n
                    changes += 1
                    hit = True
                    break
            if hit:
                continue
            # Terms: exact (case insensitive), or a close spelling of one word or of two joined words.
            replaced = False
            for term, tkey in term_keys.items():
                if not tkey:
                    continue
                for n in (1, 2):
                    if i + n > len(words):
                        continue
                    parts = [_norm(w.text) for w in words[i : i + n]]
                    heard = "".join(parts)
                    if not heard:
                        continue
                    # Joining two words ("graph ana" -> Grafana) is only plausible when the
                    # first word is itself the start of the term. Otherwise "to" + "Postgres"
                    # looks 91% like "Postgres" and the stray word gets swallowed.
                    if n == 2 and (
                        not parts[0] or not parts[1]
                        or difflib.SequenceMatcher(None, parts[0], tkey[: len(parts[0])]).ratio() < 0.8
                        or len(parts[0]) < 3
                    ):
                        continue
                    exact = heard == tkey
                    close = (
                        len(tkey) >= 5 and heard != tkey
                        and abs(len(heard) - len(tkey)) <= 2
                        and difflib.SequenceMatcher(None, heard, tkey).ratio() >= fuzzy
                    )
                    if exact or close:
                        merged = self._merge(words[i : i + n], term)
                        if merged.text != " ".join(w.text for w in words[i : i + n]):
                            changes += 1
                        out.append(merged)
                        i += n
                        replaced = True
                        break
                if replaced:
                    break
            if not replaced:
                out.append(words[i])
                i += 1
        return out, changes

    @staticmethod
    def _merge(group: list[Word], replacement: str) -> Word:
        lead, _, _ = _split_edges(group[0].text)
        _, _, trail = _split_edges(group[-1].text)
        return Word(
            text=f"{lead}{replacement}{trail}",
            start=group[0].start,
            end=group[-1].end,
            confidence=min(w.confidence for w in group),
            speaker=group[0].speaker,
        )
