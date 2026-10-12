"""Label-free cleaning of English GPD speech files before windowing (stage 1b).

The raw files are left untouched; cleaning is applied at load time and is
versioned (`CLEANING_VERSION`), so every encode run records which rule it used.

Removed:
- metadata lines (`Name of politician: ...`, `Title of Speech: ...`, `Date: ...`,
  `Source: http...`, ...). Any whitelisted key is removed in the leading and
  trailing lines of the file (first 12, last 8 non-empty lines); a stricter key list is removed anywhere, on
  short lines only (source/link lines containing a URL at any length);
- lines that are only a URL;
- lines whose non-space characters are at least 20% CJK ideographs (the Chinese
  originals bundled into some Taiwan files), and stray ideographs elsewhere;
- the byte-order mark.
Everything else, including speaker labels inside transcripts, is kept.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

CLEANING_VERSION = "v4_clean_r1"

# Keys that are unambiguous metadata wherever they appear on a short line.
META_KEYS_ANYWHERE = (
    "name of politician", "name of executive", "name of chief executive", "name of the speech",
    "the name of the speech", "title of speech", "title of the speech", "chinese title of the speech",
    "date of speech", "date of the speech", "speech category", "category", "link", "links", "source",
    "accessed at", "delivered on", "words", "url", "name of speech", "type of speech", "body of the speech",
)
# Additional keys treated as metadata only in the leading or trailing block.
META_KEYS_EDGE = META_KEYS_ANYWHERE + (
    "speaker", "leader", "country", "type", "date", "time", "location", "title", "name",
    "occasion", "where", "when", "info", "brief description", "person", "speech", "youtube",
    "full transcript", "transcript", "famous speech", "special report",
)
_ANYWHERE = re.compile(r"^\s*(?:" + "|".join(re.escape(k) for k in META_KEYS_ANYWHERE) + r")\s*:", re.I)
_EDGE = re.compile(r"^\s*(?:" + "|".join(re.escape(k) for k in META_KEYS_EDGE) + r")\s*:", re.I)
_SOURCE_WITH_URL = re.compile(r"^\s*(?:source|links?|accessed at|url)\s*:.*(?:https?://|www\.)", re.I)
_URL_LINE = re.compile(r"^\s*[\"'(<]?\s*(?:https?://|www\.)\S+\s*[\"')>]?\s*$", re.I)
_IDEOGRAPH = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_CJK_PUNCT = re.compile(r"[　-〿！-／：-＠［-｀｛-･]")
MAX_ANYWHERE_LINE_CHARS = 200
CJK_LINE_SHARE = 0.20
HEAD_LINES = 12
TAIL_LINES = 8


@dataclass
class CleaningLog:
    removed_metadata: list[str] = field(default_factory=list)
    removed_url: list[str] = field(default_factory=list)
    removed_cjk_lines: int = 0
    removed_cjk_chars: int = 0
    stripped_ideographs: int = 0

    def as_dict(self) -> dict:
        return {
            "removed_metadata": self.removed_metadata,
            "removed_url": self.removed_url,
            "removed_cjk_lines": self.removed_cjk_lines,
            "removed_cjk_chars": self.removed_cjk_chars,
            "stripped_ideographs": self.stripped_ideographs,
        }


def _ideograph_share(line: str) -> float:
    chars = [ch for ch in line if not ch.isspace()]
    if not chars:
        return 0.0
    return sum(1 for ch in chars if _IDEOGRAPH.match(ch)) / len(chars)


def _is_edge_metadata(line: str) -> bool:
    return bool(_EDGE.match(line) or _URL_LINE.match(line))


def clean_speech_text(text: str) -> tuple[str, CleaningLog]:
    log = CleaningLog()
    lines = str(text).replace("﻿", "").split("\n")

    # 1. Chinese lines and stray ideographs.
    kept: list[str] = []
    for line in lines:
        if _IDEOGRAPH.search(line) and _ideograph_share(line) >= CJK_LINE_SHARE:
            log.removed_cjk_lines += 1
            log.removed_cjk_chars += len(line.strip())
            continue
        if _IDEOGRAPH.search(line):
            log.stripped_ideographs += len(_IDEOGRAPH.findall(line))
            line = _CJK_PUNCT.sub(" ", _IDEOGRAPH.sub("", line))
        kept.append(line)
    lines = kept

    # 2. Metadata lines within the first HEAD_LINES and last TAIL_LINES non-empty lines.
    def strip_edge(sequence: list[str], limit: int) -> list[str]:
        out: list[str] = []
        seen = 0
        for position, line in enumerate(sequence):
            if seen >= limit:
                out.extend(sequence[position:])
                break
            if line.strip():
                seen += 1
                if _is_edge_metadata(line) and len(line.strip()) <= MAX_ANYWHERE_LINE_CHARS:
                    target = log.removed_url if _URL_LINE.match(line) else log.removed_metadata
                    target.append(line.strip())
                    continue
            out.append(line)
        return out

    lines = strip_edge(lines, HEAD_LINES)
    lines = list(reversed(strip_edge(list(reversed(lines)), TAIL_LINES)))

    # 3. Strict metadata keys and bare URL lines anywhere.
    kept = []
    for line in lines:
        if _URL_LINE.match(line):
            log.removed_url.append(line.strip())
            continue
        if _SOURCE_WITH_URL.match(line) or (
            _ANYWHERE.match(line) and len(line.strip()) <= MAX_ANYWHERE_LINE_CHARS
        ):
            log.removed_metadata.append(line.strip())
            continue
        kept.append(line)
    return "\n".join(kept).strip(), log
