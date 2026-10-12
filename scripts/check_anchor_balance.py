#!/usr/bin/env python3
"""Surface-balance check for the pipeline anchor and nuisance sentence files (CPU, label-free).

For every pole of every channel (or nuisance set) it reports mean word count,
VADER compound and negative mass, and the share of sentences using they/them,
we/us/our, you/your and I/me/my. Pair-level flags mark large length or
sentiment gaps, pronoun-class mismatches and near-duplicate sentences.

Gates (anchor files, channels P and E only):
  |mean words populist - mainstream|      <= 2
  |pronoun-class share difference|       <= 0.10 for each class
  |mean VADER compound difference|        <= 0.15
M is reported but not gated: moral condemnation is negative by construction.
Nuisance files are reported without gates: the nuisance member differs by design.
Exit code 1 when a gate fails.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).resolve().parents[1]
PRONOUNS = {
    "they": re.compile(r"\b(they|them|their|theirs|themselves)\b", re.I),
    "we": re.compile(r"\b(we|us|our|ours|ourselves)\b", re.I),
    "you": re.compile(r"\b(you|your|yours)\b", re.I),
    "i": re.compile(r"\b(i|me|my|mine)\b", re.I),
}
GATED_CHANNELS = ("P", "E")
MAX_WORD_GAP = 2.0
MAX_PRONOUN_GAP = 0.10
MAX_VADER_GAP = 0.15
PAIR_WORD_FLAG = 4
PAIR_VADER_FLAG = 0.40
DUPLICATE_JACCARD = 0.60

_vader = SentimentIntensityAnalyzer()


def features(sentence: str) -> dict:
    scores = _vader.polarity_scores(sentence)
    out = {"words": len(sentence.split()), "vader": scores["compound"], "neg": scores["neg"]}
    out.update({name: bool(pattern.search(sentence)) for name, pattern in PRONOUNS.items()})
    return out


def pole_summary(sentences: list[str]) -> dict:
    rows = [features(sentence) for sentence in sentences]
    return {
        "n": len(rows),
        "words": float(np.mean([r["words"] for r in rows])),
        "vader": float(np.mean([r["vader"] for r in rows])),
        "neg": float(np.mean([r["neg"] for r in rows])),
        **{name: float(np.mean([r[name] for r in rows])) for name in PRONOUNS},
    }


def _tokens(sentence: str) -> set[str]:
    return set(re.findall(r"[a-z']+", sentence.lower()))


def near_duplicates(sentences: list[tuple[str, str]]) -> list[dict]:
    out = []
    for (id_a, a), (id_b, b) in itertools.combinations(sentences, 2):
        ta, tb = _tokens(a), _tokens(b)
        jaccard = len(ta & tb) / max(1, len(ta | tb))
        if jaccard > DUPLICATE_JACCARD:
            out.append({"a": id_a, "b": id_b, "jaccard": round(jaccard, 3)})
    return out


def load_poles(payload: dict) -> tuple[str, dict[str, dict]]:
    """Return (kind, {group: {'left_name','right_name','left','right','ids'}})."""
    schema = payload.get("schema", "")
    groups: dict[str, dict] = {}
    if schema.startswith("populism_axis.v4.pair_anchors"):
        for channel, block in payload["channels"].items():
            pairs = block["pairs"]
            groups[channel] = {
                "left_name": "populist", "right_name": "matched_mainstream",
                "left": [p["populist"] for p in pairs], "right": [p["matched_mainstream"] for p in pairs],
                "ids": [p["pair_id"] for p in pairs], "paired": True,
            }
        return "anchors", groups
    if schema.startswith("populism_axis.v3.proposition_anchors"):
        for channel, block in payload["channels"].items():
            groups[channel] = {
                "left_name": "populist", "right_name": "mainstream",
                "left": block["populist"], "right": block["mainstream"],
                "ids": [f"{channel}{i:02d}" for i in range(1, len(block["populist"]) + 1)], "paired": False,
            }
        return "anchors", groups
    if schema.startswith("populism_axis.v4.nuisance_sets"):
        for name, block in payload["sets"].items():
            pairs = block["pairs"]
            groups[name] = {
                "left_name": "nuisance", "right_name": "neutral",
                "left": [p["nuisance"] for p in pairs], "right": [p["neutral"] for p in pairs],
                "ids": [p["pair_id"] for p in pairs], "paired": True,
            }
        return "nuisance", groups
    raise SystemExit(f"unsupported schema {schema!r}")


def check(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    kind, groups = load_poles(payload)
    report: dict = {"file": str(path), "kind": kind, "schema": payload.get("schema"), "groups": {}, "gates_passed": True}
    for name, group in groups.items():
        left, right = pole_summary(group["left"]), pole_summary(group["right"])
        gaps = {key: left[key] - right[key] for key in ("words", "vader", "neg", *PRONOUNS)}
        gated = kind == "anchors" and name in GATED_CHANNELS
        gates = {}
        if gated:
            gates = {
                "words": abs(gaps["words"]) <= MAX_WORD_GAP,
                "vader": abs(gaps["vader"]) <= MAX_VADER_GAP,
                **{f"pronoun_{p}": abs(gaps[p]) <= MAX_PRONOUN_GAP for p in PRONOUNS},
            }
            if not all(gates.values()):
                report["gates_passed"] = False
        pair_flags = []
        if group["paired"]:
            for pair_id, a, b in zip(group["ids"], group["left"], group["right"]):
                fa, fb = features(a), features(b)
                flags = []
                if abs(fa["words"] - fb["words"]) > PAIR_WORD_FLAG:
                    flags.append(f"length {fa['words']} vs {fb['words']}")
                if gated and abs(fa["vader"] - fb["vader"]) > PAIR_VADER_FLAG:
                    flags.append(f"vader {fa['vader']:+.2f} vs {fb['vader']:+.2f}")
                mismatched = [p for p in PRONOUNS if fa[p] != fb[p]]
                if mismatched:
                    flags.append("pronoun " + ",".join(mismatched))
                if flags:
                    pair_flags.append({"pair_id": pair_id, "flags": flags})
        duplicates = near_duplicates(list(zip(group["ids"], group["left"])))
        report["groups"][name] = {
            "left_name": group["left_name"], "right_name": group["right_name"],
            "left": left, "right": right, "gaps": gaps, "gated": gated, "gates": gates,
            "pair_flags": pair_flags, "near_duplicates_left": duplicates,
        }
    return report


def to_markdown(reports: list[dict]) -> str:
    lines = ["# Anchor balance report", "",
             f"Gates for anchor channels {', '.join(GATED_CHANNELS)}: |Δ words| ≤ {MAX_WORD_GAP:g}, "
             f"|Δ VADER| ≤ {MAX_VADER_GAP:g}, |Δ pronoun share| ≤ {MAX_PRONOUN_GAP:g}. "
             "M and nuisance sets are reported without gates.", ""]
    for report in reports:
        lines += [f"## {report['file']}", "", f"Gates passed: **{report['gates_passed']}**", "",
                  "| Group | n | Words (L/R) | VADER (L/R) | Neg (L/R) | they (L/R) | we (L/R) | you (L/R) | I (L/R) | Gate |",
                  "|---|---:|---|---|---|---|---|---|---|---|"]
        for name, g in report["groups"].items():
            l, r = g["left"], g["right"]
            gate = "–" if not g["gated"] else ("pass" if all(g["gates"].values()) else
                                                "FAIL: " + ", ".join(k for k, ok in g["gates"].items() if not ok))
            lines.append(
                f"| {name} ({g['left_name']} / {g['right_name']}) | {l['n']} | {l['words']:.1f} / {r['words']:.1f} | "
                f"{l['vader']:+.2f} / {r['vader']:+.2f} | {l['neg']:.3f} / {r['neg']:.3f} | "
                f"{l['they']:.0%} / {r['they']:.0%} | {l['we']:.0%} / {r['we']:.0%} | "
                f"{l['you']:.0%} / {r['you']:.0%} | {l['i']:.0%} / {r['i']:.0%} | {gate} |"
            )
        flagged = [(name, f) for name, g in report["groups"].items() for f in g["pair_flags"]]
        dups = [(name, d) for name, g in report["groups"].items() for d in g["near_duplicates_left"]]
        lines += ["", f"Pair flags: {len(flagged)}"]
        lines += [f"- {name} {f['pair_id']}: {'; '.join(f['flags'])}" for name, f in flagged]
        if dups:
            lines += ["", "Near-duplicate sentences:"] + [f"- {name}: {d['a']} ~ {d['b']} (Jaccard {d['jaccard']})" for name, d in dups]
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()
    reports = [check(path) for path in args.files]
    text = to_markdown(reports)
    if args.out_md:
        args.out_md.write_text(text, encoding="utf-8")
    if args.out_json:
        args.out_json.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print(text)
    gated = [r for r in reports if r["kind"] == "anchors" and "v4" in (r["schema"] or "")]
    sys.exit(0 if all(r["gates_passed"] for r in gated) else 1)


if __name__ == "__main__":
    main()
