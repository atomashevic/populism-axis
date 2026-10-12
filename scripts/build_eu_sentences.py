#!/usr/bin/env python3
"""Rebuild data/benchmarks/eu_sentences.csv from the EU-OOD dataset on the Hugging Face Hub.

The EU-OOD sentences (Chalkidis et al. 2025, coastalcph/populism-eu-speeches) are
licensed CC BY-NC-ND 4.0, so the public repository does not redistribute them. This
script downloads the dataset (or reads a local copy of its train.csv), applies the
project's re-shaping (row ids, English text column, speaker groups), and checks the
result against the SHA-256 recorded in data/benchmarks/freeze.json.

    uv run --extra cpu python scripts/build_eu_sentences.py
    uv run --extra cpu python scripts/build_eu_sentences.py --source path/to/train.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "benchmarks" / "eu_sentences.csv"
REPO_ID = "coastalcph/populism-eu-speeches"


def build(source: pd.DataFrame) -> pd.DataFrame:
    """Same rows and order as the source; English (DeepL) text is the scored text; speakers are the groups."""
    out = pd.DataFrame({
        "row_id": [f"eu_{i:05d}" for i in range(len(source))],
        "text": source["translated_sentence"],
        "sentence": source["sentence"],
        "translated_sentence": source["translated_sentence"],
        "pop_code": source["pop_code"],
        "speaker": source["speaker"],
        "speech_id": source["speech_id"],
        "group_id": source["speaker"],
    })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, help="Local train.csv of the dataset; downloads it from the Hub if omitted")
    parser.add_argument("--revision", default=None, help="Hub revision of the dataset (default: main)")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    source_path = args.source
    if source_path is None:
        from huggingface_hub import hf_hub_download

        source_path = Path(hf_hub_download(REPO_ID, "train.csv", repo_type="dataset", revision=args.revision))
    frame = build(pd.read_csv(source_path))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    expected = json.loads((ROOT / "data" / "benchmarks" / "freeze.json").read_text())["datasets"]["eu_sentences"]["sha256"]
    if digest != expected:
        print(f"WARNING: {args.out} has sha256 {digest}, frozen table has {expected}; the Hub dataset may have changed.")
        return 1
    print(f"wrote {args.out} ({len(frame)} rows); matches the frozen table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
