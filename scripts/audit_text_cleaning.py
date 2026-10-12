#!/usr/bin/env python3
"""Apply the speech-text cleaning to every English GPD speech and write an audit (CPU).

Outputs (tracked, small):
  results/step1b_text_cleaning/cleaning_audit.json   per-speech counts and cleaned-text SHA-256
  results/step1b_text_cleaning/removed_lines.csv     every removed metadata / URL line, for review
The raw speech files are not modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from populism_axis.run_manifest import write_run_manifest  # noqa: E402
from populism_axis.text import CLEANING_VERSION, clean_speech_text  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--english", type=Path, default=ROOT / "data" / "benchmarks" / "english_gpd_speeches.csv")
    parser.add_argument("--split", type=Path, default=ROOT / "data" / "benchmarks" / "english_split.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "step1b_text_cleaning")
    args = parser.parse_args()

    english = pd.read_csv(args.english)
    split = pd.read_csv(args.split).set_index("id")["split"]
    per_speech = []
    removed_rows = []
    for row in english.itertuples(index=False):
        raw = (ROOT / str(row.path)).read_text(encoding="utf-8", errors="ignore")
        cleaned, log = clean_speech_text(raw)
        per_speech.append(
            {
                "id": row.id,
                "split": split[row.id],
                "words_raw": len(raw.split()),
                "words_clean": len(cleaned.split()),
                "removed_metadata_lines": len(log.removed_metadata),
                "removed_url_lines": len(log.removed_url),
                "removed_cjk_lines": log.removed_cjk_lines,
                "removed_cjk_chars": log.removed_cjk_chars,
                "stripped_ideographs": log.stripped_ideographs,
                "clean_sha256": hashlib.sha256(cleaned.encode("utf-8")).hexdigest(),
            }
        )
        removed_rows += [{"id": row.id, "kind": "metadata", "line": line} for line in log.removed_metadata]
        removed_rows += [{"id": row.id, "kind": "url", "line": line} for line in log.removed_url]

    frame = pd.DataFrame(per_speech)
    changed = frame[(frame["words_raw"] != frame["words_clean"]) | (frame["removed_cjk_lines"] > 0)]
    summary = {
        "cleaning_version": CLEANING_VERSION,
        "n_speeches": int(len(frame)),
        "n_changed": int(len(changed)),
        "n_changed_by_split": {k: int(v) for k, v in changed["split"].value_counts().items()},
        "words_raw": int(frame["words_raw"].sum()),
        "words_clean": int(frame["words_clean"].sum()),
        "removed_metadata_lines": int(frame["removed_metadata_lines"].sum()),
        "removed_url_lines": int(frame["removed_url_lines"].sum()),
        "speeches_with_cjk_lines_removed": changed.loc[changed["removed_cjk_lines"] > 0, "id"].tolist(),
        "removed_cjk_lines": int(frame["removed_cjk_lines"].sum()),
        "removed_cjk_chars": int(frame["removed_cjk_chars"].sum()),
        "stripped_ideographs": int(frame["stripped_ideographs"].sum()),
        "cleaned_corpus_sha256": hashlib.sha256("".join(frame["clean_sha256"]).encode("utf-8")).hexdigest(),
        "per_speech": per_speech,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "cleaning_audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(removed_rows).to_csv(args.out_dir / "removed_lines.csv", index=False)
    write_run_manifest(
        args.out_dir,
        script="scripts/audit_text_cleaning.py",
        inputs={"english": args.english, "split": args.split, "cleaner": ROOT / "src" / "populism_axis" / "text.py"},
        extra={"cleaning_version": CLEANING_VERSION},
        filename="cleaning_audit.manifest.json",
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "per_speech"}, indent=2))


if __name__ == "__main__":
    main()
