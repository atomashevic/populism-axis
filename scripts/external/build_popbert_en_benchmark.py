#!/usr/bin/env python3
"""Attach the GPT-4o English translations of the PopBERT sentences to the frozen PopBERT split (CPU).

Translations: Wang, Yu, Wang & He (2026), "The Sound of Populism", Harvard Dataverse
doi:10.7910/DVN/VJGS7H (CC0), file gpt_translations.csv. Every sentence of the
repository's frozen PopBERT benchmark (data/benchmark_popbert/{train,test}.csv,
freeze pa_popbert_benchmark_freeze_r1) has a translation; the German text and the
majority-vote gold labels are identical to the frozen files. Writes
data/benchmark_popbert/{train,test}_en.csv (frozen columns + text_en) and a freeze note.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from populism_axis.run_manifest import file_sha256, write_run_manifest  # noqa: E402

SRC = ROOT / "data" / "external" / "sound_of_populism" / "gpt_translations.csv"
BENCH = ROOT / "data" / "benchmark_popbert"
OUT = ROOT / "results" / "external" / "popbert_en"


def majority(cell: str) -> int:
    votes = json.loads(cell)
    return int(sum(votes.values()) * 2 >= len(votes))


def main() -> None:
    translations = pd.read_csv(SRC)
    summary = {"schema": "populism_axis.v4.external_benchmark_freeze.v1", "freeze_id": "pa_v4_external_popbert_en_r1",
               "source": {"translations": "Wang et al. (2026), The Sound of Populism, doi:10.7910/DVN/VJGS7H (CC0), gpt_translations.csv (GPT-4o)",
                          "sentences_and_labels": "PopBERT (Erhard et al. 2025), frozen split pa_popbert_benchmark_freeze_r1 in data/benchmark_popbert/"},
               "splits": {}}
    for split in ("train", "test"):
        frame = pd.read_csv(BENCH / f"{split}.csv")
        merged = frame.merge(translations[["id", "text", "gpt_translation", "anti_elitism", "people_centrism"]], on="id", how="left", suffixes=("", "_sop"))
        if merged["gpt_translation"].isna().any() or not (merged["text"] == merged["text_sop"]).all():
            raise SystemExit(f"{split}: translation join failed")
        if not ((merged["anti_elitism"].map(majority) == merged["elite_gold"]).all() and (merged["people_centrism"].map(majority) == merged["centr_gold"]).all()):
            raise SystemExit(f"{split}: majority labels differ from the frozen gold")
        out = frame.copy()
        out["text_en"] = merged["gpt_translation"].astype(str).str.strip()
        out.to_csv(BENCH / f"{split}_en.csv", index=False)
        summary["splits"][split] = {"n": int(len(out)), "elite_pos": int(out["elite_gold"].sum()), "centr_pos": int(out["centr_gold"].sum()),
                                    "sha256": file_sha256(BENCH / f"{split}_en.csv"), "frozen_source_sha256": file_sha256(BENCH / f"{split}.csv"),
                                    "text_en_median_words": float(out["text_en"].str.split().str.len().median())}
    summary["published"] = {
        "popbert_gbert_5fold_cv_f1": {"elite": 0.847, "centr": 0.696, "macro4": 0.733},
        "sound_of_populism_roberta_large_on_gpt_translations_f1": {"elite": 0.85, "centr": 0.71, "left": 0.72, "right": 0.70, "macro4": 0.75,
                                                                    "note": "own split; not the frozen split used here (see README/notebook notes in the report)"},
        "v3_multilingual_e5_axis_on_german": {"elite_auroc": 0.794, "centr_auroc": 0.756, "elite_f1_train_threshold": 0.431, "centr_f1_train_threshold": 0.204,
                                              "source": "results/popbert/evaluation.json"},
        "human_coder_vs_loo_majority_f1": {"elite": 0.576, "centr": 0.327, "source": "results/popbert/freeze.json"},
    }
    summary["canonical_sha256"] = hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "benchmark_freeze.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_run_manifest(OUT, script="scripts/external/build_popbert_en_benchmark.py", inputs={"translations": SRC, "test": BENCH / "test.csv", "train": BENCH / "train.csv"},
                       extra={"freeze_id": summary["freeze_id"]}, filename="build.manifest.json")
    print(json.dumps(summary["splits"], indent=2))


if __name__ == "__main__":
    main()
