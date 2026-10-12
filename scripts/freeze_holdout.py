#!/usr/bin/env python3
"""Freeze the English development/holdout split (country level).

Writes `data/benchmarks/english_split.csv` and
`results/step0_env_holdout/holdout_freeze.json`, and refuses to overwrite
either. EU-OOD and the English holdout are scored once, after stage 6.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from populism_axis.run_manifest import file_sha256, write_run_manifest  # noqa: E402
from populism_axis.split import SplitRule, balance_distance, choose_country_split, split_frame  # noqa: E402
from benchmarks import english_null_baselines  # noqa: E402

SCHEMA = "populism_axis.v4.holdout_freeze.v1"
FREEZE_ID = "pa_v4_english_holdout_r1"


def _describe(frame: pd.DataFrame) -> dict:
    scores = frame["source_score"].astype(float)
    return {
        "n_speeches": int(len(frame)),
        "n_leaders": int(frame["leader"].nunique()),
        "n_countries": int(frame["country"].nunique()),
        "n_leader_terms": int(frame["leader_term_id"].nunique()),
        "extreme_positive": int((frame["extreme_label"] == 1).sum()),
        "extreme_negative": int((frame["extreme_label"] == 0).sum()),
        "nonzero_score": int((scores > 0).sum()),
        "mean_score": float(scores.mean()),
        "median_score": float(scores.median()),
        "speechtype_counts": {str(k): int(v) for k, v in frame["speechtype"].value_counts().sort_index().items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--english", type=Path, default=ROOT / "data" / "benchmarks" / "english_gpd_speeches.csv")
    parser.add_argument("--gpd-csv", type=Path, default=ROOT / "data" / "gpd_v2_20220427.csv")
    parser.add_argument("--v3-freeze", type=Path, default=ROOT / "data" / "benchmarks" / "freeze.json")
    parser.add_argument("--split-out", type=Path, default=ROOT / "data" / "benchmarks" / "english_split.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "step0_env_holdout")
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()

    freeze_path = args.out_dir / "holdout_freeze.json"
    for path in (args.split_out, freeze_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite frozen file: {path}")

    english = pd.read_csv(args.english)
    rule = SplitRule()
    choice = choose_country_split(english, rule)
    split = split_frame(english, choice["holdout_countries"])
    mask = english["country"].astype(str).isin(choice["holdout_countries"]).to_numpy()

    args.split_out.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(args.split_out, index=False)

    per_split = {}
    for name, sub_mask in (("dev", ~mask), ("holdout", mask)):
        sub = english[sub_mask].reset_index(drop=True)
        nulls = english_null_baselines(sub, args.gpd_csv, args.bootstrap_draws, rule.seed)
        per_split[name] = {
            **_describe(sub),
            "countries": sorted(sub["country"].astype(str).unique().tolist()),
            "human_ceiling_spearman": nulls["human_ceiling_spearman"],
            "human_coder_pairs": nulls["human_coder_pairs"],
            "word_count_spearman": nulls["confound_baselines"]["word_count"]["spearman"],
            "word_count_spearman_ci95": nulls["confound_baselines"]["word_count"]["spearman_ci95"],
        }

    v3_freeze = json.loads(args.v3_freeze.read_text(encoding="utf-8"))
    payload = {
        "schema": SCHEMA,
        "freeze_id": FREEZE_ID,
        "rule": rule.as_dict(),
        "selection": (
            "Sample country subsets (each country included with the rule's probability), keep those "
            "meeting every constraint, choose the one minimizing |mean score dev - holdout| + KS distance. "
            "Labels are used only for balance, never for method choice."
        ),
        "n_candidates_unique": choice["n_candidates_unique"],
        "n_candidates_passed": choice["n_candidates_passed"],
        "balance_distance": choice["balance_distance"],
        "holdout_countries": choice["holdout_countries"],
        "split_file": {"path": str(args.split_out.relative_to(ROOT)), "sha256": file_sha256(args.split_out)},
        "source_dataset": {
            "path": str(args.english.relative_to(ROOT)),
            "file_sha256": file_sha256(args.english),
            "v3_freeze_canonical_sha256": v3_freeze["canonical_sha256"],
            "v3_frame_sha256": v3_freeze["datasets"]["english_gpd"]["sha256"],
        },
        "splits": per_split,
        "access_policy": [
            "Development decisions use English dev rows and Agora only.",
            "English holdout rows and EU-OOD are scored exactly once, after stage 6, via --final.",
            "Every holdout or EU access is appended to results/holdout_access_log.json.",
            "The holdout was visible to initial configuration choices (the initial configuration used all 258 speeches for development); it is unseen by the stepwise decisions only.",
        ],
    }
    payload["canonical_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_run_manifest(
        args.out_dir,
        script="scripts/freeze_holdout.py",
        inputs={"english": args.english, "gpd_csv": args.gpd_csv, "v3_freeze": args.v3_freeze},
        extra={"rule": rule.as_dict(), "split_sha256": payload["split_file"]["sha256"]},
    )

    print(f"holdout countries ({len(choice['holdout_countries'])}): {', '.join(choice['holdout_countries'])}")
    print(f"candidates unique={choice['n_candidates_unique']} passed={choice['n_candidates_passed']} "
          f"balance distance={choice['balance_distance']:.4f}")
    for name, stats in per_split.items():
        print(f"{name:8s} n={stats['n_speeches']:3d} leaders={stats['n_leaders']:2d} countries={stats['n_countries']:2d} "
              f"extreme+={stats['extreme_positive']:2d} nonzero={stats['nonzero_score']:3d} "
              f"mean={stats['mean_score']:.3f} ceiling={stats['human_ceiling_spearman']:.3f} "
              f"wordcount rho={stats['word_count_spearman']:.3f}")
    print(f"wrote {args.split_out} and {freeze_path}")


if __name__ == "__main__":
    main()
