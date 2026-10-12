#!/usr/bin/env python3
"""Paired comparison of a candidate configuration against the carried baseline (CPU).

English-dev: leader-clustered paired bootstrap of the Spearman difference.
Agora: politician-clustered. Reuses `paired_delta` from the paired-comparison script. Applies
the adoption rule of docs/development-protocol.md and writes `paired_vs_previous.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from populism_axis.run_manifest import write_run_manifest  # noqa: E402
from paired_comparison import paired_delta  # noqa: E402

BENCH = ROOT / "data" / "benchmarks"
TARGETS = {
    "english": ("english_gpd_speeches.csv", "id", "source_score", "leader"),
    "agora": ("agora_paragraphs.csv", "row_id", "populism_human", "politician"),
}
MIN_PRIMARY = 0.01
MIN_SECONDARY = -0.01


def aligned(dataset: str, baseline_dir: Path, candidate_dir: Path, system: str, candidate_system: str):
    file_name, id_column, target_column, group_column = TARGETS[dataset]
    frame = pd.read_csv(BENCH / file_name).set_index(id_column)
    base = pd.read_csv(baseline_dir / f"scores_{dataset}.csv").set_index("id")
    cand = pd.read_csv(candidate_dir / f"scores_{dataset}.csv").set_index("id")
    if list(base.index) != list(cand.index):
        raise SystemExit(f"{dataset}: baseline and candidate score different documents")
    frame = frame.loc[base.index]
    return (
        cand[candidate_system].to_numpy(dtype=float),
        base[system].to_numpy(dtype=float),
        frame[target_column].to_numpy(dtype=float),
        frame[group_column].astype(str).to_numpy(),
    )


def decide(primary: dict, secondary: dict, rule: str) -> dict:
    checks = {
        "primary_delta_at_least_+0.01": primary["delta_spearman"] >= MIN_PRIMARY,
        "secondary_delta_at_least_-0.01": secondary["delta_spearman"] >= MIN_SECONDARY,
        "primary_ci_not_entirely_below_0": primary["ci95"][1] >= 0,
        "secondary_ci_not_entirely_below_0": secondary["ci95"][1] >= 0,
    }
    passes = all(checks.values())
    if rule == "defect_fix":
        outcome = "ADOPT (defect fix, adopted regardless of the rule)"
    else:
        outcome = "ADOPT" if passes else "KEEP PREVIOUS"
    return {"rule": rule, "checks": checks, "rule_passed": passes, "outcome": outcome}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--system", default="PE__frozen__soft")
    parser.add_argument("--candidate-system", default=None)
    parser.add_argument("--rule", choices=("standard", "defect_fix"), default="standard")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    candidate_system = args.candidate_system or args.system

    results: dict = {
        "baseline_dir": str(args.baseline_dir),
        "candidate_dir": str(args.candidate_dir),
        "system": args.system,
        "candidate_system": candidate_system,
        "draws": args.draws,
        "seed": args.seed,
        "paired": {},
    }
    for dataset in ("english", "agora"):
        left, right, target, groups = aligned(dataset, args.baseline_dir, args.candidate_dir, args.system, candidate_system)
        payload = paired_delta(left, right, target, groups, args.draws, args.seed)
        # The two-sided p (2 * min tail share) counts draws with delta exactly 0 in
        # both tails and can exceed 1 when many draws tie; a p-value is capped at 1.
        payload["p_two_sided_raw_v3_formula"] = payload["p_two_sided"]
        payload["p_two_sided"] = min(1.0, payload["p_two_sided"])
        payload["n_documents"] = int(len(target))
        payload["n_clusters"] = int(len(np.unique(groups)))
        payload["cluster"] = TARGETS[dataset][3]
        results["paired"][dataset] = payload
        print(
            f"{dataset:8s} delta={payload['delta_spearman']:+.4f} "
            f"CI[{payload['ci95'][0]:+.4f},{payload['ci95'][1]:+.4f}] p={payload['p_two_sided']:.4f} "
            f"(n={payload['n_documents']}, clusters={payload['n_clusters']})"
        )
    results["decision"] = decide(results["paired"]["english"], results["paired"]["agora"], args.rule)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_run_manifest(
        args.out.parent,
        script="scripts/compare_steps.py",
        inputs={
            "baseline_english": args.baseline_dir / "scores_english.csv",
            "candidate_english": args.candidate_dir / "scores_english.csv",
        },
        filename=args.out.stem + ".manifest.json",
    )
    print(f"decision: {results['decision']['outcome']}")


if __name__ == "__main__":
    main()
