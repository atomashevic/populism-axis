#!/usr/bin/env python3
"""stage 2 grid: label-free reference transform x nuisance residualization (CPU).

Cells: n_pcs in {0,1,2,4,8} x residualize in {off,on} x whitening in {none, shrinkage .5}.
Pre-declared primary cell: n_pcs=2, residualize on, no whitening.
Every cell is scored on English-dev and Agora with scripts/run_config.py,
compared with the carried step-1b configuration by scripts/compare_steps.py,
and summarized in results/step2_whitening/grid.{json,md}. The selection rule
of docs/development-protocol.md is applied to the whole grid and to the primary cell.

Label-free diagnostic per cell: Spearman correlation between window-level E and M
projections and VADER negativity of the window text (English-dev and Agora).
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from populism_axis.run_manifest import write_run_manifest  # noqa: E402
from compare_steps import MIN_PRIMARY, MIN_SECONDARY  # noqa: E402

STEP = ROOT / "results" / "step2_whitening"
N_PCS = (0, 1, 2, 4, 8)
RESIDUALIZE = (False, True)
WHITEN = (None, 0.5)
PRIMARY = (2, True, None)
SYSTEM = "PE__frozen__soft"


def cell_label(n_pcs: int, residualize: bool, whiten: float | None) -> str:
    return f"d{n_pcs}_{'res' if residualize else 'nores'}_{'zca50' if whiten is not None else 'nowhite'}"


def vader_negativity(windows_path: Path, cache: Path) -> np.ndarray:
    if cache.exists():
        return np.load(cache)["neg"]
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    texts = pd.read_parquet(windows_path)["window"].tolist()
    neg = np.asarray([analyzer.polarity_scores(text)["neg"] for text in texts], dtype=np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, neg=neg)
    return neg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carried", type=Path, default=ROOT / "results" / "step1b_text_cleaning" / "carried_config.json")
    parser.add_argument("--nuisance", type=Path, required=True, help="nuisance sentence npz (encoded, reviewed set)")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--only", default="", help="comma-separated cell labels (debugging)")
    args = parser.parse_args()

    carried = json.loads(args.carried.read_text(encoding="utf-8"))
    english_dir = ROOT / carried["embeddings_dirs"]["english"]
    agora_dir = ROOT / carried["embeddings_dirs"]["agora"]
    anchors = ROOT / carried["anchors"]
    baseline_dir = ROOT / carried["result_dir"]
    only = {item for item in args.only.split(",") if item}

    split = pd.read_csv(ROOT / "data" / "benchmarks" / "english_split.csv").set_index("id")["split"]
    english = pd.read_csv(ROOT / "data" / "benchmarks" / "english_gpd_speeches.csv")
    dev_docs = set(np.flatnonzero(english["id"].map(split).eq("dev").to_numpy()))
    english_windows = pd.read_parquet(english_dir / "windows_english.parquet")
    dev_window_mask = english_windows["doc_index"].isin(dev_docs).to_numpy()
    neg = {
        "english": vader_negativity(english_dir / "windows_english.parquet", STEP / "cache" / "vader_neg_english.npz")[dev_window_mask],
        "agora": vader_negativity(agora_dir / "windows_agora.parquet", STEP / "cache" / "vader_neg_agora.npz"),
    }

    # Reference row: the carried pole-mean configuration, re-scored to save projections for the diagnostic.
    reference_dir = STEP / "grid" / "reference_v3_rule"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_config.py"), "--label", "reference_v3_rule",
         "--embeddings-dir", str(english_dir), "--dataset-dir", f"agora={agora_dir}", "--anchors", str(anchors),
         "--out-dir", str(reference_dir), "--draws", str(args.draws), "--save-projections"],
        check=True,
    )
    reference_vader = {}
    for dataset in ("english", "agora"):
        projections = np.load(reference_dir / f"projections_{dataset}.npz")
        for channel in ("E", "M"):
            reference_vader[f"{dataset}_{channel}"] = float(spearmanr(projections[channel], neg[dataset]).statistic)

    rows = []
    for n_pcs, residualize, whiten in itertools.product(N_PCS, RESIDUALIZE, WHITEN):
        label = cell_label(n_pcs, residualize, whiten)
        if only and label not in only:
            continue
        out_dir = STEP / "grid" / label
        command = [
            sys.executable, str(ROOT / "scripts" / "run_config.py"),
            "--label", label, "--embeddings-dir", str(english_dir), "--dataset-dir", f"agora={agora_dir}",
            "--anchors", str(anchors), "--out-dir", str(out_dir), "--transform", "ref",
            "--n-pcs", str(n_pcs), "--draws", str(args.draws), "--save-projections",
        ]
        if residualize:
            command += ["--nuisance", str(args.nuisance)]
        if whiten is not None:
            command += ["--whiten-shrinkage", str(whiten)]
        subprocess.run(command, check=True)
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "compare_steps.py"),
             "--baseline-dir", str(baseline_dir), "--candidate-dir", str(out_dir),
             "--out", str(out_dir / "paired_vs_previous.json")],
            check=True,
        )
        evaluation = json.loads((out_dir / "evaluation.json").read_text(encoding="utf-8"))["systems"]
        paired = json.loads((out_dir / "paired_vs_previous.json").read_text(encoding="utf-8"))
        geometry = json.loads((out_dir / "geometry.json").read_text(encoding="utf-8"))
        vader_corr = {}
        for dataset in ("english", "agora"):
            projections = np.load(out_dir / f"projections_{dataset}.npz")
            for channel in ("E", "M"):
                vader_corr[f"{dataset}_{channel}"] = float(spearmanr(projections[channel], neg[dataset]).statistic)
        row = {
            "cell": label, "n_pcs": n_pcs, "residualize": residualize, "whiten_shrinkage": whiten,
            "primary_cell": (n_pcs, residualize, whiten) == PRIMARY,
            "english_rho": evaluation[SYSTEM]["english"]["spearman"],
            "english_ci95": evaluation[SYSTEM]["english"]["spearman_ci95"],
            "agora_rho": evaluation[SYSTEM]["agora"]["spearman"],
            "english_extreme_auroc": evaluation[SYSTEM]["english"]["extreme_auroc"],
            "english_leader_term_rho": evaluation[SYSTEM]["english"]["leader_term_spearman"],
            "agora_high_low_auroc": evaluation[SYSTEM]["agora"]["high_vs_low_auroc"],
            "P_english_rho": evaluation["P__frozen__soft"]["english"]["spearman"],
            "PEM_english_rho": evaluation["PEM__frozen__soft"]["english"]["spearman"],
            "delta_english": paired["paired"]["english"]["delta_spearman"],
            "delta_english_ci95": paired["paired"]["english"]["ci95"],
            "delta_agora": paired["paired"]["agora"]["delta_spearman"],
            "delta_agora_ci95": paired["paired"]["agora"]["ci95"],
            "rule_passed": paired["decision"]["rule_passed"],
            "vader_neg_spearman": vader_corr,
            "english_channel_gram": geometry["per_dataset"]["english"]["channel_axis_gram"],
            "english_residualization": geometry["per_dataset"]["english"].get("residualization"),
            "removed_pc_variance_english": geometry["per_dataset"]["english"]["transform"]["explained_variance_of_removed_pcs"],
        }
        rows.append(row)
        print(f"{label:24s} ENG {row['english_rho']:.4f} (Δ {row['delta_english']:+.4f})  "
              f"AG {row['agora_rho']:.4f} (Δ {row['delta_agora']:+.4f})  rule={row['rule_passed']}", flush=True)

    passing = [row for row in rows if row["rule_passed"]]

    def complexity(row: dict) -> tuple:
        return (row["n_pcs"], row["residualize"], row["whiten_shrinkage"] is not None)

    selected = None
    if passing:
        best = max((r["delta_english"] + r["delta_agora"]) / 2 for r in passing)
        tied = [r for r in passing if (r["delta_english"] + r["delta_agora"]) / 2 >= best - 0.005]
        selected = min(tied, key=complexity)["cell"]
    primary = next((r for r in rows if r["primary_cell"]), None)
    summary = {
        "baseline": str(baseline_dir.relative_to(ROOT)),
        "rule": {"min_delta_english": MIN_PRIMARY, "min_delta_agora": MIN_SECONDARY,
                 "selection": "max mean(delta_english, delta_agora) among passing cells; ties within .005 to the simpler cell"},
        "primary_cell": primary["cell"] if primary else None,
        "primary_rule_passed": primary["rule_passed"] if primary else None,
        "selected_cell": selected,
        "reference_v3_rule_vader_neg_spearman": reference_vader,
        "cells": rows,
    }
    (STEP / "grid.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = ["# stage 2 grid", "",
             f"Baseline: `{summary['baseline']}`. System `{SYSTEM}`. Δ = cell minus baseline, paired cluster bootstrap.",
             f"Pre-declared primary cell: `{summary['primary_cell']}`. Selected cell: `{selected}`.", "",
             "| Cell | English-dev ρ | Δ English [95% CI] | Agora ρ | Δ Agora [95% CI] | Rule | E~neg (Eng) | M~neg (Eng) | E~neg (Agora) |",
             "|---|---:|---|---:|---|---|---:|---:|---:|",
             f"| carried baseline (pole-mean rule) | – | – | – | – | – | {reference_vader['english_E']:+.3f} | "
             f"{reference_vader['english_M']:+.3f} | {reference_vader['agora_E']:+.3f} |"]
    for r in rows:
        mark = " (primary)" if r["primary_cell"] else ""
        lines.append(
            f"| `{r['cell']}`{mark} | {r['english_rho']:.3f} | {r['delta_english']:+.3f} "
            f"[{r['delta_english_ci95'][0]:+.3f}, {r['delta_english_ci95'][1]:+.3f}] | {r['agora_rho']:.3f} | "
            f"{r['delta_agora']:+.3f} [{r['delta_agora_ci95'][0]:+.3f}, {r['delta_agora_ci95'][1]:+.3f}] | "
            f"{'pass' if r['rule_passed'] else 'fail'} | {r['vader_neg_spearman']['english_E']:+.3f} | "
            f"{r['vader_neg_spearman']['english_M']:+.3f} | {r['vader_neg_spearman']['agora_E']:+.3f} |"
        )
    (STEP / "grid.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_run_manifest(
        STEP,
        script="scripts/sweep_whitening.py",
        inputs={"carried": args.carried, "nuisance": args.nuisance, "anchors": anchors},
        extra={"cells": [r["cell"] for r in rows], "selected": selected},
        filename="grid.manifest.json",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
