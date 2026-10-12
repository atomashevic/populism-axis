#!/usr/bin/env python3
"""stage 4 grid: window size x document pooling rule (CPU).

Cells: sentences per window in {1, 2, 3} (stride 1, 60-word cap per sentence)
x pooling in {k:1, k:2, k:3, k:5, k:8, frac:0.10, softcount:0.9}.
Pre-declared primary cell: spw 1, k:3 (the carried configuration), so the grid
asks whether any other cell beats it under docs/development-protocol.md.
Also reports, per window size, how the pervasiveness readout (softcount:0.9)
relates to the carried top-3 score.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from populism_axis.run_manifest import write_run_manifest  # noqa: E402

STEP = ROOT / "results" / "step4_windows_topk"
EMB = ROOT / "results" / "embeddings" / "e5-large-v2"
SPW = (1, 2, 3)
RULES = ("k:1", "k:2", "k:3", "k:5", "k:8", "frac:0.10", "softcount:0.9")
PRIMARY = (1, "k:3")
SYSTEM = "PE__frozen__soft"


def dirs_for(spw: int) -> tuple[Path, Path]:
    if spw == 1:
        return EMB / "spw1_cap60_clean", EMB / "spw1_cap60"
    return EMB / f"spw{spw}_cap60_clean", EMB / f"spw{spw}_cap60_clean"


def cell_label(spw: int, rule: str) -> str:
    return f"spw{spw}_{rule.replace(':', '').replace('.', 'p')}"


def complexity(row: dict) -> tuple:
    rule_rank = {"k:3": 0, "k:2": 1, "k:5": 1, "k:1": 2, "k:8": 2, "frac:0.10": 3, "softcount:0.9": 3}
    return (row["spw"], rule_rank[row["rule"]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carried", type=Path, default=ROOT / "results" / "step3_pair_anchors" / "carried_config.json")
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()
    carried = json.loads(args.carried.read_text(encoding="utf-8"))
    anchors = ROOT / carried["anchors"]
    baseline = ROOT / carried["result_dir"]

    rows = []
    for spw, rule in itertools.product(SPW, RULES):
        label = cell_label(spw, rule)
        out_dir = STEP / "grid" / label
        english_dir, agora_dir = dirs_for(spw)
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_config.py"), "--label", label,
             "--embeddings-dir", str(english_dir), "--dataset-dir", f"agora={agora_dir}", "--anchors", str(anchors),
             "--out-dir", str(out_dir), "--top-rule", rule, "--draws", str(args.draws)],
            check=True,
        )
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "compare_steps.py"), "--baseline-dir", str(baseline),
             "--candidate-dir", str(out_dir), "--out", str(out_dir / "paired_vs_previous.json")],
            check=True,
        )
        evaluation = json.loads((out_dir / "evaluation.json").read_text(encoding="utf-8"))
        systems = evaluation["systems"]
        paired = json.loads((out_dir / "paired_vs_previous.json").read_text(encoding="utf-8"))
        row = {
            "cell": label, "spw": spw, "rule": rule, "primary_cell": (spw, rule) == PRIMARY,
            "english_rho": systems[SYSTEM]["english"]["spearman"],
            "english_ci95": systems[SYSTEM]["english"]["spearman_ci95"],
            "english_extreme_auroc": systems[SYSTEM]["english"]["extreme_auroc"],
            "english_leader_term_rho": systems[SYSTEM]["english"]["leader_term_spearman"],
            "agora_rho": systems[SYSTEM]["agora"]["spearman"],
            "agora_high_low_auroc": systems[SYSTEM]["agora"]["high_vs_low_auroc"],
            "P_english_rho": systems["P__frozen__soft"]["english"]["spearman"],
            "delta_english": paired["paired"]["english"]["delta_spearman"],
            "delta_english_ci95": paired["paired"]["english"]["ci95"],
            "delta_agora": paired["paired"]["agora"]["delta_spearman"],
            "delta_agora_ci95": paired["paired"]["agora"]["ci95"],
            "rule_passed": paired["decision"]["rule_passed"],
            "windows": evaluation["window_counts"],
        }
        rows.append(row)
        print(f"{label:22s} ENG {row['english_rho']:.4f} (Δ {row['delta_english']:+.4f})  "
              f"AG {row['agora_rho']:.4f} (Δ {row['delta_agora']:+.4f})  rule={row['rule_passed']}", flush=True)

    pervasiveness = {}
    for spw in SPW:
        for dataset in ("english",):
            top3 = pd.read_csv(STEP / "grid" / cell_label(spw, "k:3") / f"scores_{dataset}.csv")[SYSTEM]
            soft = pd.read_csv(STEP / "grid" / cell_label(spw, "softcount:0.9") / f"scores_{dataset}.csv")[SYSTEM]
            pervasiveness[f"spw{spw}_{dataset}"] = {"spearman_softcount_vs_top3": float(spearmanr(top3, soft).statistic)}

    passing = [row for row in rows if row["rule_passed"]]
    selected = None
    if passing:
        best = max((r["delta_english"] + r["delta_agora"]) / 2 for r in passing)
        tied = [r for r in passing if (r["delta_english"] + r["delta_agora"]) / 2 >= best - 0.005]
        selected = min(tied, key=complexity)["cell"]
    summary = {
        "baseline": str(baseline.relative_to(ROOT)), "primary_cell": cell_label(*PRIMARY), "selected_cell": selected,
        "selection": "max mean(delta_english, delta_agora) among cells passing the adoption rule; ties within .005 to the simpler cell",
        "pervasiveness": pervasiveness, "cells": rows,
    }
    (STEP / "grid.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# stage 4 grid: window size x pooling", "",
             f"Baseline `{summary['baseline']}`, system `{SYSTEM}`. Primary cell `{summary['primary_cell']}`. Selected cell: `{selected}`.", "",
             "| Cell | English-dev ρ | Δ English [95% CI] | Leader-term ρ | Agora ρ | Δ Agora [95% CI] | Rule |",
             "|---|---:|---|---:|---:|---|---|"]
    for r in rows:
        lines.append(
            f"| `{r['cell']}`{' (primary)' if r['primary_cell'] else ''} | {r['english_rho']:.3f} | "
            f"{r['delta_english']:+.3f} [{r['delta_english_ci95'][0]:+.3f}, {r['delta_english_ci95'][1]:+.3f}] | "
            f"{r['english_leader_term_rho']:.3f} | {r['agora_rho']:.3f} | {r['delta_agora']:+.3f} "
            f"[{r['delta_agora_ci95'][0]:+.3f}, {r['delta_agora_ci95'][1]:+.3f}] | {'pass' if r['rule_passed'] else 'fail'} |"
        )
    (STEP / "grid.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_run_manifest(STEP, script="scripts/sweep_windows_topk.py",
                       inputs={"carried": args.carried, "anchors": anchors},
                       extra={"selected": selected}, filename="grid.manifest.json")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
