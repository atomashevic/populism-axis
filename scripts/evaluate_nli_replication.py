#!/usr/bin/env python3
"""Claim-check replication (CPU): evaluate cheaper entailment models against BART-large-MNLI.

Plan and rule: research/development-log/claim-check-replication-plan.md. Everything except the
entailment model is held fixed: hypotheses, stored premises, per-window rule,
pooling, and the .5/.5 z-mean combination with the axis.

  dev    English-dev and Agora. For each candidate: NLI-only and combined scores,
         paired bootstrap of the combined score against B6 (BART combined), the
         non-inferiority rule (Δ >= -.01 on both), and low-tail calibration
         diagnostics. Writes results/nli_replication/dev_results.{json,md} and
         passing.json.
  spent  For candidates listed in passing.json (or --candidates): the primary
         system of every spent corpus, scored once beside BART, with a paired
         cluster bootstrap. Writes spent_results.{json,md}.

Corpus loading, scoring and metrics are shared with scripts/frozen_reference.py
(transductive path only).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

from populism_axis.run_manifest import write_run_manifest  # noqa: E402
from populism_axis.geometry import pole_mean_axes  # noqa: E402
from run_config import load_anchor_groups  # noqa: E402
from frozen_reference import ANCHORS, auroc, combine, corpora, document_quantities, nli_window_quantities, paired_delta, spearman  # noqa: E402

R = ROOT / "results"
OUT = R / "nli_replication"
BART = "bart-large-mnli"
CANDIDATES = {
    "DeBERTa-v3-base-mnli-fever-anli": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    "nli-deberta-v3-small": "cross-encoder/nli-deberta-v3-small",
    "nli-deberta-v3-xsmall": "cross-encoder/nli-deberta-v3-xsmall",
}
RULE_MARGIN = -0.01


def score(item: dict, label: str, axes: dict) -> tuple[dict, pd.DataFrame, dict]:
    directory, name = item["nli"]
    requests, window_nli = nli_window_quantities(directory, name, label)
    docs, _ = document_quantities(item["matrix"], item["doc_index"], item["n"], item["rule"], requests, window_nli, axes, None)
    return combine(docs, item["rule"], None), requests, window_nli


def tail_diagnostics(window_nli: dict, bart_window: dict, requests: pd.DataFrame) -> dict:
    out = {}
    for c in ("P", "E"):
        v, b = window_nli[f"{c}_nli"], bart_window[f"{c}_nli"]
        out[c] = {"median": float(np.median(v)), "p99": float(np.percentile(v, 99)), "share_gt_0.5": float((v > 0.5).mean()),
                  "window_spearman_vs_bart": float(spearmanr(v, b).statistic)}
    out["PE_window_spearman_vs_bart"] = float(spearmanr(window_nli["PE_nli"], bart_window["PE_nli"]).statistic)
    return out


def stage_dev(draws: int, seed: int) -> None:
    axes = pole_mean_axes(load_anchor_groups(ANCHORS))
    items = {i["name"]: i for i in corpora() if i["name"] in ("english_dev", "agora")}
    results: dict = {"rule": f"paired delta of combined vs B6 >= {RULE_MARGIN} on English-dev and Agora", "draws": draws, "seed": seed, "candidates": {}}
    bart = {name: score(item, BART, axes) for name, item in items.items()}
    for label, model in CANDIDATES.items():
        entry: dict = {"model": model, "corpora": {}}
        passes = True
        for name, item in items.items():
            cand, requests, window_nli = score(item, label, axes)
            ref_scores, _, ref_window = bart[name]
            ev = item["evals"][0]
            t, g = ev["target"], ev["groups"]
            paired = paired_delta(cand["PE_axis_plus_nli"], ref_scores["PE_axis_plus_nli"], t, g, spearman, draws, seed)
            paired_nli = paired_delta(cand["PE_nli"], ref_scores["PE_nli"], t, g, spearman, draws, seed)
            entry["corpora"][name] = {
                "combined_spearman": spearman(t, cand["PE_axis_plus_nli"]), "bart_combined_spearman": spearman(t, ref_scores["PE_axis_plus_nli"]),
                "nli_only_spearman": spearman(t, cand["PE_nli"]), "bart_nli_only_spearman": spearman(t, ref_scores["PE_nli"]),
                "axis_only_spearman": spearman(t, cand["PE_axis"]),
                "paired_combined_vs_b6": paired, "paired_nli_only_vs_bart_nli": paired_nli,
                "doc_spearman_nli_vs_bart": float(spearmanr(cand["PE_nli"], ref_scores["PE_nli"]).statistic),
                "tail": tail_diagnostics(window_nli, ref_window, requests),
            }
            passes &= paired["delta"] >= RULE_MARGIN
            pd.DataFrame({"doc_index": np.arange(item["n"]), "PE_nli": cand["PE_nli"], "PE_axis_plus_nli": cand["PE_axis_plus_nli"]}).to_csv(
                OUT / f"scores_{name}__{label}.csv", index=False)
            print(f"{label:34s} {name:12s} combined {entry['corpora'][name]['combined_spearman']:.3f} vs B6 {entry['corpora'][name]['bart_combined_spearman']:.3f} "
                  f"Δ {paired['delta']:+.3f} [{paired['ci95'][0]:+.3f}, {paired['ci95'][1]:+.3f}]  nli-only {entry['corpora'][name]['nli_only_spearman']:.3f}", flush=True)
        entry["non_inferior"] = bool(passes)
        results["candidates"][label] = entry
    passing = [label for label, e in results["candidates"].items() if e["non_inferior"]]
    results["passing"] = passing
    (OUT / "dev_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (OUT / "passing.json").write_text(json.dumps({"passing": passing, "rule": results["rule"]}, indent=2) + "\n", encoding="utf-8")

    lines = ["| Candidate | English-dev combined | Δ vs B6 [95% CI] | Agora combined | Δ vs B6 [95% CI] | English-dev NLI only | Agora NLI only | Non-inferior |",
             "|---|---:|---|---:|---|---:|---:|---|"]
    b = {n: bart[n][0] for n in items}
    lines.insert(0, f"B6 reference (BART-large-MNLI): English-dev combined {spearman(items['english_dev']['evals'][0]['target'], b['english_dev']['PE_axis_plus_nli']):.3f}, "
                    f"NLI only {spearman(items['english_dev']['evals'][0]['target'], b['english_dev']['PE_nli']):.3f}; Agora combined "
                    f"{spearman(items['agora']['evals'][0]['target'], b['agora']['PE_axis_plus_nli']):.3f}, NLI only {spearman(items['agora']['evals'][0]['target'], b['agora']['PE_nli']):.3f}.\n")
    for label, e in results["candidates"].items():
        en, ag = e["corpora"]["english_dev"], e["corpora"]["agora"]
        pe, pa = en["paired_combined_vs_b6"], ag["paired_combined_vs_b6"]
        lines.append(f"| {label} | {en['combined_spearman']:.3f} | {pe['delta']:+.3f} [{pe['ci95'][0]:+.3f}, {pe['ci95'][1]:+.3f}] | {ag['combined_spearman']:.3f} | "
                     f"{pa['delta']:+.3f} [{pa['ci95'][0]:+.3f}, {pa['ci95'][1]:+.3f}] | {en['nli_only_spearman']:.3f} | {ag['nli_only_spearman']:.3f} | {'yes' if e['non_inferior'] else 'no'} |")
    lines += ["", "Low-tail calibration on English-dev windows (P channel / E channel): median entailment, 99th percentile, share above .5, window-level Spearman with BART.", "",
              "| Candidate | P median | P p99 | P > .5 | P ρ vs BART | E median | E p99 | E > .5 | E ρ vs BART | PE window ρ vs BART | doc NLI ρ vs BART (Eng / Agora) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for label, e in results["candidates"].items():
        t = e["corpora"]["english_dev"]["tail"]
        lines.append(f"| {label} | {t['P']['median']:.4f} | {t['P']['p99']:.3f} | {t['P']['share_gt_0.5']:.4f} | {t['P']['window_spearman_vs_bart']:.3f} | "
                     f"{t['E']['median']:.4f} | {t['E']['p99']:.3f} | {t['E']['share_gt_0.5']:.4f} | {t['E']['window_spearman_vs_bart']:.3f} | {t['PE_window_spearman_vs_bart']:.3f} | "
                     f"{e['corpora']['english_dev']['doc_spearman_nli_vs_bart']:.3f} / {e['corpora']['agora']['doc_spearman_nli_vs_bart']:.3f} |")
    (OUT / "dev_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_run_manifest(OUT, script="scripts/evaluate_nli_replication.py", inputs={"hypotheses": R / "step6_nli" / "hypotheses.json"},
                       extra={"stage": "dev", "candidates": CANDIDATES, "passing": passing}, filename="dev.manifest.json")
    print("\n".join(lines))
    print("passing:", passing)


def stage_spent(labels: list[str], draws: int, seed: int) -> None:
    axes = pole_mean_axes(load_anchor_groups(ANCHORS))
    items = [i for i in corpora() if i["name"] not in ("english_dev", "agora")]
    results: dict = {"draws": draws, "seed": seed, "candidates": labels, "rows": []}
    for item in items:
        bart_scores, _, bart_window = score(item, BART, axes)
        cand_scores = {label: score(item, label, axes) for label in labels}
        for ev in item["evals"]:
            s = ev["system"]
            if not s.endswith("_plus_nli"):
                continue
            nli_system = s.replace("_axis_plus_nli", "_nli")
            metric = spearman if ev["metric"] == "spearman" else auroc
            mask = ev.get("mask", np.ones(item["n"], bool))
            t, g = ev["target"][mask], ev["groups"][mask]
            row = {"corpus": item["name"], "system": s, "label": ev.get("label", ""), "metric": ev["metric"], "n": int(mask.sum()),
                   "bart": metric(t, bart_scores[s][mask]), "bart_nli_only": metric(t, bart_scores[nli_system][mask]), "candidates": {}}
            for label, (cs, _, cw) in cand_scores.items():
                row["candidates"][label] = {"combined": metric(t, cs[s][mask]), "nli_only": metric(t, cs[nli_system][mask]),
                                            "paired_vs_bart": paired_delta(cs[s][mask], bart_scores[s][mask], t, g, metric, draws, seed),
                                            "doc_nli_spearman_vs_bart": float(spearmanr(cs[nli_system][mask], bart_scores[nli_system][mask]).statistic)}
            results["rows"].append(row)
            print(item["name"], s, row["label"], {k: round(v["combined"], 3) for k, v in row["candidates"].items()}, "bart", round(row["bart"], 3), flush=True)
    (OUT / "spent_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    head = "| Corpus | System | Target | Metric | n | BART combined | " + " | ".join(f"{l} combined, Δ vs BART [95% CI]" for l in labels) + " | BART NLI only | " + " | ".join(f"{l} NLI only" for l in labels) + " |"
    lines = [head, "|---|---|---|---|---:|---:|" + "---|" * len(labels) + "---:|" + "---:|" * len(labels)]
    for r in results["rows"]:
        cells = [f"{r['candidates'][l]['combined']:.3f}, {r['candidates'][l]['paired_vs_bart']['delta']:+.3f} [{r['candidates'][l]['paired_vs_bart']['ci95'][0]:+.3f}, {r['candidates'][l]['paired_vs_bart']['ci95'][1]:+.3f}]" for l in labels]
        nli = [f"{r['candidates'][l]['nli_only']:.3f}" for l in labels]
        lines.append(f"| {r['corpus']} | {r['system']} | {r['label']} | {r['metric']} | {r['n']} | {r['bart']:.3f} | " + " | ".join(cells) + f" | {r['bart_nli_only']:.3f} | " + " | ".join(nli) + " |")
    (OUT / "spent_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_run_manifest(OUT, script="scripts/evaluate_nli_replication.py", inputs={"hypotheses": R / "step6_nli" / "hypotheses.json"},
                       extra={"stage": "spent", "candidates": labels}, filename="spent.manifest.json")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("dev", "spent"))
    parser.add_argument("--candidates", default="", help="spent: comma-separated labels (default: passing.json)")
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "dev":
        stage_dev(args.draws, args.seed)
    else:
        labels = [l for l in args.candidates.split(",") if l] or json.loads((OUT / "passing.json").read_text())["passing"]
        if not labels:
            raise SystemExit("no passing candidates; nothing to score on spent sets")
        stage_spent(labels, args.draws, args.seed)


if __name__ == "__main__":
    main()
