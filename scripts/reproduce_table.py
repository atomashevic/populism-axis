#!/usr/bin/env python3
"""Recompute the README benchmark table from committed per-item scores (CPU, seconds).

Reads only the score tables under ``results/`` and the gold labels under ``data/``;
it loads no model and writes nothing unless ``--json`` is given. Each cell is compared
with the value reported in the README (three decimals), and the paired bootstrap
intervals are recomputed with the seeds and draw counts of the scripts that produced
them (``scripts/final_holdout.py``, ``scripts/external/score_*.py``).

The EU-OOD rows need ``data/benchmarks/eu_sentences.csv``; when it is absent (the
dataset is CC BY-NC-ND and is not redistributed), they are marked ``skipped`` and
``scripts/build_eu_sentences.py`` rebuilds it from the Hugging Face dataset.

Exit status is 1 if any recomputed cell differs from the reported value.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from evaluate_eu_paper_protocol import SURNAME, best_f1_threshold, macro_f1  # noqa: E402
from paired_comparison import paired_delta  # noqa: E402

SEED = 20260923
COLUMNS = ("axis", "combined", "roberta")
# README "Benchmarks" table: (row label, axis, axis + claim check, RoBERTa-large).
REPORTED = {
    "holdout": ("GPD English holdout, Spearman", .417, .518, .440),
    "eu_auroc": ("EU-OOD, AUROC 3-vs-0", .890, .942, .900),
    "eu_f1": ("EU-OOD, macro-F1 (leave one speaker out)", .533, .545, .555),
    "agora": ("AgoraSpeech (development set), Spearman", .359, .382, .372),
    "usgov": ("US governors 2010-18, Spearman", .458, .526, .326),
    "tweets": ("Party anti-elitism tweets, E, AUROC", .754, .756, .571),
    "popbert_elite": ("PopBERT anti-elitism, E, AUROC", .781, .806, .669),
    "popbert_centr": ("PopBERT people-centrism, P, AUROC", .754, .771, .690),
    "portugal_elite": ("Portuguese parliament anti-elitism, E, AUROC", .770, .798, .867),
    "portugal_centr": ("Portuguese parliament people-centrism, P, AUROC", .668, .695, .582),
}
EU_SPEAKERS = ["Le Pen", "Weidel", "Zemmour", "Kickl", "Tsipras"]


def read(path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / path)


def auroc_paired(a, b, y, groups, draws: int, style: str) -> dict:
    """AUROC difference a - b with the resampling scheme of the producing script."""
    rng = np.random.default_rng(SEED)
    deltas = []
    if style == "cluster_choice":  # score_antielite_tweets.py
        unique = np.unique(groups)
        index_by = {g: np.flatnonzero(groups == g) for g in unique}
    elif style == "cluster_integers":  # score_portugal.py
        codes, inverse = np.unique(groups, return_inverse=True)
        members = [np.flatnonzero(inverse == i) for i in range(len(codes))]
    for _ in range(draws):
        if style == "cluster_choice":
            idx = np.concatenate([index_by[g] for g in rng.choice(unique, size=len(unique), replace=True)])
        elif style == "cluster_integers":
            idx = np.concatenate([members[i] for i in rng.integers(0, len(codes), len(codes))])
        else:  # score_popbert_en.py: sentences resampled independently
            idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    low, high = np.percentile(deltas, [2.5, 97.5])
    return {"delta": float(roc_auc_score(y, a) - roc_auc_score(y, b)), "ci95": [float(low), float(high)]}


def compute() -> tuple[dict, dict, dict]:
    cells, paired, notes = {}, {}, {}

    # English holdout (GPD): final axis and combined scores; leader-clustered paired bootstrap.
    split = read("data/benchmarks/english_split.csv").set_index("id")["split"]
    english = read("data/benchmarks/english_gpd_speeches.csv")
    holdout = english[english["id"].map(split).eq("holdout")].reset_index(drop=True)
    axis = read("results/final/B4_axis/scores_english.csv")
    combined = read("results/final/B6_axis_plus_nli/scores_english.csv")
    roberta = read("results/final/roberta/scores_english.csv")
    assert list(axis.id) == list(holdout.id) == list(combined.id) == list(roberta.id)
    y = holdout.source_score.to_numpy(float)
    a, c = axis["PE__frozen__soft"].to_numpy(float), combined["PE__frozen__soft"].to_numpy(float)
    cells["holdout"] = {"axis": spearmanr(a, y).statistic, "combined": spearmanr(c, y).statistic,
                        "roberta": spearmanr(roberta.RoBERTa_anti_elitism_top3, y).statistic}
    paired["holdout: axis + claim check vs axis"] = paired_delta(c, a, y, holdout.leader.astype(str).to_numpy(), 4000, SEED)

    # EU-OOD: AUROC on codes 3 vs 0, and the paper's leave-one-speaker-out macro-F1.
    eu_path = ROOT / "data/benchmarks/eu_sentences.csv"
    if eu_path.exists():
        eu = pd.read_csv(eu_path)
        axis_eu = read("results/final/B4_axis/scores_eu.csv")
        combined_eu = read("results/final/B6_axis_plus_nli/scores_eu.csv")
        roberta_eu = read("results/final/roberta/scores_eu.csv")
        assert list(combined_eu.id) == list(eu.row_id) == list(roberta_eu.id)
        mask = eu.pop_code.isin([0, 3]).to_numpy()
        label = (eu.pop_code[mask] == 3).astype(int)
        cells["eu_auroc"] = {"axis": roc_auc_score(label, axis_eu["PE__frozen__soft"].to_numpy()[mask]),
                             "combined": roc_auc_score(label, combined_eu["PE__frozen__soft"].to_numpy()[mask]),
                             "roberta": roc_auc_score(label, roberta_eu.RoBERTa_populist_total.to_numpy()[mask])}
        notes["eu_auroc"] = f"n = {int(mask.sum())} sentences coded 0 or 3"
        channels = read("results/frozen_reference/scores_eu.csv")  # per-dimension scores; PE columns equal the final ones
        codes = eu.pop_code.to_numpy(int)
        truth = {"AE": np.isin(codes, [1, 3]).astype(int), "PC": np.isin(codes, [2, 3]).astype(int)}
        speaker = eu.speaker.map(SURNAME).fillna(eu.speaker).to_numpy()

        def loso(scores: dict) -> float:
            predicted = {k: np.zeros(len(eu), int) for k in truth}
            for held in EU_SPEAKERS:
                test = speaker == held
                for k in truth:
                    cut = best_f1_threshold(scores[k][~test], truth[k][~test])
                    predicted[k][test] = (scores[k][test] > cut).astype(int)
            return macro_f1(truth, predicted, np.ones(len(eu), bool))["macro"]

        cells["eu_f1"] = {
            "axis": loso({"AE": channels.E_axis__transductive.to_numpy(), "PC": channels.P_axis__transductive.to_numpy()}),
            "combined": loso({"AE": channels.E_axis_plus_nli__transductive.to_numpy(),
                              "PC": channels.P_axis_plus_nli__transductive.to_numpy()}),
            "roberta": loso({"AE": roberta_eu.RoBERTa_anti_elitism.to_numpy(), "PC": roberta_eu.RoBERTa_people_centrism.to_numpy()}),
        }
    else:
        notes["eu_auroc"] = notes["eu_f1"] = "skipped: run scripts/build_eu_sentences.py first"

    # AgoraSpeech (development set), paragraph level.
    agora = read("data/benchmarks/agora_paragraphs.csv")
    agora_scores = read("results/final/scores_agora.csv")
    assert list(agora_scores.row_id) == list(agora.row_id)
    cells["agora"] = {name: spearmanr(agora_scores[col], agora.populism_human).statistic for name, col in
                      (("axis", "axis"), ("combined", "combined"), ("roberta", "RoBERTa_populist_total"))}

    # US governors 2010-18 (speech level), group-clustered paired bootstrap.
    usgov = read("results/external/usgov/final_scores_speeches.csv")
    governors = usgov[usgov.subset == "governors_2010_2018"].reset_index(drop=True)
    cells["usgov"] = {name: spearmanr(governors[col], governors.score).statistic for name, col in
                      (("axis", "B4_axis"), ("combined", "B6_axis_plus_nli"), ("roberta", "RoBERTa_anti_elitism_top3"))}
    paired["US governors: axis + claim check vs RoBERTa"] = paired_delta(
        governors.B6_axis_plus_nli.to_numpy(), governors.RoBERTa_anti_elitism_top3.to_numpy(),
        governors.score.to_numpy(float), governors.group.astype(str).to_numpy(), 4000, SEED)

    # Party tweets, test split, E dimension; country-clustered.
    tweets = read("results/external/antielite_tweets/final_scores.csv")
    test = tweets[tweets.split == "test"].reset_index(drop=True)
    yt = test.y_general.to_numpy(int)
    cells["tweets"] = {name: roc_auc_score(yt, test[col]) for name, col in
                       (("axis", "E_axis"), ("combined", "E_axis_plus_nli"), ("roberta", "RoBERTa_anti_elitism"))}
    paired["tweets: axis + claim check vs RoBERTa"] = auroc_paired(
        test.E_axis_plus_nli.to_numpy(), test.RoBERTa_anti_elitism.to_numpy(), yt,
        test.country.astype(str).to_numpy(), 2000, "cluster_choice")

    # PopBERT test split (English translation) and Portuguese parliament, per dimension.
    popbert = read("results/external/popbert_en/final_scores.csv")
    popbert = popbert[popbert.split == "test"].reset_index(drop=True)
    portugal = read("results/external/portugal/final_scores.csv")
    for frame, name, groups, style in ((popbert, "popbert", None, "iid"),
                                       (portugal, "portugal", portugal.actor.astype(str).to_numpy(), "cluster_integers")):
        for dim, ch in (("elite", "E"), ("centr", "P")):
            yy = frame[f"{dim}_gold"].to_numpy(int)
            cells[f"{name}_{dim}"] = {col: roc_auc_score(yy, frame[key]) for col, key in
                                      (("axis", f"{ch}_axis"), ("combined", f"{ch}_axis_plus_nli"), ("roberta", f"{ch}_roberta"))}
            paired[f"{name} {dim}: axis + claim check vs RoBERTa"] = auroc_paired(
                frame[f"{ch}_axis_plus_nli"].to_numpy(), frame[f"{ch}_roberta"].to_numpy(), yy, groups, 2000, style)

    return cells, paired, notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, help="Also write the recomputed values to this JSON file")
    args = parser.parse_args()
    cells, paired, notes = compute()

    mismatches, out = 0, {"cells": {}, "paired": paired, "notes": notes}
    print("| Benchmark | Populism axis | Axis + claim check | RoBERTa-large |")
    print("|---|---:|---:|---:|")
    for row, (label, *reported) in REPORTED.items():
        shown = []
        for col, expected in zip(COLUMNS, reported):
            value = cells.get(row, {}).get(col)
            if value is None:
                shown.append(f"{expected:.3f} (skipped)")
                status = "skipped"
            else:
                status = "ok" if round(float(value), 3) == expected else "MISMATCH"
                shown.append(f"{value:.3f}" + ("" if status == "ok" else f" (reported {expected:.3f}) MISMATCH"))
            mismatches += status == "MISMATCH"
            out["cells"][f"{row}/{col}"] = {"reported": expected, "status": status, "value": None if value is None else float(value)}
        print(f"| {label} | " + " | ".join(shown) + " |")
    print()
    for name, result in paired.items():
        delta = result.get("delta", result.get("delta_spearman"))
        low, high = result["ci95"]
        print(f"paired {name}: {delta:+.3f} [{low:+.3f}, {high:+.3f}]")
    for row, note in notes.items():
        print(f"note {row}: {note}")
    counts = {s: sum(c["status"] == s for c in out["cells"].values()) for s in ("ok", "skipped", "MISMATCH")}
    print(f"\n{counts['ok']} recomputed and matching, {counts['skipped']} skipped, {counts['MISMATCH']} mismatching.")
    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + "\n")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
