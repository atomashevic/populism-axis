#!/usr/bin/env python3
"""Secondary RoBERTa cells in the external development logs, softmax vs corrected sigmoid.

Uses the same evaluation functions as scripts/external/score_*.py on the stored axis/NLI
columns; only the RoBERTa columns change. US-governor sentence rows are out of scope (their
cache was not re-encoded). Writes secondary_tables.json next to this script.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("compare_roberta", HERE / "compare_roberta.py")
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
EXT, SEED = c.EXT, c.SEED


def round_tree(value):
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {k: round_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [round_tree(v) for v in value]
    return value


def main() -> None:
    out: dict = {}
    # US governors: every speech subset with the RoBERTa anti-elitism top-3 system.
    speeches = pd.read_csv(ROOT / "data" / "benchmark_usgov" / "speeches.csv")
    stored = pd.read_csv(EXT / "usgov" / "final_scores_speeches.csv")
    for version in ("softmax", "sigmoid"):
        rob = c.pooled("usgov_speeches", version, len(speeches), 1, "k:3")
        block = {}
        for subset in list(c.usgov.SUBSETS) + ["all_pooled"]:
            mask = np.ones(len(speeches), bool) if subset == "all_pooled" else (speeches["subset"] == subset).to_numpy()
            frame = speeches[mask].reset_index(drop=True)
            metrics = c.usgov.speech_metrics(rob[mask], frame, 4000, SEED)
            paired = c.paired_delta(stored["B6_axis_plus_nli"].to_numpy(float)[mask], rob[mask], frame["score"].to_numpy(float),
                                    frame["group"].astype(str).to_numpy(), 4000, SEED)
            block[subset] = {"metrics": metrics, "B6_minus_RoBERTa": paired}
        out.setdefault("usgov", {})[version] = block

    # Tweets: both targets on the test split, the full paired list (shared RNG) and countries.
    tw = pd.read_csv(EXT / "antielite_tweets" / "final_scores.csv")
    test = (tw["split"] == "test").to_numpy(); train = ~test
    for version in ("softmax", "sigmoid"):
        rob = c.pooled("antielite_tweets", version, len(tw), 1, "k:1")
        block = {target: c.tweets.evaluate(rob[test], tw[test].reset_index(drop=True), target, rob[train], tw[train].reset_index(drop=True), 2000, SEED)
                 for target in ("y_general", "y_any")}
        systems = {col: tw[col].to_numpy(float) for col in ("E_axis", "E_axis_plus_nli", "PE_axis_plus_nli", "XLMT_released_classifier")}
        systems["RoBERTa_anti_elitism"] = rob
        y = tw.loc[test, "y_general"].to_numpy(int); groups = tw.loc[test, "country"].astype(str).to_numpy()
        rng = np.random.default_rng(SEED); unique = np.unique(groups); index_by = {g: np.flatnonzero(groups == g) for g in unique}
        pairs = [("E_axis_plus_nli", "E_axis"), ("E_axis_plus_nli", "PE_axis_plus_nli"), ("E_axis_plus_nli", "RoBERTa_anti_elitism"),
                 ("E_axis", "RoBERTa_anti_elitism"), ("E_axis_plus_nli", "XLMT_released_classifier"), ("RoBERTa_anti_elitism", "XLMT_released_classifier")]
        paired = {}
        for left, right in pairs:
            l, r = systems[left][test], systems[right][test]; samples = []
            for _ in range(2000):
                idx = np.concatenate([index_by[g] for g in rng.choice(unique, size=len(unique), replace=True)])
                if len(np.unique(y[idx])) < 2:
                    continue
                samples.append(roc_auc_score(y[idx], l[idx]) - roc_auc_score(y[idx], r[idx]))
            arr = np.asarray(samples)
            paired[f"{left} - {right}"] = {"delta": float(roc_auc_score(y, l) - roc_auc_score(y, r)),
                                           "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
                                           "p_two_sided": float(min(1.0, 2 * min((arr <= 0).mean(), (arr >= 0).mean())))}
        countries = {}
        for country in np.unique(groups):
            sel = groups == country
            if len(np.unique(y[sel])) == 2:
                countries[country] = float(roc_auc_score(y[sel], rob[test][sel]))
        out.setdefault("tweets", {})[version] = {"evaluate": block, "paired": paired, "by_country_auroc": countries}

    # PopBERT: both labelling rules, E and P RoBERTa columns.
    pb = pd.read_csv(EXT / "popbert_en" / "final_scores.csv")
    n_by_split = {s: int((pb["split"] == s).sum()) for s in ("train", "test")}
    test = (pb["split"] == "test").to_numpy(); train = ~test
    for version in ("softmax", "sigmoid"):
        block = {}
        for dim, component in (("elite", 1), ("centr", 2)):
            rob = np.concatenate([c.pooled(f"popbert_en_{s}", version, n_by_split[s], component, "k:1") for s in ("train", "test")])
            for rule, column in (("majority", f"{dim}_gold"), ("any_coder", f"{dim}_any")):
                block[f"{dim}_{rule}"] = c.popbert.evaluate(rob, pb.loc[test, column].to_numpy(int), pb.loc[train, column].to_numpy(int), test, train, 2000, SEED)
        out.setdefault("popbert", {})[version] = block

    # Portugal: anti-elitism, any populism and people-centrism columns; issue and overlap breakdowns.
    pt = pd.read_csv(EXT / "portugal" / "final_scores.csv")
    groups = pt["actor"].astype(str).to_numpy()
    for version in ("softmax", "sigmoid"):
        cols = {"E_roberta": c.pooled("portugal_en", version, len(pt), 1, "k:1"),
                "PE_roberta_total": c.pooled("portugal_en", version, len(pt), -1, "k:1"),
                "P_roberta": c.pooled("portugal_en", version, len(pt), 2, "k:1")}
        block = {"elite": {k: c.portugal.evaluate(cols[k], pt["elite_gold"].to_numpy(int), groups, 2000, SEED) for k in ("E_roberta", "PE_roberta_total")},
                 "centr": {"P_roberta": c.portugal.evaluate(cols["P_roberta"], pt["centr_gold"].to_numpy(int), groups, 2000, SEED)}}
        y = pt["elite_gold"].to_numpy(int); ours = pt["E_axis_plus_nli"].to_numpy(float); rob = cols["E_roberta"]
        issues = {}
        for issue in ("banking_finance", "law_order", "political_system", "corruption"):
            sel = pt["issue"].astype(str).eq(issue).to_numpy()
            if sel.any() and len(np.unique(y[sel])) == 2:
                issues[issue] = float(roc_auc_score(y[sel], rob[sel]))
        k = int(y.sum())
        top_ours, top_rob = set(np.argsort(-ours)[:k]), set(np.argsort(-rob, kind="stable")[:k])
        positives = set(np.flatnonzero(y == 1))
        block["breakdowns"] = {"issue_auroc": issues, "spearman_with_E_axis_plus_nli": float(spearmanr(ours, rob).statistic),
                               "positives_in_top_k": {"k": k, "ours": len(top_ours & positives), "roberta": len(top_rob & positives),
                                                      "both": len(top_ours & top_rob & positives)}}
        out.setdefault("portugal", {})[version] = block

    (HERE / "secondary_tables.json").write_text(json.dumps(round_tree(out), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote secondary_tables.json")


if __name__ == "__main__":
    main()
