#!/usr/bin/env python3
"""Score the frozen configurations on the party anti-elitism tweets benchmark (CPU).

Stage `axis`: pole-mean axes from the frozen e5-large-v2 anchor embeddings, projected on
the tweet embeddings (each tweet is one document; windows pooled with the max, i.e. k:1).
Writes per-tweet axis scores and NLI requests (top window per tweet).
Stage `final`: adds NLI, combined, RoBERTa and XLM-T comparators; evaluates on the
paper's fixed test split (n=1,059) and on all 5,401 tweets.

Pre-declared systems. The benchmark labels general elite criticism, so the E channel is
the primary score; the carried PE system is reported as secondary.
  E_axis           E-channel projection (z within dataset)
  PE_axis          carried window score (PE soft conjunction)
  E_nli            max entailment of the five E anchor claims
  E_axis_plus_nli  mean of z(E_axis) and z(E_nli)         <- primary axis + claim check
  PE_axis_plus_nli mean of z(PE_axis) and z(PE nli)       <- carried configuration
Thresholds for F1 use only the training split (prevalence-matched, and F1-optimal on
train), so the test metrics are comparable to a classifier that saw the training labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

from populism_axis.contextual_axis import CHANNELS, WindowProjections, aggregate_from_window_scores, window_conjunction_scores  # noqa: E402
from populism_axis.run_manifest import write_run_manifest  # noqa: E402
from populism_axis.roberta_comparator import load_scores  # noqa: E402
from populism_axis.geometry import center_normalize, pole_mean_axes  # noqa: E402
from evaluate_benchmarks import _auroc_metric, cluster_bootstrap  # noqa: E402
from run_config import load_anchor_groups, load_embeddings  # noqa: E402
from evaluate_nli import zscore  # noqa: E402

LABELS = ROOT / "data" / "benchmark_antielite_tweets" / "tweets_labels.csv"
OUT = ROOT / "results" / "external" / "antielite_tweets"
EMB = ROOT / "results" / "embeddings_external" / "antielite_tweets" / "e5-large-v2" / "spw3_cap60"
ANCHORS = ROOT / "results" / "embeddings" / "e5-large-v2" / "sentences" / "v3_proposition__ea98e5b2.npz"
NAME = "antielite_tweets"


def stage_axis() -> None:
    frame = pd.read_csv(LABELS)
    axes = pole_mean_axes(load_anchor_groups(ANCHORS))
    matrix, doc_index, windows = load_embeddings(EMB, NAME)
    if len(frame) != int(doc_index.max()) + 1:
        raise SystemExit("embeddings and labels disagree on the number of tweets")
    normalized, _ = center_normalize(matrix)
    projections = WindowProjections(doc_index=doc_index, windows=windows["window"].tolist(),
                                    projections={c: normalized @ axes[c] for c in CHANNELS})
    pe = window_conjunction_scores(projections, {"P": 1.0, "E": 1.0, "M": 0.0}, conjunction="soft")
    n = len(frame)
    out = frame[["row_id", "id", "split", "labeling", "y_general", "y_any"]].copy()
    out["PE_axis"] = aggregate_from_window_scores(pe, doc_index, n, "k:1")
    for c in ("P", "E"):
        z = (projections.projections[c] - projections.projections[c].mean()) / projections.projections[c].std()
        out[f"{c}_axis"] = aggregate_from_window_scores(z, doc_index, n, "k:1")
    out.to_csv(OUT / "axis_scores.csv", index=False)
    # NLI requests: the top PE window of each tweet (almost always the whole tweet).
    order = np.lexsort((-pe, doc_index)); sorted_docs = doc_index[order]
    rows = []
    for doc in range(n):
        position = order[np.searchsorted(sorted_docs, doc)]
        rows.append({"request_id": doc, "doc_index": doc, "doc_id": frame.loc[doc, "row_id"], "window_rank": 0,
                     "window_position": int(position), "window_score": float(pe[position]), "premise": windows.loc[position, "window"]})
    pd.DataFrame(rows).to_parquet(OUT / f"requests_{NAME}.parquet", index=False)
    write_run_manifest(OUT, script="scripts/external/score_antielite_tweets.py", inputs={"anchors": ANCHORS, "labels": LABELS},
                       extra={"stage": "axis", "windows": int(len(pe))}, filename="axis.manifest.json")
    print("tweets", n, "windows", len(pe))


def nli_channel_scores(model_label: str, n: int) -> dict[str, np.ndarray]:
    hypotheses = json.loads((OUT / "hypotheses.json").read_text(encoding="utf-8"))
    requests = pd.read_parquet(OUT / f"requests_{NAME}.parquet")
    payload = np.load(OUT / f"nli_{NAME}__{model_label}.npz")
    if not np.array_equal(payload["request_id"], requests["request_id"].to_numpy()):
        raise SystemExit("NLI output does not match requests")
    entail = payload["probs"][:, :, 2].astype(np.float64)
    cols = {c: [i for i, h in enumerate(hypotheses) if h["channel"] == c] for c in ("P", "E")}
    p_max, e_max = entail[:, cols["P"]].max(axis=1), entail[:, cols["E"]].max(axis=1)
    out = {"P_nli": np.zeros(n), "E_nli": np.zeros(n), "PE_nli": np.zeros(n)}
    idx = requests["doc_index"].to_numpy()
    out["P_nli"][idx], out["E_nli"][idx], out["PE_nli"][idx] = p_max, e_max, np.sqrt(p_max * e_max)
    return out


def threshold_metrics(values: np.ndarray, y: np.ndarray, train_values: np.ndarray, train_y: np.ndarray) -> dict:
    out = {}
    # prevalence-matched on train (label count only), F1-optimal on train, oracle on the evaluated split
    rate = float(train_y.mean())
    cuts = {"prevalence_train": float(np.quantile(values, 1.0 - rate))}
    grid = np.unique(np.quantile(train_values, np.linspace(0.3, 0.995, 250)))
    cuts["f1opt_train"] = float(max(grid, key=lambda c: f1_score(train_y, (train_values >= c).astype(int), zero_division=0)))
    grid = np.unique(np.quantile(values, np.linspace(0.3, 0.995, 250)))
    cuts["oracle"] = float(max(grid, key=lambda c: f1_score(y, (values >= c).astype(int), zero_division=0)))
    for name, cut in cuts.items():
        pred = (values >= cut).astype(int)
        out[name] = {"f1_pos": float(f1_score(y, pred, zero_division=0)), "precision_pos": float(precision_score(y, pred, zero_division=0)),
                     "recall_pos": float(recall_score(y, pred, zero_division=0)),
                     "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)), "accuracy": float((pred == y).mean())}
    return out


def evaluate(values: np.ndarray, frame: pd.DataFrame, target: str, train_values: np.ndarray, train_frame: pd.DataFrame, draws: int, seed: int) -> dict:
    y = frame[target].to_numpy(dtype=int)
    groups = frame["country"].astype(str).to_numpy()
    low, high = cluster_bootstrap(_auroc_metric, values, y, groups, draws, seed)
    return {"n": int(len(y)), "n_positive": int(y.sum()), "auroc": float(roc_auc_score(y, values)), "auroc_ci95_country": [low, high],
            "average_precision": float(average_precision_score(y, values)),
            "thresholds": threshold_metrics(values, y, train_values, train_frame[target].to_numpy(dtype=int))}


def stage_final(model_label: str, draws: int, seed: int) -> None:
    frame = pd.read_csv(LABELS)
    axis = pd.read_csv(OUT / "axis_scores.csv")
    if list(axis["row_id"]) != list(frame["row_id"]):
        raise SystemExit("axis scores and labels disagree on order")
    n = len(frame)
    nli = nli_channel_scores(model_label, n)
    rob = load_scores(OUT / "roberta" / f"sigmoid_{NAME}.npz")
    rob_ae = aggregate_from_window_scores(rob["probabilities"][:, 1].astype(float), rob["doc_index"].astype(int), n, "k:1")
    rob_total = aggregate_from_window_scores(1.0 - rob["probabilities"][:, 0].astype(float), rob["doc_index"].astype(int), n, "k:1")
    systems = {
        "E_axis": axis["E_axis"].to_numpy(dtype=float),
        "E_axis_plus_nli": (zscore(axis["E_axis"].to_numpy(dtype=float)) + zscore(nli["E_nli"])) / 2,
        "E_nli": nli["E_nli"],
        "PE_axis": axis["PE_axis"].to_numpy(dtype=float),
        "PE_axis_plus_nli": (zscore(axis["PE_axis"].to_numpy(dtype=float)) + zscore(nli["PE_nli"])) / 2,
        "P_axis": axis["P_axis"].to_numpy(dtype=float),
        "RoBERTa_anti_elitism": rob_ae,
        "RoBERTa_populist_total": rob_total,
    }
    xlmt_path = OUT / "xlmt" / "xlmt_probabilities.csv"
    xlmt_check = None
    if xlmt_path.exists():
        xl = pd.read_csv(xlmt_path)  # same row order as the text file the classifier read (ids repeat for 48 duplicate rows)
        if len(xl) != n or list(xl["id"]) != list(frame["id"]):
            raise SystemExit("XLM-T output does not align with the label file")
        systems["XLMT_released_classifier"] = xl["prob_general"].to_numpy(dtype=float)
        pred = (systems["XLMT_released_classifier"] >= 0.5).astype(int)
        yt = frame["y_general"].to_numpy(dtype=int); te = (frame["split"] == "test").to_numpy()
        xlmt_check = {"threshold": 0.5, "f1_pos_test": float(f1_score(yt[te], pred[te])), "f1_macro_test": float(f1_score(yt[te], pred[te], average="macro")),
                      "accuracy_test": float((pred[te] == yt[te]).mean()), "published_f1_pos": 0.611, "published_f1_macro": 0.745, "published_accuracy": 0.815}
    scores = frame[["row_id", "id", "country", "party", "split", "labeling", "y_general", "y_any"]].copy()
    for name, values in systems.items():
        scores[name] = values
    scores.to_csv(OUT / "final_scores.csv", index=False)

    test = (frame["split"] == "test").to_numpy(); train = ~test
    results: dict = {"nli_model": model_label, "test": {}, "all": {}, "paired_auroc_test": {}}
    for target in ("y_general", "y_any"):
        results["test"][target] = {name: evaluate(v[test], frame[test].reset_index(drop=True), target, v[train], frame[train].reset_index(drop=True), draws, seed)
                                   for name, v in systems.items() if not np.isnan(v[test]).any()}
        results["all"][target] = {name: evaluate(v, frame, target, v[train], frame[train].reset_index(drop=True), draws, seed)
                                  for name, v in systems.items() if not np.isnan(v).any()}
    # paired AUROC differences on the test split (country-clustered bootstrap)
    y = frame.loc[test, "y_general"].to_numpy(dtype=int); groups = frame.loc[test, "country"].astype(str).to_numpy()
    rng = np.random.default_rng(seed); unique = np.unique(groups); index_by = {g: np.flatnonzero(groups == g) for g in unique}
    pairs = [("E_axis_plus_nli", "E_axis"), ("E_axis_plus_nli", "PE_axis_plus_nli"), ("E_axis_plus_nli", "RoBERTa_anti_elitism"), ("E_axis", "RoBERTa_anti_elitism")]
    if "XLMT_released_classifier" in systems:
        pairs += [("E_axis_plus_nli", "XLMT_released_classifier"), ("RoBERTa_anti_elitism", "XLMT_released_classifier")]
    for left, right in pairs:
        l, r = systems[left][test], systems[right][test]
        observed = roc_auc_score(y, l) - roc_auc_score(y, r); samples = []
        for _ in range(draws):
            idx = np.concatenate([index_by[g] for g in rng.choice(unique, size=len(unique), replace=True)])
            if len(np.unique(y[idx])) < 2:
                continue
            samples.append(roc_auc_score(y[idx], l[idx]) - roc_auc_score(y[idx], r[idx]))
        arr = np.asarray(samples)
        results["paired_auroc_test"][f"{left} - {right}"] = {"delta_auroc": float(observed), "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
                                                              "p_two_sided": float(min(1.0, 2 * min((arr <= 0).mean(), (arr >= 0).mean()))), "n_draws": int(len(arr))}
    freeze = json.loads((OUT / "benchmark_freeze.json").read_text(encoding="utf-8"))
    results["published"] = freeze["published_results"]
    results["xlmt_reproduction_at_0.5"] = xlmt_check
    (OUT / "final_results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_run_manifest(OUT, script="scripts/external/score_antielite_tweets.py",
                       inputs={"axis": OUT / "axis_scores.csv", "nli": OUT / f"nli_{NAME}__{model_label}.npz", "labels": LABELS},
                       extra={"stage": "final", "xlmt_included": xlmt_path.exists()}, filename="final.manifest.json")

    lines = []
    for split in ("test", "all"):
        for target in ("y_general", "y_any"):
            block = results[split][target]; any_row = next(iter(block.values()))
            lines += [f"### {split} split, target {target} (n={any_row['n']}, positives={any_row['n_positive']})", "",
                      "| System | AUROC [95% CI, country-clustered] | AP | F1 pos / macro @ train-prevalence cut | F1 pos / macro @ train-F1-optimal cut | F1 pos / macro @ oracle cut |",
                      "|---|---|---:|---|---|---|"]
            for name, r in block.items():
                t = r["thresholds"]
                lines.append(f"| {name} | {r['auroc']:.3f} [{r['auroc_ci95_country'][0]:.3f}, {r['auroc_ci95_country'][1]:.3f}] | {r['average_precision']:.3f} | "
                             f"{t['prevalence_train']['f1_pos']:.3f} / {t['prevalence_train']['f1_macro']:.3f} | {t['f1opt_train']['f1_pos']:.3f} / {t['f1opt_train']['f1_macro']:.3f} | "
                             f"{t['oracle']['f1_pos']:.3f} / {t['oracle']['f1_macro']:.3f} |")
            lines.append("")
    lines += ["### Paired AUROC differences, test split, target y_general", "", "| Comparison | Δ AUROC | 95% CI | p |", "|---|---:|---|---:|"]
    for key, p in results["paired_auroc_test"].items():
        lines.append(f"| {key} | {p['delta_auroc']:+.3f} | [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] | {p['p_two_sided']:.3f} |")
    (OUT / "final_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("axis", "final"))
    parser.add_argument("--nli-model-label", default="bart-large-mnli")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stage_axis() if args.stage == "axis" else stage_final(args.nli_model_label, args.draws, args.seed)


if __name__ == "__main__":
    main()
