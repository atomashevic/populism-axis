#!/usr/bin/env python3
"""Score the frozen configurations on PopBERT sentences in English translation (CPU).

Stage `axis`: pole-mean axes from the frozen e5-large-v2 anchor embeddings, projected on
the translated sentences of the frozen PopBERT split (train 7,036 / test 1,759). Each
sentence is one document (max over its windows). Centering and z-scoring pool train and
test, as the earlier multilingual PopBERT run did, so thresholds fitted on train transfer to test.
Writes per-sentence axis scores and NLI requests (top window per sentence).
Stage `final`: adds NLI, combined, and the RoBERTa comparator; evaluates on the test
split against two label rules:
  majority   >= 50% of the five coders (PopBERT's gold; the frozen benchmark)
  any_coder  at least one coder (the rule used by Wang et al. 2026 for their RoBERTa)
Channel mapping is fixed in advance: E -> anti-elitism (elite), P -> people-centrism (centr).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

from populism_axis.contextual_axis import CHANNELS, WindowProjections, aggregate_from_window_scores, window_conjunction_scores  # noqa: E402
from populism_axis.run_manifest import write_run_manifest  # noqa: E402
from populism_axis.roberta_comparator import load_scores  # noqa: E402
from populism_axis.geometry import center_normalize, pole_mean_axes  # noqa: E402
from run_config import load_anchor_groups, load_embeddings  # noqa: E402
from evaluate_nli import zscore  # noqa: E402

BENCH = ROOT / "data" / "benchmark_popbert"
OUT = ROOT / "results" / "external" / "popbert_en"
EMB = ROOT / "results" / "embeddings_external" / "popbert_en" / "e5-large-v2" / "spw3_cap60"
ANCHORS = ROOT / "results" / "embeddings" / "e5-large-v2" / "sentences" / "v3_proposition__ea98e5b2.npz"
DIMENSIONS = {"elite": "E", "centr": "P"}


def load_split_frames() -> pd.DataFrame:
    parts = []
    for split in ("train", "test"):
        frame = pd.read_csv(BENCH / f"{split}_en.csv"); frame["split"] = split; parts.append(frame)
    frame = pd.concat(parts, ignore_index=True)
    for dim in DIMENSIONS:
        frame[f"{dim}_any"] = (frame[f"{dim}_vote_share"] > 0).astype(int)
    return frame


def stage_axis() -> None:
    frame = load_split_frames()
    axes = pole_mean_axes(load_anchor_groups(ANCHORS))
    matrices, docs, windows_all, offset = [], [], [], 0
    for split in ("train", "test"):
        m, d, w = load_embeddings(EMB, f"popbert_en_{split}")
        n_split = int((frame["split"] == split).sum())
        if int(d.max()) + 1 != n_split:
            raise SystemExit(f"{split}: embeddings do not match the frame")
        matrices.append(m); docs.append(d + offset); windows_all.append(w); offset += n_split
    matrix, doc_index, windows = np.vstack(matrices), np.concatenate(docs), pd.concat(windows_all, ignore_index=True)
    normalized, _ = center_normalize(matrix)
    projections = WindowProjections(doc_index=doc_index, windows=windows["window"].tolist(), projections={c: normalized @ axes[c] for c in CHANNELS})
    pe = window_conjunction_scores(projections, {"P": 1.0, "E": 1.0, "M": 0.0}, conjunction="soft")
    n = len(frame)
    out = frame[["id", "split", "elite_gold", "centr_gold", "elite_any", "centr_any", "elite_vote_share", "centr_vote_share"]].copy()
    out["PE_axis"] = aggregate_from_window_scores(pe, doc_index, n, "k:1")
    for c in ("P", "E"):
        z = (projections.projections[c] - projections.projections[c].mean()) / projections.projections[c].std()
        out[f"{c}_axis"] = aggregate_from_window_scores(z, doc_index, n, "k:1")
    out.to_csv(OUT / "axis_scores.csv", index=False)
    order = np.lexsort((-pe, doc_index)); sorted_docs = doc_index[order]
    rows = []
    for doc in range(n):
        position = order[np.searchsorted(sorted_docs, doc)]
        rows.append({"request_id": doc, "doc_index": doc, "doc_id": str(frame.loc[doc, "id"]), "window_rank": 0,
                     "window_position": int(position), "window_score": float(pe[position]), "premise": windows.loc[position, "window"]})
    pd.DataFrame(rows).to_parquet(OUT / "requests_popbert_en.parquet", index=False)
    write_run_manifest(OUT, script="scripts/external/score_popbert_en.py", inputs={"anchors": ANCHORS, "test": BENCH / "test_en.csv", "train": BENCH / "train_en.csv"},
                       extra={"stage": "axis", "windows": int(len(pe))}, filename="axis.manifest.json")
    print("sentences", n, "windows", len(pe))


def nli_scores(model_label: str, n: int) -> dict[str, np.ndarray]:
    hypotheses = json.loads((OUT / "hypotheses.json").read_text(encoding="utf-8"))
    requests = pd.read_parquet(OUT / "requests_popbert_en.parquet")
    payload = np.load(OUT / f"nli_popbert_en__{model_label}.npz")
    if not np.array_equal(payload["request_id"], requests["request_id"].to_numpy()):
        raise SystemExit("NLI output does not match requests")
    entail = payload["probs"][:, :, 2].astype(np.float64)
    cols = {c: [i for i, h in enumerate(hypotheses) if h["channel"] == c] for c in ("P", "E")}
    idx = requests["doc_index"].to_numpy()
    out = {"P_nli": np.zeros(n), "E_nli": np.zeros(n)}
    out["P_nli"][idx] = entail[:, cols["P"]].max(axis=1); out["E_nli"][idx] = entail[:, cols["E"]].max(axis=1)
    out["PE_nli"] = np.sqrt(out["P_nli"] * out["E_nli"])
    return out


def roberta_scores(n_by_split: dict[str, int], component: int) -> np.ndarray:
    parts = []
    for split in ("train", "test"):
        payload = load_scores(OUT / "roberta" / f"sigmoid_popbert_en_{split}.npz")
        values = payload["probabilities"][:, component].astype(float) if component >= 0 else 1.0 - payload["probabilities"][:, 0].astype(float)
        parts.append(aggregate_from_window_scores(values, payload["doc_index"].astype(int), n_by_split[split], "k:1"))
    return np.concatenate(parts)


def f1_at(values: np.ndarray, y: np.ndarray, cut: float) -> dict:
    pred = (values >= cut).astype(int)
    return {"f1": float(f1_score(y, pred, zero_division=0)), "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)), "threshold": float(cut)}


def best_f1_cut(values: np.ndarray, y: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, values)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def evaluate(values: np.ndarray, y_test: np.ndarray, y_train: np.ndarray, test: np.ndarray, train: np.ndarray, draws: int, seed: int) -> dict:
    vt, vtr = values[test], values[train]
    rng = np.random.default_rng(seed); samples = []
    for _ in range(draws):
        idx = rng.integers(0, len(vt), len(vt))
        if len(np.unique(y_test[idx])) == 2:
            samples.append(roc_auc_score(y_test[idx], vt[idx]))
    out = {"n": int(len(vt)), "n_positive": int(y_test.sum()), "auroc": float(roc_auc_score(y_test, vt)),
           "auroc_ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
           "average_precision": float(average_precision_score(y_test, vt)),
           "f1_train_prevalence": f1_at(vt, y_test, float(np.quantile(vt, 1.0 - y_train.mean()))),
           "f1_train_optimal": f1_at(vt, y_test, best_f1_cut(vtr, y_train)),
           "f1_test_oracle": f1_at(vt, y_test, best_f1_cut(vt, y_test))}
    return out


def stage_final(model_label: str, draws: int, seed: int) -> None:
    frame = load_split_frames()
    axis = pd.read_csv(OUT / "axis_scores.csv")
    if list(axis["id"]) != list(frame["id"]):
        raise SystemExit("axis scores and frame disagree on order")
    n = len(frame); n_by_split = {s: int((frame["split"] == s).sum()) for s in ("train", "test")}
    nli = nli_scores(model_label, n)
    channel_scores = {
        "E": {"axis": axis["E_axis"].to_numpy(float), "axis_plus_nli": (zscore(axis["E_axis"].to_numpy(float)) + zscore(nli["E_nli"])) / 2,
              "nli": nli["E_nli"], "roberta": roberta_scores(n_by_split, 1)},
        "P": {"axis": axis["P_axis"].to_numpy(float), "axis_plus_nli": (zscore(axis["P_axis"].to_numpy(float)) + zscore(nli["P_nli"])) / 2,
              "nli": nli["P_nli"], "roberta": roberta_scores(n_by_split, 2)},
    }
    shared = {"PE_axis": axis["PE_axis"].to_numpy(float), "PE_axis_plus_nli": (zscore(axis["PE_axis"].to_numpy(float)) + zscore(nli["PE_nli"])) / 2,
              "roberta_populist_total": roberta_scores(n_by_split, -1)}
    scores = frame[["id", "split", "elite_gold", "centr_gold", "elite_any", "centr_any"]].copy()
    for c, systems in channel_scores.items():
        for name, v in systems.items():
            scores[f"{c}_{name}"] = v
    for name, v in shared.items():
        scores[name] = v
    scores.to_csv(OUT / "final_scores.csv", index=False)

    test = (frame["split"] == "test").to_numpy(); train = ~test
    results: dict = {"nli_model": model_label, "test": {}}
    for dim, channel in DIMENSIONS.items():
        results["test"][dim] = {}
        for rule, column in (("majority", f"{dim}_gold"), ("any_coder", f"{dim}_any")):
            y_test, y_train = frame.loc[test, column].to_numpy(int), frame.loc[train, column].to_numpy(int)
            block = {}
            for name, v in channel_scores[channel].items():
                block[f"{channel}_{name}"] = evaluate(v, y_test, y_train, test, train, draws, seed)
            for name, v in shared.items():
                block[name] = evaluate(v, y_test, y_train, test, train, draws, seed)
            results["test"][dim][rule] = block
    # Paired AUROC differences on the test split (sentence bootstrap; no speaker ids in the frozen split).
    results["paired_auroc_test"] = {}
    for dim, channel in DIMENSIONS.items():
        y = frame.loc[test, f"{dim}_gold"].to_numpy(int)
        systems = {f"{channel}_{name}": v for name, v in channel_scores[channel].items()}
        pairs = [(f"{channel}_axis_plus_nli", f"{channel}_axis"), (f"{channel}_nli", f"{channel}_axis_plus_nli"),
                 (f"{channel}_axis_plus_nli", f"{channel}_roberta"), (f"{channel}_axis_plus_nli", "PE_axis_plus_nli")]
        for a, b in pairs:
            va = (systems | shared)[a][test]; vb = (systems | shared)[b][test]
            rng = np.random.default_rng(seed); deltas = []
            for _ in range(draws):
                idx = rng.integers(0, len(y), len(y))
                if len(np.unique(y[idx])) == 2:
                    deltas.append(roc_auc_score(y[idx], va[idx]) - roc_auc_score(y[idx], vb[idx]))
            deltas = np.asarray(deltas); point = float(roc_auc_score(y, va) - roc_auc_score(y, vb))
            results["paired_auroc_test"][f"{dim}: {a} - {b}"] = {
                "delta": point, "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
                "p_two_sided": float(min(1.0, 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())))}
    freeze = json.loads((OUT / "benchmark_freeze.json").read_text(encoding="utf-8"))
    results["published"] = freeze["published"]
    (OUT / "final_results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_run_manifest(OUT, script="scripts/external/score_popbert_en.py", inputs={"axis": OUT / "axis_scores.csv", "nli": OUT / f"nli_popbert_en__{model_label}.npz"},
                       extra={"stage": "final"}, filename="final.manifest.json")
    lines = []
    for dim in DIMENSIONS:
        for rule in ("majority", "any_coder"):
            block = results["test"][dim][rule]; any_row = next(iter(block.values()))
            lines += [f"### {dim}, {rule} labels (test n={any_row['n']}, positives={any_row['n_positive']})", "",
                      "| System | AUROC [95% CI] | AP | F1 @ train-prevalence cut | F1 @ train-optimal cut | F1 @ test-oracle cut |", "|---|---|---:|---:|---:|---:|"]
            for name, r in block.items():
                lines.append(f"| {name} | {r['auroc']:.3f} [{r['auroc_ci95'][0]:.3f}, {r['auroc_ci95'][1]:.3f}] | {r['average_precision']:.3f} | "
                             f"{r['f1_train_prevalence']['f1']:.3f} | {r['f1_train_optimal']['f1']:.3f} | {r['f1_test_oracle']['f1']:.3f} |")
            lines.append("")
    lines += ["### Paired AUROC differences on the test split, majority labels (sentence bootstrap)", "", "| Comparison | Δ AUROC | 95% CI | p |", "|---|---:|---|---:|"]
    for name, r in results["paired_auroc_test"].items():
        lines.append(f"| {name} | {r['delta']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['p_two_sided']:.3f} |")
    lines.append("")
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
