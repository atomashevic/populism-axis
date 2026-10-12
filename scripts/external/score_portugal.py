#!/usr/bin/env python3
"""Score the frozen configurations on the Portuguese parliament paragraphs (CPU).

Stage `axis`: pole-mean axes projected on three window encodings of the 6,214 paragraphs:
  e5_en      e5-large-v2 on the Marian English translation (the frozen translation route; primary)
  me5_plain  multilingual-e5-large-instruct, plain template, on the Portuguese original
  me5_instr  multilingual-e5-large-instruct with its instruction, on the Portuguese original
Paragraphs are documents: three-sentence windows, PE soft conjunction, mean of the top 8
windows (B4), and per-channel top-8 means. Writes axis scores and NLI requests (top-8
windows of the e5_en route).
Stage `final`: adds the NLI stage, the RoBERTa comparator on the translation and an
in-domain supervised reference fitted here (logistic regression on mean e5 window
embeddings, 5-fold grouped by speaker), and evaluates against the four hand-coded
dimensions. Pre-declared targets: P channel -> people-centrism, E channel -> anti-elitism.
No train split exists, so thresholds are prevalence-matched (label count only) or oracle.
CIs and paired differences use a speaker-clustered bootstrap.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

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

BENCH = ROOT / "data" / "benchmark_portugal"
OUT = ROOT / "results" / "external" / "portugal"
EMB = ROOT / "results" / "embeddings_external" / "portugal"
ANCHOR_ROOT = ROOT / "results" / "embeddings"
ROUTES = {
    "e5_en": ("e5-large-v2", "portugal_en"),
    "me5_plain": ("me5-large-instruct-noinstr", "portugal_pt"),
    "me5_instr": ("me5-large-instruct", "portugal_pt"),
}
DIMENSIONS = ["centr", "elite", "sovereignty", "exclusion"]
PRIMARY = {"centr": "P", "elite": "E"}
TOP_RULE = "k:8"


def project(route: str, n: int):
    encoder, name = ROUTES[route]
    axes = pole_mean_axes(load_anchor_groups(ANCHOR_ROOT / encoder / "sentences" / "v3_proposition__ea98e5b2.npz"))
    matrix, doc_index, windows = load_embeddings(EMB / encoder / "spw3_cap60", name)
    if int(doc_index.max()) + 1 != n:
        raise SystemExit(f"{route}: embeddings cover {int(doc_index.max()) + 1} documents, expected {n}")
    normalized, _ = center_normalize(matrix)
    projections = WindowProjections(doc_index=doc_index, windows=windows["window"].tolist(),
                                    projections={c: normalized @ axes[c] for c in CHANNELS})
    pe = window_conjunction_scores(projections, {"P": 1.0, "E": 1.0, "M": 0.0}, conjunction="soft")
    return projections, pe, windows, matrix


def stage_axis() -> None:
    frame = pd.read_csv(BENCH / "paragraphs_en.csv")
    n = len(frame)
    out = frame[["id", "actor", "party", "date", "issue"] + [f"{d}_gold" for d in DIMENSIONS]].copy()
    counts = {}
    for route in ROUTES:
        projections, pe, windows, matrix = project(route, n)
        counts[route] = int(len(pe))
        out[f"{route}__PE_axis"] = aggregate_from_window_scores(pe, projections.doc_index, n, TOP_RULE)
        for c in ("P", "E"):
            z = zscore(projections.projections[c])
            out[f"{route}__{c}_axis"] = aggregate_from_window_scores(z, projections.doc_index, n, TOP_RULE)
        if route == "e5_en":
            order = np.lexsort((-pe, projections.doc_index)); sorted_docs = projections.doc_index[order]
            rows = []
            for doc in range(n):
                lo, hi = np.searchsorted(sorted_docs, doc), np.searchsorted(sorted_docs, doc, side="right")
                for rank, position in enumerate(order[lo:hi][:8]):
                    rows.append({"doc_index": doc, "doc_id": frame.loc[doc, "id"], "window_rank": rank, "window_position": int(position),
                                 "window_score": float(pe[position]), "premise": windows.loc[position, "window"]})
            requests = pd.DataFrame(rows); requests.insert(0, "request_id", np.arange(len(requests)))
            requests.to_parquet(OUT / "requests_portugal.parquet", index=False)
            # Mean raw window embedding per paragraph, the feature vector of the supervised reference.
            sums = np.zeros((n, matrix.shape[1])); np.add.at(sums, projections.doc_index, matrix)
            np.save(OUT / "doc_mean_embeddings_e5_en.npy", (sums / np.bincount(projections.doc_index, minlength=n)[:, None]).astype(np.float32))
    out.to_csv(OUT / "axis_scores.csv", index=False)
    write_run_manifest(OUT, script="scripts/external/score_portugal.py", inputs={"paragraphs_en": BENCH / "paragraphs_en.csv"},
                       extra={"stage": "axis", "windows": counts, "top_rule": TOP_RULE}, filename="axis.manifest.json")
    print("windows per route:", counts, "| NLI requests:", len(pd.read_parquet(OUT / "requests_portugal.parquet")))


def nli_channel_scores(model_label: str, n: int) -> dict[str, np.ndarray]:
    hypotheses = json.loads((OUT / "hypotheses.json").read_text(encoding="utf-8"))
    requests = pd.read_parquet(OUT / "requests_portugal.parquet")
    payload = np.load(OUT / f"nli_portugal__{model_label}.npz")
    if not np.array_equal(payload["request_id"], requests["request_id"].to_numpy()):
        raise SystemExit("NLI output does not match requests")
    entail = payload["probs"][:, :, 2].astype(np.float64)
    cols = {c: [i for i, h in enumerate(hypotheses) if h["channel"] == c] for c in ("P", "E")}
    per_window = {"P": entail[:, cols["P"]].max(axis=1), "E": entail[:, cols["E"]].max(axis=1)}
    per_window["PE"] = np.sqrt(per_window["P"] * per_window["E"])
    doc = requests["doc_index"].to_numpy()
    return {f"{c}_nli": pd.Series(v).groupby(doc).mean().reindex(range(n)).fillna(0.0).to_numpy() for c, v in per_window.items()}


def roberta_scores(n: int) -> dict[str, np.ndarray]:
    payload = load_scores(OUT / "roberta" / "sigmoid_portugal_en.npz")
    probs, doc_index = payload["probabilities"].astype(float), payload["doc_index"].astype(int)
    return {"E_roberta": aggregate_from_window_scores(probs[:, 1], doc_index, n, "k:1"),
            "P_roberta": aggregate_from_window_scores(probs[:, 2], doc_index, n, "k:1"),
            "PE_roberta_total": aggregate_from_window_scores(1.0 - probs[:, 0], doc_index, n, "k:1")}


def supervised_reference(y: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    """Out-of-fold logistic-regression scores on mean e5 window embeddings, folds grouped by speaker."""
    features = np.load(OUT / "doc_mean_embeddings_e5_en.npy").astype(np.float64)
    scores = np.zeros(len(y))
    for train, test in GroupKFold(n_splits=5).split(features, y, groups):
        scaler = StandardScaler().fit(features[train])
        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=seed)
        model.fit(scaler.transform(features[train]), y[train])
        scores[test] = model.decision_function(scaler.transform(features[test]))
    return scores


def cluster_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    codes, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(codes))]
    drawn = rng.integers(0, len(codes), len(codes))
    return np.concatenate([members[g] for g in drawn])


def f1_at(values: np.ndarray, y: np.ndarray, cut: float) -> dict:
    pred = (values >= cut).astype(int)
    return {"f1": float(f1_score(y, pred, zero_division=0)), "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)), "threshold": float(cut)}


def oracle_cut(values: np.ndarray, y: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, values)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def evaluate(values: np.ndarray, y: np.ndarray, groups: np.ndarray, draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed); samples = []
    for _ in range(draws):
        idx = cluster_indices(groups, rng)
        if len(np.unique(y[idx])) == 2:
            samples.append(roc_auc_score(y[idx], values[idx]))
    return {"n": int(len(y)), "n_positive": int(y.sum()), "n_clusters": int(len(np.unique(groups))),
            "auroc": float(roc_auc_score(y, values)), "auroc_ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
            "average_precision": float(average_precision_score(y, values)),
            "f1_prevalence_cut": f1_at(values, y, float(np.quantile(values, 1.0 - y.mean()))),
            "f1_oracle_cut": f1_at(values, y, oracle_cut(values, y))}


def paired(a: np.ndarray, b: np.ndarray, y: np.ndarray, groups: np.ndarray, draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed); deltas = []
    for _ in range(draws):
        idx = cluster_indices(groups, rng)
        if len(np.unique(y[idx])) == 2:
            deltas.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    deltas = np.asarray(deltas)
    return {"delta": float(roc_auc_score(y, a) - roc_auc_score(y, b)),
            "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
            "p_two_sided": float(min(1.0, 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())))}


def stage_final(model_label: str, draws: int, seed: int) -> None:
    frame = pd.read_csv(BENCH / "paragraphs_en.csv")
    axis = pd.read_csv(OUT / "axis_scores.csv")
    if list(axis["id"]) != list(frame["id"]):
        raise SystemExit("axis scores and paragraphs disagree on order")
    n = len(frame); groups = frame["actor"].astype(str).to_numpy()
    nli = nli_channel_scores(model_label, n); rob = roberta_scores(n)
    systems: dict[str, np.ndarray] = {}
    for c in ("P", "E", "PE"):
        a = axis[f"e5_en__{c}_axis"].to_numpy(float)
        systems[f"{c}_axis"] = a
        systems[f"{c}_axis_plus_nli"] = (zscore(a) + zscore(nli[f"{c}_nli"])) / 2
        systems[f"{c}_nli"] = nli[f"{c}_nli"]
        systems[f"{c}_axis_me5_plain"] = axis[f"me5_plain__{c}_axis"].to_numpy(float)
        systems[f"{c}_axis_me5_instr"] = axis[f"me5_instr__{c}_axis"].to_numpy(float)
        # Sensitivity: multilingual axis on the original plus the NLI stage on the translation.
        systems[f"{c}_axis_me5_plain_plus_nli"] = (zscore(systems[f"{c}_axis_me5_plain"]) + zscore(nli[f"{c}_nli"])) / 2
    systems.update(rob)
    for dim in PRIMARY:
        systems[f"supervised_lr_{dim}"] = supervised_reference(frame[f"{dim}_gold"].to_numpy(int), groups, seed)
    scores = frame[["id", "actor", "party", "date", "issue"] + [f"{d}_gold" for d in DIMENSIONS]].copy()
    for name, v in systems.items():
        scores[name] = v
    scores.to_csv(OUT / "final_scores.csv", index=False)

    results: dict = {"nli_model": model_label, "primary": {}, "paired_auroc": {}, "discriminant_auroc": {}}
    for dim, channel in PRIMARY.items():
        y = frame[f"{dim}_gold"].to_numpy(int)
        names = [f"{channel}_axis_plus_nli", f"{channel}_axis", f"{channel}_nli", f"{channel}_axis_me5_plain", f"{channel}_axis_me5_instr",
                 f"{channel}_axis_me5_plain_plus_nli", "PE_axis_plus_nli", "PE_axis", f"{channel}_roberta", "PE_roberta_total", f"supervised_lr_{dim}"]
        results["primary"][dim] = {name: evaluate(systems[name], y, groups, draws, seed) for name in names}
        head = f"{channel}_axis_plus_nli"
        results["paired_auroc"][dim] = {}
        for other in (f"{channel}_axis", f"{channel}_nli", f"{channel}_axis_me5_plain", f"{channel}_axis_me5_plain_plus_nli", "PE_axis_plus_nli", f"{channel}_roberta", f"supervised_lr_{dim}"):
            results["paired_auroc"][dim][f"{head} - {other}"] = paired(systems[head], systems[other], y, groups, draws, seed)
    for dim in DIMENSIONS:
        y = frame[f"{dim}_gold"].to_numpy(int)
        results["discriminant_auroc"][dim] = {name: float(roc_auc_score(y, systems[name])) for name in
                                             ("P_axis", "E_axis", "PE_axis", "P_axis_plus_nli", "E_axis_plus_nli", "P_nli", "E_nli")}
        results["discriminant_auroc"][dim]["n_positive"] = int(y.sum())
    results["by_party"] = {}
    for dim, channel in PRIMARY.items():
        results["by_party"][dim] = {}
        for party, sub in frame.groupby("party"):
            y = sub[f"{dim}_gold"].to_numpy(int)
            if y.sum() >= 5 and len(y) - y.sum() >= 5:
                results["by_party"][dim][party] = {"n": int(len(y)), "n_positive": int(y.sum()),
                                                   "auroc": float(roc_auc_score(y, systems[f"{channel}_axis_plus_nli"][sub.index]))}
    freeze = json.loads((OUT / "benchmark_freeze.json").read_text(encoding="utf-8"))
    results["krippendorff_alpha_published"] = freeze["krippendorff_alpha_published"]
    (OUT / "final_results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_run_manifest(OUT, script="scripts/external/score_portugal.py",
                       inputs={"axis": OUT / "axis_scores.csv", "nli": OUT / f"nli_portugal__{model_label}.npz",
                               "roberta": OUT / "roberta" / "sigmoid_portugal_en.npz"},
                       extra={"stage": "final", "draws": draws, "seed": seed}, filename="final.manifest.json")

    lines = []
    for dim, block in results["primary"].items():
        any_row = next(iter(block.values()))
        lines += [f"### {dim} (n={any_row['n']}, positives={any_row['n_positive']}, speakers={any_row['n_clusters']})", "",
                  "| System | AUROC [95% CI] | AP | F1 @ prevalence cut | F1 @ oracle cut |", "|---|---|---:|---:|---:|"]
        for name, r in block.items():
            lines.append(f"| {name} | {r['auroc']:.3f} [{r['auroc_ci95'][0]:.3f}, {r['auroc_ci95'][1]:.3f}] | {r['average_precision']:.3f} | "
                         f"{r['f1_prevalence_cut']['f1']:.3f} | {r['f1_oracle_cut']['f1']:.3f} |")
        lines += ["", "| Paired Δ AUROC (speaker bootstrap) | Δ | 95% CI | p |", "|---|---:|---|---:|"]
        for key, p in results["paired_auroc"][dim].items():
            lines.append(f"| {key} | {p['delta']:+.3f} | [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] | {p['p_two_sided']:.3f} |")
        lines.append("")
    lines += ["### Discriminant AUROC: every channel against every dimension", "",
              "| Dimension (positives) | P axis | E axis | PE axis | P axis+NLI | E axis+NLI | P NLI | E NLI |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for dim, r in results["discriminant_auroc"].items():
        lines.append(f"| {dim} ({r['n_positive']}) | {r['P_axis']:.3f} | {r['E_axis']:.3f} | {r['PE_axis']:.3f} | {r['P_axis_plus_nli']:.3f} | "
                     f"{r['E_axis_plus_nli']:.3f} | {r['P_nli']:.3f} | {r['E_nli']:.3f} |")
    lines.append("")
    (OUT / "final_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(json.dumps(results["by_party"], indent=1))


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
