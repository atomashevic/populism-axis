#!/usr/bin/env python3
"""Score the frozen configurations on the US governors / 2016 candidates benchmark (CPU).

Two stages, both label-free until the metrics are computed:

  axis   Build pole-mean axes from the frozen e5-large-v2 anchor embeddings, project the
         stored window embeddings, and write the axis document scores:
           B0  initial configuration: one-sentence windows, no cap, PE soft, mean of top 3
           B4  populism axis: 3-sentence windows capped at 60 words, PE soft, mean of top 8
         Also writes the NLI requests (top-8 windows per speech, top window per sentence)
         in the step-6 format so scripts/nli_stage.py runs unchanged.
  final  Add B6 = mean of z-scores of B4 and the NLI document score, the RoBERTa
         comparator, and the package's released SetFit shares; compute metrics per
         subset with cluster-bootstrap CIs and paired differences; write
         final_results.{json,md}.

Centering and z-scoring use all speeches pooled (one external dataset), and all
sentences pooled; metrics are reported per subset and pooled.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

from populism_axis.contextual_axis import CHANNELS, WindowProjections, aggregate_from_window_scores, window_conjunction_scores  # noqa: E402
from populism_axis.run_manifest import write_run_manifest  # noqa: E402
from populism_axis.roberta_comparator import load_scores  # noqa: E402
from populism_axis.geometry import center_normalize, pole_mean_axes  # noqa: E402
from evaluate_benchmarks import _spearman_metric, cluster_bootstrap  # noqa: E402
from paired_comparison import paired_delta  # noqa: E402
from run_config import load_anchor_groups, load_embeddings  # noqa: E402
from evaluate_nli import zscore  # noqa: E402

DATA = ROOT / "data" / "benchmark_usgov"
OUT = ROOT / "results" / "external" / "usgov"
EMB = ROOT / "results" / "embeddings_external" / "usgov" / "e5-large-v2"
ANCHORS = ROOT / "results" / "embeddings" / "e5-large-v2" / "sentences" / "v3_proposition__ea98e5b2.npz"
WEIGHTS = {"P": 1.0, "E": 1.0, "M": 0.0}
BINARY_CUT = 0.499
SUBSETS = ("governors_2010_2018", "governors_2018_2022", "presidential_2016")


def project(embeddings_dir: Path, name: str, axes: dict) -> tuple[WindowProjections, np.ndarray, pd.DataFrame]:
    matrix, doc_index, windows = load_embeddings(embeddings_dir, name)
    normalized, _ = center_normalize(matrix)
    projections = WindowProjections(doc_index=doc_index, windows=windows["window"].tolist(),
                                    projections={c: normalized @ axes[c] for c in CHANNELS})
    scores = window_conjunction_scores(projections, WEIGHTS, conjunction="soft")
    return projections, scores, windows


def top_windows(frame: pd.DataFrame, id_column: str, doc_index: np.ndarray, scores: np.ndarray, windows: pd.DataFrame, k: int) -> pd.DataFrame:
    order = np.lexsort((-scores, doc_index))
    sorted_docs = doc_index[order]
    rows = []
    for doc in range(len(frame)):
        lo, hi = np.searchsorted(sorted_docs, doc), np.searchsorted(sorted_docs, doc, side="right")
        for rank, position in enumerate(order[lo:hi][:k]):
            rows.append({"doc_index": doc, "doc_id": frame.loc[doc, id_column], "window_rank": rank,
                         "window_position": int(position), "window_score": float(scores[position]),
                         "premise": windows.loc[position, "window"]})
    requests = pd.DataFrame(rows)
    requests.insert(0, "request_id", np.arange(len(requests)))
    return requests


def stage_axis() -> None:
    speeches = pd.read_csv(DATA / "speeches.csv")
    sentences = pd.read_csv(DATA / "sentences.csv")
    axes = pole_mean_axes(load_anchor_groups(ANCHORS))

    proj0, scores0, _ = project(EMB / "spw1_cap0", "usgov_speeches", axes)
    proj4, scores4, windows4 = project(EMB / "spw3_cap60", "usgov_speeches", axes)
    n = len(speeches)
    out = speeches[["id", "subset", "group", "score", "binary_populist"]].copy()
    out["B0_axis_v3"] = aggregate_from_window_scores(scores0, proj0.doc_index, n, "k:3")
    out["B4_axis"] = aggregate_from_window_scores(scores4, proj4.doc_index, n, "k:8")
    for channel in ("P", "E"):
        z = (proj4.projections[channel] - proj4.projections[channel].mean()) / proj4.projections[channel].std()
        out[f"{channel}_only_k8"] = aggregate_from_window_scores(z, proj4.doc_index, n, "k:8")
    out.to_csv(OUT / "axis_scores_speeches.csv", index=False)
    top_windows(speeches, "id", proj4.doc_index, scores4, windows4, 8).to_parquet(OUT / "requests_usgov_speeches.parquet", index=False)

    proj_s, scores_s, windows_s = project(EMB / "spw3_cap60", "usgov_sentences", axes)
    m = len(sentences)
    out_s = sentences[["id", "subset", "label"]].copy()
    out_s["B4_axis"] = aggregate_from_window_scores(scores_s, proj_s.doc_index, m, "k:1")
    for channel in ("P", "E"):
        z = (proj_s.projections[channel] - proj_s.projections[channel].mean()) / proj_s.projections[channel].std()
        out_s[f"{channel}_only"] = aggregate_from_window_scores(z, proj_s.doc_index, m, "k:1")
    out_s.to_csv(OUT / "axis_scores_sentences.csv", index=False)
    top_windows(sentences, "id", proj_s.doc_index, scores_s, windows_s, 1).to_parquet(OUT / "requests_usgov_sentences.parquet", index=False)
    write_run_manifest(OUT, script="scripts/external/score_usgov.py", inputs={"anchors": ANCHORS, "speeches": DATA / "speeches.csv", "sentences": DATA / "sentences.csv"},
                       extra={"stage": "axis", "windows": {"speeches_spw1": int(len(scores0)), "speeches_spw3": int(len(scores4)), "sentences": int(len(scores_s))}},
                       filename="axis.manifest.json")
    print("speech windows:", len(scores0), len(scores4), "| sentence windows:", len(scores_s))
    print("NLI requests:", len(pd.read_parquet(OUT / "requests_usgov_speeches.parquet")), "speech windows,",
          len(pd.read_parquet(OUT / "requests_usgov_sentences.parquet")), "sentence windows")


def nli_scores(name: str, n_docs: int, k: int, model_label: str) -> np.ndarray:
    hypotheses = json.loads((OUT / "hypotheses.json").read_text(encoding="utf-8"))
    requests = pd.read_parquet(OUT / f"requests_{name}.parquet")
    payload = np.load(OUT / f"nli_{name}__{model_label}.npz")
    if not np.array_equal(payload["request_id"], requests["request_id"].to_numpy()):
        raise SystemExit(f"{name}: NLI output does not match requests")
    entail = payload["probs"][:, :, 2].astype(np.float64)
    p_cols = [i for i, h in enumerate(hypotheses) if h["channel"] == "P"]
    e_cols = [i for i, h in enumerate(hypotheses) if h["channel"] == "E"]
    window = np.sqrt(entail[:, p_cols].max(axis=1) * entail[:, e_cols].max(axis=1))
    kept = requests["window_rank"].to_numpy() < k
    return pd.Series(window[kept]).groupby(requests["doc_index"].to_numpy()[kept]).mean().reindex(range(n_docs)).fillna(0.0).to_numpy()


def roberta_scores(name: str, n_docs: int, rule: str, component: int) -> np.ndarray:
    payload = load_scores(OUT / "roberta" / f"sigmoid_{name}.npz")
    doc_index, probs = payload["doc_index"].astype(int), payload["probabilities"].astype(float)
    values = 1.0 - probs[:, 0] if component < 0 else probs[:, component]
    return aggregate_from_window_scores(values, doc_index, n_docs, rule)


def in_sample_cutoff_metrics(values: np.ndarray, binary: np.ndarray) -> dict:
    """The paper's protocol: F1-optimal cutoff chosen on the same data; report accuracy, F1, balanced accuracy."""
    grid = np.unique(np.quantile(values, np.linspace(0.0, 0.995, 200)))
    best, best_f1 = grid[0], -1.0
    for cut in grid:
        f1 = f1_score(binary, (values >= cut).astype(int), zero_division=0)
        if f1 > best_f1:
            best, best_f1 = cut, f1
    pred = (values >= best).astype(int)
    tp = int(((pred == 1) & (binary == 1)).sum()); tn = int(((pred == 0) & (binary == 0)).sum())
    fp = int(((pred == 1) & (binary == 0)).sum()); fn = int(((pred == 0) & (binary == 1)).sum())
    recall = tp / max(1, tp + fn); specificity = tn / max(1, tn + fp)
    return {"accuracy": float((pred == binary).mean()), "f1": float(best_f1), "precision": tp / max(1, tp + fp),
            "recall": recall, "balanced_accuracy": (recall + specificity) / 2}


def speech_metrics(values: np.ndarray, frame: pd.DataFrame, draws: int, seed: int) -> dict:
    target = frame["score"].to_numpy(dtype=float)
    binary = frame["binary_populist"].to_numpy(dtype=int)
    groups = frame["group"].astype(str).to_numpy()
    low, high = cluster_bootstrap(_spearman_metric, values, target, groups, draws, seed)
    out = {"n": int(len(frame)), "n_groups": int(len(np.unique(groups))), "n_positive": int(binary.sum()),
           "spearman": float(spearmanr(values, target).statistic), "spearman_ci95": [low, high],
           "pearson_r2": float(pearsonr(values, target)[0] ** 2)}
    if 0 < binary.sum() < len(binary):
        out["auroc"] = float(roc_auc_score(binary, values))
        out["average_precision"] = float(average_precision_score(binary, values))
        out["in_sample_cutoff"] = in_sample_cutoff_metrics(values, binary)
    return out


def sentence_metrics(values: np.ndarray, frame: pd.DataFrame) -> dict:
    label = frame["label"].to_numpy()
    out = {"n": int(len(frame)), "counts": {k: int(v) for k, v in pd.Series(label).value_counts().items()}}
    pop = (label == "populist").astype(int)
    if 0 < pop.sum() < len(pop):
        out["auroc_populist_vs_rest"] = float(roc_auc_score(pop, values))
        out["ap_populist_vs_rest"] = float(average_precision_score(pop, values))
    mask = np.isin(label, ["populist", "pluralist"])
    if mask.sum() and 0 < pop[mask].sum() < mask.sum():
        out["auroc_populist_vs_pluralist"] = float(roc_auc_score(pop[mask], values[mask]))
    mask = np.isin(label, ["populist", "neutral"])
    if mask.sum() and 0 < pop[mask].sum() < mask.sum():
        out["auroc_populist_vs_neutral"] = float(roc_auc_score(pop[mask], values[mask]))
    return out


def stage_final(model_label: str, draws: int, seed: int) -> None:
    # Both caches must be corrected before any speech/sentence result write.
    for name in ("usgov_speeches", "usgov_sentences"):
        load_scores(OUT / "roberta" / f"sigmoid_{name}.npz")
    speeches = pd.read_csv(DATA / "speeches.csv")
    axis = pd.read_csv(OUT / "axis_scores_speeches.csv")
    comparator = pd.read_csv(DATA / "comparator_predictions.csv").set_index("id")["setfit_pop_share"]
    if list(axis["id"]) != list(speeches["id"]):
        raise SystemExit("axis scores and speeches disagree on order")
    n = len(speeches)
    nli = nli_scores("usgov_speeches", n, 8, model_label)
    systems = {
        "B0_v3_config": axis["B0_axis_v3"].to_numpy(dtype=float),
        "B4_axis": axis["B4_axis"].to_numpy(dtype=float),
        "B6_axis_plus_nli": (zscore(axis["B4_axis"].to_numpy(dtype=float)) + zscore(nli)) / 2,
        "NLI_only": nli,
        "RoBERTa_anti_elitism_top3": roberta_scores("usgov_speeches", n, "k:3", 1),
        "RoBERTa_populist_total_max": roberta_scores("usgov_speeches", n, "k:1", -1),
        "SetFit_share_published": speeches["id"].map(comparator).to_numpy(dtype=float),
        "P_only_k8": axis["P_only_k8"].to_numpy(dtype=float),
        "E_only_k8": axis["E_only_k8"].to_numpy(dtype=float),
    }
    scores_out = speeches[["id", "subset", "group", "score", "binary_populist"]].copy()
    for name, values in systems.items():
        scores_out[name] = values
    scores_out.to_csv(OUT / "final_scores_speeches.csv", index=False)

    results: dict = {"speeches": {}, "paired_spearman": {}, "sentences": {}}
    for subset in list(SUBSETS) + ["all_pooled"]:
        mask = np.ones(n, dtype=bool) if subset == "all_pooled" else (speeches["subset"] == subset).to_numpy()
        frame = speeches[mask].reset_index(drop=True)
        results["speeches"][subset] = {}
        for name, values in systems.items():
            v = values[mask]
            if np.isnan(v).any():
                keep = ~np.isnan(v)
                results["speeches"][subset][name] = speech_metrics(v[keep], frame[keep].reset_index(drop=True), draws, seed)
                results["speeches"][subset][name]["n_scored"] = int(keep.sum())
            else:
                results["speeches"][subset][name] = speech_metrics(v, frame, draws, seed)
        target = frame["score"].to_numpy(dtype=float); groups = frame["group"].astype(str).to_numpy()
        results["paired_spearman"][subset] = {}
        for left, right in (("B4_axis", "B0_v3_config"), ("B6_axis_plus_nli", "B0_v3_config"), ("B6_axis_plus_nli", "B4_axis"),
                            ("B6_axis_plus_nli", "SetFit_share_published"), ("B6_axis_plus_nli", "RoBERTa_anti_elitism_top3"),
                            ("B4_axis", "SetFit_share_published")):
            l, r = systems[left][mask], systems[right][mask]
            keep = ~(np.isnan(l) | np.isnan(r))
            payload = paired_delta(l[keep], r[keep], target[keep], groups[keep], draws, seed)
            payload["p_two_sided"] = min(1.0, payload["p_two_sided"])
            results["paired_spearman"][subset][f"{left} - {right}"] = payload

    sentences = pd.read_csv(DATA / "sentences.csv")
    axis_s = pd.read_csv(OUT / "axis_scores_sentences.csv")
    m = len(sentences)
    nli_s = nli_scores("usgov_sentences", m, 1, model_label)
    systems_s = {
        "B4_axis": axis_s["B4_axis"].to_numpy(dtype=float),
        "B6_axis_plus_nli": (zscore(axis_s["B4_axis"].to_numpy(dtype=float)) + zscore(nli_s)) / 2,
        "NLI_only": nli_s,
        "RoBERTa_populist_total_max": roberta_scores("usgov_sentences", m, "k:1", -1),
        "P_only": axis_s["P_only"].to_numpy(dtype=float), "E_only": axis_s["E_only"].to_numpy(dtype=float),
    }
    scores_s = sentences[["id", "subset", "label"]].copy()
    for name, values in systems_s.items():
        scores_s[name] = values
    scores_s.to_csv(OUT / "final_scores_sentences.csv", index=False)
    for subset in list(SUBSETS[:1]) + [SUBSETS[2], "all_pooled"]:
        mask = np.ones(m, dtype=bool) if subset == "all_pooled" else (sentences["subset"] == subset).to_numpy()
        results["sentences"][subset] = {name: sentence_metrics(values[mask], sentences[mask].reset_index(drop=True)) for name, values in systems_s.items()}

    freeze = json.loads((OUT / "benchmark_freeze.json").read_text(encoding="utf-8"))
    results["published"] = freeze["published_results"]
    results["nli_model"] = model_label
    (OUT / "final_results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_run_manifest(OUT, script="scripts/external/score_usgov.py",
                       inputs={"axis_scores": OUT / "axis_scores_speeches.csv", "nli_speeches": OUT / f"nli_usgov_speeches__{model_label}.npz",
                               "roberta_speeches": OUT / "roberta" / "sigmoid_usgov_speeches.npz", "comparator": DATA / "comparator_predictions.csv"},
                       extra={"stage": "final"}, filename="final.manifest.json")

    lines = []
    for subset in list(SUBSETS) + ["all_pooled"]:
        block = results["speeches"][subset]
        any_row = next(iter(block.values()))
        lines += [f"### Speeches: {subset} (n={any_row['n']}, groups={any_row['n_groups']}, populist={any_row['n_positive']})", "",
                  "| System | Spearman [95% CI] | Pearson r² | AUROC | AP | In-sample cutoff: acc / F1 / bal.acc |", "|---|---|---:|---:|---:|---|"]
        for name, r in block.items():
            cut = r.get("in_sample_cutoff", {})
            lines.append(f"| {name} | {r['spearman']:.3f} [{r['spearman_ci95'][0]:.3f}, {r['spearman_ci95'][1]:.3f}] | {r['pearson_r2']:.3f} | "
                         f"{r.get('auroc', float('nan')):.3f} | {r.get('average_precision', float('nan')):.3f} | "
                         f"{cut.get('accuracy', float('nan')):.2f} / {cut.get('f1', float('nan')):.2f} / {cut.get('balanced_accuracy', float('nan')):.2f} |")
        lines += ["", "| Paired Δ Spearman | Δ | 95% CI | p |", "|---|---:|---|---:|"]
        for key, p in results["paired_spearman"][subset].items():
            lines.append(f"| {key} | {p['delta_spearman']:+.3f} | [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] | {p['p_two_sided']:.3f} |")
        lines.append("")
    for subset, block in results["sentences"].items():
        any_row = next(iter(block.values()))
        lines += [f"### Sentences: {subset} (n={any_row['n']}, {any_row['counts']})", "",
                  "| System | AUROC populist vs rest | AP | AUROC vs pluralist | AUROC vs neutral |", "|---|---:|---:|---:|---:|"]
        for name, r in block.items():
            lines.append(f"| {name} | {r.get('auroc_populist_vs_rest', float('nan')):.3f} | {r.get('ap_populist_vs_rest', float('nan')):.3f} | "
                         f"{r.get('auroc_populist_vs_pluralist', float('nan')):.3f} | {r.get('auroc_populist_vs_neutral', float('nan')):.3f} |")
        lines.append("")
    (OUT / "final_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("axis", "final"))
    parser.add_argument("--nli-model-label", default="bart-large-mnli")
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "axis":
        stage_axis()
    else:
        stage_final(args.nli_model_label, args.draws, args.seed)


if __name__ == "__main__":
    main()
