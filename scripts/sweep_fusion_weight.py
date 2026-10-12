#!/usr/bin/env python3
"""Fusion-weight sweep for the populism axis + claim check combination (CPU; stored scores only).

The pre-declared axis + claim check score is the mean of within-corpus z-scores of the axis
document score and the NLI document score, i.e. weight .5 on each. This script forms

    score(w) = w * z(axis) + (1 - w) * z(nli),   w in {0, .1, ..., 1}

from the per-document scores already stored under results/ and reports each
corpus's primary metric at every weight, with a cluster-bootstrap paired interval
against w = .5. w = 1 is the axis alone and w = 0 the NLI stage alone.

No embeddings, NLI probabilities or GPU are touched. The holdout, EU-OOD and the four
external sets are spent, so their curves are sensitivity analyses. Any weight other
than .5 could only be adopted on English-dev and Agora under the step-6 rule.

Corpora (axis score, NLI score, target, primary metric, clusters):
  english_dev       B4 spw3_k8 axis, NLI top-8       GPD rubric   Spearman  leader
  agora             B4 spw3_k8 axis, NLI top-8       human score  Spearman  politician
  english_holdout   B4 axis, NLI top-8               GPD rubric   Spearman  leader
  eu_3v0            B4 axis, NLI top window          pop 3 vs 0   AUROC     speaker
  eu_AE / eu_PC     E / P top-window proj., E / P NLI   paper labels AUROC  speaker
  usgov_gov2010_18  B4 axis, NLI top-8 (z pooled over all 359 speeches)  Spearman  state-term
  usgov_pooled      same, all 359                    rubric       Spearman  speaker group
  tweets_general    E axis, E NLI (z over all 5,401; test split scored)  AUROC  country
  popbert_AE / _PC  E / P axis, E / P NLI (z over train+test; test scored)  AUROC  sentence
  portugal_AE / _PC E / P axis, E / P NLI            hand codes   AUROC     speaker

Outputs: results/fusion_weight_sweep/{sweep.json, sweep.md, run_manifest.json}.
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

from populism_axis.run_manifest import write_run_manifest  # noqa: E402
from evaluate_nli import zscore  # noqa: E402

R = ROOT / "results"
OUT = R / "fusion_weight_sweep"
SYSTEM = "PE__frozen__soft"
WEIGHTS = [round(w, 1) for w in np.linspace(0.0, 1.0, 11)]
REFERENCE = 0.5


def spearman(target: np.ndarray, values: np.ndarray) -> float:
    return float(spearmanr(values, target).statistic)


def auroc(target: np.ndarray, values: np.ndarray) -> float:
    return float(roc_auc_score(target, values))


def window_max(doc_index: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    out = np.full(n, -np.inf)
    np.maximum.at(out, doc_index, values)
    out[~np.isfinite(out)] = 0.0
    return out


def read_scores(path: Path, column: str) -> np.ndarray:
    return pd.read_csv(path)[column].to_numpy(dtype=float)


def load_corpora() -> dict[str, dict]:
    """Each corpus: z_axis, z_nli (z-scored over the same population the reports used),
    an evaluation mask, target, metric, and cluster ids for the bootstrap."""
    corpora: dict[str, dict] = {}
    split = pd.read_csv(ROOT / "data" / "benchmarks" / "english_split.csv").set_index("id")["split"]
    english = pd.read_csv(ROOT / "data" / "benchmarks" / "english_gpd_speeches.csv")

    # English-dev and Agora: carried B4 axis (step 4 grid cell spw3_k8) and step-6 NLI top-8.
    carried = R / "step4_windows_topk" / "grid" / "spw3_k8"
    nli_dev = R / "step6_nli" / "nli_only_k8"
    dev = english[english["id"].map(split).eq("dev")].reset_index(drop=True)
    ids = pd.read_csv(carried / "scores_english.csv")["id"]
    if list(ids) != list(dev["id"]) or list(pd.read_csv(nli_dev / "scores_english.csv")["id"]) != list(dev["id"]):
        raise SystemExit("english_dev: stored score order does not match the split")
    corpora["english_dev"] = dict(axis=read_scores(carried / "scores_english.csv", SYSTEM), nli=read_scores(nli_dev / "scores_english.csv", SYSTEM),
                                  target=dev["source_score"].to_numpy(float), groups=dev["leader"].astype(str).to_numpy(), metric="spearman",
                                  n=len(dev), spent=False)
    agora = pd.read_csv(ROOT / "data" / "benchmarks" / "agora_paragraphs.csv")
    if list(pd.read_csv(carried / "scores_agora.csv")["id"]) != list(agora["row_id"]):
        raise SystemExit("agora: stored score order does not match the frame")
    corpora["agora"] = dict(axis=read_scores(carried / "scores_agora.csv", SYSTEM), nli=read_scores(nli_dev / "scores_agora.csv", SYSTEM),
                            target=agora["populism_human"].to_numpy(float), groups=agora["politician"].astype(str).to_numpy(), metric="spearman",
                            n=len(agora), spent=False)

    # English holdout and EU-OOD: final B4 axis and the NLI-only column stored beside B6.
    final = R / "final"
    holdout = english[english["id"].map(split).eq("holdout")].reset_index(drop=True)
    if list(pd.read_csv(final / "B4_axis" / "scores_english.csv")["id"]) != list(holdout["id"]):
        raise SystemExit("english_holdout: stored score order does not match the split")
    corpora["english_holdout"] = dict(axis=read_scores(final / "B4_axis" / "scores_english.csv", SYSTEM),
                                      nli=read_scores(final / "B6_axis_plus_nli" / "scores_english.csv", "nli_only"),
                                      target=holdout["source_score"].to_numpy(float), groups=holdout["leader"].astype(str).to_numpy(), metric="spearman",
                                      n=len(holdout), spent=True)
    eu = pd.read_csv(ROOT / "data" / "benchmarks" / "eu_sentences.csv")
    mask = eu["pop_code"].isin([0, 3]).to_numpy()
    corpora["eu_3v0"] = dict(axis=read_scores(final / "B4_axis" / "scores_eu.csv", SYSTEM), nli=read_scores(final / "B6_axis_plus_nli" / "scores_eu.csv", "nli_only"),
                             target=(eu["pop_code"].to_numpy() == 3).astype(int), groups=eu["speaker"].astype(str).to_numpy(), metric="auroc",
                             mask=mask, n=len(eu), spent=True)
    # EU paper-protocol channels: top-window projection per channel, top-window NLI per channel.
    proj = np.load(final / "B4_axis" / "projections_eu.npz")
    idx = proj["doc_index"].astype(int)
    hyp = json.loads((R / "step6_nli" / "hypotheses.json").read_text(encoding="utf-8"))
    req = pd.read_parquet(R / "step6_nli" / "requests_eu.parquet")
    payload = np.load(R / "step6_nli" / "nli_eu__bart-large-mnli.npz")
    if not np.array_equal(payload["request_id"], req["request_id"].to_numpy()):
        raise SystemExit("eu: NLI output does not match requests")
    entail = payload["probs"][:, :, 2].astype(np.float64)
    top = (req["window_rank"] == 0).to_numpy()
    pop = eu["pop_code"].to_numpy(int)
    for key, channel, positives in (("AE", "E", [1, 3]), ("PC", "P", [2, 3])):
        cols = [i for i, h in enumerate(hyp) if h["channel"] == channel]
        nli_doc = np.zeros(len(eu))
        nli_doc[req.loc[top, "doc_index"].to_numpy()] = entail[top][:, cols].max(axis=1)
        corpora[f"eu_{key}"] = dict(axis=window_max(idx, proj[channel].astype(float), len(eu)), nli=nli_doc,
                                    target=np.isin(pop, positives).astype(int), groups=eu["speaker"].astype(str).to_numpy(), metric="auroc",
                                    n=len(eu), spent=True)

    # US governors: z pooled over all 359 speeches (as scored); evaluate the main subset and the pool.
    us = pd.read_csv(R / "external" / "usgov" / "final_scores_speeches.csv")
    for key, mask in (("usgov_gov2010_18", (us["subset"] == "governors_2010_2018").to_numpy()), ("usgov_pooled", np.ones(len(us), bool))):
        corpora[key] = dict(axis=us["B4_axis"].to_numpy(float), nli=us["NLI_only"].to_numpy(float), target=us["score"].to_numpy(float),
                            groups=us["group"].astype(str).to_numpy(), metric="spearman", mask=mask, n=len(us), spent=True)

    # Party tweets: E channel, z over all 5,401, test split scored.
    tw = pd.read_csv(R / "external" / "antielite_tweets" / "final_scores.csv")
    corpora["tweets_general"] = dict(axis=tw["E_axis"].to_numpy(float), nli=tw["E_nli"].to_numpy(float), target=tw["y_general"].to_numpy(int),
                                     groups=tw["country"].astype(str).to_numpy(), metric="auroc", mask=(tw["split"] == "test").to_numpy(), n=len(tw), spent=True)

    # PopBERT in English: per channel, z over train + test, test scored, sentence bootstrap.
    pb = pd.read_csv(R / "external" / "popbert_en" / "final_scores.csv")
    for key, channel, gold in (("popbert_AE", "E", "elite_gold"), ("popbert_PC", "P", "centr_gold")):
        corpora[key] = dict(axis=pb[f"{channel}_axis"].to_numpy(float), nli=pb[f"{channel}_nli"].to_numpy(float), target=pb[gold].to_numpy(int),
                            groups=np.arange(len(pb)).astype(str), metric="auroc", mask=(pb["split"] == "test").to_numpy(), n=len(pb), spent=True)

    # Portuguese parliament: per channel, translation route, speaker clusters.
    pt = pd.read_csv(R / "external" / "portugal" / "final_scores.csv")
    for key, channel, gold in (("portugal_AE", "E", "elite_gold"), ("portugal_PC", "P", "centr_gold")):
        corpora[key] = dict(axis=pt[f"{channel}_axis"].to_numpy(float), nli=pt[f"{channel}_nli"].to_numpy(float), target=pt[gold].to_numpy(int),
                            groups=pt["actor"].astype(str).to_numpy(), metric="auroc", n=len(pt), spent=True)

    for name, c in corpora.items():
        c["z_axis"], c["z_nli"] = zscore(c["axis"]), zscore(c["nli"])
        c.setdefault("mask", np.ones(c["n"], bool))
    return corpora


def fused(c: dict, w: float) -> np.ndarray:
    return w * c["z_axis"] + (1.0 - w) * c["z_nli"]


def check_reference(corpora: dict[str, dict]) -> dict[str, float]:
    """The w = .5 score must reproduce the stored combined column exactly (rank-identical)."""
    stored = {
        "english_dev": read_scores(R / "step6_nli" / "combined_k8" / "scores_english.csv", SYSTEM),
        "agora": read_scores(R / "step6_nli" / "combined_k8" / "scores_agora.csv", SYSTEM),
        "english_holdout": read_scores(R / "final" / "B6_axis_plus_nli" / "scores_english.csv", SYSTEM),
        "eu_3v0": read_scores(R / "final" / "B6_axis_plus_nli" / "scores_eu.csv", SYSTEM),
        "usgov_pooled": read_scores(R / "external" / "usgov" / "final_scores_speeches.csv", "B6_axis_plus_nli"),
        "tweets_general": read_scores(R / "external" / "antielite_tweets" / "final_scores.csv", "E_axis_plus_nli"),
        "popbert_AE": read_scores(R / "external" / "popbert_en" / "final_scores.csv", "E_axis_plus_nli"),
        "popbert_PC": read_scores(R / "external" / "popbert_en" / "final_scores.csv", "P_axis_plus_nli"),
        "portugal_AE": read_scores(R / "external" / "portugal" / "final_scores.csv", "E_axis_plus_nli"),
        "portugal_PC": read_scores(R / "external" / "portugal" / "final_scores.csv", "P_axis_plus_nli"),
    }
    out = {}
    for name, values in stored.items():
        diff = float(np.abs(fused(corpora[name], REFERENCE) - values).max())
        if diff > 1e-9:
            raise SystemExit(f"{name}: w=.5 does not reproduce the stored combined score (max abs diff {diff:.2e})")
        out[name] = diff
    return out


def paired_ci(c: dict, w: float, draws: int, seed: int) -> tuple[float, float, float]:
    metric = spearman if c["metric"] == "spearman" else auroc
    mask = c["mask"]
    left, right, target, groups = fused(c, w)[mask], fused(c, REFERENCE)[mask], c["target"][mask], c["groups"][mask]
    observed = metric(target, left) - metric(target, right)
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    index_by = {g: np.flatnonzero(groups == g) for g in unique}
    samples = []
    for _ in range(draws):
        idx = np.concatenate([index_by[g] for g in rng.choice(unique, size=len(unique), replace=True)])
        if len(np.unique(target[idx])) < 2:
            continue
        try:
            d = metric(target[idx], left[idx]) - metric(target[idx], right[idx])
        except ValueError:
            continue
        if np.isfinite(d):
            samples.append(d)
    arr = np.asarray(samples)
    return observed, float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    corpora = load_corpora()
    reproduction = check_reference(corpora)

    results: dict = {"weights": WEIGHTS, "reference_weight": REFERENCE, "draws": args.draws, "seed": args.seed,
                     "reproduction_max_abs_diff_at_reference": reproduction, "corpora": {}}
    for name, c in corpora.items():
        metric = spearman if c["metric"] == "spearman" else auroc
        mask = c["mask"]
        curve = {str(w): metric(c["target"][mask], fused(c, w)[mask]) for w in WEIGHTS}
        paired = {}
        for w in WEIGHTS:
            if w == REFERENCE:
                continue
            d, lo, hi = paired_ci(c, w, args.draws, args.seed)
            paired[str(w)] = {"delta": d, "ci95": [lo, hi]}
        values = np.array([curve[str(w)] for w in WEIGHTS])
        best = WEIGHTS[int(values.argmax())]
        results["corpora"][name] = {
            "metric": c["metric"], "n_evaluated": int(mask.sum()), "n_clusters": int(len(np.unique(c["groups"][mask]))), "spent": c["spent"],
            "window_spearman_axis_vs_nli": float(spearmanr(c["z_axis"][mask], c["z_nli"][mask]).statistic),
            "curve": curve, "paired_vs_reference": paired, "best_weight": best, "best_value": float(values.max()),
            "value_at_reference": curve[str(REFERENCE)], "gain_of_best_over_reference": float(values.max() - curve[str(REFERENCE)]),
            "flat_band_0.3_0.7_range": float(values[3:8].max() - values[3:8].min()),
        }
        print(f"{name:18s} {c['metric']:8s} ref {curve[str(REFERENCE)]:.3f}  best w={best} {values.max():.3f}  axis {curve['1.0']:.3f}  nli {curve['0.0']:.3f}", flush=True)
    (OUT / "sweep.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    head = "| Corpus | Metric | n | " + " | ".join(f"w={w}" for w in WEIGHTS) + " | best w | gain over .5 | flat .3–.7 |"
    sep = "|---|---|---:|" + "---:|" * len(WEIGHTS) + "---:|---:|---:|"
    lines = ["Weight w on the axis z-score, 1 − w on the NLI z-score; w = 1 is the axis alone, w = 0 the NLI stage alone. "
             "Bold: w = .5, the pre-declared axis + claim check score. Metrics as in the reports: Spearman against the rubric or human score, "
             "AUROC for binary labels. Every corpus except English-dev and Agora is spent; those curves are sensitivity analyses.", "",
             head, sep]
    for name, r in results["corpora"].items():
        cells = []
        for w in WEIGHTS:
            v = f"{r['curve'][str(w)]:.3f}"
            cells.append(f"**{v}**" if w == REFERENCE else v)
        lines.append(f"| {name} | {r['metric']} | {r['n_evaluated']} | " + " | ".join(cells) +
                     f" | {r['best_weight']} | {r['gain_of_best_over_reference']:+.3f} | {r['flat_band_0.3_0.7_range']:.3f} |")
    lines += ["", "Paired difference against w = .5 (cluster bootstrap, 95% CI). Rows: corpus; columns: weight.", "",
              "| Corpus | " + " | ".join(f"w={w}" for w in WEIGHTS if w != REFERENCE) + " |",
              "|---|" + "---|" * (len(WEIGHTS) - 1)]
    for name, r in results["corpora"].items():
        cells = []
        for w in WEIGHTS:
            if w == REFERENCE:
                continue
            p = r["paired_vs_reference"][str(w)]
            star = "*" if (p["ci95"][0] > 0 or p["ci95"][1] < 0) else ""
            cells.append(f"{p['delta']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]{star}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "`*` marks an interval that excludes zero."]
    (OUT / "sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_run_manifest(OUT, script="scripts/sweep_fusion_weight.py",
                       inputs={"carried_axis_english": R / "step4_windows_topk" / "grid" / "spw3_k8" / "scores_english.csv",
                               "nli_dev_english": R / "step6_nli" / "nli_only_k8" / "scores_english.csv",
                               "final_b4_english": R / "final" / "B4_axis" / "scores_english.csv",
                               "final_b6_eu": R / "final" / "B6_axis_plus_nli" / "scores_eu.csv",
                               "usgov": R / "external" / "usgov" / "final_scores_speeches.csv",
                               "tweets": R / "external" / "antielite_tweets" / "final_scores.csv",
                               "popbert": R / "external" / "popbert_en" / "final_scores.csv",
                               "portugal": R / "external" / "portugal" / "final_scores.csv"},
                       extra={"weights": WEIGHTS, "reference_weight": REFERENCE, "draws": args.draws})
    print("\n".join(lines))


if __name__ == "__main__":
    main()
