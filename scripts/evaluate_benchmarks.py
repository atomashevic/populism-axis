#!/usr/bin/env python3
"""Evaluate contextual P/E/M axis scores against the frozen benchmark.

The metric functions are the shared evaluation library for run_config.py,
final_holdout.py and the external comparisons. The command-line entry point
follows the same split rules as run_config.py:
  dev      English rows with split == dev only (the default).
  all      all English speeches; needs --access-reason.
  holdout  English holdout rows; requires --final.
EU is held out and requires --final. Every non-dev English access and every EU
access is appended to results/holdout_access_log.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from populism_axis.contextual_axis import CHANNELS, WindowProjections, aggregate_scores  # noqa: E402

SYSTEMS = ("P", "E", "M", "PE", "PM", "EM", "PEM")
ACCESS_LOG = ROOT / "results" / "holdout_access_log.json"

# The V2.1 inventory writes E with the pathology as its positive pole but P and
# M with the virtue as theirs. "frozen" reproduces the V2.1 implicit all-+1
# convention; "construct" orients every channel toward the populist pole, which
# means reading M as moral accusation rather than as professed virtue.
SIGN_SCHEMES = {
    "frozen": {"P": 1.0, "E": 1.0, "M": 1.0},
    "construct": {"P": 1.0, "E": 1.0, "M": -1.0},
}


def load_projections(path: Path) -> WindowProjections:
    payload = np.load(path)
    return WindowProjections(
        doc_index=payload["doc_index"].astype(int),
        windows=[],
        projections={channel: payload[channel].astype(float) for channel in CHANNELS},
    )


def select_documents(projections: WindowProjections, frame: pd.DataFrame, keep: np.ndarray) -> tuple[WindowProjections, pd.DataFrame]:
    """Keep only the documents flagged in `keep` and re-index their windows."""
    if projections.doc_index.max(initial=-1) >= len(frame):
        raise SystemExit("projections index more documents than the benchmark frame")
    window_mask = keep[projections.doc_index]
    old_to_new = np.cumsum(keep) - 1
    selected = WindowProjections(
        doc_index=old_to_new[projections.doc_index[window_mask]],
        windows=[],
        projections={channel: values[window_mask] for channel, values in projections.projections.items()},
    )
    return selected, frame.loc[keep].reset_index(drop=True)


def log_access(entry: dict) -> None:
    ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    log = json.loads(ACCESS_LOG.read_text(encoding="utf-8")) if ACCESS_LOG.exists() else []
    log.append(entry)
    ACCESS_LOG.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")


def cluster_bootstrap(
    metric_fn,
    values: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    index_by_group = {group: np.flatnonzero(groups == group) for group in unique}
    samples = []
    for _ in range(draws):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([index_by_group[group] for group in drawn])
        try:
            value = metric_fn(target[indices], values[indices])
        except ValueError:
            continue
        if value is not None and np.isfinite(value):
            samples.append(value)
    if len(samples) < 20:
        return float("nan"), float("nan")
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _spearman_metric(target: np.ndarray, values: np.ndarray) -> float:
    if len(np.unique(target)) < 2 or len(np.unique(values)) < 2:
        raise ValueError("degenerate")
    return float(spearmanr(values, target).statistic)


def _auroc_metric(target: np.ndarray, values: np.ndarray) -> float:
    if len(np.unique(target)) < 2:
        raise ValueError("degenerate")
    return float(roc_auc_score(target, values))


def evaluate_english(
    scores: np.ndarray,
    frame: pd.DataFrame,
    draws: int,
    seed: int,
) -> dict:
    target = frame["source_score"].to_numpy(dtype=float)
    groups = frame["leader"].to_numpy()
    rho = float(spearmanr(scores, target).statistic)
    low, high = cluster_bootstrap(_spearman_metric, scores, target, groups, draws, seed)

    mask = frame["extreme_label"].to_numpy() >= 0
    labels = frame.loc[mask, "extreme_label"].to_numpy(dtype=int)
    extreme_scores = scores[mask]
    extreme_groups = frame.loc[mask, "leader"].to_numpy()
    auroc = float(roc_auc_score(labels, extreme_scores))
    a_low, a_high = cluster_bootstrap(_auroc_metric, extreme_scores, labels, extreme_groups, draws, seed)

    # GPD's native unit: aggregate speech scores inside each leader-term.
    term_frame = frame.copy()
    term_frame["score"] = scores
    grouped = term_frame.groupby("leader_term_id").agg(
        score=("score", "mean"),
        label=("leader_term_score", "first"),
        leader=("leader", "first"),
    )
    grouped = grouped[grouped["label"].notna()]
    term_rho = float(spearmanr(grouped["score"], grouped["label"]).statistic)
    t_low, t_high = cluster_bootstrap(
        _spearman_metric,
        grouped["score"].to_numpy(dtype=float),
        grouped["label"].to_numpy(dtype=float),
        grouped["leader"].to_numpy(),
        draws,
        seed,
    )
    return {
        "spearman": rho,
        "spearman_ci95": [low, high],
        "extreme_auroc": auroc,
        "extreme_auroc_ci95": [a_low, a_high],
        "extreme_ap": float(average_precision_score(labels, extreme_scores)),
        "leader_term_spearman": term_rho,
        "leader_term_spearman_ci95": [t_low, t_high],
        "n_leader_terms": int(len(grouped)),
    }


def evaluate_eu(scores: np.ndarray, frame: pd.DataFrame, draws: int, seed: int) -> dict:
    mask = frame["pop_code"].isin([0, 3]).to_numpy()
    labels = (frame.loc[mask, "pop_code"] == 3).astype(int).to_numpy()
    subset = scores[mask]
    speakers = frame.loc[mask, "speaker"].to_numpy()
    auroc = float(roc_auc_score(labels, subset))
    low, high = cluster_bootstrap(_auroc_metric, subset, labels, speakers, draws, seed)

    # Within-speaker AUROC removes the speaker main effect entirely.
    within = []
    for speaker in np.unique(speakers):
        sel = speakers == speaker
        if len(np.unique(labels[sel])) == 2:
            within.append(
                {
                    "speaker": str(speaker),
                    "n": int(sel.sum()),
                    "n_positive": int(labels[sel].sum()),
                    "auroc": float(roc_auc_score(labels[sel], subset[sel])),
                }
            )
    weights = np.asarray([item["n_positive"] for item in within], dtype=float)
    pooled = float(np.average([item["auroc"] for item in within], weights=weights)) if within else float("nan")
    return {
        "auroc": auroc,
        "auroc_speaker_clustered_ci95": [low, high],
        "average_precision": float(average_precision_score(labels, subset)),
        "within_speaker_auroc": within,
        "pooled_within_speaker_auroc": pooled,
    }


def evaluate_agora(scores: np.ndarray, frame: pd.DataFrame, draws: int, seed: int) -> dict:
    target = frame["populism_human"].to_numpy(dtype=float)
    groups = frame["politician"].to_numpy()
    rho = float(spearmanr(scores, target).statistic)
    low, high = cluster_bootstrap(_spearman_metric, scores, target, groups, draws, seed)

    mask = frame["class_label"].isin(["low", "high"]).to_numpy()
    labels = (frame.loc[mask, "class_label"] == "high").astype(int).to_numpy()
    subset = scores[mask]
    auroc = float(roc_auc_score(labels, subset))

    within = []
    speakers = frame.loc[mask, "politician"].to_numpy()
    for speaker in np.unique(speakers):
        sel = speakers == speaker
        if len(np.unique(labels[sel])) == 2:
            within.append(
                {
                    "speaker": str(speaker),
                    "n": int(sel.sum()),
                    "n_positive": int(labels[sel].sum()),
                    "auroc": float(roc_auc_score(labels[sel], subset[sel])),
                }
            )
    weights = np.asarray([item["n_positive"] for item in within], dtype=float)
    pooled = float(np.average([item["auroc"] for item in within], weights=weights)) if within else float("nan")
    return {
        "spearman": rho,
        "spearman_ci95": [low, high],
        "high_vs_low_auroc": auroc,
        "within_speaker_high_vs_low_auroc": within,
        "pooled_within_speaker_auroc": pooled,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="e5-large-v2")
    parser.add_argument("--contextual-dir", type=Path, default=ROOT / "results" / "v3" / "contextual")
    parser.add_argument("--bench-dir", type=Path, default=ROOT / "data" / "benchmarks")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "v3" / "contextual")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--datasets", default="english,agora", help="comma-separated subset of english, eu, agora")
    parser.add_argument("--english-split", choices=("dev", "all", "holdout"), default="dev")
    parser.add_argument("--final", action="store_true", help="required for holdout English or EU")
    parser.add_argument("--access-reason", default="")
    args = parser.parse_args()

    files = {"english": "english_gpd_speeches.csv", "eu": "eu_sentences.csv", "agora": "agora_paragraphs.csv"}
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    unknown = sorted(set(datasets) - set(files))
    if unknown:
        raise SystemExit(f"unknown datasets: {unknown}")
    touches_holdout = ("english" in datasets and args.english_split != "dev") or "eu" in datasets
    if touches_holdout:
        if args.english_split == "holdout" or "eu" in datasets:
            if not args.final:
                raise SystemExit("held-out data (English holdout or EU) requires --final")
        if args.english_split == "all" and not args.access_reason:
            raise SystemExit("--english-split all needs --access-reason (declared reproduction check only)")

    base = args.contextual_dir / args.label
    evaluators = {"english": evaluate_english, "eu": evaluate_eu, "agora": evaluate_agora}

    results: dict = {"label": args.label, "english_split": args.english_split, "top_k": args.top_k, "systems": {}}
    score_store: dict[str, dict[str, np.ndarray]] = {}
    for dataset in datasets:
        projection_path = base / f"projections_{dataset}.npz"
        if not projection_path.exists():
            print(f"skipping {dataset}: no projections at {projection_path}", flush=True)
            continue
        frame = pd.read_csv(args.bench_dir / files[dataset])
        projections = load_projections(projection_path)
        if dataset == "english" and args.english_split != "all":
            split = pd.read_csv(args.bench_dir / "english_split.csv").set_index("id")["split"]
            frame_split = frame["id"].map(split).to_numpy()
            if (pd.isna(frame_split)).any():
                raise SystemExit("english split file does not cover every benchmark row")
            projections, frame = select_documents(projections, frame, frame_split == args.english_split)
        score_store[dataset] = {}
        for system in SYSTEMS:
            weights = {channel: (1.0 if channel in system else 0.0) for channel in CHANNELS}
            for scheme, signs in SIGN_SCHEMES.items():
                # Sign only matters for systems that actually contain M.
                if "M" not in system and scheme != "frozen":
                    continue
                for conjunction in ("hard", "soft"):
                    key = f"{system}__{scheme}__{conjunction}"
                    scores = aggregate_scores(
                        projections,
                        n_documents=len(frame),
                        weights=weights,
                        top_k=args.top_k if dataset != "eu" else 1,
                        conjunction=conjunction,
                        signs=signs,
                    )
                    score_store[dataset][key] = scores
                    metrics = evaluators[dataset](scores, frame, args.draws, args.seed)
                    results["systems"].setdefault(key, {})[dataset] = metrics
        print(f"scored {dataset}", flush=True)

    for dataset, store in score_store.items():
        pd.DataFrame(store).to_csv(base / f"scores_{dataset}.csv", index=False)

    out_path = args.out_dir / args.label / "evaluation.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if touches_holdout:
        log_access(
            {
                "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "script": "scripts/evaluate_benchmarks.py",
                "label": args.label,
                "datasets": datasets,
                "english_split": args.english_split,
                "final": args.final,
                "reason": args.access_reason,
                "out_path": str(out_path),
            }
        )

    print(f"\n{'system':24s} {'ENG rho':>9s} {'ENG CI':>18s} {'ENG AUROC':>10s} "
          f"{'EU AUROC':>9s} {'EU within':>10s} {'AG rho':>8s}")
    for key, payload in results["systems"].items():
        e = payload.get("english")
        u = payload.get("eu")
        a = payload.get("agora", {})
        eng = (f"{e['spearman']:+9.4f} [{e['spearman_ci95'][0]:+.3f},{e['spearman_ci95'][1]:+.3f}] {e['extreme_auroc']:10.4f}"
               if e else f"{'-':>9s} {'-':>18s} {'-':>10s}")
        eu_auroc = f"{u['auroc']:9.4f}" if u else f"{'-':>9s}"
        eu_within = f"{u['pooled_within_speaker_auroc']:10.4f}" if u else f"{'-':>10s}"
        ag = f"{a['spearman']:+8.4f}" if a else f"{'-':>8s}"
        print(f"{key:24s} {eng} {eu_auroc} {eu_within} {ag}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
