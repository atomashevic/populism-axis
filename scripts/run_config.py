#!/usr/bin/env python3
"""Build axes, project windows, aggregate and evaluate one configuration (CPU).

Input: stored raw embeddings (`results/embeddings/<encoder>/<setting>/`) and
anchor sentence embeddings. Output directory gets `config.json`,
`geometry.json`, `evaluation.json`, `scores_<dataset>.csv`,
`evidence_english.json` and `run_manifest.json`.

English split handling:
  dev      English rows with split == dev only (the development set).
  all      all 258 speeches; allowed once, as the declared reproduction check of the initial configuration.
  holdout  English holdout rows; requires --final.
Centering and z-scoring use only the windows of the documents being scored, so
dev scores never depend on holdout text. Every non-dev English access and
every EU access is appended to results/holdout_access_log.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from populism_axis.contextual_axis import (  # noqa: E402
    CHANNELS,
    WindowProjections,
    aggregate_scores,
    top_evidence_windows,
)
from populism_axis.run_manifest import write_run_manifest  # noqa: E402
from populism_axis.geometry import (  # noqa: E402
    apply_transform,
    axis_gram,
    center_normalize,
    difference_parallelism,
    fit_reference_transform,
    nuisance_directions,
    pole_mean_axes,
    residualize_axes,
    transformed_pole_axes,
)
from evaluate_benchmarks import evaluate_agora, evaluate_english, evaluate_eu  # noqa: E402

BENCH = ROOT / "data" / "benchmarks"
SPLIT_PATH = ROOT / "data" / "benchmarks" / "english_split.csv"
ACCESS_LOG = ROOT / "results" / "holdout_access_log.json"
FRAMES = {
    "english": "english_gpd_speeches.csv",
    "agora": "agora_paragraphs.csv",
    "eu": "eu_sentences.csv",
}
EVALUATORS = {"english": evaluate_english, "agora": evaluate_agora, "eu": evaluate_eu}
SYSTEMS = ("P", "E", "M", "PE", "PM", "EM", "PEM")
CARRIED_SYSTEM = "PE__frozen__soft"


def load_anchor_groups(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path)
    return {str(name): payload[str(name)].astype(np.float64) for name in payload["group_names"]}


def load_embeddings(run_dir: Path, dataset: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    payload = np.load(run_dir / f"embeddings_{dataset}.npz")
    windows = pd.read_parquet(run_dir / f"windows_{dataset}.parquet")
    return payload["embeddings"].astype(np.float64), payload["doc_index"].astype(int), windows


def select_documents(
    frame: pd.DataFrame, matrix: np.ndarray, doc_index: np.ndarray, windows: pd.DataFrame, keep: np.ndarray
):
    """Keep rows of `frame` where `keep` is True; re-index the window arrays accordingly."""
    old_to_new = -np.ones(len(frame), dtype=int)
    old_to_new[np.flatnonzero(keep)] = np.arange(int(keep.sum()))
    window_mask = keep[doc_index]
    return (
        frame[keep].reset_index(drop=True),
        matrix[window_mask],
        old_to_new[doc_index[window_mask]],
        windows[window_mask].reset_index(drop=True),
    )


def log_access(entry: dict) -> None:
    ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    log = json.loads(ACCESS_LOG.read_text(encoding="utf-8")) if ACCESS_LOG.exists() else []
    log.append(entry)
    ACCESS_LOG.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")


def score_systems(
    projections: WindowProjections, n_documents: int, top_k: int, top_rule: str | None = None
) -> dict[str, np.ndarray]:
    scores: dict[str, np.ndarray] = {}
    for system in SYSTEMS:
        weights = {channel: (1.0 if channel in system else 0.0) for channel in CHANNELS}
        scores[f"{system}__frozen__soft"] = aggregate_scores(
            projections, n_documents, weights, top_k=top_k, conjunction="soft", top_rule=top_rule
        )
    scores["PE__frozen__hard"] = aggregate_scores(
        projections, n_documents, {"P": 1.0, "E": 1.0, "M": 0.0}, top_k=top_k, conjunction="hard", top_rule=top_rule
    )
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-dir", action="append", default=[],
        help="per-dataset override, e.g. agora=results/embeddings/e5-large-v2/spw1_cap60",
    )
    parser.add_argument("--anchors", type=Path, action="append", required=True,
                        help="sentence npz from encode_sentences.py; repeat to pool anchor sets")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--datasets", default="english,agora")
    parser.add_argument("--english-split", choices=("dev", "all", "holdout"), default="dev")
    parser.add_argument("--transform", default="v3", choices=("v3", "ref"),
                        help="v3: pole-mean rule; ref: label-free reference transform applied to windows and anchors")
    parser.add_argument("--n-pcs", type=int, default=0, help="ref: top principal directions removed")
    parser.add_argument("--whiten-shrinkage", type=float, default=None, help="ref: ZCA whitening with this shrinkage")
    parser.add_argument("--nuisance", type=Path, default=None, help="nuisance sentence npz; residualize axes on it")
    parser.add_argument("--nuisance-sets", default="", help="comma-separated subset of nuisance sets (default: all)")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--top-rule", default=None, help="document pooling rule: k:<n>, frac:<f>, softcount:<q>, mean")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--final", action="store_true", help="required for holdout English or EU")
    parser.add_argument("--access-reason", default="")
    parser.add_argument("--save-projections", action="store_true", help="write projections_<dataset>.npz (gitignored)")
    args = parser.parse_args()

    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    dataset_dirs = {name: args.embeddings_dir for name in datasets}
    for item in args.dataset_dir:
        name, _, path = item.partition("=")
        dataset_dirs[name.strip()] = Path(path.strip())
    touches_holdout = ("english" in datasets and args.english_split != "dev") or "eu" in datasets
    if touches_holdout:
        if args.english_split == "holdout" or "eu" in datasets:
            if not args.final:
                raise SystemExit("held-out data (English holdout or EU) requires --final")
        if args.english_split == "all" and not args.access_reason:
            raise SystemExit("--english-split all needs --access-reason (declared reproduction check only)")

    if args.transform == "v3" and (args.n_pcs or args.whiten_shrinkage is not None or args.nuisance):
        raise SystemExit("--n-pcs, --whiten-shrinkage and --nuisance require --transform ref")
    groups: dict[str, np.ndarray] = {}
    for anchor_path in args.anchors:
        for name, values in load_anchor_groups(anchor_path).items():
            groups[name] = np.vstack([groups[name], values]) if name in groups else values
    nuisance_groups = load_anchor_groups(args.nuisance) if args.nuisance else None
    axes = pole_mean_axes(groups)
    geometry = {
        "anchors": [str(path) for path in args.anchors],
        "anchor_counts": {name: int(len(values)) for name, values in groups.items()},
        "transform": args.transform,
        "axis_rule": (
            "normalize(mean(unit populist) - mean(unit mainstream)); anchors not centered (pole-mean rule)"
            if args.transform == "v3"
            else "normalize(mean(T(populist)) - mean(T(mainstream))) with T fitted on the scored dataset's windows"
        ),
        "window_transform": (
            "center on scored documents' window mean, then L2-normalize (pole-mean rule)"
            if args.transform == "v3"
            else f"center, remove {args.n_pcs} PCs, whiten={args.whiten_shrinkage}, L2-normalize"
        ),
        "nuisance": str(args.nuisance) if args.nuisance else None,
        "per_dataset": {},
        "v3_rule_channel_axis_gram": axis_gram(axes),
        "difference_parallelism": {
            channel: difference_parallelism(groups[f"{channel}__populist"], groups[f"{channel}__mainstream"])
            for channel in CHANNELS
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        k: ([str(item) for item in v] if isinstance(v, list) else str(v) if isinstance(v, Path) else v)
        for k, v in vars(args).items()
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    evaluation: dict = {"label": args.label, "english_split": args.english_split, "top_k": args.top_k,
                        "top_rule": args.top_rule or f"k:{args.top_k}", "systems": {}}
    window_counts: dict[str, int] = {}
    inputs = {f"anchors_{index}": path for index, path in enumerate(args.anchors)}
    inputs["split"] = SPLIT_PATH
    if args.nuisance:
        inputs["nuisance"] = args.nuisance
    for dataset in datasets:
        frame = pd.read_csv(BENCH / FRAMES[dataset])
        matrix, doc_index, windows = load_embeddings(dataset_dirs[dataset], dataset)
        inputs[f"embeddings_{dataset}"] = dataset_dirs[dataset] / f"embeddings_{dataset}.npz"
        if dataset == "english":
            split = pd.read_csv(SPLIT_PATH).set_index("id")["split"]
            frame_split = frame["id"].map(split).to_numpy()
            if (pd.isna(frame_split)).any():
                raise SystemExit("english split file does not cover every benchmark row")
            keep = np.ones(len(frame), dtype=bool) if args.english_split == "all" else frame_split == args.english_split
            frame, matrix, doc_index, windows = select_documents(frame, matrix, doc_index, windows, keep)
        window_counts[dataset] = int(len(matrix))
        if args.transform == "v3":
            normalized, _ = center_normalize(matrix)
            dataset_axes = axes
        else:
            transform = fit_reference_transform(matrix, args.n_pcs, args.whiten_shrinkage)
            normalized = apply_transform(matrix, transform)
            dataset_axes = transformed_pole_axes(groups, transform)
            entry: dict = {"transform": transform.info, "channel_axis_gram_before_residualization": axis_gram(dataset_axes)}
            if nuisance_groups is not None:
                directions = nuisance_directions(nuisance_groups, transform)
                if args.nuisance_sets:
                    keep = {item.strip() for item in args.nuisance_sets.split(",") if item.strip()}
                    unknown = keep - set(directions)
                    if unknown:
                        raise SystemExit(f"unknown nuisance sets: {sorted(unknown)}")
                    directions = {name: vec for name, vec in directions.items() if name in keep}
                dataset_axes, diagnostics = residualize_axes(dataset_axes, directions)
                entry["residualization"] = diagnostics
                entry["nuisance_direction_gram"] = axis_gram(directions)
            entry["channel_axis_gram"] = axis_gram(dataset_axes)
            entry["cosine_to_v3_rule_axes"] = {c: float(dataset_axes[c] @ axes[c]) for c in CHANNELS}
            geometry["per_dataset"][dataset] = entry
        projections = WindowProjections(
            doc_index=doc_index,
            windows=windows["window"].tolist(),
            projections={channel: normalized @ dataset_axes[channel] for channel in CHANNELS},
        )
        if args.save_projections:
            np.savez_compressed(
                args.out_dir / f"projections_{dataset}.npz",
                doc_index=doc_index.astype(np.int32),
                **{channel: projections.projections[channel].astype(np.float32) for channel in CHANNELS},
            )
        scores = score_systems(
            projections, len(frame), args.top_k if dataset != "eu" else 1, args.top_rule if dataset != "eu" else None
        )
        for key, values in scores.items():
            evaluation["systems"].setdefault(key, {})[dataset] = EVALUATORS[dataset](values, frame, args.draws, args.seed)
        id_column = "id" if dataset == "english" else "row_id"
        out_scores = pd.DataFrame({"id": frame[id_column]})
        for key, values in scores.items():
            out_scores[key] = values
        out_scores.to_csv(args.out_dir / f"scores_{dataset}.csv", index=False)

        if dataset == "english":
            carried = scores[CARRIED_SYSTEM]
            order = np.argsort(-carried)[:5]
            evidence = []
            for position in order:
                rows = top_evidence_windows(projections, int(position), {"P": 1.0, "E": 1.0, "M": 0.0}, limit=3)
                evidence.append(
                    {
                        "id": str(frame.loc[position, "id"]),
                        "gpd_score": float(frame.loc[position, "source_score"]),
                        "score": float(carried[position]),
                        "windows": rows,
                    }
                )
            (args.out_dir / "evidence_english.json").write_text(
                json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
            )
        print(f"scored {dataset}: {len(frame)} documents, {len(matrix)} windows", flush=True)

    evaluation["window_counts"] = window_counts
    (args.out_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out_dir / "geometry.json").write_text(json.dumps(geometry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = write_run_manifest(
        args.out_dir,
        script="scripts/run_config.py",
        inputs=inputs,
        extra={"label": args.label, "english_split": args.english_split},
    )
    if touches_holdout:
        log_access(
            {
                "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "label": args.label,
                "datasets": datasets,
                "english_split": args.english_split,
                "final": args.final,
                "reason": args.access_reason,
                "out_dir": str(args.out_dir),
                "manifest": str(manifest),
            }
        )

    carried = evaluation["systems"].get(CARRIED_SYSTEM, {})
    line = [f"{args.label}: {CARRIED_SYSTEM}"]
    if "english" in carried:
        e = carried["english"]
        line.append(f"English({args.english_split}) rho={e['spearman']:+.4f} [{e['spearman_ci95'][0]:+.3f},{e['spearman_ci95'][1]:+.3f}]")
    if "agora" in carried:
        line.append(f"Agora rho={carried['agora']['spearman']:+.4f}")
    print("  ".join(line))


if __name__ == "__main__":
    main()
