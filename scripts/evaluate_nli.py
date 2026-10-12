#!/usr/bin/env python3
"""stage 6, stage 3 (CPU): evaluate the NLI second stage against the carried axis.

Per window: P_nli = max entailment probability over the P hypotheses, E_nli the
same over E; window NLI score = sqrt(P_nli * E_nli), a conjunction like the axis's.
Per document: mean over its top-K windows (K = 8, the carried pooling; K = 3 as
sensitivity). Combined = mean of the within-dataset z-scores of the carried axis
document score and the NLI document score (pre-declared in the development plan).

Rows written under results/step6_nli/<row>/ with `scores_<dataset>.csv` whose
`PE__frozen__soft` column holds that row's score, so compare_steps.py and
render_step_report.py work unchanged. The axis score stays the primary measure;
adoption applies to "combined vs axis-only".
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from populism_axis.run_manifest import write_run_manifest  # noqa: E402
from evaluate_benchmarks import evaluate_agora, evaluate_english  # noqa: E402

STEP = ROOT / "results" / "step6_nli"
BENCH = ROOT / "data" / "benchmarks"
SYSTEM = "PE__frozen__soft"


def zscore(values: np.ndarray) -> np.ndarray:
    std = float(values.std())
    return (values - values.mean()) / std if std > 1e-12 else np.zeros_like(values)


def document_nli(requests: pd.DataFrame, probs: np.ndarray, hypotheses: list[dict], n_docs: int, k: int) -> tuple[np.ndarray, pd.DataFrame]:
    entail = probs[:, :, 2].astype(np.float64)
    p_cols = [i for i, h in enumerate(hypotheses) if h["channel"] == "P"]
    e_cols = [i for i, h in enumerate(hypotheses) if h["channel"] == "E"]
    window = requests[["doc_index", "window_rank"]].copy()
    window["P_nli"] = entail[:, p_cols].max(axis=1)
    window["E_nli"] = entail[:, e_cols].max(axis=1)
    window["P_best"] = [hypotheses[p_cols[i]]["hypothesis_id"] for i in entail[:, p_cols].argmax(axis=1)]
    window["E_best"] = [hypotheses[e_cols[i]]["hypothesis_id"] for i in entail[:, e_cols].argmax(axis=1)]
    window["nli"] = np.sqrt(window["P_nli"] * window["E_nli"])
    kept = window[window["window_rank"] < k]
    doc = kept.groupby("doc_index")["nli"].mean().reindex(range(n_docs)).fillna(0.0).to_numpy()
    return doc, window


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carried", type=Path, default=ROOT / "results" / "step5_encoders" / "carried_config.json")
    parser.add_argument("--model-label", default="bart-large-mnli")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    carried = json.loads(args.carried.read_text(encoding="utf-8"))
    baseline = ROOT / carried["result_dir"]
    hypotheses = json.loads((STEP / "hypotheses.json").read_text(encoding="utf-8"))
    split = pd.read_csv(ROOT / "data" / "benchmarks" / "english_split.csv").set_index("id")["split"]

    frames = {
        "english": pd.read_csv(BENCH / "english_gpd_speeches.csv"),
        "agora": pd.read_csv(BENCH / "agora_paragraphs.csv"),
    }
    frames["english"] = frames["english"][frames["english"]["id"].map(split).eq("dev")].reset_index(drop=True)
    names = {"english": "english_dev", "agora": "agora"}
    evaluators = {"english": evaluate_english, "agora": evaluate_agora}

    rows = {"nli_only_k8": {}, "nli_only_k3": {}, "combined_k8": {}}
    windows_by_dataset = {}
    for dataset, frame in frames.items():
        requests = pd.read_parquet(STEP / f"requests_{names[dataset]}.parquet")
        payload = np.load(STEP / f"nli_{names[dataset]}__{args.model_label}.npz")
        if not np.array_equal(payload["request_id"], requests["request_id"].to_numpy()):
            raise SystemExit(f"{dataset}: NLI output does not match the request file")
        axis = pd.read_csv(baseline / f"scores_{dataset}.csv")
        id_column = "id" if dataset == "english" else "row_id"
        if list(axis["id"]) != list(frame[id_column]):
            raise SystemExit(f"{dataset}: axis scores and frame disagree on document order")
        axis_score = axis[SYSTEM].to_numpy(dtype=float)
        nli8, windows = document_nli(requests, payload["probs"], hypotheses, len(frame), 8)
        nli3, _ = document_nli(requests, payload["probs"], hypotheses, len(frame), 3)
        windows_by_dataset[dataset] = windows.assign(premise=requests["premise"].to_numpy(), doc_id=requests["doc_id"].to_numpy())
        scores = {"nli_only_k8": nli8, "nli_only_k3": nli3, "combined_k8": (zscore(axis_score) + zscore(nli8)) / 2}
        for row, values in scores.items():
            rows[row][dataset] = (values, frame, id_column)

    summary = {}
    for row, per_dataset in rows.items():
        out_dir = STEP / row
        out_dir.mkdir(parents=True, exist_ok=True)
        evaluation = {"label": row, "english_split": "dev", "systems": {SYSTEM: {}}}
        for dataset, (values, frame, id_column) in per_dataset.items():
            evaluation["systems"][SYSTEM][dataset] = evaluators[dataset](values, frame, args.draws, args.seed)
            pd.DataFrame({"id": frame[id_column], SYSTEM: values}).to_csv(out_dir / f"scores_{dataset}.csv", index=False)
        (out_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_run_manifest(out_dir, script="scripts/evaluate_nli.py",
                           inputs={"hypotheses": STEP / "hypotheses.json",
                                   "nli_english": STEP / f"nli_english_dev__{args.model_label}.npz",
                                   "nli_agora": STEP / f"nli_agora__{args.model_label}.npz",
                                   "axis_english": baseline / "scores_english.csv"},
                           extra={"row": row, "model": args.model_label})
        subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_steps.py"), "--baseline-dir", str(baseline),
                        "--candidate-dir", str(out_dir), "--out", str(out_dir / "paired_vs_previous.json")], check=True)
        e = evaluation["systems"][SYSTEM]
        summary[row] = {"english_rho": e["english"]["spearman"], "agora_rho": e["agora"]["spearman"],
                        "english_extreme_auroc": e["english"]["extreme_auroc"],
                        "agora_high_low_auroc": e["agora"]["high_vs_low_auroc"]}
        print(row, {k: round(v, 4) for k, v in summary[row].items()}, flush=True)

    # Diagnostics: window-level agreement between axis and NLI, hypothesis usage, evidence.
    diagnostics = {}
    for dataset, windows in windows_by_dataset.items():
        from scipy.stats import spearmanr

        requests = pd.read_parquet(STEP / f"requests_{names[dataset]}.parquet")
        diagnostics[dataset] = {
            "window_spearman_axis_vs_nli": float(spearmanr(requests["window_score"], windows["nli"]).statistic),
            "share_windows_P_entail_gt_0.5": float((windows["P_nli"] > 0.5).mean()),
            "share_windows_E_entail_gt_0.5": float((windows["E_nli"] > 0.5).mean()),
            "share_windows_both_gt_0.5": float(((windows["P_nli"] > 0.5) & (windows["E_nli"] > 0.5)).mean()),
            "best_P_hypothesis_counts": windows["P_best"].value_counts().to_dict(),
            "best_E_hypothesis_counts": windows["E_best"].value_counts().to_dict(),
        }
    combined = pd.read_csv(STEP / "combined_k8" / "scores_english.csv")
    english_windows = windows_by_dataset["english"]
    evidence = []
    for doc in combined[SYSTEM].to_numpy().argsort()[::-1][:5]:
        w = english_windows[english_windows["doc_index"] == doc].sort_values("nli", ascending=False).head(2)
        evidence.append({"id": frames["english"].loc[doc, "id"], "gpd_score": float(frames["english"].loc[doc, "source_score"]),
                         "windows": [{"premise": r.premise, "P_nli": float(r.P_nli), "E_nli": float(r.E_nli),
                                      "P_best": r.P_best, "E_best": r.E_best} for r in w.itertuples()]})
    (STEP / "diagnostics.json").write_text(json.dumps({"summary": summary, "diagnostics": diagnostics, "evidence_english_dev": evidence}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
