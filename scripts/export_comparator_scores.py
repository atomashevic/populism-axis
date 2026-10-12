#!/usr/bin/env python3
"""Export compact per-item score tables from the gitignored caches (CPU, seconds).

Five cells of the README benchmark table had no committed per-item scores: RoBERTa on
the English holdout, EU-OOD (AUROC and macro-F1) and AgoraSpeech, and the AgoraSpeech
axis + claim check score. This script reads the caches that hold them and writes ids
and scores only, so ``scripts/reproduce_table.py`` can recompute those cells:

- ``results/final/roberta/scores_english.csv``: English holdout, anti-elitism
  probability, mean of the top 3 windows (``scripts/final_holdout.py``).
- ``results/final/roberta/scores_eu.csv``: EU-OOD, window maximum of the populist
  total (1 - p0), anti-elitism and people-centrism probabilities.
- ``results/final/scores_agora.csv``: AgoraSpeech axis, claim check, combined score and
  RoBERTa populist total (1 - p0, window maximum).

The RoBERTa pooling is that of
``results/roberta-correction/rc-20261002T1415Z-sigmoid/compare_roberta.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from populism_axis.roberta_comparator import load_scores  # noqa: E402
from populism_axis.run_manifest import write_run_manifest  # noqa: E402
from evaluate_eu_paper_protocol import window_max  # noqa: E402

R = ROOT / "results"
SIGMOID = R / "roberta-correction" / "rc-20261002T1415Z-sigmoid"
SYSTEM = "PE__frozen__soft"
AGORA = {"axis": R / "step4_windows_topk" / "grid" / "spw3_k8" / "scores_agora.csv",
         "claim_check": R / "step6_nli" / "nli_only_k8" / "scores_agora.csv",
         "combined": R / "step6_nli" / "combined_k8" / "scores_agora.csv"}


def windows(name: str, n_documents: int) -> tuple[np.ndarray, np.ndarray]:
    payload = load_scores(SIGMOID / f"sigmoid_{name}.npz")
    doc_index, probs = payload["doc_index"].astype(int), payload["probabilities"].astype(float)
    if doc_index.max() >= n_documents:
        raise SystemExit(f"{name}: window index exceeds the {n_documents} documents of the benchmark table")
    return doc_index, probs


def write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    print(f"wrote {path.relative_to(ROOT)}: {len(frame)} rows")


def main() -> None:
    out = R / "final"

    split = pd.read_csv(ROOT / "data" / "benchmarks" / "english_split.csv").set_index("id")["split"]
    english = pd.read_csv(ROOT / "data" / "benchmarks" / "english_gpd_speeches.csv")
    is_holdout = english["id"].map(split).eq("holdout").to_numpy()
    doc_index, probs = windows("english", len(english))
    top3 = np.zeros(len(english))
    for i in range(len(english)):
        v = np.sort(probs[doc_index == i, 1])[::-1]
        top3[i] = v[:3].mean() if len(v) else 0.0
    holdout_ids = english.loc[is_holdout, "id"].to_numpy()
    if list(holdout_ids) != list(pd.read_csv(out / "B4_axis" / "scores_english.csv")["id"]):
        raise SystemExit("english: holdout order does not match results/final/B4_axis/scores_english.csv")
    write(pd.DataFrame({"id": holdout_ids, "RoBERTa_anti_elitism_top3": top3[is_holdout]}), out / "roberta" / "scores_english.csv")

    eu_ids = pd.read_csv(out / "B4_axis" / "scores_eu.csv")["id"].to_numpy()
    doc_index, probs = windows("eu", len(eu_ids))
    write(pd.DataFrame({"id": eu_ids,
                        "RoBERTa_populist_total": window_max(doc_index, 1.0 - probs[:, 0], len(eu_ids)),
                        "RoBERTa_anti_elitism": window_max(doc_index, probs[:, 1], len(eu_ids)),
                        "RoBERTa_people_centrism": window_max(doc_index, probs[:, 2], len(eu_ids))}),
          out / "roberta" / "scores_eu.csv")

    agora_ids = pd.read_csv(ROOT / "data" / "benchmarks" / "agora_paragraphs.csv", usecols=["row_id"])["row_id"].to_numpy()
    table = {"row_id": agora_ids}
    for column, path in AGORA.items():
        scores = pd.read_csv(path)
        if list(scores["id"]) != list(agora_ids):
            raise SystemExit(f"agora: {path.relative_to(ROOT)} does not match the benchmark table")
        table[column] = scores[SYSTEM].to_numpy(float)
    doc_index, probs = windows("agora", len(agora_ids))
    table["RoBERTa_populist_total"] = window_max(doc_index, 1.0 - probs[:, 0], len(agora_ids))
    write(pd.DataFrame(table), out / "scores_agora.csv")

    write_run_manifest(out, script="scripts/export_comparator_scores.py",
                       inputs={f"sigmoid_{n}": SIGMOID / f"sigmoid_{n}.npz" for n in ("english", "eu", "agora")}
                       | {f"agora_{k}": v for k, v in AGORA.items()},
                       extra={"agora_system": SYSTEM}, filename="export_comparator_scores.manifest.json")


if __name__ == "__main__":
    main()
