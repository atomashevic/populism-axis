#!/usr/bin/env python3
"""stage 3: minimal-pair anchors vs the carried proposition anchors (CPU).

Rows (all with the carried embeddings and the pole-mean axis rule):
  B1c        carried configuration, proposition anchors (already scored in step 1b)
  pairs      minimal-pair anchors (pre-declared primary)
  pooled     proposition + pair anchors pooled (sensitivity)
Outputs in results/step3_pair_anchors/: per-row run directories, paired
comparisons against B1c, `diagnostics.json` (window-level coupling of E and M
with VADER negativity; anchor QA: pair parallelism, per-pair cosine to its
axis, leave-one-pair-out axis stability, proposition-vs-pair axis cosine) and
`attribution_english_dev.json` (top windows of the top dev speeches with the
anchor pairs that contribute most to each window's projection).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from populism_axis.contextual_axis import _standardize  # noqa: E402
from populism_axis.run_manifest import write_run_manifest  # noqa: E402
from populism_axis.geometry import axis_gram, center_normalize, normalize, pole_mean_axes, unit_rows  # noqa: E402
from run_config import load_anchor_groups, load_embeddings, select_documents  # noqa: E402
from sweep_whitening import vader_negativity  # noqa: E402

STEP = ROOT / "results" / "step3_pair_anchors"
CARRIED = ROOT / "results" / "step2_whitening" / "carried_config.json"
PAIRS_JSON = ROOT / "anchors" / "pair_anchors.json"
PAIRS_NPZ = ROOT / "results" / "embeddings" / "e5-large-v2" / "sentences" / "v4_pairs_r1__24f70345.npz"
CHANNELS = ("P", "E", "M")


def run(label: str, anchors: list[Path], carried: dict) -> Path:
    out_dir = STEP / label
    command = [sys.executable, str(ROOT / "scripts" / "run_config.py"), "--label", label,
               "--embeddings-dir", str(ROOT / carried["embeddings_dirs"]["english"]),
               "--dataset-dir", f"agora={ROOT / carried['embeddings_dirs']['agora']}",
               "--out-dir", str(out_dir), "--save-projections"]
    for path in anchors:
        command += ["--anchors", str(path)]
    subprocess.run(command, check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_steps.py"),
                    "--baseline-dir", str(ROOT / carried["result_dir"]), "--candidate-dir", str(out_dir),
                    "--out", str(out_dir / "paired_vs_previous.json")], check=True)
    return out_dir


def anchor_qa(groups: dict[str, np.ndarray], pair_meta: dict) -> dict:
    out = {}
    for channel in CHANNELS:
        differences = unit_rows(groups[f"{channel}__populist"]) - unit_rows(groups[f"{channel}__mainstream"])
        axis = normalize(differences.mean(axis=0))
        unit_diff = unit_rows(differences)
        n = len(unit_diff)
        gram = unit_diff @ unit_diff.T
        loo = []
        for index in range(n):
            reduced = normalize(np.delete(differences, index, axis=0).mean(axis=0))
            loo.append(float(reduced @ axis))
        cos_to_axis = unit_diff @ axis
        ids = [pair["pair_id"] for pair in pair_meta[channel]]
        order = np.argsort(cos_to_axis)
        out[channel] = {
            "n_pairs": n,
            "pair_parallelism": float((gram.sum() - n) / (n * (n - 1))),
            "cosine_to_axis_quantiles": {str(q): float(np.quantile(cos_to_axis, q)) for q in (0.0, 0.1, 0.5, 0.9, 1.0)},
            "lowest_cosine_pairs": [{"pair_id": ids[i], "cosine": float(cos_to_axis[i])} for i in order[:5]],
            "min_leave_one_out_axis_cosine": float(min(loo)),
            "pairs_with_negative_cosine": [ids[i] for i in range(n) if cos_to_axis[i] < 0],
        }
    return out


def main() -> None:
    carried = json.loads(CARRIED.read_text(encoding="utf-8"))
    v3_npz = ROOT / carried["anchors"]
    STEP.mkdir(parents=True, exist_ok=True)
    rows = {
        "B1c_v3_anchors": ROOT / carried["result_dir"],
        "pairs_v4": run("pairs_v4", [PAIRS_NPZ], carried),
        "pooled_v3_v4": run("pooled_v3_v4", [v3_npz, PAIRS_NPZ], carried),
    }

    # The carried result dir has no saved projections; re-score it into the step folder for the diagnostic.
    reference = STEP / "B1c_v3_anchors_projections"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "run_config.py"), "--label", "B1c_reference",
                    "--embeddings-dir", str(ROOT / carried["embeddings_dirs"]["english"]),
                    "--dataset-dir", f"agora={ROOT / carried['embeddings_dirs']['agora']}",
                    "--anchors", str(v3_npz), "--out-dir", str(reference), "--save-projections", "--draws", "50"],
                   check=True)
    projection_dirs = {"B1c_v3_anchors": reference, "pairs_v4": rows["pairs_v4"], "pooled_v3_v4": rows["pooled_v3_v4"]}

    english_dir = ROOT / carried["embeddings_dirs"]["english"]
    agora_dir = ROOT / carried["embeddings_dirs"]["agora"]
    english = pd.read_csv(ROOT / "data" / "benchmarks" / "english_gpd_speeches.csv")
    split = pd.read_csv(ROOT / "data" / "benchmarks" / "english_split.csv").set_index("id")["split"]
    dev_mask_docs = english["id"].map(split).eq("dev").to_numpy()
    matrix, doc_index, windows = load_embeddings(english_dir, "english")
    dev_frame, dev_matrix, dev_doc_index, dev_windows = select_documents(english, matrix, doc_index, windows, dev_mask_docs)
    cache = ROOT / "results" / "step2_whitening" / "cache"
    neg_all = vader_negativity(english_dir / "windows_english.parquet", cache / "vader_neg_english.npz")
    neg = {"english": neg_all[dev_mask_docs[doc_index]],
           "agora": vader_negativity(agora_dir / "windows_agora.parquet", cache / "vader_neg_agora.npz")}

    diagnostics: dict = {"vader_neg_spearman": {}, "pe_window_score_vader_spearman": {}}
    for label, directory in projection_dirs.items():
        diagnostics["vader_neg_spearman"][label] = {}
        for dataset in ("english", "agora"):
            payload = np.load(directory / f"projections_{dataset}.npz")
            for channel in CHANNELS:
                diagnostics["vader_neg_spearman"][label][f"{dataset}_{channel}"] = float(
                    spearmanr(payload[channel], neg[dataset]).statistic)
            gate = np.minimum(_standardize(payload["P"]), _standardize(payload["E"]))
            diagnostics["pe_window_score_vader_spearman"].setdefault(label, {})[dataset] = float(
                spearmanr(gate, neg[dataset]).statistic)

    pair_json = json.loads(PAIRS_JSON.read_text(encoding="utf-8"))
    pair_meta = {channel: block["pairs"] for channel, block in pair_json["channels"].items()}
    v4_groups = load_anchor_groups(PAIRS_NPZ)
    v3_groups = load_anchor_groups(v3_npz)
    v4_axes, v3_axes = pole_mean_axes(v4_groups), pole_mean_axes(v3_groups)
    diagnostics["anchor_qa_v4"] = anchor_qa(v4_groups, pair_meta)
    diagnostics["v3_vs_v4_axis_cosine"] = {c: float(v3_axes[c] @ v4_axes[c]) for c in CHANNELS}
    diagnostics["channel_axis_gram_v4"] = axis_gram(v4_axes)
    diagnostics["channel_axis_gram_v3"] = axis_gram(v3_axes)
    (STEP / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")

    # Per-anchor attribution: projection of a centered unit window w on axis a = sum_i w.d_i / (n |mean d|).
    normalized, _ = center_normalize(dev_matrix)
    scores = pd.read_csv(rows["pairs_v4"] / "scores_english.csv").set_index("id")["PE__frozen__soft"]
    z = {c: _standardize(normalized @ v4_axes[c]) for c in ("P", "E")}
    gate = np.minimum(z["P"], z["E"])
    attribution = []
    for speech_id in scores.sort_values(ascending=False).index[:5]:
        position = int(np.flatnonzero(dev_frame["id"].to_numpy() == speech_id)[0])
        window_positions = np.flatnonzero(dev_doc_index == position)
        top = window_positions[np.argsort(-gate[window_positions])[:3]]
        entry = {"id": speech_id, "gpd_score": float(dev_frame.loc[position, "source_score"]),
                 "score": float(scores[speech_id]), "windows": []}
        for w in top:
            window_entry = {"text": dev_windows.loc[w, "window"], "z": {c: float(z[c][w]) for c in ("P", "E")}, "top_pairs": {}}
            for channel in ("P", "E"):
                differences = unit_rows(v4_groups[f"{channel}__populist"]) - unit_rows(v4_groups[f"{channel}__mainstream"])
                scale = len(differences) * np.linalg.norm(differences.mean(axis=0))
                contributions = differences @ normalized[w] / scale
                best = np.argsort(-contributions)[:3]
                window_entry["top_pairs"][channel] = [
                    {"pair_id": pair_meta[channel][i]["pair_id"], "populist": pair_meta[channel][i]["populist"],
                     "contribution": float(contributions[i])} for i in best]
            entry["windows"].append(window_entry)
        attribution.append(entry)
    (STEP / "attribution_english_dev.json").write_text(json.dumps(attribution, indent=2) + "\n", encoding="utf-8")

    summary = {}
    for label, directory in rows.items():
        evaluation = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))["systems"]
        summary[label] = {system: {d: evaluation[system][d]["spearman"] for d in ("english", "agora")}
                          for system in ("PE__frozen__soft", "P__frozen__soft", "E__frozen__soft", "M__frozen__soft", "PEM__frozen__soft")}
    (STEP / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_run_manifest(STEP, script="scripts/run_step3_anchors.py",
                       inputs={"pairs_json": PAIRS_JSON, "pairs_npz": PAIRS_NPZ, "v3_npz": v3_npz, "carried": CARRIED},
                       filename="step3.manifest.json")
    print(json.dumps(summary, indent=2))
    print(json.dumps({k: diagnostics[k] for k in ("vader_neg_spearman", "pe_window_score_vader_spearman", "v3_vs_v4_axis_cosine")}, indent=2))


if __name__ == "__main__":
    main()
