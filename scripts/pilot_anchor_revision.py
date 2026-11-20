#!/usr/bin/env python3
"""Anchor-revision pilot on the development sets (CPU): frozen V4 anchors against candidate anchor sets.

Only English-dev (174 speeches) and Agora (5,272 paragraphs) are scored. The English window file
holds all 258 speeches; the holdout rows are dropped on load. EU-OOD and the external benchmarks
are not read. Plan: research/development-log/anchor-revision-pilot-plan.md.

Subcommands:
  prepare   label-free geometry and hypothesis-rotation audit; axis scores for every arm, with
            V4 checked against the committed step-4 scores and step-6 windows; one deduplicated
            request file for the unmodified scripts/nli_stage.py.
  evaluate  remaps the BART-large-MNLI output to each arm, forms the combined score, checks that
            the V4 rerun reproduces the committed B6 scores, and writes paired differences
            against B6, the adoption-rule reading, NLI diagnostics, spot checks (window text only
            under the gitignored disagreement/), results.json and REPORT.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from claim_check_arms import tail_counts  # noqa: E402
from evaluate_benchmarks import evaluate_agora, evaluate_english  # noqa: E402
from evaluate_nli import zscore  # noqa: E402
from paired_comparison import paired_delta  # noqa: E402
from populism_axis.contextual_axis import (  # noqa: E402
    CHANNELS,
    WindowProjections,
    aggregate_from_window_scores,
    window_conjunction_scores,
)
from populism_axis.geometry import (  # noqa: E402
    axis_gram,
    center_normalize,
    difference_parallelism,
    normalize,
    pole_mean_axes,
    unit_rows,
)
from populism_axis.run_manifest import file_sha256, write_run_manifest  # noqa: E402
from populism_axis.sentences import load_sentence_groups  # noqa: E402
from run_config import BENCH, SPLIT_PATH, load_anchor_groups, load_embeddings, select_documents  # noqa: E402
from sweep_whitening import vader_negativity  # noqa: E402

DEFAULT_OUT_DIR = ROOT / "results" / "anchor_revision_pilot"
CANDIDATE_DIR = ROOT / "anchors" / "candidates"
DEFAULT_ARMS = "v5_green,v5_red,v5_green_red"

V4 = "v4"
V4_ANCHORS_JSON = ROOT / "anchors" / "proposition_anchors.json"
V4_ANCHORS_NPZ = ROOT / "results" / "embeddings" / "e5-large-v2" / "sentences" / "v3_proposition__ea98e5b2.npz"
V4_TOP5_P = ["P00", "P05", "P01", "P06", "P08"]
V4_TOP5_E = ["E05", "E00", "E02", "E04", "E06"]

EMB_MAIN = ROOT / "results" / "embeddings" / "e5-large-v2" / "spw3_cap60_clean"
STEP4 = ROOT / "results" / "step4_windows_topk" / "grid" / "spw3_k8"
STEP6 = ROOT / "results" / "step6_nli"

DATASETS = ("english_dev", "agora")
SCORE_FILE = {"english_dev": "scores_english.csv", "agora": "scores_agora.csv"}
ID_COLUMN = {"english_dev": "id", "agora": "row_id"}
EVALUATORS = {"english_dev": evaluate_english, "agora": evaluate_agora}
TARGET = {"english_dev": ("source_score", "leader"), "agora": ("populism_human", "politician")}
SYSTEM = "PE__frozen__soft"
RULE = "k:8"
TOP_K = 8
REQUESTS_NAME = "dev_dedup"
NLI_LABEL = "bart-large-mnli"

# Spearman of the frozen configuration on the development sets (README, development log step 6).
REPORTED = {
    "axis": {"english_dev": 0.517, "agora": 0.359},
    "combined": {"english_dev": 0.537, "agora": 0.382},
}
CHECK_TOLERANCE = 0.001
CI_DRAWS = 2000
PAIRED_DRAWS = 4000

# docs/development-protocol.md, adoption rule.
ADOPT_ENGLISH_DEV_MIN = 0.01
ADOPT_AGORA_MIN = -0.01

SPOTCHECK_EXACT = ("UK_Brown_Campaign_1", "Philippines_Arroyo_Famous_2")
SPOTCHECK_PREFIX = "South Africa_Mbeki_"

# Each stage of a candidate arm is compared with this stage of the committed V4 scores.
REFERENCE_STAGE = {
    "axis": "axis",
    "nli_only": "nli_only",
    "combined": "combined",
    "nli_only_v4hyp": "nli_only",
    "combined_v4hyp": "combined",
}


# ---------------------------------------------------------------------------
# Helpers covered by tests/test_pilot_anchor_revision.py
# ---------------------------------------------------------------------------


def adoption_reading(english_dev: Mapping[str, Any], agora: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the development protocol's adoption rule to two paired differences against B6.

    Passes when the English-dev difference is at least +.01, the Agora difference is at least
    -.01, and neither 95% interval lies entirely below 0.
    """
    d_en = float(english_dev["delta_spearman"])
    d_ag = float(agora["delta_spearman"])
    conditions = {
        "english_dev_delta_at_least_+.01": d_en >= ADOPT_ENGLISH_DEV_MIN,
        "agora_delta_at_least_-.01": d_ag >= ADOPT_AGORA_MIN,
        "english_dev_ci_not_below_0": float(english_dev["ci95"][1]) >= 0.0,
        "agora_ci_not_below_0": float(agora["ci95"][1]) >= 0.0,
    }
    return {
        "conditions": conditions,
        "passes": bool(all(conditions.values())),
        "mean_delta": (d_en + d_ag) / 2.0,
    }


def deduplicate_nli_requests(
    window_rows: pd.DataFrame,
    arm_hypotheses: Mapping[str, list[dict]],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], dict[str, list[dict]]]:
    """Deduplicate premise strings and build union hypothesis table.

    Args:
        window_rows: DataFrame with columns
            ["arm", "dataset", "doc_index", "doc_id", "window_rank", "premise"]
            (optional extra columns such as "window_position", "window_score" preserved).
        arm_hypotheses: Mapping arm_name -> list of hypothesis dicts
            (each with "hypothesis_id", "channel" in {"P", "E"}, "text").

    Returns:
        requests_df: DataFrame["request_id", "premise"] with unique premises.
        arm_window_map: DataFrame without "premise", with "request_id" mapped.
        union_hypotheses: list of dicts [{"hypothesis_id": "U_P00", "channel": "P", "text": ...}, ...].
        mapped_arm_hypotheses: copy of arm_hypotheses where each item has
            "union_hypothesis_id" and "union_index".
    """
    required = {"arm", "dataset", "doc_index", "doc_id", "window_rank", "premise"}
    missing = required - set(window_rows.columns)
    if missing:
        raise ValueError(f"window_rows missing required columns: {sorted(missing)}")

    premises = window_rows["premise"].astype(str).tolist()
    unique_premises = list(dict.fromkeys(premises))
    premise_to_id = {text: idx for idx, text in enumerate(unique_premises)}
    requests_df = pd.DataFrame(
        {
            "request_id": np.arange(len(unique_premises), dtype=np.int64),
            "premise": pd.Series(unique_premises, dtype="string"),
        }
    )

    arm_window_map = window_rows.drop(columns=["premise"]).copy().reset_index(drop=True)
    arm_window_map["arm"] = arm_window_map["arm"].astype(str)
    arm_window_map["dataset"] = arm_window_map["dataset"].astype(str)
    arm_window_map["doc_index"] = arm_window_map["doc_index"].astype(np.int64)
    arm_window_map["doc_id"] = arm_window_map["doc_id"].astype(str)
    arm_window_map["window_rank"] = arm_window_map["window_rank"].astype(np.int64)
    arm_window_map["request_id"] = np.asarray([premise_to_id[p] for p in premises], dtype=np.int64)

    union_hypotheses: list[dict] = []
    key_to_union: dict[tuple[str, str], tuple[int, str]] = {}
    for channel in ("P", "E"):
        ch_idx = 0
        for arm in arm_hypotheses:
            for hyp in arm_hypotheses[arm]:
                if hyp["channel"] != channel:
                    continue
                text = str(hyp["text"])
                key = (channel, text)
                if key not in key_to_union:
                    uid = f"U_{channel}{ch_idx:02d}"
                    u_idx = len(union_hypotheses)
                    key_to_union[key] = (u_idx, uid)
                    union_hypotheses.append(
                        {"hypothesis_id": uid, "channel": channel, "text": text}
                    )
                    ch_idx += 1

    mapped_arm_hypotheses: dict[str, list[dict]] = {}
    for arm, hyps in arm_hypotheses.items():
        mapped_list = []
        for hyp in hyps:
            channel = str(hyp["channel"])
            text = str(hyp["text"])
            u_idx, uid = key_to_union[(channel, text)]
            item = dict(hyp)
            item["union_hypothesis_id"] = uid
            item["union_index"] = int(u_idx)
            mapped_list.append(item)
        mapped_arm_hypotheses[arm] = mapped_list

    return requests_df, arm_window_map, union_hypotheses, mapped_arm_hypotheses


def remap_arm_nli_scores(
    arm_window_map: pd.DataFrame,
    request_ids: np.ndarray,
    entail_probs: np.ndarray,
    union_hypothesis_ids: list[str],
    arm_hyps: list[dict],
    doc_indices: Sequence[int] | np.ndarray,
    k: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Remap deduplicated NLI entailment probabilities to one arm's windows & documents.

    Args:
        arm_window_map: Subset of arm_window_map for one (arm, dataset).
        request_ids: 1D array of request_ids matching rows of entail_probs.
        entail_probs: 2D array (n_unique_requests, n_union_hypotheses) of entailment probs (probs[:, :, 2]).
        union_hypothesis_ids: Ordered hypothesis_ids matching columns of entail_probs.
        arm_hyps: Arm's hypothesis dicts (5 P, 5 E), each referencing union_hypothesis_id.
        doc_indices: Ordered sequence of doc_index values for the output document arrays.
        k: Number of top windows (window_rank < k) to average per document.

    Returns:
        doc_scores: {"P_nli": np.ndarray, "E_nli": np.ndarray, "PE_nli": np.ndarray}
            aligned to doc_indices (missing docs filled with 0.0).
        window_df: DataFrame with columns
            ["doc_index", "doc_id", "window_rank", "request_id",
             "P_nli", "E_nli", "PE_nli", "P_best", "E_best"].
    """
    entail = np.asarray(entail_probs, dtype=np.float64)
    req_arr = np.asarray(request_ids, dtype=np.int64)
    if entail.ndim != 2 or entail.shape[0] != len(req_arr) or entail.shape[1] != len(union_hypothesis_ids):
        raise ValueError(
            f"entail_probs shape {entail.shape} incompatible with {len(req_arr)} requests and {len(union_hypothesis_ids)} hypotheses"
        )
    doc_idx_arr = np.asarray(doc_indices, dtype=np.int64)
    cols_out = ["doc_index", "doc_id", "window_rank", "request_id", "P_nli", "E_nli", "PE_nli", "P_best", "E_best"]
    if len(arm_window_map) == 0:
        empty_scores = {c: np.zeros(len(doc_idx_arr), dtype=np.float64) for c in ("P_nli", "E_nli", "PE_nli")}
        return empty_scores, pd.DataFrame(columns=cols_out)

    uid_to_col = {str(uid): idx for idx, uid in enumerate(union_hypothesis_ids)}
    req_to_row = {int(rid): idx for idx, rid in enumerate(req_arr)}
    p_hyps = [h for h in arm_hyps if h["channel"] == "P"]
    e_hyps = [h for h in arm_hyps if h["channel"] == "E"]
    if not p_hyps or not e_hyps:
        raise ValueError("arm_hyps must contain both P and E hypotheses")

    p_cols = [uid_to_col[str(h["union_hypothesis_id"])] for h in p_hyps]
    e_cols = [uid_to_col[str(h["union_hypothesis_id"])] for h in e_hyps]

    req_vals = arm_window_map["request_id"].to_numpy(dtype=np.int64)
    row_pos = np.asarray([req_to_row[int(rid)] for rid in req_vals], dtype=np.int64)
    sub_entail = entail[row_pos]

    p_mat = sub_entail[:, p_cols]
    e_mat = sub_entail[:, e_cols]
    p_nli = p_mat.max(axis=1)
    e_nli = e_mat.max(axis=1)
    pe_nli = np.sqrt(np.clip(p_nli, 0.0, None) * np.clip(e_nli, 0.0, None))
    p_best = [str(p_hyps[int(idx)]["hypothesis_id"]) for idx in p_mat.argmax(axis=1)]
    e_best = [str(e_hyps[int(idx)]["hypothesis_id"]) for idx in e_mat.argmax(axis=1)]

    window_df = pd.DataFrame(
        {
            "doc_index": arm_window_map["doc_index"].to_numpy(dtype=np.int64),
            "doc_id": arm_window_map["doc_id"].astype(str).to_numpy(),
            "window_rank": arm_window_map["window_rank"].to_numpy(dtype=np.int64),
            "request_id": req_vals,
            "P_nli": p_nli,
            "E_nli": e_nli,
            "PE_nli": pe_nli,
            "P_best": p_best,
            "E_best": e_best,
        }
    )

    kept = window_df[window_df["window_rank"].to_numpy() < int(k)]
    doc_scores: dict[str, np.ndarray] = {}
    for col in ("P_nli", "E_nli", "PE_nli"):
        if len(kept):
            grouped = kept.groupby("doc_index")[col].mean()
            doc_scores[col] = grouped.reindex(doc_idx_arr).fillna(0.0).to_numpy(dtype=np.float64)
        else:
            doc_scores[col] = np.zeros(len(doc_idx_arr), dtype=np.float64)

    return doc_scores, window_df


# ---------------------------------------------------------------------------
# Arms and data
# ---------------------------------------------------------------------------


def _resolve_arms(names: str, emb_dir: Path) -> dict[str, tuple[Path, Path]]:
    """V4 plus the named candidates; each candidate's embeddings must match its current text."""
    arms: dict[str, tuple[Path, Path]] = {V4: (V4_ANCHORS_JSON, V4_ANCHORS_NPZ)}
    for name in [n.strip() for n in names.split(",") if n.strip()]:
        if name in arms:
            raise SystemExit(f"arm named twice or reserved: {name}")
        json_path = CANDIDATE_DIR / f"{name}.json"
        if not json_path.is_file():
            raise SystemExit(f"missing candidate anchors: {json_path}")
        npz_path = emb_dir / f"{name}__{file_sha256(json_path)[:8]}.npz"
        if not npz_path.is_file():
            raise SystemExit(
                f"no embeddings for the current text of {json_path} (expected {npz_path}); "
                f"encode with scripts/encode_sentences.py --set-id {name}"
            )
        arms[name] = (json_path, npz_path)
    if len(arms) < 2:
        raise SystemExit("no candidate arms given")
    return arms


def _load_frame(dataset: str) -> pd.DataFrame:
    """Benchmark rows of one development set, in the order of the stored scores."""
    if dataset == "english_dev":
        frame = pd.read_csv(BENCH / "english_gpd_speeches.csv")
        split = frame["id"].astype(str).map(pd.read_csv(SPLIT_PATH).set_index("id")["split"])
        if split.isna().any():
            raise SystemExit("english_split.csv does not cover every benchmark row")
        return frame[split.eq("dev").to_numpy()].reset_index(drop=True)
    if dataset == "agora":
        return pd.read_csv(BENCH / "agora_paragraphs.csv")
    raise ValueError(f"not a development set: {dataset}")


def _load_dataset(dataset: str) -> dict[str, Any]:
    """Frame, centered and normalized window embeddings, doc_index and window text."""
    if dataset == "english_dev":
        full = pd.read_csv(BENCH / "english_gpd_speeches.csv")
        split = full["id"].astype(str).map(pd.read_csv(SPLIT_PATH).set_index("id")["split"])
        if split.isna().any():
            raise SystemExit("english_split.csv does not cover every benchmark row")
        matrix, doc_index, windows = load_embeddings(EMB_MAIN, "english")
        frame, matrix, doc_index, windows = select_documents(
            full, matrix, doc_index, windows, split.eq("dev").to_numpy()
        )
    elif dataset == "agora":
        frame = pd.read_csv(BENCH / "agora_paragraphs.csv")
        matrix, doc_index, windows = load_embeddings(EMB_MAIN, "agora")
    else:
        raise ValueError(f"not a development set: {dataset}")
    if int(doc_index.max()) + 1 != len(frame):
        raise SystemExit(f"{dataset}: window doc_index does not cover the frame")
    normalized, _ = center_normalize(matrix)
    return {"frame": frame, "normalized": normalized, "doc_index": doc_index, "windows": windows}


def _score_arm(axes: Mapping[str, np.ndarray], data: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """PE window scores (soft conjunction) and document scores (mean of the top 8 windows)."""
    doc_index = data["doc_index"]
    n_docs = len(data["frame"])
    proj = {c: data["normalized"] @ axes[c] for c in CHANNELS}
    wp = WindowProjections(doc_index=doc_index, windows=[], projections=proj)
    pe_win = window_conjunction_scores(wp, {"P": 1.0, "E": 1.0, "M": 0.0}, conjunction="soft")
    docs = {
        "PE": aggregate_from_window_scores(pe_win, doc_index, n_docs, RULE),
        "P": aggregate_from_window_scores(zscore(proj["P"]), doc_index, n_docs, RULE),
        "E": aggregate_from_window_scores(zscore(proj["E"]), doc_index, n_docs, RULE),
    }
    return pe_win, docs


def _top_windows_rows(
    arm: str,
    dataset: str,
    doc_ids: Sequence[str],
    doc_index: np.ndarray,
    pe_window_scores: np.ndarray,
    windows: pd.DataFrame,
    k: int,
) -> list[dict]:
    """Top-k PE windows of every document, in the stable order of prepare_nli_requests.py."""
    order = np.lexsort((-pe_window_scores, doc_index))
    sorted_docs = doc_index[order]
    win_texts = windows["window"].astype(str).tolist()
    rows: list[dict] = []
    for doc in range(len(doc_ids)):
        lo = int(np.searchsorted(sorted_docs, doc))
        hi = int(np.searchsorted(sorted_docs, doc, side="right"))
        for rank, position in enumerate(order[lo:hi][:k]):
            pos = int(position)
            rows.append(
                {
                    "arm": arm,
                    "dataset": dataset,
                    "doc_index": int(doc),
                    "doc_id": str(doc_ids[doc]),
                    "window_rank": int(rank),
                    "window_position": pos,
                    "window_score": float(pe_window_scores[pos]),
                    "premise": win_texts[pos],
                }
            )
    return rows


def _load_v4_committed() -> dict[str, dict[str, np.ndarray]]:
    """Committed PE scores of the frozen configuration: axis (B4), claim check and combined (B6)."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for dataset in DATASETS:
        name = SCORE_FILE[dataset]
        axis = pd.read_csv(STEP4 / name)
        nli = pd.read_csv(STEP6 / "nli_only_k8" / name)
        combined = pd.read_csv(STEP6 / "combined_k8" / name)
        frame_ids = _load_frame(dataset)[ID_COLUMN[dataset]].astype(str).tolist()
        for label, table in (("axis", axis), ("nli_only", nli), ("combined", combined)):
            if table["id"].astype(str).tolist() != frame_ids:
                raise SystemExit(f"{dataset}: committed {label} scores and the benchmark disagree on order")
        out[dataset] = {
            "axis": axis[SYSTEM].to_numpy(dtype=np.float64),
            "nli_only": nli[SYSTEM].to_numpy(dtype=np.float64),
            "combined": combined[SYSTEM].to_numpy(dtype=np.float64),
        }
    return out


# ---------------------------------------------------------------------------
# Label-free audit: geometry, negativity, hypothesis rotation
# ---------------------------------------------------------------------------


def _leave_one_sentence_out_min_cosine(pop: np.ndarray, main: np.ndarray, full_axis: np.ndarray) -> float:
    u_pop, u_main = unit_rows(pop), unit_rows(main)
    cosines: list[float] = []
    for idx in range(len(u_pop)):
        axis_loo = normalize(np.delete(u_pop, idx, axis=0).mean(axis=0) - u_main.mean(axis=0))
        cosines.append(float(axis_loo @ full_axis))
    for idx in range(len(u_main)):
        axis_loo = normalize(u_pop.mean(axis=0) - np.delete(u_main, idx, axis=0).mean(axis=0))
        cosines.append(float(axis_loo @ full_axis))
    return float(min(cosines))


def _top_hypotheses_for_arm(
    groups: Mapping[str, np.ndarray],
    axes: Mapping[str, np.ndarray],
    sentences: Mapping[str, list[str]],
    per_channel: int = 5,
) -> tuple[list[dict], dict[str, list[dict]]]:
    selected: list[dict] = []
    full_ranking: dict[str, list[dict]] = {}
    for channel in ("P", "E"):
        cosines = unit_rows(groups[f"{channel}__populist"]) @ axes[channel]
        order = np.argsort(-cosines)
        full_ranking[channel] = [
            {
                "hypothesis_id": f"{channel}{int(idx):02d}",
                "cosine_to_axis": float(cosines[idx]),
            }
            for idx in order
        ]
        for idx in order[:per_channel]:
            selected.append(
                {
                    "hypothesis_id": f"{channel}{int(idx):02d}",
                    "channel": channel,
                    "text": sentences[f"{channel}__populist"][int(idx)],
                    "cosine_to_axis": float(cosines[idx]),
                }
            )
    return selected, full_ranking


def _vader_neg_arrays(out_dir: Path) -> dict[str, np.ndarray]:
    cache_dir = out_dir / "cache"
    eng_cache = cache_dir / "vader_neg_english_spw3.npz"
    ag_cache = cache_dir / "vader_neg_agora_spw3.npz"

    english = pd.read_csv(BENCH / "english_gpd_speeches.csv")
    split = pd.read_csv(SPLIT_PATH).set_index("id")["split"]
    dev_docs = english["id"].astype(str).map(split).eq("dev").to_numpy()
    _, doc_index_all, _ = load_embeddings(EMB_MAIN, "english")
    neg_eng_all = vader_negativity(EMB_MAIN / "windows_english.parquet", eng_cache)
    neg_agora = vader_negativity(EMB_MAIN / "windows_agora.parquet", ag_cache)
    return {
        "english_dev": neg_eng_all[dev_docs[doc_index_all]],
        "agora": neg_agora,
    }


def _revised_slot_projections(
    v4_groups: Mapping[str, np.ndarray],
    v4_sentences: Mapping[str, list[str]],
    groups: Mapping[str, np.ndarray],
    sentences: Mapping[str, list[str]],
    v4_axes: Mapping[str, np.ndarray],
) -> list[dict]:
    """For each revised P or E sentence: its projection and the replaced sentence's projection on
    the V4 axis of its own channel built without that slot, and on the other channel's V4 axis.

    Contrast sentences are expected on the negative side of their own axis.
    """
    rows: list[dict] = []
    for channel, other in (("P", "E"), ("E", "P")):
        pop = unit_rows(v4_groups[f"{channel}__populist"])
        main = unit_rows(v4_groups[f"{channel}__mainstream"])
        for pole in ("populist", "mainstream"):
            key = f"{channel}__{pole}"
            old_unit = unit_rows(v4_groups[key])
            new_unit = unit_rows(groups[key])
            for idx, (old, new) in enumerate(zip(v4_sentences[key], sentences[key])):
                if old == new:
                    continue
                if pole == "populist":
                    own = normalize(np.delete(pop, idx, axis=0).mean(axis=0) - main.mean(axis=0))
                else:
                    own = normalize(pop.mean(axis=0) - np.delete(main, idx, axis=0).mean(axis=0))
                rows.append(
                    {
                        "slot": f"{channel}{'+' if pole == 'populist' else '-'}{idx:02d}",
                        "own_axis_v4_sentence": float(old_unit[idx] @ own),
                        "own_axis_revised_sentence": float(new_unit[idx] @ own),
                        "other_axis_v4_sentence": float(old_unit[idx] @ v4_axes[other]),
                        "other_axis_revised_sentence": float(new_unit[idx] @ v4_axes[other]),
                        "cosine_v4_revised": float(old_unit[idx] @ new_unit[idx]),
                    }
                )
    return rows


def _rotation_entry(top_hyps: list[dict], full_ranking: dict, v4_sentences: Mapping[str, list[str]]) -> dict:
    entry: dict[str, Any] = {}
    for channel, v4_top in (("P", V4_TOP5_P), ("E", V4_TOP5_E)):
        ids = [h["hypothesis_id"] for h in top_hyps if h["channel"] == channel]
        ranked = full_ranking[channel]
        entry[f"top5_{channel}"] = ids
        entry[f"top5_{channel}_vs_v4"] = {
            "retained_ids": [x for x in ids if x in v4_top],
            "added_ids": [x for x in ids if x not in v4_top],
            "dropped_ids": [x for x in v4_top if x not in ids],
            "same_set_as_v4": set(ids) == set(v4_top),
            "ids_with_revised_text": [
                h["hypothesis_id"]
                for h in top_hyps
                if h["channel"] == channel
                and h["text"] != v4_sentences[f"{channel}__populist"][int(h["hypothesis_id"][1:])]
            ],
            "cosine_gap_rank5_rank6": float(ranked[4]["cosine_to_axis"] - ranked[5]["cosine_to_axis"]),
        }
    entry["full_ranking"] = full_ranking
    return entry


def _geometry_audit(
    arms: Mapping[str, tuple[Path, Path]],
    out_dir: Path,
    data: Mapping[str, Mapping[str, Any]],
) -> tuple[dict, dict, dict[str, dict[str, np.ndarray]], dict[str, list[dict]], dict[str, dict]]:
    neg = _vader_neg_arrays(out_dir)
    arm_groups = {arm: load_anchor_groups(npz_path) for arm, (_, npz_path) in arms.items()}
    arm_sentences = {arm: load_sentence_groups(json_path)[1] for arm, (json_path, _) in arms.items()}
    arm_axes = {arm: pole_mean_axes(groups) for arm, groups in arm_groups.items()}
    v4_axes = arm_axes[V4]

    geometry: dict[str, Any] = {}
    selected_hyps: dict[str, list[dict]] = {}
    rotation: dict[str, dict] = {}
    for arm, axes in arm_axes.items():
        groups = arm_groups[arm]
        vader: dict[str, float] = {}
        pe_vader: dict[str, float] = {}
        for dataset in DATASETS:
            normalized = data[dataset]["normalized"]
            if len(neg[dataset]) != len(normalized):
                raise SystemExit(f"{dataset}: VADER cache and window embeddings differ in length")
            proj = {c: normalized @ axes[c] for c in CHANNELS}
            for c in CHANNELS:
                vader[f"{dataset}_{c}"] = float(spearmanr(proj[c], neg[dataset]).statistic)
            wp = WindowProjections(doc_index=np.zeros(len(normalized), dtype=int), windows=[], projections=proj)
            pe_win = window_conjunction_scores(wp, {"P": 1.0, "E": 1.0, "M": 0.0}, conjunction="soft")
            pe_vader[dataset] = float(spearmanr(pe_win, neg[dataset]).statistic)
        geometry[arm] = {
            "cosine_to_v4": {c: float(axes[c] @ v4_axes[c]) for c in CHANNELS},
            "cross_channel_cosine_PE": float(axes["P"] @ axes["E"]),
            "channel_axis_gram": axis_gram(axes),
            "leave_one_out_min_cosine": {
                c: _leave_one_sentence_out_min_cosine(groups[f"{c}__populist"], groups[f"{c}__mainstream"], axes[c])
                for c in CHANNELS
            },
            "difference_parallelism": {
                c: difference_parallelism(groups[f"{c}__populist"], groups[f"{c}__mainstream"]) for c in CHANNELS
            },
            "vader_neg_spearman": vader,
            "pe_window_vader_neg_spearman": pe_vader,
            "revised_slots": _revised_slot_projections(
                arm_groups[V4], arm_sentences[V4], groups, arm_sentences[arm], v4_axes
            ),
        }
        top_hyps, full_ranking = _top_hypotheses_for_arm(groups, axes, arm_sentences[arm], per_channel=5)
        selected_hyps[arm] = top_hyps
        rotation[arm] = _rotation_entry(top_hyps, full_ranking, arm_sentences[V4])

    if rotation[V4]["top5_P"] != V4_TOP5_P or rotation[V4]["top5_E"] != V4_TOP5_E:
        raise SystemExit(f"V4 top-5 hypotheses mismatch: P={rotation[V4]['top5_P']} E={rotation[V4]['top5_E']}")
    between_arms = {
        c: {a: {b: float(arm_axes[a][c] @ arm_axes[b][c]) for b in arm_axes} for a in arm_axes} for c in ("P", "E")
    }
    return geometry, between_arms, arm_axes, selected_hyps, rotation


# ---------------------------------------------------------------------------
# Shared evaluation helpers
# ---------------------------------------------------------------------------


def _paired(dataset: str, frame: pd.DataFrame, left: np.ndarray, right: np.ndarray, seed: int) -> dict:
    """Paired Δ Spearman (left minus right), leaders or politicians resampled."""
    target_col, group_col = TARGET[dataset]
    res = paired_delta(
        np.asarray(left, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
        frame[target_col].to_numpy(dtype=float),
        frame[group_col].astype(str).to_numpy(),
        PAIRED_DRAWS,
        seed,
    )
    res["p_two_sided"] = min(1.0, res["p_two_sided"])
    return res


def _candidate_pairs(candidates: Sequence[str]) -> list[tuple[str, str]]:
    """Every pair of candidate arms, the later-listed arm minus the earlier one."""
    return [(candidates[j], candidates[i]) for i in range(len(candidates)) for j in range(i + 1, len(candidates))]


def _nli_window_diagnostics(
    window_df: pd.DataFrame,
    arm_map_ds: pd.DataFrame,
    entail_submatrix: np.ndarray,
) -> dict[str, Any]:
    pe = window_df["PE_nli"].to_numpy(dtype=np.float64)
    p = window_df["P_nli"].to_numpy(dtype=np.float64)
    e = window_df["E_nli"].to_numpy(dtype=np.float64)
    merged_ws = arm_map_ds["window_score"].to_numpy(dtype=np.float64)
    tc = tail_counts(entail_submatrix.astype(np.float32))
    return {
        "n_windows": int(len(window_df)),
        "PE_median": float(np.median(pe)),
        "PE_p90": float(np.percentile(pe, 90)),
        "PE_p99": float(np.percentile(pe, 99)),
        "P_median": float(np.median(p)),
        "P_p90": float(np.percentile(p, 90)),
        "P_p99": float(np.percentile(p, 99)),
        "E_median": float(np.median(e)),
        "E_p90": float(np.percentile(e, 90)),
        "E_p99": float(np.percentile(e, 99)),
        "share_P_gt_0.5": float((p > 0.5).mean()),
        "share_E_gt_0.5": float((e > 0.5).mean()),
        "share_both_gt_0.5": float(((p > 0.5) & (e > 0.5)).mean()),
        "distinct_float32_values": int(tc["distinct"]),
        "exact_zero": int(tc["exact_zero"]),
        "exact_one": int(tc["exact_one"]),
        "window_spearman_axis_vs_nli": float(spearmanr(merged_ws, pe).statistic),
        "best_P_hypothesis_counts": {
            str(k): int(v) for k, v in window_df["P_best"].value_counts().to_dict().items()
        },
        "best_E_hypothesis_counts": {
            str(k): int(v) for k, v in window_df["E_best"].value_counts().to_dict().items()
        },
    }


def _spotchecks(
    frame_en_dev: pd.DataFrame,
    scores: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    arm_window_dfs: Mapping[str, pd.DataFrame],
    requests_df: pd.DataFrame,
    arm_map_en: pd.DataFrame,
) -> tuple[list[dict], list[dict]]:
    """Scores of the low-scored spot-check speeches (tracked) and their windows with text (not tracked)."""
    doc_ids = frame_en_dev["id"].astype(str).tolist()
    spot_indices = [
        i for i, did in enumerate(doc_ids) if did in SPOTCHECK_EXACT or did.startswith(SPOTCHECK_PREFIX)
    ]
    premise_of = dict(zip(requests_df["request_id"].astype(int), requests_df["premise"].astype(str)))
    numeric_rows: list[dict] = []
    window_rows: list[dict] = []
    for idx in spot_indices:
        gpd = float(frame_en_dev.loc[idx, "source_score"])
        entry_num: dict[str, Any] = {"doc_id": doc_ids[idx], "gpd_score": gpd, "arms": {}}
        entry_win: dict[str, Any] = {"doc_id": doc_ids[idx], "gpd_score": gpd, "arms": {}}
        for arm, by_stage in scores.items():
            entry_num["arms"][arm] = {}
            for stage in ("axis", "nli_only", "combined"):
                vec = by_stage[stage]["english_dev"]
                entry_num["arms"][arm][stage] = float(vec[idx])
                entry_num["arms"][arm][f"{stage}_pct"] = float((vec <= vec[idx]).mean())
            wdf = arm_window_dfs[arm]
            ws_sub = arm_map_en[(arm_map_en["arm"] == arm) & (arm_map_en["doc_index"] == idx)]
            ws_map = dict(zip(ws_sub["window_rank"].astype(int), ws_sub["window_score"].astype(float)))
            entry_win["arms"][arm] = [
                {
                    "window_rank": int(r.window_rank),
                    "axis_window_score": ws_map.get(int(r.window_rank), float("nan")),
                    "P_nli": float(r.P_nli),
                    "E_nli": float(r.E_nli),
                    "PE_nli": float(r.PE_nli),
                    "P_best": str(r.P_best),
                    "E_best": str(r.E_best),
                    "premise": premise_of[int(r.request_id)],
                }
                for r in wdf[wdf["doc_index"] == idx].sort_values("window_rank").itertuples()
            ]
        numeric_rows.append(entry_num)
        window_rows.append(entry_win)
    return numeric_rows, window_rows


# ---------------------------------------------------------------------------
# Subcommand: prepare
# ---------------------------------------------------------------------------


def _v4_prepare_checks(
    metrics: Mapping[str, Mapping[str, Any]],
    doc_scores: Mapping[str, np.ndarray],
    window_rows: Mapping[str, pd.DataFrame],
    v4_hyps: list[dict],
) -> dict[str, Any]:
    """V4 recomputed here must equal the committed axis scores, top windows and claim hypotheses."""
    checks: dict[str, Any] = {"tolerance_spearman": CHECK_TOLERANCE, "datasets": {}}
    committed = _load_v4_committed()
    for dataset in DATASETS:
        observed = float(metrics[dataset]["spearman"])
        reported = REPORTED["axis"][dataset]
        score_diff = float(np.abs(doc_scores[dataset] - committed[dataset]["axis"]).max())
        stored = pd.read_parquet(STEP6 / f"requests_{dataset}.parquet")
        rows = window_rows[dataset]
        same_windows = bool(
            len(stored) == len(rows)
            and all(
                np.array_equal(stored[col].to_numpy(dtype=np.int64), rows[col].to_numpy(dtype=np.int64))
                for col in ("doc_index", "window_rank", "window_position")
            )
            and stored["premise"].astype(str).tolist() == rows["premise"].astype(str).tolist()
        )
        entry = {
            "spearman_reported": reported,
            "spearman_recomputed": observed,
            "max_abs_diff_vs_committed_axis_scores": score_diff,
            "top_windows_equal_step6_requests": same_windows,
            "n_windows": int(len(rows)),
        }
        checks["datasets"][dataset] = entry
        if abs(observed - reported) > CHECK_TOLERANCE:
            raise SystemExit(f"V4 axis check failed on {dataset}: Spearman {observed:.4f}, reported {reported:.3f}")
        if score_diff > 1e-8:
            raise SystemExit(f"V4 axis scores differ from {STEP4 / SCORE_FILE[dataset]} by {score_diff:.2e}")
        if not same_windows:
            raise SystemExit(f"V4 top windows differ from {STEP6 / f'requests_{dataset}.parquet'}")
    stored_hyps = json.loads((STEP6 / "hypotheses.json").read_text(encoding="utf-8"))
    same_hyps = [(h["hypothesis_id"], h["text"]) for h in stored_hyps] == [
        (h["hypothesis_id"], h["text"]) for h in v4_hyps
    ]
    checks["claim_hypotheses_equal_step6"] = bool(same_hyps)
    if not same_hyps:
        raise SystemExit("V4 claim hypotheses differ from results/step6_nli/hypotheses.json")
    checks["passed"] = True
    return checks


def run_prepare(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "subset_ids.json").exists():
        raise SystemExit(
            f"{out_dir} holds subset_ids.json, which only the all-benchmark version of this script "
            "(commit 9e0201c) wrote; move or delete that run's outputs first"
        )
    emb_dir = args.emb_dir or out_dir / "embeddings" / "e5-large-v2" / "sentences"
    arms = _resolve_arms(args.arms, emb_dir)
    candidates = [arm for arm in arms if arm != V4]

    data = {dataset: _load_dataset(dataset) for dataset in DATASETS}
    geometry, between_arms, arm_axes, selected_hyps, rotation = _geometry_audit(arms, out_dir, data)

    score_tables: list[pd.DataFrame] = []
    doc_scores: dict[str, dict[str, np.ndarray]] = {arm: {} for arm in arms}
    metrics: dict[str, dict[str, Any]] = {arm: {} for arm in arms}
    window_tables: dict[str, dict[str, pd.DataFrame]] = {arm: {} for arm in arms}
    for dataset in DATASETS:
        d = data[dataset]
        frame = d["frame"]
        doc_ids = frame[ID_COLUMN[dataset]].astype(str).tolist()
        for arm in arms:
            pe_win, docs = _score_arm(arm_axes[arm], d)
            doc_scores[arm][dataset] = docs["PE"]
            metrics[arm][dataset] = EVALUATORS[dataset](docs["PE"], frame, CI_DRAWS, args.bootstrap_seed)
            score_tables.append(
                pd.DataFrame(
                    {
                        "arm": arm,
                        "dataset": dataset,
                        "doc_index": np.arange(len(frame), dtype=np.int64),
                        "doc_id": pd.Series(doc_ids, dtype="string"),
                        "PE_axis": docs["PE"],
                        "P_axis": docs["P"],
                        "E_axis": docs["E"],
                    }
                )
            )
            window_tables[arm][dataset] = pd.DataFrame(
                _top_windows_rows(arm, dataset, doc_ids, d["doc_index"], pe_win, d["windows"], TOP_K)
            )

    v4_checks = _v4_prepare_checks(metrics[V4], doc_scores[V4], window_tables[V4], selected_hyps[V4])

    paired: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        frame = data[dataset]["frame"]
        paired[dataset] = {
            f"{left} - {right}": _paired(
                dataset, frame, doc_scores[left][dataset], doc_scores[right][dataset], args.bootstrap_seed
            )
            for left, right in [(arm, V4) for arm in candidates] + _candidate_pairs(candidates)
        }

    window_df_all = pd.concat(
        [window_tables[arm][dataset] for arm in arms for dataset in DATASETS], ignore_index=True
    )
    requests_df, arm_window_map, union_hyps, mapped_arm_hyps = deduplicate_nli_requests(
        window_df_all, {arm: selected_hyps[arm] for arm in arms}
    )
    nli_in_dir = out_dir / "nli_in"
    nli_in_dir.mkdir(parents=True, exist_ok=True)
    requests_path = nli_in_dir / f"requests_{REQUESTS_NAME}.parquet"
    requests_df.to_parquet(requests_path, index=False)
    hyp_path = nli_in_dir / "hypotheses.json"
    hyp_path.write_text(json.dumps(union_hyps, indent=2) + "\n", encoding="utf-8")
    arm_window_map.to_parquet(out_dir / "arm_window_map.parquet", index=False)
    (out_dir / "arm_hypotheses.json").write_text(json.dumps(mapped_arm_hyps, indent=2) + "\n", encoding="utf-8")
    pd.concat(score_tables, ignore_index=True).to_parquet(out_dir / "stage1_scores.parquet", index=False)

    export = {
        "window_rows_all_arms": int(len(window_df_all)),
        "unique_premises": int(len(requests_df)),
        "union_hypotheses": int(len(union_hyps)),
        "pairs": int(len(requests_df) * len(union_hyps)),
        "windows_per_arm": {
            arm: {dataset: int(len(window_tables[arm][dataset])) for dataset in DATASETS} for arm in arms
        },
        "requests_sha256": file_sha256(requests_path),
        "hypotheses_sha256": file_sha256(hyp_path),
    }
    summary = {
        "config": {
            "datasets": list(DATASETS),
            "bootstrap_seed": int(args.bootstrap_seed),
            "arms": {
                arm: {
                    "anchors_json": str(json_path.relative_to(ROOT)),
                    "anchors_json_sha256": file_sha256(json_path),
                    "anchors_npz": npz_path.name,
                    "anchors_npz_sha256": file_sha256(npz_path),
                }
                for arm, (json_path, npz_path) in arms.items()
            },
        },
        "v4_checks": v4_checks,
        "geometry_audit": geometry,
        "axis_cosine_between_arms": between_arms,
        "hypothesis_rotation": rotation,
        "axis_metrics": metrics,
        "axis_paired_deltas": paired,
        "nli_export": export,
    }
    (out_dir / "stage1_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    inputs: dict[str, Path] = {"english_split": SPLIT_PATH}
    for arm, (json_path, npz_path) in arms.items():
        inputs[f"{arm}_anchors"] = json_path
        inputs[f"{arm}_npz"] = npz_path
    write_run_manifest(
        out_dir,
        script="scripts/pilot_anchor_revision.py",
        inputs=inputs,
        extra={"subcommand": "prepare", "bootstrap_seed": int(args.bootstrap_seed), "nli_export": export},
        filename="prepare.manifest.json",
    )

    print("V4 checks passed: axis Spearman, committed axis scores, step-6 windows and claim hypotheses.")
    for dataset in DATASETS:
        row = " ".join(f"{arm}={metrics[arm][dataset]['spearman']:.4f}" for arm in arms)
        print(f"  axis Spearman {dataset}: {row}")
    print(
        f"NLI export: {export['unique_premises']} unique premises x {export['union_hypotheses']} hypotheses "
        f"= {export['pairs']:,} pairs"
    )


# ---------------------------------------------------------------------------
# Subcommand: evaluate
# ---------------------------------------------------------------------------


def _fmt_ci(value: float, ci: Sequence[float]) -> str:
    return f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def _fmt_delta(payload: Mapping[str, Any]) -> str:
    lo, hi = payload["ci95"]
    return f"{payload['delta_spearman']:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def _render_report(results: Mapping[str, Any], summary: Mapping[str, Any]) -> str:
    arms = list(summary["config"]["arms"])
    candidates = [arm for arm in arms if arm != V4]
    geom = summary["geometry_audit"]
    rotation = summary["hypothesis_rotation"]
    lines = [
        "# Anchor-revision pilot: development sets",
        "",
        "English-dev (174 speeches, leaders resampled) and Agora (5,272 paragraphs, politicians resampled). "
        f"Paired differences: {PAIRED_DRAWS} draws, seed {results['metadata']['bootstrap_seed']}. "
        "No held-out or external set is scored.",
        "",
        "## Checks",
        "",
    ]
    for dataset in DATASETS:
        c = summary["v4_checks"]["datasets"][dataset]
        r = results["v4_rerun_check"][dataset]
        lines.append(
            f"- {dataset}: V4 axis Spearman {c['spearman_recomputed']:.4f} (reported {c['spearman_reported']:.3f}); "
            f"top windows equal the step-6 requests: {c['top_windows_equal_step6_requests']}; "
            f"V4 claim-check rerun combined Spearman {r['combined_spearman_rerun']:.4f} "
            f"(committed {r['combined_spearman_committed']:.4f}, max score difference {r['max_abs_diff_combined']:.2e})."
        )
    lines += [
        "",
        "## Label-free geometry",
        "",
        "| Arm | cos(P, V4) | cos(E, V4) | cos(P, E) | VADER neg. ρ, PE windows (English-dev / Agora) | Claim hypotheses added / dropped |",
        "|---|---:|---:|---:|---|---|",
    ]
    for arm in arms:
        g = geom[arm]
        rot = rotation[arm]
        moved = "; ".join(
            f"{ch}: +{','.join(rot[f'top5_{ch}_vs_v4']['added_ids']) or '–'} −{','.join(rot[f'top5_{ch}_vs_v4']['dropped_ids']) or '–'}"
            for ch in ("P", "E")
        )
        lines.append(
            f"| `{arm}` | {g['cosine_to_v4']['P']:.4f} | {g['cosine_to_v4']['E']:.4f} | {g['cross_channel_cosine_PE']:.4f} | "
            f"{g['pe_window_vader_neg_spearman']['english_dev']:+.3f} / {g['pe_window_vader_neg_spearman']['agora']:+.3f} | {moved} |"
        )
    lines += [
        "",
        "Revised sentences on the V4 axis of their own channel, built without the slot (V4 sentence → revised sentence). "
        "Contrast slots (−) belong on the negative side.",
        "",
        "| Arm | Slot | Own axis | Other channel's axis |",
        "|---|---|---|---|",
    ]
    for arm in candidates:
        for row in geom[arm]["revised_slots"]:
            lines.append(
                f"| `{arm}` | {row['slot']} | {row['own_axis_v4_sentence']:+.3f} → {row['own_axis_revised_sentence']:+.3f} | "
                f"{row['other_axis_v4_sentence']:+.3f} → {row['other_axis_revised_sentence']:+.3f} |"
            )
    stage_labels = {
        "axis": "Axis",
        "nli_only": "Claim check, arm's hypotheses",
        "combined": "Combined, arm's hypotheses",
        "nli_only_v4hyp": "Claim check, V4 hypotheses",
        "combined_v4hyp": "Combined, V4 hypotheses",
    }
    lines += ["", "## Spearman with the human score", "", "| Arm | Score | English-dev [95% CI] | Agora [95% CI] |", "|---|---|---|---|"]
    for arm in arms:
        for stage, label in stage_labels.items():
            if stage not in results["metrics"][arm]:
                continue
            m = results["metrics"][arm][stage]
            lines.append(
                f"| `{arm}` | {label} | {_fmt_ci(m['english_dev']['spearman'], m['english_dev']['spearman_ci95'])} | "
                f"{_fmt_ci(m['agora']['spearman'], m['agora']['spearman_ci95'])} |"
            )
    lines += [
        "",
        "## Paired difference against the committed V4 scores",
        "",
        "| Arm | Score | English-dev Δ [95% CI] | Agora Δ [95% CI] |",
        "|---|---|---|---|",
    ]
    for arm in candidates:
        for stage, label in stage_labels.items():
            p = results["paired_vs_v4"][arm][stage]
            lines.append(f"| `{arm}` | {label} | {_fmt_delta(p['english_dev'])} | {_fmt_delta(p['agora'])} |")
    lines += [
        "",
        "## Adoption rule (combined score, arm's hypotheses, against B6)",
        "",
        f"English-dev Δ at least {ADOPT_ENGLISH_DEV_MIN:+.2f}, Agora Δ at least {ADOPT_AGORA_MIN:+.2f}, neither interval entirely below 0.",
        "",
        "| Arm | English-dev Δ | Agora Δ | Mean Δ | Passes |",
        "|---|---:|---:|---:|---|",
    ]
    for arm in candidates:
        a = results["adoption_rule"][arm]
        p = results["paired_vs_v4"][arm]["combined"]
        lines.append(
            f"| `{arm}` | {p['english_dev']['delta_spearman']:+.3f} | {p['agora']['delta_spearman']:+.3f} | "
            f"{a['mean_delta']:+.3f} | {'yes' if a['passes'] else 'no'} |"
        )
    lines += ["", "## Contrasts between candidate arms (left minus right)", "", "| Contrast | Score | English-dev Δ [95% CI] | Agora Δ [95% CI] |", "|---|---|---|---|"]
    for stage in ("axis", "combined"):
        for contrast, p in results["paired_between_arms"][stage].items():
            lines.append(f"| {contrast} | {stage_labels[stage]} | {_fmt_delta(p['english_dev'])} | {_fmt_delta(p['agora'])} |")
    lines += [
        "",
        "## Claim-check diagnostics (arm's windows and hypotheses)",
        "",
        "| Arm | Set | windows | PE median | PE p99 | share P > .5 | share E > .5 | window ρ axis vs claim check |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        for dataset in DATASETS:
            h = results["nli_health"]["per_arm"][arm][dataset]
            lines.append(
                f"| `{arm}` | {dataset} | {h['n_windows']} | {h['PE_median']:.4f} | {h['PE_p99']:.3f} | "
                f"{h['share_P_gt_0.5']:.4f} | {h['share_E_gt_0.5']:.4f} | {h['window_spearman_axis_vs_nli']:.3f} |"
            )
    lines += [
        "",
        "## Spot-check speeches (English-dev, percentile of the combined score; 1.0 is the top speech)",
        "",
        "| Speech | GPD score | " + " | ".join(f"`{arm}`" for arm in arms) + " |",
        "|---|---:|" + "---:|" * len(arms),
    ]
    for row in results["spotcheck_numeric"]:
        cells = " | ".join(f"{row['arms'][arm]['combined_pct']:.3f}" for arm in arms)
        lines.append(f"| {row['doc_id']} | {row['gpd_score']:.2f} | {cells} |")
    return "\n".join(lines) + "\n"


def run_evaluate(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    nli_in_dir = out_dir / "nli_in"
    nli_out_dir = out_dir / "nli_out"
    paths = {
        "nli_npz": nli_out_dir / f"nli_{REQUESTS_NAME}__{NLI_LABEL}.npz",
        "hypotheses": nli_in_dir / "hypotheses.json",
        "requests": nli_in_dir / f"requests_{REQUESTS_NAME}.parquet",
        "arm_window_map": out_dir / "arm_window_map.parquet",
        "arm_hypotheses": out_dir / "arm_hypotheses.json",
        "stage1_scores": out_dir / "stage1_scores.parquet",
        "stage1_summary": out_dir / "stage1_summary.json",
    }
    for label, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")

    summary = json.loads(paths["stage1_summary"].read_text(encoding="utf-8"))
    if summary["config"]["datasets"] != list(DATASETS):
        raise SystemExit("stage1_summary.json was not written by the development-set prepare step")
    arms = list(summary["config"]["arms"])
    candidates = [arm for arm in arms if arm != V4]
    union_hyps = json.loads(paths["hypotheses"].read_text(encoding="utf-8"))
    union_ids = [str(h["hypothesis_id"]) for h in union_hyps]
    requests_df = pd.read_parquet(paths["requests"])
    arm_window_map = pd.read_parquet(paths["arm_window_map"])
    arm_hyps = json.loads(paths["arm_hypotheses"].read_text(encoding="utf-8"))
    stage1 = pd.read_parquet(paths["stage1_scores"])

    payload = np.load(paths["nli_npz"])
    if not np.array_equal(payload["request_id"], requests_df["request_id"].to_numpy()):
        raise SystemExit("NLI output request_id does not match the request file")
    if list(payload["hypothesis_ids"].astype(str)) != union_ids:
        raise SystemExit("NLI output hypothesis_ids do not match hypotheses.json")
    entail = payload["probs"][:, :, 2].astype(np.float64)
    request_ids = payload["request_id"].astype(np.int64)
    row_of = {int(rid): i for i, rid in enumerate(request_ids)}
    nli_manifest_path = nli_out_dir / f"nli__{NLI_LABEL}.manifest.json"
    nli_manifest = json.loads(nli_manifest_path.read_text(encoding="utf-8")) if nli_manifest_path.exists() else None

    committed = _load_v4_committed()
    frames = {dataset: _load_frame(dataset) for dataset in DATASETS}

    # scores[arm][stage][dataset]; V4 holds the committed scores (B4 axis, B6 combined).
    scores: dict[str, dict[str, dict[str, np.ndarray]]] = {arm: {} for arm in arms}
    for stage in ("axis", "nli_only", "combined"):
        scores[V4][stage] = {dataset: committed[dataset][stage] for dataset in DATASETS}
    rerun_check: dict[str, Any] = {"tolerance_spearman": CHECK_TOLERANCE}
    nli_health: dict[str, Any] = {
        "stored_entailment": tail_counts(payload["probs"][:, :, 2]),
        "per_arm": {arm: {} for arm in arms},
    }
    window_dfs_en: dict[str, pd.DataFrame] = {}
    export_rows: list[pd.DataFrame] = []

    for dataset in DATASETS:
        frame = frames[dataset]
        n_docs = len(frame)
        doc_indices = np.arange(n_docs, dtype=np.int64)
        target_col, _ = TARGET[dataset]
        target = frame[target_col].to_numpy(dtype=float)
        for arm in arms:
            s1 = stage1[(stage1["arm"] == arm) & (stage1["dataset"] == dataset)].sort_values("doc_index")
            if s1["doc_id"].astype(str).tolist() != frame[ID_COLUMN[dataset]].astype(str).tolist():
                raise SystemExit(f"{dataset}/{arm}: stage-1 scores and the benchmark disagree on order")
            axis = s1["PE_axis"].to_numpy(dtype=np.float64)
            arm_map = arm_window_map[(arm_window_map["arm"] == arm) & (arm_window_map["dataset"] == dataset)]
            own, win_df = remap_arm_nli_scores(
                arm_map, request_ids, entail, union_ids, arm_hyps[arm], doc_indices, TOP_K
            )
            rows = np.asarray([row_of[int(rid)] for rid in arm_map["request_id"].to_numpy()], dtype=int)
            cols = [int(h["union_index"]) for h in arm_hyps[arm]]
            nli_health["per_arm"][arm][dataset] = _nli_window_diagnostics(win_df, arm_map, entail[np.ix_(rows, cols)])
            if dataset == "english_dev":
                window_dfs_en[arm] = win_df

            if arm == V4:
                if float(np.abs(axis - committed[dataset]["axis"]).max()) > 1e-8:
                    raise SystemExit(f"{dataset}: stage-1 V4 axis scores differ from the committed scores")
                rerun = (zscore(axis) + zscore(own["PE_nli"])) / 2.0
                rho_rerun = float(spearmanr(rerun, target).statistic)
                rho_committed = float(spearmanr(committed[dataset]["combined"], target).statistic)
                rerun_check[dataset] = {
                    "combined_spearman_committed": rho_committed,
                    "combined_spearman_reported": REPORTED["combined"][dataset],
                    "combined_spearman_rerun": rho_rerun,
                    "max_abs_diff_claim_check": float(np.abs(own["PE_nli"] - committed[dataset]["nli_only"]).max()),
                    "max_abs_diff_combined": float(np.abs(rerun - committed[dataset]["combined"]).max()),
                    "spearman_rerun_vs_committed_combined": float(
                        spearmanr(rerun, committed[dataset]["combined"]).statistic
                    ),
                }
                if abs(rho_committed - REPORTED["combined"][dataset]) > CHECK_TOLERANCE:
                    raise SystemExit(f"{dataset}: committed B6 Spearman {rho_committed:.4f} is not the reported value")
                if abs(rho_rerun - rho_committed) > CHECK_TOLERANCE:
                    raise SystemExit(
                        f"{dataset}: V4 claim-check rerun gives combined Spearman {rho_rerun:.4f}, "
                        f"committed {rho_committed:.4f}; the rerun does not reproduce B6"
                    )
                continue

            v4hyp, _ = remap_arm_nli_scores(
                arm_map, request_ids, entail, union_ids, arm_hyps[V4], doc_indices, TOP_K
            )
            by_stage = {
                "axis": axis,
                "nli_only": own["PE_nli"],
                "combined": (zscore(axis) + zscore(own["PE_nli"])) / 2.0,
                "nli_only_v4hyp": v4hyp["PE_nli"],
                "combined_v4hyp": (zscore(axis) + zscore(v4hyp["PE_nli"])) / 2.0,
            }
            for stage, values in by_stage.items():
                scores[arm].setdefault(stage, {})[dataset] = values
            export_rows.append(
                pd.DataFrame(
                    {
                        "arm": arm,
                        "dataset": dataset,
                        "doc_index": doc_indices,
                        "doc_id": s1["doc_id"].to_numpy(),
                        **{f"PE_{stage}": values for stage, values in by_stage.items()},
                    }
                )
            )
    rerun_check["passed"] = True

    seed = int(args.bootstrap_seed)
    metrics = {
        arm: {
            stage: {
                dataset: EVALUATORS[dataset](by_dataset[dataset], frames[dataset], CI_DRAWS, seed)
                for dataset in DATASETS
            }
            for stage, by_dataset in scores[arm].items()
        }
        for arm in arms
    }
    paired_vs_v4 = {
        arm: {
            stage: {
                dataset: _paired(
                    dataset, frames[dataset], scores[arm][stage][dataset], scores[V4][reference][dataset], seed
                )
                for dataset in DATASETS
            }
            for stage, reference in REFERENCE_STAGE.items()
        }
        for arm in candidates
    }
    paired_between = {
        stage: {
            f"{left} − {right}": {
                dataset: _paired(dataset, frames[dataset], scores[left][stage][dataset], scores[right][stage][dataset], seed)
                for dataset in DATASETS
            }
            for left, right in _candidate_pairs(candidates)
        }
        for stage in ("axis", "combined")
    }
    adoption = {
        arm: adoption_reading(paired_vs_v4[arm]["combined"]["english_dev"], paired_vs_v4[arm]["combined"]["agora"])
        for arm in candidates
    }

    spot_numeric, spot_windows = _spotchecks(
        frames["english_dev"],
        scores,
        window_dfs_en,
        requests_df,
        arm_window_map[arm_window_map["dataset"] == "english_dev"],
    )
    disagreement_dir = out_dir / "disagreement"
    disagreement_dir.mkdir(parents=True, exist_ok=True)
    (disagreement_dir / "spotcheck_windows.json").write_text(json.dumps(spot_windows, indent=2) + "\n", encoding="utf-8")
    pd.concat(export_rows, ignore_index=True).to_parquet(out_dir / "pilot_scores.parquet", index=False)

    results = {
        "metadata": {
            "datasets": list(DATASETS),
            "arms": arms,
            "bootstrap_seed": seed,
            "paired_draws": PAIRED_DRAWS,
            "ci_draws": CI_DRAWS,
            "reference": "committed V4 scores: step4 spw3_k8 (axis), step6 nli_only_k8 and combined_k8 (B6)",
            "nli_manifest": nli_manifest,
        },
        "v4_rerun_check": rerun_check,
        "metrics": metrics,
        "paired_vs_v4": paired_vs_v4,
        "paired_between_arms": paired_between,
        "adoption_rule": adoption,
        "nli_health": nli_health,
        "spotcheck_numeric": spot_numeric,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    report = _render_report(results, summary)
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")

    inputs = dict(paths)
    if nli_manifest_path.exists():
        inputs["nli_manifest"] = nli_manifest_path
    write_run_manifest(
        out_dir,
        script="scripts/pilot_anchor_revision.py",
        inputs=inputs,
        extra={"subcommand": "evaluate", "bootstrap_seed": seed},
        filename="evaluate.manifest.json",
    )
    print(report)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subcommand", choices=("prepare", "evaluate"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--arms", default=DEFAULT_ARMS,
                        help="comma-separated candidate names; each is anchors/candidates/<name>.json (prepare only)")
    parser.add_argument("--emb-dir", type=Path, default=None,
                        help="directory of the candidates' anchor embeddings (default: <out-dir>/embeddings/e5-large-v2/sentences)")
    parser.add_argument("--bootstrap-seed", type=int, default=20260923)
    args = parser.parse_args()
    if args.subcommand == "prepare":
        run_prepare(args)
    else:
        run_evaluate(args)


if __name__ == "__main__":
    main()
