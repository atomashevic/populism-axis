#!/usr/bin/env python3
"""Post-hoc sensitivity analyses for the manuscript (CPU; stored scores and local caches).

None of these analyses changes the frozen configuration. The English holdout, EU-OOD and the four
external benchmarks are spent, so every result here is a sensitivity analysis; no setting below may
be adopted on their evidence (docs/development-protocol.md). Each recomputation first reproduces the
stored score it varies and stops if it does not.

1. length       Document scores against the number of passages in the speech corpora, and agreement
                with the coders after partialling out that number.
2. hypotheses   For each corpus, the share of selected passages in which each claim hypothesis gives the
                largest entailment probability; the claim-check and combined scores recomputed with the
                hypotheses rated Direct only (P00, P01, P08; E00, E02, E05) and without P05 and E04.
3. conjunction  The minimum of the two component supports instead of their geometric mean (joint targets).
4. axis_form    The axis score recomputed from the stored window embeddings with t in {.1, .5, 1}, without
                the square, with the hard rule, and with the mean or the minimum of the two standardized
                projections (joint targets; axis only, because other selections have no stored entailment).
5. exclusionism Portuguese paragraphs: co-coding of the exclusionist paragraphs, their party, and AUROC
                of each score among paragraphs that carry no other populism code.
6. selection    Component targets: documents with more passages than K, and documents whose component
                and joint selections differ.
7. eu_macro_f1  Bootstrap intervals for RoBERTa's EU-OOD macro-F1 and its paired difference with the
                measure, resampling sentences within speakers and conditional on the fixed
                leave-one-speaker-out predictions; for comparison, also resampling whole speakers.
8. tone         VADER negativity against the projections of the English development passages, for the
                one-sentence passages of development step 2 and the three-sentence passages of the measure.
9. token_limit  Tweet and PopBERT test passages longer than the 128 tokens they were encoded with.
10. specificity Whether the restricted hypothesis sets of part 2 make the claim check respond less to
                content outside the construct: AUROC of the component claim checks for Portuguese
                exclusionism, and the passage-level Spearman correlation of the component supports with
                VADER negativity among the selected English development passages.

Outputs: results/posthoc_sensitivity/{sensitivity.json, sensitivity.md, run_manifest.json}.
Run from the repository root:  uv run --extra cpu python scripts/posthoc_sensitivity.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_apsi_comparison import comparison, loso_predictions, metric  # noqa: E402
from evaluate_nli import zscore  # noqa: E402
from populism_axis.contextual_axis import (  # noqa: E402
    CHANNELS,
    WindowProjections,
    _nonzero_median,
    _standardize,
    aggregate_from_window_scores,
    window_conjunction_scores,
)
from populism_axis.geometry import center_normalize, pole_mean_axes  # noqa: E402
from populism_axis.model_revisions import E5_LARGE_V2, PINNED_REVISIONS  # noqa: E402
from populism_axis.resampling import sampler  # noqa: E402
from populism_axis.run_manifest import write_run_manifest  # noqa: E402
from run_config import load_anchor_groups, load_embeddings, select_documents  # noqa: E402
from sweep_fusion_weight import load_corpora, window_max  # noqa: E402

R = ROOT / "results"
OUT = R / "posthoc_sensitivity"
WEIGHTS = {"P": 1.0, "E": 1.0, "M": 0.0}
SOFTNESS = 0.25
DIRECT = {"P": ["P00", "P01", "P08"], "E": ["E00", "E02", "E05"]}
BROAD = {"P": "P05", "E": "E04"}
ANCHOR_EMB = R / "embeddings/e5-large-v2/sentences/v3_proposition__ea98e5b2.npz"
EN_DIR = R / "embeddings/e5-large-v2/spw3_cap60_clean"
EN_DEV_STEP2_DIR = R / "embeddings/e5-large-v2/spw1_cap60_clean"
USGOV_DIR = R / "embeddings_external/usgov/e5-large-v2/spw3_cap60"
EU_DIR = R / "embeddings_eu/e5-large-v2/spw3_cap60"
PT_DIR = R / "embeddings_external/portugal/e5-large-v2/spw3_cap60"
TW_DIR = R / "embeddings_external/antielite_tweets/e5-large-v2/spw3_cap60"
PB_DIR = R / "embeddings_external/popbert_en/e5-large-v2/spw3_cap60"
ENGLISH = ROOT / "data/benchmarks/english_gpd_speeches.csv"
ENGLISH_SPLIT = ROOT / "data/benchmarks/english_split.csv"
STEP2_TONE = R / "step2_whitening/grid.json"
EU_SENTENCES = ROOT / "data/benchmarks/eu_sentences.csv"
EU_OURS = R / "apsi_comparison/cache/baseline_eu.parquet"
EU_ROBERTA = R / "roberta-correction/rc-20261002T1415Z-sigmoid/sigmoid_eu.npz"
STORED_MACRO_F1 = R / "final/final_results.json"

# corpus key in sweep_fusion_weight.load_corpora -> (entailment file, requests, scored claim, K)
CLAIM_CHECK = {
    "english_dev": (R / "step6_nli/nli_english_dev__bart-large-mnli.npz", R / "step6_nli/requests_english_dev.parquet", "PE", 8),
    "agora": (R / "step6_nli/nli_agora__bart-large-mnli.npz", R / "step6_nli/requests_agora.parquet", "PE", 8),
    "english_holdout": (R / "step6_nli/nli_english_holdout__bart-large-mnli.npz", R / "step6_nli/requests_english_holdout.parquet", "PE", 8),
    "eu_3v0": (R / "step6_nli/nli_eu__bart-large-mnli.npz", R / "step6_nli/requests_eu.parquet", "PE", 1),
    "usgov_gov2010_18": (R / "external/usgov/nli_usgov_speeches__bart-large-mnli.npz", R / "external/usgov/requests_usgov_speeches.parquet", "PE", 8),
    "tweets_general": (R / "external/antielite_tweets/nli_antielite_tweets__bart-large-mnli.npz", R / "external/antielite_tweets/requests_antielite_tweets.parquet", "E", 1),
    "popbert_AE": (R / "external/popbert_en/nli_popbert_en__bart-large-mnli.npz", R / "external/popbert_en/requests_popbert_en.parquet", "E", 1),
    "popbert_PC": (R / "external/popbert_en/nli_popbert_en__bart-large-mnli.npz", R / "external/popbert_en/requests_popbert_en.parquet", "P", 1),
    "portugal_AE": (R / "external/portugal/nli_portugal__bart-large-mnli.npz", R / "external/portugal/requests_portugal.parquet", "E", 8),
    "portugal_PC": (R / "external/portugal/nli_portugal__bart-large-mnli.npz", R / "external/portugal/requests_portugal.parquet", "P", 8),
}
JOINT = ["english_dev", "agora", "english_holdout", "eu_3v0", "usgov_gov2010_18"]


def evaluated(c: dict, values: np.ndarray) -> float:
    return metric(c["metric"], c["target"][c["mask"]], values[c["mask"]])


def paired(c: dict, left: np.ndarray, right: np.ndarray, draws: int, seed: int) -> dict:
    """Paired bootstrap on the evaluated documents, texts resampled within strata (left minus right)."""
    m = c["mask"]
    r = comparison(c["target"][m], left[m], right[m], c["groups"][m], c["metric"], draws, seed)
    return {k: r[k] for k in ("left", "right", "delta", "left_ci95", "delta_ci95", "n_clusters", "valid_draws")}


def same(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b))))


# --------------------------------------------------------------------------- claim check
class ClaimCheck:
    """Stored entailment probabilities of the selected passages of one corpus."""

    def __init__(self, npz: Path, requests: Path, n: int, k: int):
        payload = np.load(npz)
        req = pd.read_parquet(requests)
        if not np.array_equal(payload["request_id"], req["request_id"].to_numpy()):
            raise SystemExit(f"{npz.name}: entailment rows do not match the requests")
        labels = [str(x) for x in payload["labels"]] if "labels" in payload.files else ["contradiction", "neutral", "entailment"]
        self.hyp = [str(h) for h in payload["hypothesis_ids"]]
        self.ent = payload["probs"][:, :, labels.index("entailment")].astype(np.float64)
        keep = (req["window_rank"] < k).to_numpy()
        self.ent, self.doc = self.ent[keep], req.loc[keep, "doc_index"].to_numpy(int)
        self.n = n

    def support(self, claim: str, ids: list[str] | None = None) -> np.ndarray:
        ids = ids or [h for h in self.hyp if h.startswith(claim)]
        return self.ent[:, [self.hyp.index(h) for h in ids]].max(axis=1)

    def document(self, values: np.ndarray) -> np.ndarray:
        return pd.Series(values).groupby(self.doc).mean().reindex(range(self.n)).to_numpy()

    def score(self, scored: str, sets: dict | None = None, combine: str = "geometric") -> np.ndarray:
        sets = sets or {}
        if scored != "PE":
            return self.document(self.support(scored, sets.get(scored)))
        qp, qe = self.support("P", sets.get("P")), self.support("E", sets.get("E"))
        return self.document(np.sqrt(qp * qe) if combine == "geometric" else np.minimum(qp, qe))

    def concentration(self, claim: str) -> dict:
        ids = [h for h in self.hyp if h.startswith(claim)]
        best = np.array(ids)[self.support_matrix(ids).argmax(axis=1)]
        counts = pd.Series(best).value_counts()
        return {h: round(float(counts.get(h, 0)) / len(best), 4) for h in ids}

    def support_matrix(self, ids: list[str]) -> np.ndarray:
        return self.ent[:, [self.hyp.index(h) for h in ids]]


def claim_check_analyses(corpora: dict, draws: int, seed: int) -> tuple[dict, dict]:
    hypotheses, conjunction = {}, {}
    for name, (npz, requests, scored, k) in CLAIM_CHECK.items():
        c = corpora[name]
        check = ClaimCheck(npz, requests, c["n"], k)
        gap = same(check.score(scored), c["nli"])
        if gap > 1e-6:
            raise SystemExit(f"{name}: recomputed claim-check scores differ from the stored ones (max relative gap {gap:.2e})")
        stored_combined = .5 * c["z_axis"] + .5 * c["z_nli"]
        variants = {"direct_only": {k_: DIRECT[k_] for k_ in "PE"},
                    "without_P05_E04": {k_: [h for h in check.hyp if h.startswith(k_) and h != BROAD[k_]] for k_ in "PE"}}
        rows = {}
        for label, sets in variants.items():
            nli = check.score(scored, sets)
            combined = .5 * c["z_axis"] + .5 * zscore(nli)
            rows[label] = {"hypotheses": {k_: v for k_, v in sets.items() if scored in ("PE", k_)},
                           "claim_check": paired(c, nli, c["nli"], draws, seed),
                           "combined": paired(c, combined, stored_combined, draws, seed)}
        claims = ["P", "E"] if scored == "PE" else [scored]
        hypotheses[name] = {"metric": c["metric"], "n_evaluated": int(c["mask"].sum()), "spent": c["spent"],
                            "stored": {"claim_check": evaluated(c, c["nli"]), "combined": evaluated(c, stored_combined)},
                            "reproduction_max_relative_gap": gap,
                            "share_of_selected_passages_with_the_largest_value": {k_: check.concentration(k_) for k_ in claims},
                            "n_selected_passages": int(len(check.doc)), "variants": rows}
        if scored == "PE":
            nli = check.score("PE", combine="minimum")
            combined = .5 * c["z_axis"] + .5 * zscore(nli)
            conjunction[name] = {"metric": c["metric"], "claim_check": paired(c, nli, c["nli"], draws, seed),
                                 "combined": paired(c, combined, stored_combined, draws, seed)}
        print(f"claim check  {name:18s} stored {hypotheses[name]['stored']['combined']:.3f}  "
              + "  ".join(f"{k_} {v['combined']['left']:.3f}" for k_, v in rows.items()), flush=True)
    return hypotheses, conjunction


# --------------------------------------------------------------------------- axis
def english_frame():
    frame = pd.read_csv(ENGLISH)
    split = frame["id"].map(pd.read_csv(ENGLISH_SPLIT).set_index("id")["split"]).to_numpy()
    if pd.isna(split).any():
        raise SystemExit("english split file does not cover every benchmark row")
    return frame, split


def window_corpus(name: str):
    """Window embeddings of one corpus, selected as the scorer selected them: (matrix, doc_index, windows, n, K)."""
    if name in ("english_dev", "english_holdout"):
        frame, split = english_frame()
        matrix, doc, win = load_embeddings(EN_DIR, "english")
        kept, matrix, doc, win = select_documents(frame, matrix, doc, win, split == {"english_dev": "dev", "english_holdout": "holdout"}[name])
        return matrix, doc, win, len(kept), 8
    if name == "agora":
        matrix, doc, win = load_embeddings(EN_DIR, "agora")
        return matrix, doc, win, int(doc.max()) + 1, 8
    if name == "usgov_gov2010_18":
        matrix, doc, win = load_embeddings(USGOV_DIR, "usgov_speeches")
        return matrix, doc, win, int(doc.max()) + 1, 8
    if name == "eu_3v0":
        matrix, doc, win = load_embeddings(EU_DIR, "eu")
        return matrix, doc, win, int(doc.max()) + 1, 1
    raise KeyError(name)


def projections(matrix: np.ndarray, axes: dict) -> dict:
    normalized, _ = center_normalize(matrix)
    return {c: normalized @ axes[c] for c in CHANNELS}


def soft_scores(proj: dict, doc: np.ndarray, windows: list, softness: float, conjunction: str = "soft") -> np.ndarray:
    return np.asarray(window_conjunction_scores(WindowProjections(doc_index=doc, windows=windows, projections=proj),
                                                WEIGHTS, conjunction=conjunction, softness=softness))


def top_sets(scores: np.ndarray, doc: np.ndarray, k: int) -> dict:
    """The k passages of each document with the largest scores, ties in passage order (as the scorer selects)."""
    order = np.lexsort((-scores, doc))
    ds = doc[order]
    starts = np.flatnonzero(np.r_[True, ds[1:] != ds[:-1]])
    ends = np.r_[starts[1:], len(ds)]
    return {int(ds[a]): frozenset(order[a:min(b, a + k)].tolist()) for a, b in zip(starts, ends)}


def axis_form(corpora: dict, axes: dict, draws: int, seed: int) -> dict:
    out = {}
    for name in JOINT:
        c = corpora[name]
        matrix, doc, win, n, k = window_corpus(name)
        if n != c["n"]:
            raise SystemExit(f"{name}: {n} documents in the windows, {c['n']} in the stored scores")
        proj = projections(matrix, axes)
        texts = win["window"].tolist()
        rule = f"k:{k}"
        reference = soft_scores(proj, doc, texts, SOFTNESS)
        stored = aggregate_from_window_scores(reference, doc, n, rule)
        gap = same(stored, c["axis"])
        if gap > 1e-9:
            raise SystemExit(f"{name}: recomputed axis scores differ from the stored ones (max relative gap {gap:.2e})")
        z = {ch: _standardize(proj[ch]) for ch in ("P", "E")}
        unsquared = {ch: np.log1p(np.exp(z[ch] / SOFTNESS)) for ch in ("P", "E")}
        unsquared = np.sqrt(np.prod([v / _nonzero_median(v) for v in unsquared.values()], axis=0))
        windows_by_variant = {f"t={t}": soft_scores(proj, doc, texts, t) for t in (0.1, 0.5, 1.0)}
        windows_by_variant.update({"unsquared": unsquared, "hard": soft_scores(proj, doc, texts, SOFTNESS, "hard"),
                                   "mean_of_z": (z["P"] + z["E"]) / 2, "minimum_of_z": np.minimum(z["P"], z["E"])})
        if top_sets(unsquared, doc, k) != top_sets(reference, doc, k):
            raise SystemExit(f"{name}: removing the square changed the selected passages")
        rows = {}
        for label, window_scores in windows_by_variant.items():
            doc_scores = aggregate_from_window_scores(window_scores, doc, n, rule)
            changed = sum(a != b for a, b in zip(top_sets(window_scores, doc, k).values(), top_sets(reference, doc, k).values()))
            rows[label] = {**paired(c, doc_scores, c["axis"], draws, seed), "documents_with_changed_selection": int(changed)}
        out[name] = {"metric": c["metric"], "n_evaluated": int(c["mask"].sum()), "spent": c["spent"], "k": k,
                     "stored_axis": evaluated(c, c["axis"]), "reproduction_max_relative_gap": gap, "variants": rows}
        print(f"axis form    {name:18s} stored {out[name]['stored_axis']:.3f}  "
              + "  ".join(f"{label} {r['left']:.3f}" for label, r in rows.items()), flush=True)
    return out


# --------------------------------------------------------------------------- length
def length(corpora: dict, draws: int, seed: int) -> dict:
    frame, split = english_frame()
    counts_en = np.bincount(pd.read_parquet(EN_DIR / "windows_english.parquet", columns=["doc_index"])["doc_index"].to_numpy(), minlength=len(frame))
    counts = {"english_dev": counts_en[split == "dev"], "english_holdout": counts_en[split == "holdout"],
              "usgov_gov2010_18": np.bincount(pd.read_parquet(USGOV_DIR / "windows_usgov_speeches.parquet", columns=["doc_index"])["doc_index"].to_numpy())}
    out = {}
    for name, n_passages in counts.items():
        c = corpora[name]
        if len(n_passages) != c["n"]:
            raise SystemExit(f"{name}: passage counts do not cover the stored documents")
        m = c["mask"]
        scores = {"axis": c["axis"], "claim_check": c["nli"], "combined": .5 * c["z_axis"] + .5 * c["z_nli"]}
        n_m, y, g = n_passages[m].astype(float), c["target"][m], c["groups"][m]

        def partial(score, target, size):
            rx, ry, rz = rankdata(score), rankdata(target), rankdata(size)
            res = lambda v: v - np.polyval(np.polyfit(rz, v, 1), rz)  # noqa: E731
            return float(np.corrcoef(res(rx), res(ry))[0, 1])

        rows = {}
        for label, s in scores.items():
            s_m = s[m]
            rows[label] = {"spearman_with_passages": float(spearmanr(s_m, n_m).statistic),
                           "spearman_with_target": float(spearmanr(s_m, y).statistic),
                           "partial_spearman_with_target_given_passages": partial(s_m, y, n_m)}
        rng = np.random.default_rng(seed)
        draw = sampler(g)
        boot = {"spearman_with_passages": [], "partial_minus_raw": []}
        s_m = scores["combined"][m]
        for _ in range(draws):
            ix = draw(rng)
            if len(np.unique(y[ix])) < 2:
                continue
            boot["spearman_with_passages"].append(spearmanr(s_m[ix], n_m[ix]).statistic)
            boot["partial_minus_raw"].append(partial(s_m[ix], y[ix], n_m[ix]) - spearmanr(s_m[ix], y[ix]).statistic)
        rows["combined"]["spearman_with_passages_ci95"] = np.percentile(boot["spearman_with_passages"], [2.5, 97.5]).tolist()
        rows["combined"]["partial_minus_raw_ci95"] = np.percentile(boot["partial_minus_raw"], [2.5, 97.5]).tolist()
        out[name] = {"n_evaluated": int(m.sum()), "spent": c["spent"],
                     "target_spearman_with_passages": float(spearmanr(y, n_m).statistic),
                     "passages_median": float(np.median(n_m)), "passages_min": int(n_m.min()), "passages_max": int(n_m.max()),
                     "scores": rows}
        print(f"length       {name:18s} rho(combined, passages) {rows['combined']['spearman_with_passages']:+.3f}  "
              f"partial {rows['combined']['partial_spearman_with_target_given_passages']:.3f} vs {rows['combined']['spearman_with_target']:.3f}", flush=True)
    return out


# --------------------------------------------------------------------------- exclusionism (Portugal)
def exclusionism(draws: int, seed: int) -> dict:
    pt = pd.read_csv(R / "external/portugal/final_scores.csv")
    other = ["elite_gold", "centr_gold", "sovereignty_gold"]
    excl = pt[pt["exclusion_gold"] == 1]
    plain = (pt[other].sum(axis=1) == 0).to_numpy()
    systems = ["P_axis", "E_axis", "PE_axis", "P_nli", "E_nli", "P_axis_plus_nli", "E_axis_plus_nli", "PE_axis_plus_nli"]
    y_all, y_plain = pt["exclusion_gold"].to_numpy(int), pt.loc[plain, "exclusion_gold"].to_numpy(int)
    groups = pt.loc[plain, "actor"].astype(str).to_numpy()
    rows = {}
    for s in systems:
        v = pt.loc[plain, s].to_numpy(float)
        r = comparison(y_plain, v, v, groups, "auroc", draws, seed)
        rows[s] = {"auroc_all_paragraphs": float(roc_auc_score(y_all, pt[s])), "auroc_without_other_codes": r["left"],
                   "auroc_without_other_codes_ci95": r["left_ci95"]}
    party = excl["party"].value_counts()
    top = str(party.index[0])
    within = pt[pt["party"] == top]
    return {"n_exclusionist": int(len(excl)),
            "co_coded": {col: int(excl[col].sum()) for col in other} | {"any": int((excl[other].sum(axis=1) > 0).sum())},
            "n_exclusionist_without_other_codes": int(y_plain.sum()), "n_paragraphs_without_any_code": int((y_plain == 0).sum()),
            "n_speakers_without_other_codes": int(len(np.unique(groups))),
            "party_of_exclusionist_paragraphs": {str(k): int(v) for k, v in party.items()},
            "largest_party": top, "largest_party_share_of_all_paragraphs": float((pt["party"] == top).mean()),
            "within_largest_party": {"n": int(len(within)), "n_exclusionist": int(within["exclusion_gold"].sum()),
                                     "auroc": {s: float(roc_auc_score(within["exclusion_gold"], within[s])) for s in systems}},
            "systems": rows}


# --------------------------------------------------------------------------- component and joint selection
def selection(axes: dict) -> dict:
    out = {}
    frames = {"portugal": pd.read_csv(R / "external/portugal/final_scores.csv"),
              "antielite_tweets": pd.read_csv(R / "external/antielite_tweets/final_scores.csv"),
              "popbert_en": pd.read_csv(R / "external/popbert_en/final_scores.csv")}
    for name, k in (("portugal", 8), ("antielite_tweets", 1), ("popbert_en", 1)):
        frame = frames[name]
        if name == "popbert_en":
            parts = [load_embeddings(PB_DIR, f"popbert_en_{s}") for s in ("train", "test")]
            offset = int(parts[0][1].max()) + 1
            matrix = np.vstack([parts[0][0], parts[1][0]])
            doc = np.concatenate([parts[0][1], parts[1][1] + offset])
            texts = parts[0][2]["window"].tolist() + parts[1][2]["window"].tolist()
        else:
            matrix, doc, win = load_embeddings(PT_DIR if name == "portugal" else TW_DIR, "portugal_en" if name == "portugal" else name)
            texts = win["window"].tolist()
        n = len(frame)
        proj = projections(matrix, axes)
        z = {ch: _standardize(proj[ch]) for ch in ("P", "E")}
        for ch in ("P", "E"):
            gap = same(aggregate_from_window_scores(z[ch], doc, n, f"k:{k}"), frame[f"{ch}_axis"].to_numpy(float))
            if gap > 1e-6:
                raise SystemExit(f"{name}: recomputed {ch} axis scores differ from the stored ones (max relative gap {gap:.2e})")
        joint = top_sets(soft_scores(proj, doc, texts, SOFTNESS), doc, k)
        evaluated_docs = np.flatnonzero((frame["split"] == "test").to_numpy()) if "split" in frame else np.arange(n)
        passages = np.bincount(doc, minlength=n)
        more = [int(d) for d in evaluated_docs if passages[d] > k]
        differ = {ch: sum(top_sets(z[ch], doc, k)[d] != joint[d] for d in more) for ch in ("P", "E")}
        out[name] = {"k": k, "n_evaluated_documents": int(len(evaluated_docs)), "documents_with_more_than_k_passages": len(more),
                     "documents_whose_component_selection_differs": {ch: int(v) for ch, v in differ.items()}}
        print(f"selection    {name:18s} >K {len(more)}  differ {differ}", flush=True)
    return out


# --------------------------------------------------------------------------- EU-OOD macro-F1 intervals
def eu_macro_f1(draws: int, seed: int) -> dict:
    eu = pd.read_csv(EU_SENTENCES)
    ours = pd.read_parquet(EU_OURS)
    if list(ours["id"].astype(str)) != list(eu["row_id"].astype(str)):
        raise SystemExit("EU-OOD: stored component scores are not in the order of the sentence table")
    rob = np.load(EU_ROBERTA)
    doc, probs = rob["doc_index"].astype(int), rob["probabilities"].astype(np.float64)
    e_rob, p_rob = window_max(doc, probs[:, 1], len(eu)), window_max(doc, probs[:, 2], len(eu))
    truth, pred_ours = loso_predictions(eu, ours["ours_E"].to_numpy(float), ours["ours_P"].to_numpy(float))
    _, pred_rob = loso_predictions(eu, e_rob, p_rob)
    stored = json.loads(STORED_MACRO_F1.read_text(encoding="utf-8"))["eu_paper_protocol_macro_f1_loso"]
    reproduced = {"ours": metric("macro_f1", truth, pred_ours), "roberta": metric("macro_f1", truth, pred_rob)}
    for key, ref in (("ours", stored["B6_axis_plus_nli"]["macro"]), ("roberta", stored["RoBERTa_comparator"]["macro"])):
        if abs(reproduced[key] - ref) > 1e-9:
            raise SystemExit(f"EU-OOD macro-F1 of {key} is {reproduced[key]:.4f}, stored {ref:.4f}")
    speakers = eu["speaker"].astype(str).to_numpy()
    r = comparison(truth, pred_ours, pred_rob, speakers, "macro_f1", draws, seed)
    w = comparison(truth, pred_ours, pred_rob, speakers, "macro_f1", draws, seed, scheme="strata")  # for comparison only
    return {"ours": r["left"], "ours_ci95": r["left_ci95"], "roberta": r["right"], "roberta_ci95": r["right_ci95"],
            "ours_minus_roberta": r["delta"], "ours_minus_roberta_ci95": r["delta_ci95"], "n_speakers": r["n_clusters"],
            "valid_draws": r["valid_draws"], "interval_scope": "sentences resampled within speakers, conditional on fixed LOSO predictions",
            "ours_ci95_whole_speakers": w["left_ci95"], "roberta_ci95_whole_speakers": w["right_ci95"],
            "ours_minus_roberta_ci95_whole_speakers": w["delta_ci95"]}


# --------------------------------------------------------------------------- hostile tone
def tone(axes: dict) -> dict:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    frame, split = english_frame()
    out = {}
    for label, directory in (("one_sentence_passages_step2", EN_DEV_STEP2_DIR), ("three_sentence_passages_final", EN_DIR)):
        matrix, doc, win = load_embeddings(directory, "english")
        _, matrix, doc, win = select_documents(frame, matrix, doc, win, split == "dev")
        proj = projections(matrix, axes)
        neg = np.asarray([analyzer.polarity_scores(t)["neg"] for t in win["window"].tolist()])
        out[label] = {"n_passages": int(len(neg)),
                      "spearman_with_vader_negativity": {ch: float(spearmanr(proj[ch], neg).statistic) for ch in ("P", "E")}}
        print(f"tone         {label:30s} {out[label]['spearman_with_vader_negativity']}", flush=True)
    reference = json.loads(STEP2_TONE.read_text(encoding="utf-8"))["reference_v3_rule_vader_neg_spearman"]["english_E"]
    gap = abs(out["one_sentence_passages_step2"]["spearman_with_vader_negativity"]["E"] - reference)
    if gap > .01:
        raise SystemExit(f"the step-2 tone diagnostic is not reproduced (stored {reference:.3f}, gap {gap:.3f})")
    out["step2_stored_english_E"] = reference
    return out


# --------------------------------------------------------------------------- encoder token limit
def token_limit() -> dict:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(E5_LARGE_V2, revision=PINNED_REVISIONS[E5_LARGE_V2], local_files_only=True)
    tweets = pd.read_csv(R / "external/antielite_tweets/final_scores.csv")
    tw = pd.read_parquet(TW_DIR / "windows_antielite_tweets.parquet")
    test_docs = set(np.flatnonzero((tweets["split"] == "test").to_numpy()).tolist())
    sets = {"antielite_tweets_test": tw[tw["doc_index"].isin(test_docs)]["window"].tolist(),
            "popbert_en_test": pd.read_parquet(PB_DIR / "windows_popbert_en_test.parquet")["window"].tolist()}
    out = {}
    for name, texts in sets.items():
        lengths = np.asarray([len(tokenizer("query: " + t)["input_ids"]) for t in texts])
        out[name] = {"n_passages": int(len(lengths)), "max_tokens": int(lengths.max()),
                     "passages_over_128_tokens": int((lengths > 128).sum()), "passages_over_256_tokens": int((lengths > 256).sum())}
    return out


def specificity(draws: int, seed: int) -> dict:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sets = {"all": None, "direct_only": DIRECT, "without_P05_E04": None}
    pt = pd.read_csv(R / "external/portugal/final_scores.csv")
    npz, requests, _, k = CLAIM_CHECK["portugal_AE"]
    check = ClaimCheck(npz, requests, len(pt), k)
    plain = (pt[["elite_gold", "centr_gold", "sovereignty_gold"]].sum(axis=1) == 0).to_numpy()
    y_all, y_plain = pt["exclusion_gold"].to_numpy(int), pt.loc[plain, "exclusion_gold"].to_numpy(int)
    groups = pt.loc[plain, "actor"].astype(str).to_numpy()
    exclusion = {}
    for claim in ("P", "E"):
        ids = {"all": [h for h in check.hyp if h.startswith(claim)], "direct_only": DIRECT[claim],
               "without_P05_E04": [h for h in check.hyp if h.startswith(claim) and h != BROAD[claim]]}
        stored = pt[f"{claim}_nli"].to_numpy(float)
        if same(check.score(claim, {claim: ids["all"]}), stored) > 1e-6:
            raise SystemExit(f"portugal: recomputed {claim} claim check differs from the stored one")
        exclusion[claim] = {}
        for label in sets:
            v = check.score(claim, {claim: ids[label]})
            r = comparison(y_plain, v[plain], v[plain], groups, "auroc", draws, seed)
            exclusion[claim][label] = {"auroc_all_paragraphs": float(roc_auc_score(y_all, v)),
                                       "auroc_without_other_codes": r["left"], "auroc_without_other_codes_ci95": r["left_ci95"]}
    npz, requests, _, k = CLAIM_CHECK["english_dev"]
    dev = ClaimCheck(npz, requests, int(pd.read_parquet(requests)["doc_index"].max()) + 1, k)
    premises = pd.read_parquet(requests)
    premises = premises.loc[(premises["window_rank"] < k).to_numpy(), "premise"].tolist()
    analyzer = SentimentIntensityAnalyzer()
    neg = np.asarray([analyzer.polarity_scores(t)["neg"] for t in premises])
    tone_rows = {}
    for claim in ("P", "E"):
        ids = {"all": None, "direct_only": DIRECT[claim],
               "without_P05_E04": [h for h in dev.hyp if h.startswith(claim) and h != BROAD[claim]]}
        tone_rows[claim] = {label: float(spearmanr(dev.support(claim, ids[label]), neg).statistic) for label in sets}
    print(f"specificity  exclusion {exclusion}  tone {tone_rows}", flush=True)
    return {"portugal_exclusionism": exclusion, "n_exclusionist_without_other_codes": int(y_plain.sum()),
            "english_dev_selected_passages_vader_spearman": tone_rows, "n_english_dev_selected_passages": int(len(neg))}


def markdown(results: dict) -> str:
    lines = ["# Post-hoc sensitivity analyses", "",
             "Every analysis is post hoc and changes no configuration. Paired intervals: bootstrap of texts within strata "
             f"({results['draws']} draws, seed {results['seed']}), left minus stored.", "",
             "## Claim-check hypotheses (combined score)", "", "| Corpus | Metric | Stored | Direct only | Δ [95% CI] | Without P05, E04 | Δ [95% CI] |", "|---|---|---:|---:|---|---:|---|"]
    fmt = lambda r: f"{r['delta']:+.3f} [{r['delta_ci95'][0]:+.3f}, {r['delta_ci95'][1]:+.3f}]"  # noqa: E731
    for name, h in results["hypotheses"].items():
        v = h["variants"]
        lines.append(f"| {name} | {h['metric']} | {h['stored']['combined']:.3f} | {v['direct_only']['combined']['left']:.3f} | {fmt(v['direct_only']['combined'])} | "
                     f"{v['without_P05_E04']['combined']['left']:.3f} | {fmt(v['without_P05_E04']['combined'])} |")
    lines += ["", "## Claim-check hypotheses (claim check alone)", "", "| Corpus | Stored | Direct only | Δ [95% CI] | Without P05, E04 | Δ [95% CI] |", "|---|---:|---:|---|---:|---|"]
    for name, h in results["hypotheses"].items():
        v = h["variants"]
        lines.append(f"| {name} | {h['stored']['claim_check']:.3f} | {v['direct_only']['claim_check']['left']:.3f} | {fmt(v['direct_only']['claim_check'])} | "
                     f"{v['without_P05_E04']['claim_check']['left']:.3f} | {fmt(v['without_P05_E04']['claim_check'])} |")
    lines += ["", "## Minimum instead of geometric mean of the two supports", "", "| Corpus | Claim check | Δ | Combined | Δ |", "|---|---:|---|---:|---|"]
    for name, r in results["conjunction"].items():
        lines.append(f"| {name} | {r['claim_check']['left']:.3f} | {fmt(r['claim_check'])} | {r['combined']['left']:.3f} | {fmt(r['combined'])} |")
    lines += ["", "## Axis functional form (axis score alone)", ""]
    labels = list(next(iter(results["axis_form"].values()))["variants"])
    lines += ["| Corpus | Stored | " + " | ".join(labels) + " |", "|---|---:|" + "---|" * len(labels)]
    for name, a in results["axis_form"].items():
        lines.append(f"| {name} | {a['stored_axis']:.3f} | " + " | ".join(f"{a['variants'][l]['left']:.3f} ({fmt(a['variants'][l])})" for l in labels) + " |")
    lines += ["", "## Length", "", "| Corpus | ρ(grade, passages) | ρ(axis, passages) | ρ(claim check, passages) | ρ(combined, passages) | ρ(combined, grade) | partial |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, L in results["length"].items():
        s = L["scores"]
        lines.append(f"| {name} | {L['target_spearman_with_passages']:+.3f} | {s['axis']['spearman_with_passages']:+.3f} | {s['claim_check']['spearman_with_passages']:+.3f} | "
                     f"{s['combined']['spearman_with_passages']:+.3f} | {s['combined']['spearman_with_target']:.3f} | {s['combined']['partial_spearman_with_target_given_passages']:.3f} |")
    lines += ["", "## Specificity of the restricted hypothesis sets", "", "| Claim | Hypotheses | Exclusionism AUROC (all) | Exclusionism AUROC (no other code) | ρ with VADER negativity |", "|---|---|---:|---:|---:|"]
    sp = results["specificity"]
    for claim in ("P", "E"):
        for label, r in sp["portugal_exclusionism"][claim].items():
            lines.append(f"| {claim} | {label} | {r['auroc_all_paragraphs']:.3f} | {r['auroc_without_other_codes']:.3f} | {sp['english_dev_selected_passages_vader_spearman'][claim][label]:+.3f} |")
    lines += ["", "Other analyses: see sensitivity.json (exclusionism, selection, eu_macro_f1, tone, token_limit)."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    corpora = load_corpora()
    for key in ("popbert_AE", "popbert_PC"):  # no speaker identifiers: one stratum, sentences resampled within it
        corpora[key]["groups"] = np.full(corpora[key]["n"], "all")
    axes = pole_mean_axes(load_anchor_groups(ANCHOR_EMB))
    results = {"draws": args.draws, "seed": args.seed, "resampling": "texts within strata",
               "status": "post hoc; no configuration change"}
    results["hypotheses"], results["conjunction"] = claim_check_analyses(corpora, args.draws, args.seed)
    results["length"] = length(corpora, args.draws, args.seed)
    results["axis_form"] = axis_form(corpora, axes, args.draws, args.seed)
    results["exclusionism"] = exclusionism(args.draws, args.seed)
    results["selection"] = selection(axes)
    results["eu_macro_f1"] = eu_macro_f1(args.draws, args.seed)
    results["tone"] = tone(axes)
    results["token_limit"] = token_limit()
    results["specificity"] = specificity(args.draws, args.seed)
    (OUT / "sensitivity.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    (OUT / "sensitivity.md").write_text(markdown(results), encoding="utf-8")
    inputs = {"anchors": ANCHOR_EMB, "english": ENGLISH, "english_split": ENGLISH_SPLIT, "eu_sentences": EU_SENTENCES,
              "eu_ours": EU_OURS, "eu_roberta": EU_ROBERTA, "stored_macro_f1": STORED_MACRO_F1, "step2_tone": STEP2_TONE}
    for name, (npz, requests, _, _) in CLAIM_CHECK.items():
        inputs[f"{name}_entailment"], inputs[f"{name}_requests"] = npz, requests
    for label, directory, dataset in (("english", EN_DIR, "english"), ("english_step2", EN_DEV_STEP2_DIR, "english"), ("agora", EN_DIR, "agora"),
                                      ("usgov", USGOV_DIR, "usgov_speeches"), ("eu", EU_DIR, "eu"), ("portugal", PT_DIR, "portugal_en"),
                                      ("tweets", TW_DIR, "antielite_tweets"), ("popbert_train", PB_DIR, "popbert_en_train"), ("popbert_test", PB_DIR, "popbert_en_test")):
        inputs[f"{label}_embeddings"], inputs[f"{label}_windows"] = directory / f"embeddings_{dataset}.npz", directory / f"windows_{dataset}.parquet"
    for name in ("usgov/final_scores_speeches.csv", "antielite_tweets/final_scores.csv", "popbert_en/final_scores.csv", "portugal/final_scores.csv"):
        inputs[name] = R / "external" / name
    write_run_manifest(OUT, script="scripts/posthoc_sensitivity.py", inputs=inputs,
                       extra={"draws": args.draws, "seed": args.seed, "status": "post hoc sensitivity analyses; no configuration change"})
    print(f"wrote {OUT.relative_to(ROOT)}/sensitivity.json")


if __name__ == "__main__":
    main()
