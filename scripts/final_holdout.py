#!/usr/bin/env python3
"""final step: one-time scoring of the English holdout and EU-OOD (CPU; requires --final).

Pre-declared configurations (docs/development-protocol.md, research/development-log/stage-6-claim-check.md):
  B0  initial configuration: e5-large-v2, uncapped splitter, raw text, proposition anchors and pole-mean rule, PE soft top-3
  B4  populism axis only: cleaned text, 3-sentence windows capped at 60 words/sentence, PE soft top-8
  B6  populism axis + claim check: mean of within-dataset z-scores of the B4 score and the NLI score
      (BART-large-MNLI; NLI pooled like the axis: English top-8 windows, EU top window)
  RoBERTa comparator (coastalcph/roberta-large-ft-trump-populism) on one-sentence windows, as in the initial evaluation.

Outputs in results/final/: per-configuration run directories, `final_results.json`,
`eu_paper_protocol.json`, and `final_results.md`. Every access is logged.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

from populism_axis.run_manifest import write_run_manifest  # noqa: E402
from populism_axis.roberta_comparator import load_scores  # noqa: E402
from evaluate_benchmarks import evaluate_english, evaluate_eu  # noqa: E402
from evaluate_eu_paper_protocol import PAPER_EU_OOD, SURNAME, best_f1_threshold, macro_f1, prevalence_threshold, window_max  # noqa: E402
from paired_comparison import paired_delta  # noqa: E402
from evaluate_nli import zscore  # noqa: E402
from run_config import log_access  # noqa: E402

FINAL = ROOT / "results" / "final"
EMB = ROOT / "results" / "embeddings" / "e5-large-v2"
EMB_EU = ROOT / "results" / "embeddings_eu" / "e5-large-v2"
NLI = ROOT / "results" / "step6_nli"
ANCHORS = EMB / "sentences" / "v3_proposition__ea98e5b2.npz"
SYSTEM = "PE__frozen__soft"
REASON = "Final one-time scoring of English holdout and EU-OOD (pre-declared B0, B4, B6)"
TRUMP_PREVALENCE = {"AE": 0.055, "PC": 0.034}


def run_axis(label: str, english_dir: Path, eu_dir: Path, rule: str | None) -> Path:
    out_dir = FINAL / label
    command = [sys.executable, str(ROOT / "scripts" / "run_config.py"), "--label", label,
               "--embeddings-dir", str(english_dir), "--dataset-dir", f"eu={eu_dir}", "--anchors", str(ANCHORS),
               "--out-dir", str(out_dir), "--datasets", "english,eu", "--english-split", "holdout",
               "--final", "--access-reason", REASON, "--save-projections"]
    if rule:
        command += ["--top-rule", rule]
    subprocess.run(command, check=True)
    return out_dir


def nli_document_scores(name: str, n_docs: int, k: int) -> tuple[np.ndarray, pd.DataFrame]:
    hypotheses = json.loads((NLI / "hypotheses.json").read_text(encoding="utf-8"))
    requests = pd.read_parquet(NLI / f"requests_{name}.parquet")
    payload = np.load(NLI / f"nli_{name}__bart-large-mnli.npz")
    if not np.array_equal(payload["request_id"], requests["request_id"].to_numpy()):
        raise SystemExit(f"{name}: NLI output does not match requests")
    entail = payload["probs"][:, :, 2].astype(np.float64)
    p_cols = [i for i, h in enumerate(hypotheses) if h["channel"] == "P"]
    e_cols = [i for i, h in enumerate(hypotheses) if h["channel"] == "E"]
    windows = requests[["doc_index", "window_rank"]].copy()
    windows["P_nli"] = entail[:, p_cols].max(axis=1)
    windows["E_nli"] = entail[:, e_cols].max(axis=1)
    windows["nli"] = np.sqrt(windows["P_nli"] * windows["E_nli"])
    kept = windows[windows["window_rank"] < k]
    doc = kept.groupby("doc_index")["nli"].mean().reindex(range(n_docs)).fillna(0.0).to_numpy()
    return doc, windows


def eu_protocol(frame: pd.DataFrame, candidates: dict[str, dict[str, np.ndarray]]) -> dict:
    pop_code = frame["pop_code"].to_numpy(dtype=int)
    truth = {"AE": np.isin(pop_code, [1, 3]).astype(int), "PC": np.isin(pop_code, [2, 3]).astype(int)}
    speakers = frame["speaker"].map(SURNAME).fillna(frame["speaker"]).to_numpy()
    order = ["Le Pen", "Weidel", "Zemmour", "Kickl", "Tsipras"]
    out = {"paper_reference": PAPER_EU_OOD, "ours": {}}
    for name, scores in candidates.items():
        loso = {key: np.zeros(len(frame), dtype=int) for key in ("AE", "PC")}
        for speaker in order:
            test = speakers == speaker
            for key in ("AE", "PC"):
                threshold = best_f1_threshold(scores[key][~test], truth[key][~test])
                loso[key][test] = (scores[key][test] > threshold).astype(int)
        prevalence = {key: (scores[key] > prevalence_threshold(scores[key], TRUMP_PREVALENCE[key])).astype(int) for key in ("AE", "PC")}
        out["ours"][name] = {
            "loso_f1": {"overall": macro_f1(truth, loso, np.ones(len(frame), bool)),
                        "per_speaker": {s: macro_f1(truth, loso, speakers == s)["macro"] for s in order}},
            "prevalence": {"overall": macro_f1(truth, prevalence, np.ones(len(frame), bool))},
            "average_precision": {key: float(average_precision_score(truth[key], scores[key])) for key in ("AE", "PC")},
            "auroc": {key: float(roc_auc_score(truth[key], scores[key])) for key in ("AE", "PC")},
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true", required=True)
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    # Refuse historical softmax caches before any axis subprocess or result write.
    for dataset in ("english", "eu"):
        load_scores(FINAL / "roberta" / f"sigmoid_{dataset}.npz")
    FINAL.mkdir(parents=True, exist_ok=True)

    b0 = run_axis("B0_v3_config", EMB / "spw1_cap0", EMB_EU / "spw1_cap0", None)
    b4 = run_axis("B4_axis", EMB / "spw3_cap60_clean", EMB_EU / "spw3_cap60", "k:8")

    split = pd.read_csv(ROOT / "data" / "benchmarks" / "english_split.csv").set_index("id")["split"]
    english = pd.read_csv(ROOT / "data" / "benchmarks" / "english_gpd_speeches.csv")
    holdout = english[english["id"].map(split).eq("holdout")].reset_index(drop=True)
    eu = pd.read_csv(ROOT / "data" / "benchmarks" / "eu_sentences.csv")

    scores: dict[str, dict[str, np.ndarray]] = {}
    for label, directory in (("B0_v3_config", b0), ("B4_axis", b4)):
        scores[label] = {d: pd.read_csv(directory / f"scores_{d}.csv")[SYSTEM].to_numpy(dtype=float) for d in ("english", "eu")}
        if list(pd.read_csv(directory / "scores_english.csv")["id"]) != list(holdout["id"]):
            raise SystemExit(f"{label}: holdout order mismatch")

    nli_en, _ = nli_document_scores("english_holdout", len(holdout), 8)
    nli_eu, eu_windows = nli_document_scores("eu", len(eu), 1)
    scores["B6_axis_plus_nli"] = {
        "english": (zscore(scores["B4_axis"]["english"]) + zscore(nli_en)) / 2,
        "eu": (zscore(scores["B4_axis"]["eu"]) + zscore(nli_eu)) / 2,
    }
    b6 = FINAL / "B6_axis_plus_nli"
    b6.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": holdout["id"], SYSTEM: scores["B6_axis_plus_nli"]["english"], "nli_only": nli_en}).to_csv(b6 / "scores_english.csv", index=False)
    pd.DataFrame({"id": eu["row_id"], SYSTEM: scores["B6_axis_plus_nli"]["eu"], "nli_only": nli_eu}).to_csv(b6 / "scores_eu.csv", index=False)
    log_access({"timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "label": "B6_axis_plus_nli", "datasets": ["english", "eu"], "english_split": "holdout",
                "final": True, "reason": REASON, "out_dir": str(b6.relative_to(ROOT))})

    # RoBERTa comparator on one-sentence windows (same as the initial evaluation): English anti-elitism top-3, EU populist-total max.
    roberta_dir = FINAL / "roberta"
    rob = {}
    split_all = english["id"].map(split).to_numpy()
    for dataset in ("english", "eu"):
        payload = load_scores(roberta_dir / f"sigmoid_{dataset}.npz")
        doc_index, probs = payload["doc_index"].astype(int), payload["probabilities"].astype(float)
        n = len(english) if dataset == "english" else len(eu)
        if dataset == "english":
            values = probs[:, 1]
            doc = np.zeros(n)
            for i in range(n):
                v = np.sort(values[doc_index == i])[::-1]
                doc[i] = v[:3].mean() if len(v) else 0.0
            rob[dataset] = doc[split_all == "holdout"]
        else:
            rob[dataset] = window_max(doc_index, 1.0 - probs[:, 0], n)
            rob["eu_AE"] = window_max(doc_index, probs[:, 1], n)
            rob["eu_PC"] = window_max(doc_index, probs[:, 2], n)
    scores["RoBERTa_comparator"] = {"english": rob["english"], "eu": rob["eu"]}

    results: dict = {"n_english_holdout": int(len(holdout)), "n_english_holdout_leaders": int(holdout["leader"].nunique()),
                     "n_eu": int(len(eu)), "configurations": {}, "paired_english_holdout": {}}
    for label, per in scores.items():
        results["configurations"][label] = {
            "english_holdout": evaluate_english(per["english"], holdout, 2000, args.seed),
            "eu": evaluate_eu(per["eu"], eu, 2000, args.seed),
        }
    results["configurations"]["NLI_only"] = {"english_holdout": evaluate_english(nli_en, holdout, 2000, args.seed),
                                             "eu": evaluate_eu(nli_eu, eu, 2000, args.seed)}
    target = holdout["source_score"].to_numpy(dtype=float)
    groups = holdout["leader"].astype(str).to_numpy()
    for left, right in (("B4_axis", "B0_v3_config"), ("B6_axis_plus_nli", "B0_v3_config"),
                        ("B6_axis_plus_nli", "B4_axis"), ("B6_axis_plus_nli", "RoBERTa_comparator"),
                        ("B4_axis", "RoBERTa_comparator")):
        payload = paired_delta(scores[left]["english"], scores[right]["english"], target, groups, args.draws, args.seed)
        payload["p_two_sided"] = min(1.0, payload["p_two_sided"])
        results["paired_english_holdout"][f"{left} - {right}"] = payload

    # EU paper protocol (Chalkidis et al. 2025): AE <- E channel, PC <- P channel, sentence = max over windows.
    candidates = {}
    for label, directory in (("B0_v3_config", b0), ("B4_axis", b4)):
        proj = np.load(directory / "projections_eu.npz")
        idx = proj["doc_index"].astype(int)
        candidates[label] = {"AE": window_max(idx, proj["E"].astype(float), len(eu)),
                             "PC": window_max(idx, proj["P"].astype(float), len(eu))}
    top_window = eu_windows[eu_windows["window_rank"] == 0].set_index("doc_index").reindex(range(len(eu)))
    candidates["B6_axis_plus_nli"] = {
        "AE": (zscore(candidates["B4_axis"]["AE"]) + zscore(top_window["E_nli"].fillna(0).to_numpy())) / 2,
        "PC": (zscore(candidates["B4_axis"]["PC"]) + zscore(top_window["P_nli"].fillna(0).to_numpy())) / 2,
    }
    candidates["RoBERTa_comparator"] = {"AE": rob["eu_AE"], "PC": rob["eu_PC"]}
    protocol = eu_protocol(eu, candidates)
    (FINAL / "eu_paper_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    results["eu_paper_protocol_macro_f1_loso"] = {k: v["loso_f1"]["overall"] for k, v in protocol["ours"].items()}
    freeze = json.loads((ROOT / "results" / "step0_env_holdout" / "holdout_freeze.json").read_text(encoding="utf-8"))
    results["references"] = {
        "english_holdout_human_ceiling_spearman": freeze["splits"]["holdout"]["human_ceiling_spearman"],
        "english_holdout_word_count_spearman": freeze["splits"]["holdout"]["word_count_spearman"],
        "eu_speaker_only_auroc_3v0": json.loads((ROOT / "data" / "benchmarks" / "freeze.json").read_text())["null_baselines"]["eu_sentences"]["speaker_only_auroc"],
    }
    (FINAL / "final_results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_run_manifest(FINAL, script="scripts/final_holdout.py",
                       inputs={"split": ROOT / "data" / "benchmarks" / "english_split.csv", "anchors": ANCHORS,
                               "hypotheses": NLI / "hypotheses.json"},
                       extra={"reason": REASON}, filename="final.manifest.json")

    lines = ["| Configuration | English holdout ρ [95% CI] | Extreme AUROC | Leader-term ρ | EU AUROC 3-vs-0 | EU within-speaker AUROC | EU macro-F1 (LOSO) |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for label, payload in results["configurations"].items():
        e, u = payload["english_holdout"], payload["eu"]
        f1 = results["eu_paper_protocol_macro_f1_loso"].get(label, {}).get("macro")
        lines.append(f"| {label} | {e['spearman']:.3f} [{e['spearman_ci95'][0]:.3f}, {e['spearman_ci95'][1]:.3f}] | "
                     f"{e['extreme_auroc']:.3f} | {e['leader_term_spearman']:.3f} | {u['auroc']:.3f} | "
                     f"{u['pooled_within_speaker_auroc']:.3f} | {'–' if f1 is None else f'{f1:.3f}'} |")
    lines += ["", "| English holdout paired Δρ | Δ | 95% CI | p |", "|---|---:|---|---:|"]
    for key, payload in results["paired_english_holdout"].items():
        lines.append(f"| {key} | {payload['delta_spearman']:+.3f} | [{payload['ci95'][0]:+.3f}, {payload['ci95'][1]:+.3f}] | {payload['p_two_sided']:.3f} |")
    (FINAL / "final_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(json.dumps(results["eu_paper_protocol_macro_f1_loso"], indent=2))


if __name__ == "__main__":
    main()
