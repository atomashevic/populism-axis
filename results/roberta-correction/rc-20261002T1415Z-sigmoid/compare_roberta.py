#!/usr/bin/env python3
"""Comparator-only recheck of the ten README RoBERTa rows: legacy softmax vs corrected sigmoid.

Both columns go through the same pooling, labels, splits, bootstrap settings and seeds as
the original evaluation scripts (imported unchanged from scripts/). The legacy column reads
the historical probabilities_*.npz caches (softmax, kept for comparison only); the corrected
column reads sigmoid_*.npz from this run directory via the strict load_scores contract.
Axis and claim-check scores are read from their stored result files and never recomputed.
Writes comparison.json, comparison.md and compare.manifest.json next to this script.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from populism_axis.contextual_axis import aggregate_from_window_scores  # noqa: E402
from populism_axis.roberta_comparator import load_scores  # noqa: E402
from populism_axis.run_manifest import write_run_manifest  # noqa: E402
from evaluate_benchmarks import _auroc_metric, cluster_bootstrap, evaluate_agora, evaluate_english, evaluate_eu  # noqa: E402
from evaluate_eu_paper_protocol import window_max  # noqa: E402
from paired_comparison import paired_delta  # noqa: E402

SEED = 20260923
AGORA_SEED = 20260727  # seed of the historical Agora RoBERTa evaluation
FINAL = ROOT / "results" / "final"
EXT = ROOT / "results" / "external"
LEGACY = {"english": FINAL / "roberta", "eu": FINAL / "roberta", "agora": FINAL / "roberta",
          "usgov_speeches": EXT / "usgov" / "roberta", "antielite_tweets": EXT / "antielite_tweets" / "roberta",
          "popbert_en_train": EXT / "popbert_en" / "roberta", "popbert_en_test": EXT / "popbert_en" / "roberta",
          "portugal_en": EXT / "portugal" / "roberta"}


def module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


final_holdout = module(ROOT / "scripts" / "final_holdout.py")
usgov = module(ROOT / "scripts" / "external" / "score_usgov.py")
tweets = module(ROOT / "scripts" / "external" / "score_antielite_tweets.py")
popbert = module(ROOT / "scripts" / "external" / "score_popbert_en.py")
portugal = module(ROOT / "scripts" / "external" / "score_portugal.py")


def windows(name: str, version: str) -> tuple[np.ndarray, np.ndarray]:
    if version == "softmax":
        with np.load(LEGACY[name] / f"probabilities_{name}.npz") as payload:
            return payload["doc_index"].astype(int), payload["probabilities"].astype(float)
    payload = load_scores(HERE / f"sigmoid_{name}.npz")
    return payload["doc_index"].astype(int), payload["probabilities"].astype(float)


def pooled(name: str, version: str, n: int, component: int, rule: str) -> np.ndarray:
    doc_index, probs = windows(name, version)
    values = 1.0 - probs[:, 0] if component < 0 else probs[:, component]
    return aggregate_from_window_scores(values, doc_index, n, rule)


def english_top3(version: str, n: int) -> np.ndarray:
    # Same loop as final_holdout.py (anti-elitism, top-3 mean).
    doc_index, probs = windows("english", version)
    values, doc = probs[:, 1], np.zeros(n)
    for i in range(n):
        v = np.sort(values[doc_index == i])[::-1]
        doc[i] = v[:3].mean() if len(v) else 0.0
    return doc


def tweets_paired(systems: dict[str, np.ndarray], frame: pd.DataFrame, test: np.ndarray, draws: int) -> dict:
    # Replays score_antielite_tweets.py: one RNG shared across the pair list in order.
    y = frame.loc[test, "y_general"].to_numpy(dtype=int)
    groups = frame.loc[test, "country"].astype(str).to_numpy()
    rng = np.random.default_rng(SEED); unique = np.unique(groups); index_by = {g: np.flatnonzero(groups == g) for g in unique}
    pairs = [("E_axis_plus_nli", "E_axis"), ("E_axis_plus_nli", "PE_axis_plus_nli"), ("E_axis_plus_nli", "RoBERTa_anti_elitism"),
             ("E_axis", "RoBERTa_anti_elitism")]
    out = {}
    for left, right in pairs:
        l, r = systems[left][test], systems[right][test]
        observed = roc_auc_score(y, l) - roc_auc_score(y, r); samples = []
        for _ in range(draws):
            idx = np.concatenate([index_by[g] for g in rng.choice(unique, size=len(unique), replace=True)])
            if len(np.unique(y[idx])) < 2:
                continue
            samples.append(roc_auc_score(y[idx], l[idx]) - roc_auc_score(y[idx], r[idx]))
        arr = np.asarray(samples)
        out[f"{left} - {right}"] = {"delta": float(observed), "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
                                    "p_two_sided": float(min(1.0, 2 * min((arr <= 0).mean(), (arr >= 0).mean())))}
    return out


def popbert_paired(a: np.ndarray, b: np.ndarray, y: np.ndarray, draws: int) -> dict:
    # Same loop as score_popbert_en.py (sentence bootstrap, fresh RNG per pair).
    rng = np.random.default_rng(SEED); deltas = []
    for _ in range(draws):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            deltas.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    deltas = np.asarray(deltas)
    return {"delta": float(roc_auc_score(y, a) - roc_auc_score(y, b)), "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
            "p_two_sided": float(min(1.0, 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())))}


def main() -> None:
    rows: dict[str, dict] = {}
    checks: dict[str, float] = {}

    # 1-3: English holdout and EU-OOD (scripts/final_holdout.py).
    split = pd.read_csv(ROOT / "data" / "benchmarks" / "english_split.csv").set_index("id")["split"]
    english = pd.read_csv(ROOT / "data" / "benchmarks" / "english_gpd_speeches.csv")
    is_holdout = english["id"].map(split).eq("holdout").to_numpy()
    holdout = english[is_holdout].reset_index(drop=True)
    b6_en = pd.read_csv(FINAL / "B6_axis_plus_nli" / "scores_english.csv")
    b4_en = pd.read_csv(FINAL / "B4_axis" / "scores_english.csv")
    assert list(b6_en["id"]) == list(holdout["id"]) == list(b4_en["id"])
    target, groups = holdout["source_score"].to_numpy(float), holdout["leader"].astype(str).to_numpy()
    eu = pd.read_csv(ROOT / "data" / "benchmarks" / "eu_sentences.csv")
    b6_eu = pd.read_csv(FINAL / "B6_axis_plus_nli" / "scores_eu.csv")
    stored_final = json.loads((FINAL / "final_results.json").read_text())
    stored_protocol = json.loads((FINAL / "eu_paper_protocol.json").read_text())
    for version in ("softmax", "sigmoid"):
        rob_en = english_top3(version, len(english))[is_holdout]
        ev = evaluate_english(rob_en, holdout, 2000, SEED)
        p_b6 = paired_delta(b6_en["PE__frozen__soft"].to_numpy(float), rob_en, target, groups, 4000, SEED)
        p_b4 = paired_delta(b4_en["PE__frozen__soft"].to_numpy(float), rob_en, target, groups, 4000, SEED)
        rows.setdefault("english_holdout", {})[version] = {
            "metric": ev["spearman"], "ci95": ev["spearman_ci95"],
            "paired": {"B6_axis_plus_nli - RoBERTa": p_b6, "B4_axis - RoBERTa": p_b4},
            "secondary": {"extreme_auroc": ev["extreme_auroc"], "leader_term_spearman": ev["leader_term_spearman"]}}
        doc_index, probs = windows("eu", version)
        rob_eu = window_max(doc_index, 1.0 - probs[:, 0], len(eu))
        ev = evaluate_eu(rob_eu, eu, 2000, SEED)
        rows.setdefault("eu_auroc_3v0", {})[version] = {"metric": ev["auroc"], "ci95": ev["auroc_speaker_clustered_ci95"],
                                                        "secondary": {"pooled_within_speaker_auroc": ev["pooled_within_speaker_auroc"]}}
        protocol = final_holdout.eu_protocol(eu, {"RoBERTa": {"AE": window_max(doc_index, probs[:, 1], len(eu)),
                                                              "PC": window_max(doc_index, probs[:, 2], len(eu))}})["ours"]["RoBERTa"]
        rows.setdefault("eu_loso_macro_f1", {})[version] = {"metric": protocol["loso_f1"]["overall"]["macro"], "ci95": None,
                                                            "secondary": {"loso_overall": protocol["loso_f1"]["overall"],
                                                                          "prevalence_macro_f1": protocol["prevalence"]["overall"]["macro"]}}
    ours = {"english_holdout": stored_final["configurations"]["B6_axis_plus_nli"]["english_holdout"]["spearman"],
            "eu_auroc_3v0": stored_final["configurations"]["B6_axis_plus_nli"]["eu"]["auroc"],
            "eu_loso_macro_f1": stored_protocol["ours"]["B6_axis_plus_nli"]["loso_f1"]["overall"]["macro"]}
    checks["english_softmax_vs_stored"] = rows["english_holdout"]["softmax"]["metric"] - stored_final["configurations"]["RoBERTa_comparator"]["english_holdout"]["spearman"]
    checks["english_paired_softmax_vs_stored"] = rows["english_holdout"]["softmax"]["paired"]["B6_axis_plus_nli - RoBERTa"]["ci95"][0] - stored_final["paired_english_holdout"]["B6_axis_plus_nli - RoBERTa_comparator"]["ci95"][0]
    checks["eu_softmax_vs_stored"] = rows["eu_auroc_3v0"]["softmax"]["metric"] - stored_final["configurations"]["RoBERTa_comparator"]["eu"]["auroc"]
    checks["eu_loso_softmax_vs_stored"] = rows["eu_loso_macro_f1"]["softmax"]["metric"] - stored_protocol["ours"]["RoBERTa_comparator"]["loso_f1"]["overall"]["macro"]
    checks["english_b6_recomputed_vs_stored"] = float(pd.Series(b6_en["PE__frozen__soft"]).corr(pd.Series(holdout["source_score"]), method="spearman")) - ours["english_holdout"]

    # 4: Agora (historical harness: populist total = 1 - p0, max over windows; development set).
    agora = pd.read_csv(ROOT / "data" / "benchmarks" / "agora_paragraphs.csv")
    agora_combined = pd.read_csv(ROOT / "results" / "step6_nli" / "combined_k8" / "scores_agora.csv")
    assert len(agora_combined) == len(agora)
    combined = agora_combined["PE__frozen__soft"].to_numpy(float)
    ours["agora"] = evaluate_agora(combined, agora, 2000, SEED)["spearman"]
    for version in ("softmax", "sigmoid"):
        doc_index, probs = windows("agora", version)
        rob = window_max(doc_index, 1.0 - probs[:, 0], len(agora))
        ev = evaluate_agora(rob, agora, 2000, AGORA_SEED)
        supplementary = paired_delta(combined, rob, agora["populism_human"].to_numpy(float), agora["politician"].astype(str).to_numpy(), 4000, SEED)
        rows.setdefault("agora", {})[version] = {"metric": ev["spearman"], "ci95": ev["spearman_ci95"],
                                                 "paired_supplementary_not_previously_reported": {"combined - RoBERTa": supplementary}}

    # 5: US governors 2010-18 (scripts/external/score_usgov.py, 4000 draws).
    speeches = pd.read_csv(ROOT / "data" / "benchmark_usgov" / "speeches.csv")
    stored = pd.read_csv(EXT / "usgov" / "final_scores_speeches.csv")
    assert list(stored["id"]) == list(speeches["id"])
    mask = (speeches["subset"] == "governors_2010_2018").to_numpy()
    frame = speeches[mask].reset_index(drop=True)
    b6 = stored["B6_axis_plus_nli"].to_numpy(float)[mask]
    ours["usgov"] = usgov.speech_metrics(b6, frame, 4000, SEED)["spearman"]
    for version in ("softmax", "sigmoid"):
        rob_all = pooled("usgov_speeches", version, len(speeches), 1, "k:3")
        if version == "softmax":
            checks["usgov_softmax_vs_stored_column"] = float(np.abs(rob_all - stored["RoBERTa_anti_elitism_top3"].to_numpy(float)).max())
        m = usgov.speech_metrics(rob_all[mask], frame, 4000, SEED)
        p = paired_delta(b6, rob_all[mask], frame["score"].to_numpy(float), frame["group"].astype(str).to_numpy(), 4000, SEED)
        rows.setdefault("usgov_2010_2018", {})[version] = {"metric": m["spearman"], "ci95": m["spearman_ci95"],
                                                          "paired": {"B6_axis_plus_nli - RoBERTa": p}}

    # 6: Anti-elite tweets, fixed test split (scripts/external/score_antielite_tweets.py).
    tw = pd.read_csv(EXT / "antielite_tweets" / "final_scores.csv")
    test = (tw["split"] == "test").to_numpy(); train = ~test
    ours["tweets"] = float(roc_auc_score(tw.loc[test, "y_general"], tw.loc[test, "E_axis_plus_nli"]))
    for version in ("softmax", "sigmoid"):
        rob = pooled("antielite_tweets", version, len(tw), 1, "k:1")
        if version == "softmax":
            checks["tweets_softmax_vs_stored_column"] = float(np.abs(rob - tw["RoBERTa_anti_elitism"].to_numpy(float)).max())
        ev = tweets.evaluate(rob[test], tw[test].reset_index(drop=True), "y_general", rob[train], tw[train].reset_index(drop=True), 2000, SEED)
        systems = {c: tw[c].to_numpy(float) for c in ("E_axis", "E_axis_plus_nli", "PE_axis_plus_nli")} | {"RoBERTa_anti_elitism": rob}
        p = tweets_paired(systems, tw, test, 2000)
        rows.setdefault("tweets", {})[version] = {"metric": ev["auroc"], "ci95": ev["auroc_ci95_country"],
                                                  "paired": {k: v for k, v in p.items() if "RoBERTa" in k},
                                                  "secondary": {"f1opt_train_f1_pos": ev["thresholds"]["f1opt_train"]["f1_pos"]}}

    # 7-8: PopBERT test (scripts/external/score_popbert_en.py).
    pb = pd.read_csv(EXT / "popbert_en" / "final_scores.csv")
    n_by_split = {s: int((pb["split"] == s).sum()) for s in ("train", "test")}
    assert list(pb["split"]) == ["train"] * n_by_split["train"] + ["test"] * n_by_split["test"]
    test = (pb["split"] == "test").to_numpy(); train = ~test
    for dim, channel, component in (("elite", "E", 1), ("centr", "P", 2)):
        key = f"popbert_{dim}"
        y_test, y_train = pb.loc[test, f"{dim}_gold"].to_numpy(int), pb.loc[train, f"{dim}_gold"].to_numpy(int)
        combined = pb[f"{channel}_axis_plus_nli"].to_numpy(float)
        ours[key] = float(roc_auc_score(y_test, combined[test]))
        for version in ("softmax", "sigmoid"):
            rob = np.concatenate([pooled(f"popbert_en_{s}", version, n_by_split[s], component, "k:1") for s in ("train", "test")])
            if version == "softmax":
                checks[f"{key}_softmax_vs_stored_column"] = float(np.abs(rob - pb[f"{channel}_roberta"].to_numpy(float)).max())
            ev = popbert.evaluate(rob, y_test, y_train, test, train, 2000, SEED)
            p = popbert_paired(combined[test], rob[test], y_test, 2000)
            rows.setdefault(key, {})[version] = {"metric": ev["auroc"], "ci95": ev["auroc_ci95"],
                                                 "paired": {f"{channel}_axis_plus_nli - {channel}_roberta": p},
                                                 "secondary": {"f1_train_optimal": ev["f1_train_optimal"]["f1"]}}

    # 9-10: Portugal (scripts/external/score_portugal.py; actor-clustered).
    pt = pd.read_csv(EXT / "portugal" / "final_scores.csv")
    groups = pt["actor"].astype(str).to_numpy()
    for dim, channel, component in (("elite", "E", 1), ("centr", "P", 2)):
        key = f"portugal_{dim}"
        y = pt[f"{dim}_gold"].to_numpy(int)
        combined = pt[f"{channel}_axis_plus_nli"].to_numpy(float)
        ours[key] = float(roc_auc_score(y, combined))
        for version in ("softmax", "sigmoid"):
            rob = pooled("portugal_en", version, len(pt), component, "k:1")
            if version == "softmax":
                checks[f"{key}_softmax_vs_stored_column"] = float(np.abs(rob - pt[f"{channel}_roberta"].to_numpy(float)).max())
            ev = portugal.evaluate(rob, y, groups, 2000, SEED)
            p = portugal.paired(combined, rob, y, groups, 2000, SEED)
            rows.setdefault(key, {})[version] = {"metric": ev["auroc"], "ci95": ev["auroc_ci95"],
                                                 "paired": {f"{channel}_axis_plus_nli - {channel}_roberta": p}}

    labels = {"english_holdout": "GPD English holdout, Spearman", "eu_auroc_3v0": "EU-OOD AUROC 3-vs-0",
              "eu_loso_macro_f1": "EU-OOD LOSO macro-F1", "agora": "Agora paragraphs, Spearman",
              "usgov_2010_2018": "US governors 2010-18, Spearman", "tweets": "Anti-elite tweets AUROC (E)",
              "popbert_elite": "PopBERT anti-elitism AUROC (E)", "popbert_centr": "PopBERT people-centrism AUROC (P)",
              "portugal_elite": "Portugal anti-elitism AUROC (E)", "portugal_centr": "Portugal people-centrism AUROC (P)"}
    ours_key = {"usgov_2010_2018": "usgov", "tweets": "tweets", "agora": "agora"}
    summary = []
    for key, label in labels.items():
        old, new, mine = rows[key]["softmax"]["metric"], rows[key]["sigmoid"]["metric"], ours[ours_key.get(key, key)]
        summary.append({"row": key, "label": label, "axis_plus_claim_check": mine, "roberta_softmax": old, "roberta_sigmoid": new,
                        "sigmoid_minus_softmax": new - old, "ours_better_softmax": bool(mine > old), "ours_better_sigmoid": bool(mine > new)})
    result = {"run_dir": str(HERE.relative_to(ROOT)), "seed": SEED, "agora_seed": AGORA_SEED, "summary": summary, "rows": rows,
              "axis_plus_claim_check_stored": ours, "reproduction_checks": checks,
              "wins": {"softmax": sum(r["ours_better_softmax"] for r in summary), "sigmoid": sum(r["ours_better_sigmoid"] for r in summary)}}
    (HERE / "comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def fmt_ci(ci):
        return "" if not ci else f" [{ci[0]:.3f}, {ci[1]:.3f}]"
    lines = ["| Row | Axis + claim check | RoBERTa softmax (old) | RoBERTa sigmoid (corrected) | Corrected − old | Ours better: old / corrected |",
             "|---|---:|---:|---:|---:|---|"]
    for r in summary:
        lines.append(f"| {r['label']} | {r['axis_plus_claim_check']:.3f} | {r['roberta_softmax']:.3f}{fmt_ci(rows[r['row']]['softmax']['ci95'])} | "
                     f"{r['roberta_sigmoid']:.3f}{fmt_ci(rows[r['row']]['sigmoid']['ci95'])} | {r['sigmoid_minus_softmax']:+.3f} | "
                     f"{'yes' if r['ours_better_softmax'] else 'no'} / {'yes' if r['ours_better_sigmoid'] else 'no'} |")
    lines += ["", f"Rows favouring axis + claim check: softmax {result['wins']['softmax']}/10, sigmoid {result['wins']['sigmoid']}/10.", "",
              "| Paired comparison (ours − RoBERTa) | Old Δ [95% CI] | Corrected Δ [95% CI] |", "|---|---|---|"]
    for key in labels:
        for name in rows[key]["softmax"].get("paired", {}):
            o, c = rows[key]["softmax"]["paired"][name], rows[key]["sigmoid"]["paired"][name]
            od, cd = o.get("delta", o.get("delta_spearman")), c.get("delta", c.get("delta_spearman"))
            lines.append(f"| {labels[key]}: {name} | {od:+.3f} [{o['ci95'][0]:+.3f}, {o['ci95'][1]:+.3f}] | {cd:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] |")
    for version in ("softmax", "sigmoid"):
        s = rows["agora"][version]["paired_supplementary_not_previously_reported"]["combined - RoBERTa"]
        lines.append(f"| Agora (supplementary, not previously reported), {version}: combined − RoBERTa | {s['delta_spearman']:+.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}] | |")
    (HERE / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_run_manifest(HERE, script=str((HERE / "compare_roberta.py").relative_to(ROOT)),
                       inputs={f"sigmoid_{n}": HERE / f"sigmoid_{n}.npz" for n in LEGACY} | {f"legacy_{n}": LEGACY[n] / f"probabilities_{n}.npz" for n in LEGACY},
                       extra={"seed": SEED, "agora_seed": AGORA_SEED, "wins": result["wins"]}, filename="compare.manifest.json")
    print("\n".join(lines))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
