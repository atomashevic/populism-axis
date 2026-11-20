#!/usr/bin/env python3
"""Bootstrap intervals for the main benchmark results, with texts resampled within speakers.

The intervals of the main benchmark results were computed by resampling whole clusters: GPD leaders,
US state-terms, AgoraSpeech politicians, EU-OOD speakers, tweet countries and Portuguese speakers.
This script recomputes them from the committed per-item scores with the texts resampled within
those clusters (src/populism_axis/resampling.py): every speaker stays in every resample, and the
interval describes uncertainty about the texts given the speakers of a benchmark. PopBERT has no
speaker identifiers, so its test sentences form one stratum and are resampled one by one, as before.

For each interval the script also recomputes the cluster-resampled interval with the same seed and
number of draws, and it records the stored interval that the within-speaker one replaces, with its
file and path; paper/scripts/make_numbers.py reads the within-speaker interval in its place. Every
point estimate must equal the stored one. The EU-OOD macro-F1 intervals come from
scripts/posthoc_sensitivity.py; the development steps keep the intervals used for their decisions.

Outputs: results/within_speaker_intervals/{intervals.json, intervals.md, run_manifest.json}.
Run from the repository root:  uv run --extra cpu python scripts/within_speaker_intervals.py
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from populism_axis.resampling import sampler  # noqa: E402
from populism_axis.run_manifest import write_run_manifest  # noqa: E402

OUT = ROOT / "results/within_speaker_intervals"
FINAL = "results/final/final_results.json"
CMP = "results/roberta-correction/rc-20261002T1415Z-sigmoid/comparison.json"
GOV = "results/external/usgov/final_results.json"
TW = "results/external/antielite_tweets/final_results.json"
PB = "results/external/popbert_en/final_results.json"
PT = "results/external/portugal/final_results.json"
SCORES = {"english_b6": "results/final/B6_axis_plus_nli/scores_english.csv", "english_b4": "results/final/B4_axis/scores_english.csv",
          "english_b0": "results/final/B0_v3_config/scores_english.csv", "english_roberta": "results/final/roberta/scores_english.csv",
          "eu_b6": "results/final/B6_axis_plus_nli/scores_eu.csv", "eu_roberta": "results/final/roberta/scores_eu.csv",
          "agora": "results/final/scores_agora.csv", "usgov": "results/external/usgov/final_scores_speeches.csv",
          "tweets": "results/external/antielite_tweets/final_scores.csv", "popbert": "results/external/popbert_en/final_scores.csv",
          "portugal": "results/external/portugal/final_scores.csv"}
DATA = {"english": "data/benchmarks/english_gpd_speeches.csv", "split": "data/benchmarks/english_split.csv",
        "eu": "data/benchmarks/eu_sentences.csv", "agora": "data/benchmarks/agora_paragraphs.csv"}


def spearman(y: np.ndarray, s: np.ndarray) -> float:
    return float(spearmanr(s, y).statistic)


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """AUROC from average ranks (Mann-Whitney), equal to sklearn's roc_auc_score with ties counted half."""
    positive = y == 1
    n1, n0 = int(positive.sum()), int((~positive).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((rankdata(s)[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@dataclass
class Benchmark:
    name: str
    label: str
    metric: str
    y: np.ndarray
    systems: dict[str, np.ndarray]
    strata: np.ndarray       # texts are resampled within these
    clusters: np.ndarray     # whole clusters resampled by the stored intervals
    stratum: str


def read(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel)


def benchmarks() -> list[Benchmark]:
    out = []
    split = read(DATA["split"]).set_index("id")["split"]
    english = read(DATA["english"])
    holdout = english[english["id"].map(split).eq("holdout")].reset_index(drop=True)
    b6, b4, b0, rob = (read(SCORES[k]) for k in ("english_b6", "english_b4", "english_b0", "english_roberta"))
    assert list(b6.id) == list(b4.id) == list(b0.id) == list(rob.id) == list(holdout.id)
    leaders = holdout.leader.astype(str).to_numpy()
    out.append(Benchmark("english_holdout", "GPD English evaluation speeches", "spearman", holdout.source_score.to_numpy(float),
                         {"B6_axis_plus_nli": b6.PE__frozen__soft.to_numpy(float), "B4_axis": b4.PE__frozen__soft.to_numpy(float),
                          "NLI_only": b6.nli_only.to_numpy(float), "B0_v3_config": b0.PE__frozen__soft.to_numpy(float),
                          "RoBERTa": rob.RoBERTa_anti_elitism_top3.to_numpy(float)}, leaders, leaders, "leader"))

    eu = read(DATA["eu"])
    b6_eu, rob_eu = read(SCORES["eu_b6"]), read(SCORES["eu_roberta"])
    assert list(b6_eu.id) == list(eu.row_id) == list(rob_eu.id)
    m = eu.pop_code.isin([0, 3]).to_numpy()
    speakers = eu.speaker.astype(str).to_numpy()[m]
    out.append(Benchmark("eu_both_vs_neither", "EU-OOD, both claims vs neither", "auroc", (eu.pop_code[m] == 3).astype(int).to_numpy(),
                         {"B6_axis_plus_nli": b6_eu.PE__frozen__soft.to_numpy(float)[m],
                          "RoBERTa": rob_eu.RoBERTa_populist_total.to_numpy(float)[m]}, speakers, speakers, "speaker"))

    agora, ag = read(DATA["agora"]), read(SCORES["agora"])
    assert list(ag.row_id) == list(agora.row_id)
    politicians = agora.politician.astype(str).to_numpy()
    out.append(Benchmark("agora", "AgoraSpeech paragraphs (development)", "spearman", agora.populism_human.to_numpy(float),
                         {"combined": ag.combined.to_numpy(float), "RoBERTa": ag.RoBERTa_populist_total.to_numpy(float)},
                         politicians, politicians, "politician"))

    us = read(SCORES["usgov"])
    gov = us[us.subset == "governors_2010_2018"].reset_index(drop=True)
    terms = gov.group.astype(str).to_numpy()
    out.append(Benchmark("usgov_governors_2010_2018", "US governor speeches 2010-18", "spearman", gov.score.to_numpy(float),
                         {k: gov[k].to_numpy(float) for k in ("B6_axis_plus_nli", "B4_axis", "NLI_only", "B0_v3_config")}
                         | {"RoBERTa": gov.RoBERTa_anti_elitism_top3.to_numpy(float)}, terms, terms, "state-term"))
    sf = gov[gov.SetFit_share_published.notna()].reset_index(drop=True)
    sf_terms = sf.group.astype(str).to_numpy()
    out.append(Benchmark("usgov_setfit_subset", "US governor speeches 2010-18 with SetFit predictions", "spearman", sf.score.to_numpy(float),
                         {k: sf[k].to_numpy(float) for k in ("B6_axis_plus_nli", "SetFit_share_published")}, sf_terms, sf_terms, "state-term"))

    tw = read(SCORES["tweets"])
    test = tw[tw.split == "test"].reset_index(drop=True)
    countries = test.country.astype(str).to_numpy()
    out.append(Benchmark("tweets", "Party tweets, general elite criticism", "auroc", test.y_general.to_numpy(int),
                         {k: test[k].to_numpy(float) for k in ("E_axis_plus_nli", "XLMT_released_classifier", "RoBERTa_anti_elitism")},
                         countries, countries, "country"))

    pb = read(SCORES["popbert"])
    pb = pb[pb.split == "test"].reset_index(drop=True)
    pt = read(SCORES["portugal"])
    actors = pt.actor.astype(str).to_numpy()
    for dim, ch, label in (("elite", "E", "anti-elitism"), ("centr", "P", "people-centrism")):
        out.append(Benchmark(f"popbert_{dim}", f"PopBERT test sentences, {label}", "auroc", pb[f"{dim}_gold"].to_numpy(int),
                             {k: pb[k].to_numpy(float) for k in (f"{ch}_axis_plus_nli", f"{ch}_nli", f"{ch}_roberta")},
                             np.full(len(pb), "all"), np.arange(len(pb)).astype(str), "none (one stratum)"))
        out.append(Benchmark(f"portugal_{dim}", f"Portuguese parliamentary paragraphs, {label}", "auroc", pt[f"{dim}_gold"].to_numpy(int),
                             {k: pt[k].to_numpy(float) for k in (f"{ch}_axis_plus_nli", f"supervised_lr_{dim}", f"{ch}_roberta")},
                             actors, actors, "speaker"))
    return out


# (benchmark, statistic, [(stored file, path of the stored interval)]); a statistic "a - b" is a paired difference.
ENTRIES = [
    ("english_holdout", "B6_axis_plus_nli", [(FINAL, "configurations/B6_axis_plus_nli/english_holdout/spearman_ci95")]),
    ("english_holdout", "B4_axis", [(FINAL, "configurations/B4_axis/english_holdout/spearman_ci95")]),
    ("english_holdout", "NLI_only", [(FINAL, "configurations/NLI_only/english_holdout/spearman_ci95")]),
    ("english_holdout", "B0_v3_config", [(FINAL, "configurations/B0_v3_config/english_holdout/spearman_ci95")]),
    ("english_holdout", "RoBERTa", [(FINAL, "configurations/RoBERTa_comparator/english_holdout/spearman_ci95"),
                                    (CMP, "rows/english_holdout/sigmoid/ci95")]),
    ("english_holdout", "B6_axis_plus_nli - B0_v3_config", [(FINAL, "paired_english_holdout/B6_axis_plus_nli - B0_v3_config/ci95")]),
    ("english_holdout", "B6_axis_plus_nli - B4_axis", [(FINAL, "paired_english_holdout/B6_axis_plus_nli - B4_axis/ci95")]),
    ("english_holdout", "B6_axis_plus_nli - RoBERTa", [(FINAL, "paired_english_holdout/B6_axis_plus_nli - RoBERTa_comparator/ci95"),
                                                       (CMP, "rows/english_holdout/sigmoid/paired/B6_axis_plus_nli - RoBERTa/ci95")]),
    ("english_holdout", "B4_axis - RoBERTa", [(FINAL, "paired_english_holdout/B4_axis - RoBERTa_comparator/ci95")]),
    ("english_holdout", "B4_axis - B0_v3_config", [(FINAL, "paired_english_holdout/B4_axis - B0_v3_config/ci95")]),
    ("eu_both_vs_neither", "B6_axis_plus_nli", [(FINAL, "configurations/B6_axis_plus_nli/eu/auroc_speaker_clustered_ci95")]),
    ("eu_both_vs_neither", "RoBERTa", [(FINAL, "configurations/RoBERTa_comparator/eu/auroc_speaker_clustered_ci95"),
                                       (CMP, "rows/eu_auroc_3v0/sigmoid/ci95")]),
    ("eu_both_vs_neither", "B6_axis_plus_nli - RoBERTa", []),
    ("agora", "combined", []),
    ("agora", "RoBERTa", [(CMP, "rows/agora/sigmoid/ci95")]),
    ("agora", "combined - RoBERTa", [(CMP, "rows/agora/sigmoid/paired_supplementary_not_previously_reported/combined - RoBERTa/ci95")]),
    *[("usgov_governors_2010_2018", s, [(GOV, f"speeches/governors_2010_2018/{s}/spearman_ci95")])
      for s in ("B6_axis_plus_nli", "B4_axis", "NLI_only", "B0_v3_config")],
    ("usgov_governors_2010_2018", "RoBERTa", [(CMP, "rows/usgov_2010_2018/sigmoid/ci95")]),
    ("usgov_governors_2010_2018", "B6_axis_plus_nli - RoBERTa", [(CMP, "rows/usgov_2010_2018/sigmoid/paired/B6_axis_plus_nli - RoBERTa/ci95")]),
    ("usgov_setfit_subset", "SetFit_share_published", [(GOV, "speeches/governors_2010_2018/SetFit_share_published/spearman_ci95")]),
    ("usgov_setfit_subset", "B6_axis_plus_nli - SetFit_share_published",
     [(GOV, "paired_spearman/governors_2010_2018/B6_axis_plus_nli - SetFit_share_published/ci95")]),
    ("tweets", "E_axis_plus_nli", [(TW, "test/y_general/E_axis_plus_nli/auroc_ci95_country")]),
    ("tweets", "RoBERTa_anti_elitism", [(CMP, "rows/tweets/sigmoid/ci95")]),
    ("tweets", "E_axis_plus_nli - XLMT_released_classifier", [(TW, "paired_auroc_test/E_axis_plus_nli - XLMT_released_classifier/ci95")]),
    ("tweets", "E_axis_plus_nli - RoBERTa_anti_elitism", [(CMP, "rows/tweets/sigmoid/paired/E_axis_plus_nli - RoBERTa_anti_elitism/ci95")]),
    *[e for dim, ch in (("elite", "E"), ("centr", "P")) for e in (
        (f"popbert_{dim}", f"{ch}_axis_plus_nli", [(PB, f"test/{dim}/majority/{ch}_axis_plus_nli/auroc_ci95")]),
        (f"popbert_{dim}", f"{ch}_roberta", [(CMP, f"rows/popbert_{dim}/sigmoid/ci95")]),
        (f"popbert_{dim}", f"{ch}_axis_plus_nli - {ch}_roberta", [(CMP, f"rows/popbert_{dim}/sigmoid/paired/{ch}_axis_plus_nli - {ch}_roberta/ci95")]),
        (f"portugal_{dim}", f"{ch}_axis_plus_nli", [(PT, f"primary/{dim}/{ch}_axis_plus_nli/auroc_ci95")]),
        (f"portugal_{dim}", f"{ch}_roberta", [(CMP, f"rows/portugal_{dim}/sigmoid/ci95")]),
        (f"portugal_{dim}", f"{ch}_axis_plus_nli - supervised_lr_{dim}", [(PT, f"paired_auroc/{dim}/{ch}_axis_plus_nli - supervised_lr_{dim}/ci95")]),
        (f"portugal_{dim}", f"{ch}_axis_plus_nli - {ch}_roberta", [(CMP, f"rows/portugal_{dim}/sigmoid/paired/{ch}_axis_plus_nli - {ch}_roberta/ci95")]))],
    ("popbert_elite", "E_nli - E_axis_plus_nli", [(PB, "paired_auroc_test/elite: E_nli - E_axis_plus_nli/ci95")]),
]


def get(obj, path: str):
    for part in path.split("/"):
        obj = obj[part]
    return obj


def stored_estimate(rel: str, path: str) -> float:
    """The point estimate stored next to an interval."""
    parent, key = path.rsplit("/", 1)
    siblings = get(json.loads((ROOT / rel).read_text()), parent)
    for name in (key.replace("_ci95_country", "").replace("_speaker_clustered_ci95", "").replace("_ci95", ""),
                 "metric", "delta", "delta_spearman", "delta_auroc"):
        if name in siblings and not name.endswith("ci95"):
            return float(siblings[name])
    raise KeyError(f"{rel}: no estimate next to {path}")


def resample(b: Benchmark, groups: np.ndarray, scheme: str, draws: int, seed: int) -> np.ndarray:
    """Metric of every system in each resample (rows), skipping resamples in which any is undefined."""
    metric = spearman if b.metric == "spearman" else auroc
    draw, rng = sampler(groups, scheme), np.random.default_rng(seed)
    names, rows = list(b.systems), []
    for _ in range(draws):
        ix = draw(rng)
        row = [metric(b.y[ix], b.systems[n][ix]) for n in names]
        if np.all(np.isfinite(row)):
            rows.append(row)
    return np.asarray(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    bench = {b.name: b for b in benchmarks()}
    samples, summary = {}, {}
    for b in bench.values():
        for name, s in b.systems.items():
            check = spearman(b.y, s) if b.metric == "spearman" else roc_auc_score(b.y, s)
            if b.metric == "auroc" and abs(check - auroc(b.y, s)) > 1e-12:
                raise SystemExit(f"{b.name}/{name}: rank AUROC differs from sklearn")
        samples[b.name] = {"within": resample(b, b.strata, "within", args.draws, args.seed),
                           "cluster": resample(b, b.clusters, "strata", args.draws, args.seed)}
        summary[b.name] = {"label": b.label, "metric": b.metric, "n": int(len(b.y)), "stratum": b.stratum,
                           "n_strata": int(len(np.unique(b.strata))), "n_clusters": int(len(np.unique(b.clusters))),
                           "valid_draws": {k: int(len(v)) for k, v in samples[b.name].items()}}
        print(f"{b.name:28s} n={len(b.y):5d} strata={summary[b.name]['n_strata']:4d}  valid draws {summary[b.name]['valid_draws']}", flush=True)

    entries, gaps = {}, []
    for name, statistic, replaces in ENTRIES:
        b = bench[name]
        metric = spearman if b.metric == "spearman" else auroc
        names = list(b.systems)
        parts = statistic.split(" - ")
        estimate = metric(b.y, b.systems[parts[0]]) - (metric(b.y, b.systems[parts[1]]) if len(parts) == 2 else 0.0)

        def interval(scheme):
            s = samples[name][scheme]
            v = s[:, names.index(parts[0])] - (s[:, names.index(parts[1])] if len(parts) == 2 else 0.0)
            return [float(x) for x in np.percentile(v, [2.5, 97.5])]

        record = {"benchmark": name, "statistic": statistic, "metric": b.metric, "estimate": float(estimate),
                  "within_ci95": interval("within"), "cluster_ci95": interval("cluster"), "replaces": []}
        for rel, path in replaces:
            stored = get(json.loads((ROOT / rel).read_text()), path)
            ref = stored_estimate(rel, path)
            if abs(ref - estimate) > 1e-9:
                raise SystemExit(f"{name}/{statistic}: estimate {estimate:.6f}, stored {ref:.6f} at {rel}:{path}")
            gap = max(abs(a - c) for a, c in zip(stored, record["cluster_ci95"]))
            gaps.append(gap)
            record["replaces"].append({"file": rel, "path": path, "stored_ci95": [float(x) for x in stored],
                                       "stored_estimate": ref, "max_gap_cluster_recomputed_vs_stored": float(gap)})
        entries[f"{name}:{statistic}"] = record
    if max(gaps) > .03:
        raise SystemExit(f"the recomputed cluster intervals do not reproduce the stored ones (largest gap {max(gaps):.3f})")

    result = {"method": {"resampling": "texts within strata (src/populism_axis/resampling.py)", "draws": args.draws, "seed": args.seed,
                         "interval": "percentile, 2.5 and 97.5", "paired": "both systems evaluated on the same resampled texts",
                         "comparison": "cluster_ci95 resamples whole clusters with the same seed and draws"},
              "benchmarks": summary, "entries": entries,
              "reproduction": {"max_gap_cluster_recomputed_vs_stored": float(max(gaps)), "n_replaced": len(gaps)}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "intervals.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    fmt = lambda c: f"[{c[0]:+.3f}, {c[1]:+.3f}]"  # noqa: E731
    lines = ["# Bootstrap intervals: texts resampled within speakers", "",
             f"{args.draws} resamples, seed {args.seed}. The cluster column resamples whole clusters, as the stored intervals did; "
             f"largest gap between the recomputed and stored cluster intervals {max(gaps):.3f}.", "",
             "| Benchmark | Statistic | Estimate | Within strata | Whole clusters | Strata |", "|---|---|---:|---|---|---|"]
    for r in entries.values():
        s = summary[r["benchmark"]]
        lines.append(f"| {s['label']} | {r['statistic']} | {r['estimate']:+.3f} | {fmt(r['within_ci95'])} | {fmt(r['cluster_ci95'])} | "
                     f"{s['n_strata']} ({s['stratum']}) |")
    (OUT / "intervals.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    inputs = {k: ROOT / v for k, v in (SCORES | DATA).items()} | {"script": Path(__file__), "resampling": ROOT / "src/populism_axis/resampling.py"}
    inputs |= {Path(rel).parent.name + "_stored": ROOT / rel for rel in (FINAL, CMP, GOV, TW, PB, PT)}
    write_run_manifest(OUT, script="scripts/within_speaker_intervals.py", inputs=inputs,
                       extra={"draws": args.draws, "seed": args.seed, "resampling": "texts within strata"})
    print(f"wrote {OUT.relative_to(ROOT)}: {len(entries)} intervals; largest cluster-reproduction gap {max(gaps):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
