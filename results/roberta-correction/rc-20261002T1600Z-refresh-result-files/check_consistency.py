#!/usr/bin/env python3
"""Check the refreshed result files against the comparison run (read-only on the result files).

Every RoBERTa number of rc-20261002T1415Z-sigmoid (comparison.json, full precision; secondary_tables.json,
rounded to 6 places) must equal the refreshed value in place (sigmoid) and the `superseded_softmax_roberta`
value (softmax). Writes consistency.json next to this script.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = ROOT / "results" / "roberta-correction" / "rc-20261002T1415Z-sigmoid"
FILES = {"final": "results/final/final_results.json", "protocol": "results/final/eu_paper_protocol.json",
         "usgov": "results/external/usgov/final_results.json", "tweets": "results/external/antielite_tweets/final_results.json",
         "popbert": "results/external/popbert_en/final_results.json", "portugal": "results/external/portugal/final_results.json"}
DOCS = {k: json.loads((ROOT / v).read_text(encoding="utf-8")) for k, v in FILES.items()}


def lookup(doc: str, path: tuple, version: str):
    data = DOCS[doc]
    if version == "softmax":
        superseded = data["superseded_softmax_roberta"]
        for cut in range(len(path), 0, -1):
            key = "/" + "/".join(path[:cut])
            if key in superseded:
                data, path = superseded[key], path[cut:]
                break
        else:
            raise KeyError(f"{doc}: no superseded entry for {'/'.join(path)}")
    for key in path:
        data = data[key]
    return data


def same(a, b, digits=None) -> bool:
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k], digits) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y, digits) for x, y in zip(a, b))
    if digits is not None and isinstance(b, float):
        return round(b, digits) == a
    return a == b


def main() -> None:
    comparison = json.loads((RUN / "comparison.json").read_text(encoding="utf-8"))["rows"]
    secondary = json.loads((RUN / "secondary_tables.json").read_text(encoding="utf-8"))
    checks: list[tuple[str, object, str, tuple, str, int | None]] = []
    for v in ("softmax", "sigmoid"):
        r = {k: comparison[k][v] for k in comparison}
        en = ("configurations", "RoBERTa_comparator", "english_holdout")
        checks += [
            ("english spearman", r["english_holdout"]["metric"], "final", en + ("spearman",), v, None),
            ("english ci95", r["english_holdout"]["ci95"], "final", en + ("spearman_ci95",), v, None),
            ("english extreme_auroc", r["english_holdout"]["secondary"]["extreme_auroc"], "final", en + ("extreme_auroc",), v, None),
            ("english leader_term", r["english_holdout"]["secondary"]["leader_term_spearman"], "final", en + ("leader_term_spearman",), v, None),
            ("eu auroc", r["eu_auroc_3v0"]["metric"], "final", ("configurations", "RoBERTa_comparator", "eu", "auroc"), v, None),
            ("eu ci95", r["eu_auroc_3v0"]["ci95"], "final", ("configurations", "RoBERTa_comparator", "eu", "auroc_speaker_clustered_ci95"), v, None),
            ("eu within-speaker", r["eu_auroc_3v0"]["secondary"]["pooled_within_speaker_auroc"], "final",
             ("configurations", "RoBERTa_comparator", "eu", "pooled_within_speaker_auroc"), v, None),
            ("eu loso (final)", r["eu_loso_macro_f1"]["secondary"]["loso_overall"], "final", ("eu_paper_protocol_macro_f1_loso", "RoBERTa_comparator"), v, None),
            ("eu loso (protocol)", r["eu_loso_macro_f1"]["secondary"]["loso_overall"], "protocol", ("ours", "RoBERTa_comparator", "loso_f1", "overall"), v, None),
            ("eu prevalence", r["eu_loso_macro_f1"]["secondary"]["prevalence_macro_f1"], "protocol", ("ours", "RoBERTa_comparator", "prevalence", "overall", "macro"), v, None),
            ("usgov 2010-18 spearman", r["usgov_2010_2018"]["metric"], "usgov", ("speeches", "governors_2010_2018", "RoBERTa_anti_elitism_top3", "spearman"), v, None),
            ("usgov 2010-18 ci95", r["usgov_2010_2018"]["ci95"], "usgov", ("speeches", "governors_2010_2018", "RoBERTa_anti_elitism_top3", "spearman_ci95"), v, None),
            ("tweets auroc", r["tweets"]["metric"], "tweets", ("test", "y_general", "RoBERTa_anti_elitism", "auroc"), v, None),
            ("tweets ci95", r["tweets"]["ci95"], "tweets", ("test", "y_general", "RoBERTa_anti_elitism", "auroc_ci95_country"), v, None),
            ("popbert elite auroc", r["popbert_elite"]["metric"], "popbert", ("test", "elite", "majority", "E_roberta", "auroc"), v, None),
            ("popbert elite ci95", r["popbert_elite"]["ci95"], "popbert", ("test", "elite", "majority", "E_roberta", "auroc_ci95"), v, None),
            ("popbert centr auroc", r["popbert_centr"]["metric"], "popbert", ("test", "centr", "majority", "P_roberta", "auroc"), v, None),
            ("popbert centr ci95", r["popbert_centr"]["ci95"], "popbert", ("test", "centr", "majority", "P_roberta", "auroc_ci95"), v, None),
            ("portugal elite auroc", r["portugal_elite"]["metric"], "portugal", ("primary", "elite", "E_roberta", "auroc"), v, None),
            ("portugal elite ci95", r["portugal_elite"]["ci95"], "portugal", ("primary", "elite", "E_roberta", "auroc_ci95"), v, None),
            ("portugal centr auroc", r["portugal_centr"]["metric"], "portugal", ("primary", "centr", "P_roberta", "auroc"), v, None),
            ("portugal centr ci95", r["portugal_centr"]["ci95"], "portugal", ("primary", "centr", "P_roberta", "auroc_ci95"), v, None),
        ]
        for name, key in (("B6_axis_plus_nli - RoBERTa", "B6_axis_plus_nli - RoBERTa_comparator"), ("B4_axis - RoBERTa", "B4_axis - RoBERTa_comparator")):
            p = r["english_holdout"]["paired"][name]
            checks.append((f"english paired {name}", {"delta_spearman": p["delta_spearman"], "ci95": p["ci95"], "n_draws": p["n_draws"],
                                                       "p_two_sided": p["p_two_sided"]}, "final", ("paired_english_holdout", key), v, None))
        checks.append(("usgov 2010-18 paired", r["usgov_2010_2018"]["paired"]["B6_axis_plus_nli - RoBERTa"], "usgov",
                       ("paired_spearman", "governors_2010_2018", "B6_axis_plus_nli - RoBERTa_anti_elitism_top3"), v, None))
        for name, p in r["tweets"]["paired"].items():
            checks.append((f"tweets paired {name}", {"delta_auroc": p["delta"], "ci95": p["ci95"], "p_two_sided": p["p_two_sided"]}, "tweets",
                           ("paired_auroc_test", name), v, None))
        for dim, key in (("elite", "popbert_elite"), ("centr", "popbert_centr")):
            for name, p in r[key]["paired"].items():
                checks.append((f"popbert paired {name}", p, "popbert", ("paired_auroc_test", f"{dim}: {name}"), v, None))
        for dim, key in (("elite", "portugal_elite"), ("centr", "portugal_centr")):
            for name, p in r[key]["paired"].items():
                checks.append((f"portugal paired {name}", p, "portugal", ("paired_auroc", dim, name), v, None))
        # Secondary tables (rounded to 6 places).
        s = secondary
        for subset, block in s["usgov"][v].items():
            checks.append((f"usgov {subset} metrics", block["metrics"], "usgov", ("speeches", subset, "RoBERTa_anti_elitism_top3"), v, 6))
            checks.append((f"usgov {subset} paired", block["B6_minus_RoBERTa"], "usgov", ("paired_spearman", subset, "B6_axis_plus_nli - RoBERTa_anti_elitism_top3"), v, 6))
        for target, block in s["tweets"][v]["evaluate"].items():
            checks.append((f"tweets {target}", block, "tweets", ("test", target, "RoBERTa_anti_elitism"), v, 6))
        for name, p in s["tweets"][v]["paired"].items():
            if "RoBERTa" in name:
                checks.append((f"tweets paired (secondary) {name}", {"delta_auroc": p["delta"], "ci95": p["ci95"], "p_two_sided": p["p_two_sided"]},
                               "tweets", ("paired_auroc_test", name), v, 6))
        for key, block in s["popbert"][v].items():
            dim, rule = key.split("_", 1)
            checks.append((f"popbert {key}", block, "popbert", ("test", dim, rule, "E_roberta" if dim == "elite" else "P_roberta"), v, 6))
        for dim in ("elite", "centr"):
            for name, block in s["portugal"][v][dim].items():
                checks.append((f"portugal {dim} {name}", block, "portugal", ("primary", dim, name), v, 6))
    results, failures = [], []
    for label, expected, doc, path, version, digits in checks:
        actual = lookup(doc, path, version)
        if isinstance(expected, dict) and isinstance(actual, dict):
            actual = {k: actual[k] for k in expected}
        ok = same(expected, actual, digits)
        results.append({"check": label, "version": version, "file": FILES[doc], "path": "/" + "/".join(path),
                        "precision": "exact" if digits is None else f"rounded to {digits}", "equal": ok})
        if not ok:
            failures.append((label, version, expected, actual))
    out = {"source": str(RUN.relative_to(ROOT)), "n_checks": len(results), "n_equal": sum(r["equal"] for r in results), "checks": results}
    (HERE / "consistency.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"{out['n_equal']} of {out['n_checks']} checks equal")
    for failure in failures:
        print("MISMATCH", failure)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
