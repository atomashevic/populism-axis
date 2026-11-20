#!/usr/bin/env python3
"""Same-size claim-check arms (CPU): do ModernBERT NLI models or decision models add anything to BART?

Plan: research/development-log/claim-check-arms-plan.md. Arms and contrasts: claim_check_arms.py.
Everything except the model that scores a window against a claim is held fixed: the stored
top-8 windows, the ten claims, the per-window rule (max over a channel's five claims, sqrt(P*E)),
the document mean and the .5/.5 z-mean with the axis. Only English-dev and Agora are loaded.

For each candidate arm on both development sets:
  non-inferiority  paired cluster bootstrap of the combined score against B6, margin -.01 on both
  incremental      partial Spearman of the candidate's document claim score with the human score,
                   given the B6 combined score; and the equal-weight three-input score
                   (axis, BART, candidate; 1/3 each) against B6
  contrasts        paired differences between arms (backbone, training mix, decision head)
  diagnostics      window-level rank agreement with BART, tail quantiles, ties in the stored values
  disagreement     the 50 Agora windows whose within-corpus rank differs most from BART's, with
                   the coder score of their paragraph; the table with window text is written to
                   a gitignored directory, the summary without text to results.json
  cost             sequences encoded, forward calls, wall time and peak VRAM from the GPU manifests

Outputs: results/claim_check_arms/{results.json, results.md, scores_<corpus>__<label>.csv,
run_manifest.json} and results/claim_check_arms/disagreement/top50_<label>.csv (not tracked).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from populism_axis.geometry import pole_mean_axes  # noqa: E402
from populism_axis.preset import standardize  # noqa: E402
from populism_axis.run_manifest import write_run_manifest  # noqa: E402
from claim_check_arms import ARMS, CANDIDATES, CONTRASTS, REFERENCE, tail_counts  # noqa: E402
from frozen_reference import ANCHORS, CH, EMB, combine, document_quantities, load_english, paired_delta, spearman  # noqa: E402
from run_config import load_anchor_groups, load_embeddings  # noqa: E402

R = ROOT / "results"
STEP6 = R / "step6_nli"
OUT = R / "claim_check_arms"
RULE_MARGIN = -0.01
RULE = "k:8"
TOP_DISAGREEMENT = 50


def development_sets() -> dict:
    """English-dev and Agora only; the held-out and external corpora are never loaded here."""
    frame, matrix, doc_index = load_english("dev")
    items = {"english_dev": dict(matrix=matrix, doc_index=doc_index, n=len(frame), target=frame["source_score"].to_numpy(float),
                                 groups=frame["leader"].astype(str).to_numpy())}
    agora = pd.read_csv(ROOT / "data" / "benchmarks" / "agora_paragraphs.csv")
    matrix, doc_index, _ = load_embeddings(EMB / "spw3_cap60_clean", "agora")
    items["agora"] = dict(matrix=matrix, doc_index=doc_index, n=len(agora), target=agora["populism_human"].to_numpy(float),
                          groups=agora["politician"].astype(str).to_numpy())
    return items


def window_quantities(directory: Path, name: str, label: str, requests: pd.DataFrame, hypotheses: list[dict]) -> tuple[dict, np.ndarray]:
    payload = np.load(directory / f"nli_{name}__{label}.npz")
    if not np.array_equal(payload["request_id"], requests["request_id"].to_numpy()):
        raise SystemExit(f"{name}/{label}: probabilities do not match the request file")
    if list(payload["hypothesis_ids"]) != [h["hypothesis_id"] for h in hypotheses]:
        raise SystemExit(f"{name}/{label}: hypothesis order differs from hypotheses.json")
    entail = payload["probs"][:, :, 2].astype(np.float64)
    cols = {c: [i for i, h in enumerate(hypotheses) if h["channel"] == c] for c in CH}
    out = {"P_nli": entail[:, cols["P"]].max(axis=1), "E_nli": entail[:, cols["E"]].max(axis=1)}
    out["PE_nli"] = np.sqrt(out["P_nli"] * out["E_nli"])
    return out, entail


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman correlation of x and y after removing the rank of z from both."""
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    design = np.column_stack([np.ones_like(rz), rz])
    ex = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    ey = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    denominator = np.sqrt((ex @ ex) * (ey @ ey))
    return float(ex @ ey / denominator) if denominator > 1e-12 else float("nan")


def cluster_interval(statistic, arrays: tuple, groups: np.ndarray, draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    index_by = {g: np.flatnonzero(groups == g) for g in unique}
    samples = []
    for _ in range(draws):
        idx = np.concatenate([index_by[g] for g in rng.choice(unique, size=len(unique), replace=True)])
        value = statistic(*(a[idx] for a in arrays))
        if np.isfinite(value):
            samples.append(value)
    return {"value": float(statistic(*arrays)), "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]}


def tail(window: dict, reference: dict, entail: np.ndarray) -> dict:
    out = {}
    for c in ("P", "E", "PE"):
        v, b = window[f"{c}_nli"], reference[f"{c}_nli"]
        out[c] = {"median": float(np.median(v)), "p99": float(np.percentile(v, 99)), "share_gt_0.5": float((v > 0.5).mean()),
                  "window_spearman_vs_bart": float(spearmanr(v, b).statistic)}
    out["stored"] = tail_counts(entail)
    return out


def disagreement(name: str, label: str, requests: pd.DataFrame, window: dict, reference: dict, target: np.ndarray) -> dict:
    """Windows whose rank on the PE conjunction differs most from BART's; text goes to an untracked file."""
    n = len(requests)
    cand_pct, bart_pct = rankdata(window["PE_nli"]) / n, rankdata(reference["PE_nli"]) / n
    gap = cand_pct - bart_pct
    top = np.argsort(-np.abs(gap), kind="stable")[:TOP_DISAGREEMENT]
    human = target[requests["doc_index"].to_numpy()]
    table = requests.iloc[top][["request_id", "doc_index", "doc_id", "window_rank"]].assign(
        candidate_PE=window["PE_nli"][top], bart_PE=reference["PE_nli"][top], candidate_pct=cand_pct[top], bart_pct=bart_pct[top],
        gap=gap[top], human_score=human[top], premise=requests["premise"].to_numpy()[top])
    directory = OUT / "disagreement"
    directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(directory / f"top50_{name}__{label}.csv", index=False)
    higher, lower = gap[top] > 0, gap[top] < 0
    mean = lambda mask: float(human[top][mask].mean()) if mask.any() else None  # noqa: E731
    return {"windows": int(len(top)), "candidate_ranks_higher": int(higher.sum()), "bart_ranks_higher": int(lower.sum()),
            "mean_human_where_candidate_higher": mean(higher), "mean_human_where_bart_higher": mean(lower),
            "mean_human_all_windows": float(human.mean()), "min_abs_gap": float(np.abs(gap[top]).min()),
            "spearman_gap_vs_human_all_windows": float(spearmanr(gap, human).statistic)}


def cost(directory: Path, label: str) -> dict | None:
    path = directory / f"nli__{label}.manifest.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    extra = manifest["extra"]
    agora = extra.get("agora", {})
    return {"agora_sequences": agora.get("sequences", agora.get("pairs")), "agora_forward_calls": agora.get("forward_calls"),
            "agora_seconds": agora.get("seconds"), "peak_vram_mb": extra.get("peak_vram_mb"), "gpu": extra.get("gpu"),
            "batch_size": extra.get("batch_size"), "host": manifest.get("host"), "commit": manifest.get("git", {}).get("commit"),
            "dirty": manifest.get("git", {}).get("dirty"), "model_revision": extra.get("model_revision")}


def fmt(pair: dict, key: str = "delta") -> str:
    return f"{pair[key]:+.3f} [{pair['ci95'][0]:+.3f}, {pair['ci95'][1]:+.3f}]"


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nli-dir", type=Path, default=OUT / "nli", help="directory with the arms' .npz files and manifests")
    parser.add_argument("--arms", default=",".join(CANDIDATES), help="candidate arms to evaluate, e.g. B,C")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    OUT = args.out_dir
    OUT.mkdir(parents=True, exist_ok=True)
    arms = [a for a in args.arms.split(",") if a]
    if set(arms) - set(CANDIDATES):
        parser.error(f"--arms takes {', '.join(CANDIDATES)}")

    axes = pole_mean_axes(load_anchor_groups(ANCHORS))
    hypotheses = json.loads((STEP6 / "hypotheses.json").read_text(encoding="utf-8"))
    items = development_sets()
    scores: dict = {name: {} for name in items}
    windows: dict = {name: {} for name in items}
    requests = {name: pd.read_parquet(STEP6 / f"requests_{name}.parquet") for name in items}

    def load(name: str, arm: str, directory: Path) -> None:
        item = items[name]
        window, entail = window_quantities(directory, name, ARMS[arm]["label"], requests[name], hypotheses)
        docs, _ = document_quantities(item["matrix"], item["doc_index"], item["n"], RULE, requests[name], window, axes, None)
        scores[name][arm] = combine(docs, RULE, None)
        windows[name][arm] = (window, entail)

    results: dict = {"rule": f"paired delta of combined vs B6 >= {RULE_MARGIN} on English-dev and Agora", "draws": args.draws,
                     "seed": args.seed, "arms": {a: ARMS[a] for a in ("A", *arms)}, "reference": {}, "candidates": {}, "contrasts": [],
                     "reference_rerun": {}}
    for name, item in items.items():
        load(name, "A", STEP6)  # the stored step-6 BART probabilities define B6
        t, a = item["target"], scores[name]["A"]
        results["reference"][name] = {"n": item["n"], "combined_spearman": spearman(t, a["PE_axis_plus_nli"]),
                                      "nli_only_spearman": spearman(t, a["PE_nli"]), "axis_only_spearman": spearman(t, a["PE_axis"])}
        rerun = args.nli_dir / f"nli_{name}__{REFERENCE}.npz"
        if rerun.exists():  # arm A scored again on the comparison GPU, for timing; compared with the stored file
            _, fresh = window_quantities(args.nli_dir, name, REFERENCE, requests[name], hypotheses)
            results["reference_rerun"][name] = {"max_abs_entailment_difference": float(np.abs(fresh - windows[name]["A"][1]).max())}

    for arm in arms:
        label = ARMS[arm]["label"]
        entry: dict = {"label": label, "corpora": {}, "cost": cost(args.nli_dir, label)}
        passes = True
        for name, item in items.items():
            load(name, arm, args.nli_dir)
            t, g = item["target"], item["groups"]
            cand, ref = scores[name][arm], scores[name]["A"]
            window, entail = windows[name][arm]
            b6 = ref["PE_axis_plus_nli"]
            three = (standardize(cand["PE_axis"], None) + standardize(ref["PE_nli"], None) + standardize(cand["PE_nli"], None)) / 3
            paired = paired_delta(cand["PE_axis_plus_nli"], b6, t, g, spearman, args.draws, args.seed)
            entry["corpora"][name] = {
                "combined_spearman": spearman(t, cand["PE_axis_plus_nli"]), "nli_only_spearman": spearman(t, cand["PE_nli"]),
                "paired_combined_vs_b6": paired,
                "paired_nli_only_vs_bart": paired_delta(cand["PE_nli"], ref["PE_nli"], t, g, spearman, args.draws, args.seed),
                "partial_spearman_given_b6": cluster_interval(partial_spearman, (cand["PE_nli"], t, b6), g, args.draws, args.seed),
                "three_input_spearman": spearman(t, three),
                "paired_three_input_vs_b6": paired_delta(three, b6, t, g, spearman, args.draws, args.seed),
                "doc_spearman_nli_vs_bart": float(spearmanr(cand["PE_nli"], ref["PE_nli"]).statistic),
                "tail": tail(window, windows[name]["A"][0], entail),
                "disagreement": disagreement(name, label, requests[name], window, windows[name]["A"][0], t),
            }
            passes &= paired["delta"] >= RULE_MARGIN
            pd.DataFrame({"doc_index": np.arange(item["n"]), "PE_nli": cand["PE_nli"], "PE_axis_plus_nli": cand["PE_axis_plus_nli"],
                          "PE_three_input": three}).to_csv(OUT / f"scores_{name}__{label}.csv", index=False)
        entry["non_inferior"] = bool(passes)
        agora = entry["corpora"]["agora"]
        entry["adds_information"] = bool(agora["partial_spearman_given_b6"]["ci95"][0] > 0 and agora["paired_three_input_vs_b6"]["ci95"][0] > 0)
        results["candidates"][arm] = entry

    for left, right, isolates in CONTRASTS:
        if left not in scores["agora"] or right not in scores["agora"]:
            continue
        row = {"left": left, "right": right, "isolates": isolates}
        for name, item in items.items():
            l, r = scores[name][left], scores[name][right]
            row[name] = {"combined": paired_delta(l["PE_axis_plus_nli"], r["PE_axis_plus_nli"], item["target"], item["groups"], spearman, args.draws, args.seed),
                         "nli_only": paired_delta(l["PE_nli"], r["PE_nli"], item["target"], item["groups"], spearman, args.draws, args.seed),
                         "window_spearman_PE": float(spearmanr(windows[name][left][0]["PE_nli"], windows[name][right][0]["PE_nli"]).statistic)}
        results["contrasts"].append(row)

    (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    ref = results["reference"]
    lines = [f"B6 reference (arm A, stored BART-large-MNLI): English-dev combined {ref['english_dev']['combined_spearman']:.3f}, claim check only "
             f"{ref['english_dev']['nli_only_spearman']:.3f}; Agora combined {ref['agora']['combined_spearman']:.3f}, claim check only {ref['agora']['nli_only_spearman']:.3f}.", "",
             f"Paired differences: {args.draws} draws, seed {args.seed}, leaders (English-dev) and politicians (Agora) resampled. "
             f"Non-inferior: Δ of the combined score against B6 at least {RULE_MARGIN} on both sets.", "",
             "| Arm | Model | English-dev combined | Δ vs B6 [95% CI] | Agora combined | Δ vs B6 [95% CI] | English-dev claim only | Agora claim only | Non-inferior |",
             "|---|---|---:|---|---:|---|---:|---:|---|"]
    for arm, e in results["candidates"].items():
        en, ag = e["corpora"]["english_dev"], e["corpora"]["agora"]
        lines.append(f"| {arm} | {e['label']} | {en['combined_spearman']:.3f} | {fmt(en['paired_combined_vs_b6'])} | {ag['combined_spearman']:.3f} | "
                     f"{fmt(ag['paired_combined_vs_b6'])} | {en['nli_only_spearman']:.3f} | {ag['nli_only_spearman']:.3f} | {'yes' if e['non_inferior'] else 'no'} |")
    lines += ["", "Incremental information. Partial ρ: candidate claim score with the human score given the B6 combined score. "
              "Three-input: mean of z(axis), z(BART), z(candidate).", "",
              "| Arm | Agora partial ρ [95% CI] | Agora three-input | Δ vs B6 [95% CI] | English-dev partial ρ [95% CI] | English-dev three-input | Δ vs B6 [95% CI] | Adds information (Agora) |",
              "|---|---|---:|---|---|---:|---|---|"]
    for arm, e in results["candidates"].items():
        en, ag = e["corpora"]["english_dev"], e["corpora"]["agora"]
        lines.append(f"| {arm} | {fmt(ag['partial_spearman_given_b6'], 'value')} | {ag['three_input_spearman']:.3f} | {fmt(ag['paired_three_input_vs_b6'])} | "
                     f"{fmt(en['partial_spearman_given_b6'], 'value')} | {en['three_input_spearman']:.3f} | {fmt(en['paired_three_input_vs_b6'])} | {'yes' if e['adds_information'] else 'no'} |")
    lines += ["", "Contrasts between arms (left minus right).", "",
              "| Contrast | Isolates | English-dev combined Δ [95% CI] | Agora combined Δ [95% CI] | Agora claim-only Δ [95% CI] | Agora window ρ (PE) |", "|---|---|---|---|---|---:|"]
    for row in results["contrasts"]:
        lines.append(f"| {row['left']} − {row['right']} | {row['isolates']} | {fmt(row['english_dev']['combined'])} | {fmt(row['agora']['combined'])} | "
                     f"{fmt(row['agora']['nli_only'])} | {row['agora']['window_spearman_PE']:.3f} |")
    lines += ["", "Agora windows, PE conjunction: tail, rank agreement with BART, ties in the stored values, and the 50 windows that differ most from BART in rank.", "",
              "| Arm | median | p99 | share > .5 | window ρ vs BART | doc ρ vs BART | exact 0 / exact 1 / distinct | mean coder score: candidate higher / BART higher (n) |",
              "|---|---:|---:|---:|---:|---:|---|---|"]
    for arm, e in results["candidates"].items():
        ag = e["corpora"]["agora"]
        t, s, d = ag["tail"]["PE"], ag["tail"]["stored"], ag["disagreement"]
        show = lambda v: "–" if v is None else f"{v:.2f}"  # noqa: E731
        lines.append(f"| {arm} | {t['median']:.4f} | {t['p99']:.3f} | {t['share_gt_0.5']:.4f} | {t['window_spearman_vs_bart']:.3f} | {ag['doc_spearman_nli_vs_bart']:.3f} | "
                     f"{s['exact_zero']} / {s['exact_one']} / {s['distinct']} | {show(d['mean_human_where_candidate_higher'])} / {show(d['mean_human_where_bart_higher'])} "
                     f"({d['candidate_ranks_higher']} / {d['bart_ranks_higher']}) |")
    lines += ["", "Cost on Agora (22,668 windows × 10 claims), from the GPU manifests.", "",
              "| Arm | Sequences encoded | Forward calls | Wall time (s) | Peak VRAM (MB) | GPU |", "|---|---:|---:|---:|---:|---|"]
    for arm in ("A", *results["candidates"]):
        c = cost(args.nli_dir, ARMS[arm]["label"])
        if c:
            lines.append(f"| {arm} | {c['agora_sequences']} | {c['agora_forward_calls']} | {c['agora_seconds']} | {c['peak_vram_mb']} | {c['gpu']} |")
    (OUT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_run_manifest(OUT, script="scripts/evaluate_claim_check_arms.py", inputs={"hypotheses": STEP6 / "hypotheses.json"},
                       extra={"arms": arms, "nli_dir": str(args.nli_dir), "non_inferior": [a for a, e in results["candidates"].items() if e["non_inferior"]]})
    print("\n".join(lines))


if __name__ == "__main__":
    main()
