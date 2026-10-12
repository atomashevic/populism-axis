#!/usr/bin/env python3
"""Frozen reference scales for the populism axis + claim check score (CPU; stored embeddings and NLI outputs).

Every score is transductive: each corpus supplies its own mean embedding for
centering, its own per-channel projection mean and SD, its own softplus activation
scale (nonzero median), and its own document-level mean and SD for the axis and NLI
z-scores that the combination averages. This script freezes all of those on the
English development set and re-scores every corpus with the fixed statistics:

  reference (English-dev speeches, three-sentence windows, carried B4 configuration)
    mu               mean window embedding                       (centering)
    proj[c]          mean, SD of the projection on axis c        (window z-score)
    act[c]           nonzero median of softplus(z / .25)^2       (activation scale)
    doc[rule][q]     mean, SD of document quantity q under pooling rule k:8 or k:1:
                     PE_axis, P_axis, E_axis, PE_nli, P_nli, E_nli

  scored corpora: Agora, English holdout, EU-OOD (3-vs-0 and the paper's AE / PC
  channels), US governors, party tweets, PopBERT-EN, Portuguese parliament, plus
  English-dev itself as an identity check.

Each corpus's document quantities are taken under the pooling rule its report used,
and the reference statistics are those of the same quantity under the same rule on
English-dev speeches. The NLI premises stay the stored transductive top-K windows
(NLI was run only on those); freezing changes the NLI score's scale, not its premises.

The transductive path is recomputed here and checked against the stored score files
before anything frozen is reported. Outputs: results/frozen_reference/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

from populism_axis.contextual_axis import aggregate_from_window_scores  # noqa: E402
from populism_axis.run_manifest import write_run_manifest  # noqa: E402
from populism_axis.preset import nonzero_median, standardize, window_quantities
from populism_axis.geometry import pole_mean_axes, unit_rows  # noqa: E402
from run_config import load_anchor_groups, load_embeddings, select_documents  # noqa: E402

R = ROOT / "results"
OUT = R / "frozen_reference"
EMB = R / "embeddings" / "e5-large-v2"
EXT = R / "embeddings_external"
ANCHORS = EMB / "sentences" / "v3_proposition__ea98e5b2.npz"
SOFTNESS = 0.25
CH = ("P", "E")
DOC_QUANTITIES = ("PE_axis", "P_axis", "E_axis", "PE_nli", "P_nli", "E_nli")


# ----------------------------------------------------------------------------- scoring


def nli_window_quantities(directory: Path, name: str, model_label: str = "bart-large-mnli") -> tuple[pd.DataFrame, dict]:
    hypotheses = json.loads((directory / "hypotheses.json").read_text(encoding="utf-8"))
    requests = pd.read_parquet(directory / f"requests_{name}.parquet")
    payload = np.load(directory / f"nli_{name}__{model_label}.npz")
    if not np.array_equal(payload["request_id"], requests["request_id"].to_numpy()):
        raise SystemExit(f"{name}: NLI output does not match requests")
    entail = payload["probs"][:, :, 2].astype(np.float64)
    cols = {c: [i for i, h in enumerate(hypotheses) if h["channel"] == c] for c in CH}
    out = {"P_nli": entail[:, cols["P"]].max(axis=1), "E_nli": entail[:, cols["E"]].max(axis=1)}
    out["PE_nli"] = np.sqrt(out["P_nli"] * out["E_nli"])
    return requests, out


def nli_document_quantities(requests: pd.DataFrame, window_nli: dict, n: int, k: int) -> dict:
    kept = (requests["window_rank"] < k).to_numpy()
    doc = requests["doc_index"].to_numpy()[kept]
    return {q: pd.Series(v[kept]).groupby(doc).mean().reindex(range(n)).fillna(0.0).to_numpy() for q, v in window_nli.items()}


def document_quantities(matrix, doc_index, n, rule, requests, window_nli, axes, stats):
    k = int(rule.split(":")[1])
    window, used = window_quantities(matrix, axes, stats)
    docs = {q: aggregate_from_window_scores(v, doc_index, n, rule) for q, v in window.items()}
    docs.update(nli_document_quantities(requests, window_nli, n, k))
    return docs, used


def combine(docs: dict, rule: str, stats: dict | None) -> dict:
    """Doc-level z-scores (transductive or frozen) and the axis + NLI combinations."""
    ref = None if stats is None else stats["doc"][rule]
    z = {q: standardize(docs[q], None if ref is None else tuple(ref[q])) for q in DOC_QUANTITIES}
    out = dict(docs)
    for c in ("PE", "P", "E"):
        out[f"{c}_axis_plus_nli"] = (z[f"{c}_axis"] + z[f"{c}_nli"]) / 2
    return out


# ----------------------------------------------------------------------------- metrics


def spearman(target, values):
    return float(spearmanr(values, target).statistic)


def auroc(target, values):
    return float(roc_auc_score(target, values))


def paired_delta(left, right, target, groups, metric, draws, seed):
    observed = metric(target, left) - metric(target, right)
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    index_by = {g: np.flatnonzero(groups == g) for g in unique}
    samples = []
    for _ in range(draws):
        idx = np.concatenate([index_by[g] for g in rng.choice(unique, size=len(unique), replace=True)])
        if len(np.unique(target[idx])) < 2:
            continue
        try:
            d = metric(target[idx], left[idx]) - metric(target[idx], right[idx])
        except ValueError:
            continue
        if np.isfinite(d):
            samples.append(d)
    arr = np.asarray(samples)
    return {"delta": float(observed), "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]}


# ----------------------------------------------------------------------------- corpora


def load_english(split_name: str):
    split = pd.read_csv(ROOT / "data" / "benchmarks" / "english_split.csv").set_index("id")["split"]
    frame = pd.read_csv(ROOT / "data" / "benchmarks" / "english_gpd_speeches.csv")
    matrix, doc_index, windows = load_embeddings(EMB / "spw3_cap60_clean", "english")
    keep = frame["id"].map(split).eq(split_name).to_numpy()
    frame, matrix, doc_index, _ = select_documents(frame, matrix, doc_index, windows, keep)
    return frame, matrix, doc_index


def corpora() -> list[dict]:
    """Each entry: embeddings, NLI location, pooling rule, and one or more evaluations
    (system, target, metric, clusters, evaluation mask, stored column for the transductive check)."""
    items = []
    frame, matrix, doc_index = load_english("dev")
    items.append(dict(name="english_dev", matrix=matrix, doc_index=doc_index, n=len(frame), rule="k:8",
                      nli=(R / "step6_nli", "english_dev"), spent=False,
                      evals=[dict(system=s, target=frame["source_score"].to_numpy(float), metric="spearman",
                                  groups=frame["leader"].astype(str).to_numpy(), stored=st)
                             for s, st in (("PE_axis", (R / "step4_windows_topk" / "grid" / "spw3_k8" / "scores_english.csv", "PE__frozen__soft")),
                                           ("PE_axis_plus_nli", (R / "step6_nli" / "combined_k8" / "scores_english.csv", "PE__frozen__soft")))]))

    agora = pd.read_csv(ROOT / "data" / "benchmarks" / "agora_paragraphs.csv")
    matrix, doc_index, _ = load_embeddings(EMB / "spw3_cap60_clean", "agora")
    items.append(dict(name="agora", matrix=matrix, doc_index=doc_index, n=len(agora), rule="k:8",
                      nli=(R / "step6_nli", "agora"), spent=False,
                      evals=[dict(system=s, target=agora["populism_human"].to_numpy(float), metric="spearman",
                                  groups=agora["politician"].astype(str).to_numpy(), stored=st)
                             for s, st in (("PE_axis", (R / "step4_windows_topk" / "grid" / "spw3_k8" / "scores_agora.csv", "PE__frozen__soft")),
                                           ("PE_axis_plus_nli", (R / "step6_nli" / "combined_k8" / "scores_agora.csv", "PE__frozen__soft")))]))

    frame, matrix, doc_index = load_english("holdout")
    items.append(dict(name="english_holdout", matrix=matrix, doc_index=doc_index, n=len(frame), rule="k:8",
                      nli=(R / "step6_nli", "english_holdout"), spent=True,
                      evals=[dict(system=s, target=frame["source_score"].to_numpy(float), metric="spearman",
                                  groups=frame["leader"].astype(str).to_numpy(), stored=st)
                             for s, st in (("PE_axis", (R / "final" / "B4_axis" / "scores_english.csv", "PE__frozen__soft")),
                                           ("PE_axis_plus_nli", (R / "final" / "B6_axis_plus_nli" / "scores_english.csv", "PE__frozen__soft")))]))

    eu = pd.read_csv(ROOT / "data" / "benchmarks" / "eu_sentences.csv")
    matrix, doc_index, _ = load_embeddings(R / "embeddings_eu" / "e5-large-v2" / "spw3_cap60", "eu")
    pop = eu["pop_code"].to_numpy(int)
    speakers = eu["speaker"].astype(str).to_numpy()
    items.append(dict(name="eu", matrix=matrix, doc_index=doc_index, n=len(eu), rule="k:1", nli=(R / "step6_nli", "eu"), spent=True,
                      evals=[dict(system="PE_axis", label="3v0", target=(pop == 3).astype(int), metric="auroc", groups=speakers, mask=np.isin(pop, [0, 3]),
                                  stored=(R / "final" / "B4_axis" / "scores_eu.csv", "PE__frozen__soft")),
                             dict(system="PE_axis_plus_nli", label="3v0", target=(pop == 3).astype(int), metric="auroc", groups=speakers, mask=np.isin(pop, [0, 3]),
                                  stored=(R / "final" / "B6_axis_plus_nli" / "scores_eu.csv", "PE__frozen__soft")),
                             dict(system="E_axis_plus_nli", label="AE", target=np.isin(pop, [1, 3]).astype(int), metric="auroc", groups=speakers),
                             dict(system="P_axis_plus_nli", label="PC", target=np.isin(pop, [2, 3]).astype(int), metric="auroc", groups=speakers)]))

    us = pd.read_csv(R / "external" / "usgov" / "final_scores_speeches.csv")
    matrix, doc_index, _ = load_embeddings(EXT / "usgov" / "e5-large-v2" / "spw3_cap60", "usgov_speeches")
    gov = (us["subset"] == "governors_2010_2018").to_numpy()
    items.append(dict(name="usgov", matrix=matrix, doc_index=doc_index, n=len(us), rule="k:8", nli=(R / "external" / "usgov", "usgov_speeches"), spent=True,
                      evals=[dict(system="PE_axis", label="gov2010_18", target=us["score"].to_numpy(float), metric="spearman", groups=us["group"].astype(str).to_numpy(),
                                  mask=gov, stored=(R / "external" / "usgov" / "final_scores_speeches.csv", "B4_axis")),
                             dict(system="PE_axis_plus_nli", label="gov2010_18", target=us["score"].to_numpy(float), metric="spearman", groups=us["group"].astype(str).to_numpy(),
                                  mask=gov, stored=(R / "external" / "usgov" / "final_scores_speeches.csv", "B6_axis_plus_nli")),
                             dict(system="PE_axis_plus_nli", label="pooled", target=us["score"].to_numpy(float), metric="spearman", groups=us["group"].astype(str).to_numpy())]))

    tw = pd.read_csv(R / "external" / "antielite_tweets" / "final_scores.csv")
    matrix, doc_index, _ = load_embeddings(EXT / "antielite_tweets" / "e5-large-v2" / "spw3_cap60", "antielite_tweets")
    test = (tw["split"] == "test").to_numpy()
    items.append(dict(name="tweets", matrix=matrix, doc_index=doc_index, n=len(tw), rule="k:1", nli=(R / "external" / "antielite_tweets", "antielite_tweets"), spent=True,
                      evals=[dict(system="E_axis", label="general", target=tw["y_general"].to_numpy(int), metric="auroc", groups=tw["country"].astype(str).to_numpy(), mask=test,
                                  stored=(R / "external" / "antielite_tweets" / "final_scores.csv", "E_axis")),
                             dict(system="E_axis_plus_nli", label="general", target=tw["y_general"].to_numpy(int), metric="auroc", groups=tw["country"].astype(str).to_numpy(), mask=test,
                                  stored=(R / "external" / "antielite_tweets" / "final_scores.csv", "E_axis_plus_nli"))]))

    pb = pd.read_csv(R / "external" / "popbert_en" / "final_scores.csv")
    parts, docs, offset = [], [], 0
    for split_name in ("train", "test"):
        m, d, _ = load_embeddings(EXT / "popbert_en" / "e5-large-v2" / "spw3_cap60", f"popbert_en_{split_name}")
        parts.append(m); docs.append(d + offset); offset += int(d.max()) + 1
    matrix, doc_index = np.vstack(parts), np.concatenate(docs)
    test = (pb["split"] == "test").to_numpy()
    sent = np.arange(len(pb)).astype(str)
    items.append(dict(name="popbert", matrix=matrix, doc_index=doc_index, n=len(pb), rule="k:1", nli=(R / "external" / "popbert_en", "popbert_en"), spent=True,
                      evals=[dict(system=f"{c}_axis{suffix}", label=lab, target=pb[gold].to_numpy(int), metric="auroc", groups=sent, mask=test,
                                  stored=(R / "external" / "popbert_en" / "final_scores.csv", f"{c}_axis{suffix}"))
                             for c, lab, gold in (("E", "AE", "elite_gold"), ("P", "PC", "centr_gold")) for suffix in ("", "_plus_nli")]))

    pt = pd.read_csv(R / "external" / "portugal" / "final_scores.csv")
    matrix, doc_index, _ = load_embeddings(EXT / "portugal" / "e5-large-v2" / "spw3_cap60", "portugal_en")
    actors = pt["actor"].astype(str).to_numpy()
    items.append(dict(name="portugal", matrix=matrix, doc_index=doc_index, n=len(pt), rule="k:8", nli=(R / "external" / "portugal", "portugal"), spent=True,
                      evals=[dict(system=f"{c}_axis{suffix}", label=lab, target=pt[gold].to_numpy(int), metric="auroc", groups=actors,
                                  stored=(R / "external" / "portugal" / "final_scores.csv", f"{c}_axis{suffix}"))
                             for c, lab, gold in (("E", "AE", "elite_gold"), ("P", "PC", "centr_gold")) for suffix in ("", "_plus_nli")]))
    return items


# ----------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    axes = pole_mean_axes(load_anchor_groups(ANCHORS))
    items = corpora()

    # Reference statistics from English-dev under both pooling rules.
    dev = items[0]
    requests, window_nli = nli_window_quantities(*dev["nli"])
    stats: dict = {"source": "english_dev, e5-large-v2 spw3_cap60_clean, carried B4 configuration", "n_windows": int(len(dev["matrix"])),
                   "n_documents": int(dev["n"]), "doc": {}}
    window, used = window_quantities(dev["matrix"], axes, None)
    stats["mu"] = used["mu"]; stats["proj"] = used["proj"]; stats["act"] = used["act"]
    for rule in ("k:8", "k:1"):
        docs, _ = document_quantities(dev["matrix"], dev["doc_index"], dev["n"], rule, requests, window_nli, axes, None)
        stats["doc"][rule] = {q: (float(v.mean()), float(v.std())) for q, v in docs.items()}
    np.save(OUT / "reference_mean_embedding.npy", stats["mu"].astype(np.float32))
    (OUT / "reference_stats.json").write_text(json.dumps({k: v for k, v in stats.items() if k != "mu"}, indent=2) + "\n", encoding="utf-8")

    results: dict = {"reference": {k: v for k, v in stats.items() if k != "mu"}, "corpora": {}, "draws": args.draws, "seed": args.seed}
    rows = []
    for item in items:
        name, rule = item["name"], item["rule"]
        requests, window_nli = nli_window_quantities(*item["nli"])
        docs_t, used_t = document_quantities(item["matrix"], item["doc_index"], item["n"], rule, requests, window_nli, axes, None)
        docs_f, _ = document_quantities(item["matrix"], item["doc_index"], item["n"], rule, requests, window_nli, axes, stats)
        trans, frozen = combine(docs_t, rule, None), combine(docs_f, rule, stats)
        entry: dict = {"rule": rule, "n": int(item["n"]), "n_windows": int(len(item["matrix"])), "spent": item["spent"],
                       "corpus_statistics": {"mu_distance_to_reference": float(np.linalg.norm(used_t["mu"] - stats["mu"])),
                                             "mu_cosine_to_reference": float(used_t["mu"] @ stats["mu"] / (np.linalg.norm(used_t["mu"]) * np.linalg.norm(stats["mu"]))),
                                             "proj": used_t["proj"], "act": used_t["act"],
                                             "doc": {q: (float(v.mean()), float(v.std())) for q, v in docs_t.items()}},
                       "rank_agreement_transductive_vs_frozen": {s: float(spearmanr(trans[s], frozen[s]).statistic) for s in trans},
                       "reproduction_max_abs_diff": {}, "evaluations": []}
        scores = pd.DataFrame({"doc_index": np.arange(item["n"])})
        for s in trans:
            scores[f"{s}__transductive"], scores[f"{s}__frozen"] = trans[s], frozen[s]
        scores.to_csv(OUT / f"scores_{name}.csv", index=False)
        for ev in item["evals"]:
            s = ev["system"]
            if "stored" in ev:
                path, column = ev["stored"]
                diff = float(np.abs(pd.read_csv(path)[column].to_numpy(float) - trans[s]).max())
                entry["reproduction_max_abs_diff"][s] = diff
                if diff > args.tolerance:
                    raise SystemExit(f"{name}/{s}: transductive recomputation differs from stored scores (max abs diff {diff:.3e})")
            metric = spearman if ev["metric"] == "spearman" else auroc
            mask = ev.get("mask", np.ones(item["n"], bool))
            t, g = ev["target"][mask], ev["groups"][mask]
            paired = paired_delta(frozen[s][mask], trans[s][mask], t, g, metric, args.draws, args.seed)
            row = {"system": s, "label": ev.get("label", ""), "metric": ev["metric"], "n_evaluated": int(mask.sum()),
                   "transductive": metric(t, trans[s][mask]), "frozen": metric(t, frozen[s][mask]),
                   "rank_agreement": float(spearmanr(trans[s][mask], frozen[s][mask]).statistic), "paired_frozen_minus_transductive": paired}
            entry["evaluations"].append(row)
            rows.append({"corpus": name, **row})
            print(f"{name:16s} {s:18s} {row['label']:10s} rho(t,f)={row['rank_agreement']:.3f}  {ev['metric']} {row['transductive']:.3f} -> {row['frozen']:.3f}  "
                  f"Δ {paired['delta']:+.3f} [{paired['ci95'][0]:+.3f}, {paired['ci95'][1]:+.3f}]", flush=True)
        results["corpora"][name] = entry
    (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    lines = ["Frozen reference scales: every transductive statistic replaced by its English-dev value. "
             "ρ(t, f) is the Spearman rank agreement between the transductive and frozen scores on the evaluated rows; "
             "Δ is the frozen minus transductive metric with a cluster-bootstrap 95% CI.", "",
             "| Corpus | System | Target | Metric | n | ρ(t, f) | transductive | frozen | Δ [95% CI] |", "|---|---|---|---|---:|---:|---:|---:|---|"]
    for r in rows:
        p = r["paired_frozen_minus_transductive"]
        lines.append(f"| {r['corpus']} | {r['system']} | {r['label']} | {r['metric']} | {r['n_evaluated']} | {r['rank_agreement']:.3f} | "
                     f"{r['transductive']:.3f} | {r['frozen']:.3f} | {p['delta']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] |")
    lines += ["", "Corpus statistics against the reference (English-dev). Projection mean / SD per channel on the corpus's own centering; "
              "‖μ − μ_ref‖ is the distance between the corpus mean embedding and the reference mean.", "",
              "| Corpus | windows | ‖μ − μ_ref‖ | cos(μ, μ_ref) | P proj mean / SD | E proj mean / SD | P act. scale | E act. scale |", "|---|---:|---:|---:|---|---|---:|---:|"]
    for name, e in results["corpora"].items():
        c = e["corpus_statistics"]
        lines.append(f"| {name} | {e['n_windows']} | {c['mu_distance_to_reference']:.3f} | {c['mu_cosine_to_reference']:.4f} | "
                     f"{c['proj']['P'][0]:+.4f} / {c['proj']['P'][1]:.4f} | {c['proj']['E'][0]:+.4f} / {c['proj']['E'][1]:.4f} | {c['act']['P']:.3f} | {c['act']['E']:.3f} |")
    (OUT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_run_manifest(OUT, script="scripts/frozen_reference.py", inputs={"anchors": ANCHORS, "reference_embeddings": EMB / "spw3_cap60_clean" / "embeddings_english.npz"},
                       extra={"draws": args.draws, "tolerance": args.tolerance})
    print("\n".join(lines))


if __name__ == "__main__":
    main()
