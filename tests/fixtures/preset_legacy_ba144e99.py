"""Historical numerical fixture from ba144e99, copied before scorer extraction.

Only pure helpers and formatting statements are retained; never loads models.
"""
def nonzero_median(values: np.ndarray) -> float:
    positive = values[values > 0.0]
    return float(np.median(positive)) if len(positive) else 1.0

def window_quantities(matrix: np.ndarray, axes: dict, stats: dict | None) -> tuple[dict, dict]:
    """Per-window PE conjunction score and per-channel z-projections; returns (quantities, stats used)."""
    used: dict = {}
    mu = matrix.mean(axis=0) if stats is None else np.asarray(stats["mu"])
    used["mu"] = mu
    normalized = unit_rows(matrix - mu)
    z, act, used["proj"], used["act"] = {}, {}, {}, {}
    for c in CH:
        proj = normalized @ axes[c]
        mean, sd = (float(proj.mean()), float(proj.std())) if stats is None else stats["proj"][c]
        z[c] = standardize(proj, (mean, sd))  # zero when the corpus has no projection variance
        value = np.log1p(np.exp(z[c] / SOFTNESS)) ** 2
        scale = nonzero_median(value) if stats is None else stats["act"][c]
        act[c] = value / scale
        used["proj"][c], used["act"][c] = (mean, sd), scale
    pe = np.sqrt(act["P"] * act["E"])
    return {"PE_axis": pe, "P_axis": z["P"], "E_axis": z["E"]}, used

def standardize(values: np.ndarray, ref: tuple[float, float] | None) -> np.ndarray:
    mean, sd = (float(values.mean()), float(values.std())) if ref is None else ref
    return (values - mean) / sd if sd > 1e-12 else np.zeros_like(values)

def top_windows(window_score: np.ndarray, doc_index: np.ndarray, n: int, k: int) -> dict[int, np.ndarray]:
    order = np.lexsort((-window_score, doc_index))
    sorted_docs = doc_index[order]
    out = {}
    for doc in range(n):
        lo, hi = np.searchsorted(sorted_docs, doc), np.searchsorted(sorted_docs, doc, side="right")
        out[doc] = order[lo:hi][:k]
    return out

def main():
    ref = None if stats is None or rule not in stats["doc"] else stats["doc"][rule]
    out = pd.DataFrame({"id": ids, "n_windows": np.bincount(doc_index, minlength=n)})
    out["axis"] = docs["PE_axis"]
    out["axis_P"], out["axis_E"] = docs["P_axis"], docs["E_axis"]
    out["axis_z"] = standardize(docs["PE_axis"], None if ref is None else tuple(ref["PE_axis"]))
    if not args.no_claim_check:
        out["claim_check"] = docs["PE_nli"]
        out["claim_check_P"], out["claim_check_E"] = docs["P_nli"], docs["E_nli"]
        out["claim_check_z"] = standardize(docs["PE_nli"], None if ref is None else tuple(ref["PE_nli"]))
        out["populism_axis_plus_claim_check"] = (out["axis_z"] + out["claim_check_z"]) / 2
        for c in CH:
            za = standardize(docs[f"{c}_axis"], None if ref is None else tuple(ref[f"{c}_axis"]))
            zn = standardize(docs[f"{c}_nli"], None if ref is None else tuple(ref[f"{c}_nli"]))
            out[f"{c}_axis_plus_claim_check"] = (za + zn) / 2
    out.to_csv(args.out_dir / "scores.csv", index=False)

    # Evidence: the top windows of every document with their channel values and matched claims.
    evidence = []
    position_to_row = {int(p): r for r, p in enumerate(np.concatenate([tops[d] for d in range(n)]))}
    for d in range(n):
        rows = []
        for p in tops[d]:
            row = {"window": windows[p], "P_z": float(window["P_axis"][p]), "E_z": float(window["E_axis"][p]), "axis_window_score": float(window["PE_axis"][p])}
            if not args.no_claim_check:
                r = position_to_row[int(p)]
                row.update({"P_claim": best["P"][r], "P_entailment": float(per_window["P_nli"][r]),
                            "E_claim": best["E"][r], "E_entailment": float(per_window["E_nli"][r])})
            rows.append(row)
        evidence.append({"id": ids[d], "top_windows": rows})
    (args.out_dir / "evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

