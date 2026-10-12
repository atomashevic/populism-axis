#!/usr/bin/env python3
"""Score new texts with the populism axis and the claim check (CPU or GPU).

Input: a CSV with one text per row. Output: a score table, a per-document evidence
file, and a run manifest. The configuration is the frozen one evaluated in the
development log (`docs/method.md`):

  axis         intfloat/e5-large-v2 (pinned Hub commit), three-sentence windows (stride 1, 60 words per
               sentence), proposition anchors in anchors/proposition_anchors.json,
               P and E projections, softplus activation, geometric-mean conjunction,
               mean of the top-K windows (K = 8 for speeches and paragraphs, 1 for
               sentences and tweets).
  claim check  facebook/bart-large-mnli (pinned Hub commit) entailment of the ten anchor claims in
               anchors/claim_hypotheses.json on the same top-K windows.
  combined     mean of the z-scores of the axis and claim-check document scores.

Standardization (`--reference`):
  corpus   centre, scale and z-score on the input corpus itself, as the benchmark
           evaluations do. Needs a corpus of some size; refused below --min-corpus.
  frozen   use the English development-set statistics stored under
           results/frozen_reference/ (centering mean, projection scales, activation
           scales, document scales). Works for any input size, including one text.

Example:
  uv run --extra cpu python scripts/score_texts.py --input speeches.csv --text-column text \
      --id-column id --unit speech --reference frozen --device cpu --out-dir scored/run1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from populism_axis.contextual_axis import aggregate_from_window_scores, split_windows  # noqa: E402
from populism_axis.encoders import get_spec, make_encoder  # noqa: E402
from populism_axis.geometry import pole_mean_axes  # noqa: E402
from populism_axis.model_revisions import BART_LARGE_MNLI, E5_LARGE_V2, PINNED_REVISIONS, require_revision  # noqa: E402
from populism_axis.run_manifest import write_run_manifest  # noqa: E402
from populism_axis.sentences import load_sentence_groups  # noqa: E402
from populism_axis.text import CLEANING_VERSION, clean_speech_text  # noqa: E402
from populism_axis.preset import CH, standardize, window_quantities, top_windows
from populism_axis.nli import run_nli, validate_execution, preflight_models  # noqa: E402

ANCHORS = ROOT / "anchors" / "proposition_anchors.json"
HYPOTHESES = ROOT / "anchors" / "claim_hypotheses.json"
REFERENCE = ROOT / "results" / "frozen_reference"
REFERENCE_ENCODER = "e5-large-v2"  # encoder whose English-dev statistics are stored under REFERENCE
TARGET_LABELS = ("contradiction", "neutral", "entailment")


def load_reference() -> dict:
    stats = json.loads((REFERENCE / "reference_stats.json").read_text(encoding="utf-8"))
    stats["mu"] = np.load(REFERENCE / "reference_mean_embedding.npy").astype(np.float64)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="CSV with one text per row")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--id-column", default=None, help="column to carry as the document id (default: row number)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--unit", choices=("speech", "sentence"), default="speech",
                        help="speech: mean of the top 8 windows (speeches, paragraphs); sentence: top window (sentences, tweets)")
    parser.add_argument("--top-k", type=int, default=None, help="override the pooling depth implied by --unit")
    parser.add_argument("--reference", choices=("corpus", "frozen"), default="corpus")
    parser.add_argument("--min-corpus", type=int, default=30, help="smallest input for --reference corpus")
    parser.add_argument("--clean", action="store_true", help=f"apply the {CLEANING_VERSION} speech-file cleaner before windowing")
    parser.add_argument("--no-claim-check", action="store_true", help="axis only; skip the NLI stage")
    parser.add_argument("--encoder", default="e5-large-v2")
    parser.add_argument("--nli-model", default="facebook/bart-large-mnli")
    parser.add_argument("--encoder-revision", default=None,
                        help="exact 40-character Hub commit for --encoder (default: its pinned revision; required if it has none)")
    parser.add_argument("--nli-revision", default=None,
                        help="exact 40-character Hub commit for --nli-model (default: its pinned revision; required if it has none)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--encoder-batch-size", type=int, default=None)
    parser.add_argument("--nli-batch-size", type=int, default=16)
    parser.add_argument("--allow-download", action="store_true", help="allow explicitly downloading pinned model revisions")
    args = parser.parse_args()

    validate_execution(args.device, args.threads, args.encoder_batch_size, args.nli_batch_size, not args.allow_download)
    if args.top_k is not None and args.top_k <= 0:
        raise SystemExit("--top-k must be a positive integer")
    if args.min_corpus <= 0:
        raise SystemExit("--min-corpus must be a positive integer")
    frame = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    texts = frame[args.text_column].astype(str).tolist()
    ids = frame[args.id_column].astype(str).tolist() if args.id_column else [str(i) for i in range(len(frame))]
    n = len(texts)
    if n == 0:
        raise SystemExit("no rows in input")
    if args.reference == "frozen" and args.encoder != REFERENCE_ENCODER:
        raise SystemExit(f"--reference frozen holds {REFERENCE_ENCODER} statistics only; use --encoder {REFERENCE_ENCODER} or --reference corpus")
    if args.reference == "corpus" and n < args.min_corpus:
        raise SystemExit(f"{n} documents is too few for --reference corpus (minimum {args.min_corpus}); use --reference frozen")
    try:
        encoder_revision = require_revision(get_spec(args.encoder).model_id, args.encoder_revision or get_spec(args.encoder).revision)
        nli_revision = None if args.no_claim_check else require_revision(args.nli_model, args.nli_revision)
    except (KeyError, ValueError) as error:
        raise SystemExit(str(error))
    # The stored frozen statistics were computed with the pinned e5 and BART bytes; other
    # checkpoints would get scales that do not belong to them.
    if args.reference == "frozen":
        if encoder_revision != PINNED_REVISIONS[E5_LARGE_V2]:
            raise SystemExit("--reference frozen holds statistics for the pinned e5-large-v2 commit only; drop --encoder-revision or use --reference corpus")
        if not args.no_claim_check and (args.nli_model != BART_LARGE_MNLI or nli_revision != PINNED_REVISIONS[BART_LARGE_MNLI]):
            raise SystemExit(f"--reference frozen holds claim-check statistics for the pinned {BART_LARGE_MNLI} commit only; "
                             "drop --nli-model/--nli-revision, use --no-claim-check or use --reference corpus")
    k = args.top_k or (8 if args.unit == "speech" else 1)
    rule = f"k:{k}"
    stats = load_reference() if args.reference == "frozen" else None
    if stats is not None and rule not in stats["doc"]:
        print(f"warning: no frozen document scales for {rule}; document-level z-scores fall back to the corpus", flush=True)
    preflight_models([(get_spec(args.encoder).model_id, encoder_revision)] + ([] if args.no_claim_check else [(args.nli_model, nli_revision)]), not args.allow_download)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # Windows.
    if args.clean:
        texts = [clean_speech_text(text)[0] for text in texts]
    windows, doc_index = [], []
    for index, text in enumerate(texts):
        parts = split_windows(text, 3, 1, max_words=60)
        windows.extend(parts)
        doc_index.extend([index] * len(parts))
    doc_index = np.asarray(doc_index, dtype=int)
    if not len(windows):
        raise SystemExit("no windows produced; are the texts empty?")
    print(f"{n} documents -> {len(windows)} windows", flush=True)

    # Encoder: anchors and windows in one process, identical rendering.
    spec = get_spec(args.encoder, batch_size=args.encoder_batch_size, max_length=256, revision=encoder_revision)
    encode, close = make_encoder(spec, args.device, num_threads=args.threads, log_every=0, local_files_only=not args.allow_download)
    try:
        _, groups = load_sentence_groups(ANCHORS)
        anchor_embeddings = {name: encode(sentences) for name, sentences in groups.items()}
        axes = pole_mean_axes(anchor_embeddings)
        t0 = time.time()
        matrix = encode(windows)
        resolved = {"encoder": getattr(encode, "resolved_revision", None), "nli_model": None}
    finally:
        close()
    encode_seconds = time.time() - t0
    print(f"encoded {len(windows)} windows in {encode_seconds:.0f} s", flush=True)

    # Axis stage.
    window, used = window_quantities(matrix, axes, stats)
    docs = {q: aggregate_from_window_scores(v, doc_index, n, rule) for q, v in window.items()}
    tops = top_windows(window["PE_axis"], doc_index, n, k)

    # Claim check on the top-K windows of each document.
    nli_seconds = 0.0
    hypotheses = json.loads(HYPOTHESES.read_text(encoding="utf-8"))
    if not args.no_claim_check:
        premise_positions = np.concatenate([tops[d] for d in range(n)])
        t0 = time.time()
        entail, resolved["nli_model"] = run_nli(args.nli_model, [windows[p] for p in premise_positions], [h["text"] for h in hypotheses],
                                                args.device, args.nli_batch_size, 320, args.threads, nli_revision, local_files_only=not args.allow_download)
        nli_seconds = time.time() - t0
        print(f"claim check: {entail.size} pairs in {nli_seconds:.0f} s", flush=True)
        cols = {c: [i for i, h in enumerate(hypotheses) if h["channel"] == c] for c in CH}
        per_window = {"P_nli": entail[:, cols["P"]].max(axis=1), "E_nli": entail[:, cols["E"]].max(axis=1)}
        per_window["PE_nli"] = np.sqrt(per_window["P_nli"] * per_window["E_nli"])
        best = {c: [hypotheses[cols[c][j]]["hypothesis_id"] for j in entail[:, cols[c]].argmax(axis=1)] for c in CH}
        premise_doc = doc_index[premise_positions]
        for q, v in per_window.items():
            docs[q] = pd.Series(v).groupby(premise_doc).mean().reindex(range(n)).fillna(0.0).to_numpy()

    # Document-level z-scores and the combination.
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

    seconds = time.time() - started
    write_run_manifest(args.out_dir, script="scripts/score_texts.py", inputs={"input": args.input, "anchors": ANCHORS, "hypotheses": HYPOTHESES},
                       encoder_id=spec.model_id,
                       extra={"nli_model": None if args.no_claim_check else args.nli_model, "unit": args.unit,
                              "encoder_revision": encoder_revision, "nli_revision": nli_revision, "resolved_revisions": resolved, "top_rule": rule, "reference": args.reference,
                              "device": args.device, "threads": args.threads, "n_documents": n, "n_windows": int(len(windows)),
                              "encode_seconds": round(encode_seconds, 1), "nli_seconds": round(nli_seconds, 1), "seconds_total": round(seconds, 1),
                              "standardization": {"proj": used["proj"], "act": used["act"]}})
    print(f"wrote {args.out_dir / 'scores.csv'} ({n} documents) in {seconds:.0f} s total", flush=True)


if __name__ == "__main__":
    main()
