#!/usr/bin/env python3
"""stage 6, stage 2 (GPU host): NLI probabilities for window x hypothesis pairs.

Reads requests_<name>.parquet and hypotheses.json from --in-dir and writes
nli_<name>__<model label>.npz to --out-dir (default: --in-dir) with `request_id`
and `probs` (n_requests x n_hypotheses x 3, columns [contradiction, neutral,
entailment] in that order regardless of the model's own label order). A binary
entailment head (labels entailment / not_entailment) fills the same layout as
[not_entailment, 0, entailment]; the `labels` array in the file says which.
Pairs are processed sorted by premise length to limit padding; output order
follows the request file.
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

from populism_axis.model_revisions import require_revision  # noqa: E402
from populism_axis.run_manifest import write_run_manifest  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from claim_check_arms import tail_counts  # noqa: E402

TARGET_LABELS = ("contradiction", "neutral", "entailment")
BINARY_LABELS = ("not_entailment", "unused", "entailment")


def label_layout(id2label: dict) -> tuple[list[int | None], tuple[str, ...]]:
    """Model output index for each stored column (None: filled with 0) and the stored column names."""
    index = {str(label).lower(): int(i) for i, label in id2label.items()}
    if set(index) == set(TARGET_LABELS):
        return [index[label] for label in TARGET_LABELS], TARGET_LABELS
    if set(index) == {"entailment", "not_entailment"}:
        return [index["not_entailment"], None, index["entailment"]], BINARY_LABELS
    raise ValueError(f"unsupported label set {sorted(index)}; expected 3-class NLI or entailment/not_entailment")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=ROOT / "results" / "step6_nli")
    parser.add_argument("--out-dir", type=Path, default=None, help="where the .npz and manifest go (default: --in-dir)")
    parser.add_argument("--probs-dtype", choices=("float16", "float32"), default="float16",
                        help="stored precision; float16 is the layout of the step-6 files")
    parser.add_argument("--names", default="english_dev,agora")
    parser.add_argument("--model", default="facebook/bart-large-mnli")
    parser.add_argument("--revision", default=None,
                        help="exact 40-character Hub commit; default: the pinned revision for --model (required for other models)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.in_dir.is_absolute():
        args.in_dir = Path.cwd() / args.in_dir
    out_dir = args.in_dir if args.out_dir is None else (args.out_dir if args.out_dir.is_absolute() else Path.cwd() / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        revision = require_revision(args.model, args.revision)
    except ValueError as error:
        raise SystemExit(f"{error} (--revision)")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(args.model, dtype=dtype, revision=revision).eval().to(args.device)
    try:
        column, stored_labels = label_layout(model.config.id2label)
    except ValueError as error:
        raise SystemExit(f"{args.model}: {error}")
    source = [0 if c is None else c for c in column]
    unused = [i for i, c in enumerate(column) if c is None]
    hypotheses = json.loads((args.in_dir / "hypotheses.json").read_text(encoding="utf-8"))
    hyp_texts = [h["text"] for h in hypotheses]
    label = args.model.split("/")[-1]
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    info = {}
    for name in [n for n in args.names.split(",") if n]:
        requests = pd.read_parquet(args.in_dir / f"requests_{name}.parquet")
        premises = requests["premise"].tolist()
        n_req, n_hyp = len(premises), len(hyp_texts)
        pair_premise = np.repeat(np.arange(n_req), n_hyp)
        pair_hyp = np.tile(np.arange(n_hyp), n_req)
        lengths = np.asarray([len(premises[i]) for i in pair_premise])
        order = np.argsort(lengths, kind="stable")
        probs = np.zeros((n_req * n_hyp, 3), dtype=np.float32)
        started = time.time()
        with torch.no_grad():
            for batch_number, start in enumerate(range(0, len(order), args.batch_size)):
                idx = order[start: start + args.batch_size]
                batch = tokenizer([premises[i] for i in pair_premise[idx]], [hyp_texts[j] for j in pair_hyp[idx]],
                                  padding=True, truncation="only_first", max_length=args.max_length,
                                  return_tensors="pt").to(args.device)
                logits = model(**batch).logits.float()
                probs[idx] = torch.softmax(logits, dim=-1)[:, source].cpu().numpy()
                if batch_number % 200 == 0:
                    print(f"  {name}: {start + len(idx)}/{len(order)} pairs", flush=True)
        probs[:, unused] = 0.0
        if not np.isfinite(probs).all():
            raise FloatingPointError(f"{name}: non-finite NLI probabilities")
        out = out_dir / f"nli_{name}__{label}.npz"
        stored = probs.reshape(n_req, n_hyp, 3).astype(args.probs_dtype)
        np.savez_compressed(out, request_id=requests["request_id"].to_numpy(), probs=stored,
                            hypothesis_ids=np.asarray([h["hypothesis_id"] for h in hypotheses]),
                            labels=np.asarray(stored_labels))
        info[name] = {"pairs": int(len(order)), "sequences": int(len(order)), "forward_calls": int(-(-len(order) // args.batch_size)),
                      "seconds": round(time.time() - started, 1), "output": out.name,
                      "stored_entailment": tail_counts(stored[:, :, 2])}
        print(name, info[name], flush=True)
    info["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1) if args.device.startswith("cuda") else None
    info["model"] = args.model
    info["model_revision"] = revision
    info["resolved_model_revision"] = getattr(model.config, "_commit_hash", None)
    info["id2label"] = {str(k): v for k, v in model.config.id2label.items()}
    info["stored_labels"] = list(stored_labels)
    info["probs_dtype"] = args.probs_dtype
    info["batch_size"] = args.batch_size
    info["gpu"] = torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None
    write_run_manifest(out_dir, script="scripts/nli_stage.py",
                       inputs={"hypotheses": args.in_dir / "hypotheses.json"},
                       encoder_id=args.model, extra=info, filename=f"nli__{label}.manifest.json")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
