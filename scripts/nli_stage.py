#!/usr/bin/env python3
"""stage 6, stage 2 (GPU, pvgpu1): NLI probabilities for window x hypothesis pairs.

Reads requests_<name>.parquet and hypotheses.json from --in-dir and writes
nli_<name>__<model label>.npz with `request_id` and `probs` (n_requests x
n_hypotheses x 3, float16, columns [contradiction, neutral, entailment] in that
order regardless of the model's own label order). Pairs are processed sorted by
premise length to limit padding; output order follows the request file.
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

TARGET_LABELS = ("contradiction", "neutral", "entailment")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=ROOT / "results" / "step6_nli")
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
    label_index = {label.lower(): int(index) for index, label in model.config.id2label.items()}
    column = [label_index[label] for label in TARGET_LABELS]
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
                probs[idx] = torch.softmax(logits, dim=-1)[:, column].cpu().numpy()
                if batch_number % 200 == 0:
                    print(f"  {name}: {start + len(idx)}/{len(order)} pairs", flush=True)
        if not np.isfinite(probs).all():
            raise FloatingPointError(f"{name}: non-finite NLI probabilities")
        out = args.in_dir / f"nli_{name}__{label}.npz"
        np.savez_compressed(out, request_id=requests["request_id"].to_numpy(),
                            probs=probs.reshape(n_req, n_hyp, 3).astype(np.float16),
                            hypothesis_ids=np.asarray([h["hypothesis_id"] for h in hypotheses]),
                            labels=np.asarray(TARGET_LABELS))
        info[name] = {"pairs": int(len(order)), "seconds": round(time.time() - started, 1), "output": out.name}
        print(name, info[name], flush=True)
    info["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1) if args.device.startswith("cuda") else None
    info["model"] = args.model
    info["model_revision"] = revision
    info["resolved_model_revision"] = getattr(model.config, "_commit_hash", None)
    info["id2label"] = {str(k): v for k, v in model.config.id2label.items()}
    write_run_manifest(args.in_dir, script="scripts/nli_stage.py",
                       inputs={"hypotheses": args.in_dir / "hypotheses.json"},
                       encoder_id=args.model, extra=info, filename=f"nli__{label}.manifest.json")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
