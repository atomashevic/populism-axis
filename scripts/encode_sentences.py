#!/usr/bin/env python3
"""Encode a sentence file (anchors, nuisance sets, hypotheses) with a frozen encoder.

GPU script for pvgpu1. Writes raw pooled float32 embeddings per sentence group
to `<out-root>/<encoder label>/sentences/<set-id>__<sha8>.npz`; no centering,
no axis building. `--repeat 2` encodes twice and records whether the two passes
are byte-identical (determinism check).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from populism_axis.run_manifest import file_sha256, write_run_manifest  # noqa: E402
from populism_axis.encoders import DEFAULT_TASK, get_spec, make_encoder, render_text  # noqa: E402
from populism_axis.sentences import load_sentence_groups  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", default="e5-large-v2", help="label in ENCODER_SPECS")
    parser.add_argument("--sentences", type=Path, required=True)
    parser.add_argument("--set-id", required=True)
    parser.add_argument("--out-root", type=Path, default=ROOT / "results" / "embeddings")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    args.sentences = args.sentences if args.sentences.is_absolute() else (Path.cwd() / args.sentences)
    args.out_root = args.out_root if args.out_root.is_absolute() else (Path.cwd() / args.out_root)

    import torch

    spec = get_spec(args.encoder, batch_size=args.batch_size, max_length=args.max_length)
    schema, groups = load_sentence_groups(args.sentences)
    source_sha = file_sha256(args.sentences)
    out_dir = args.out_root / spec.label / "sentences"
    out_path = out_dir / f"{args.set_id}__{source_sha[:8]}.npz"

    order = [(name, index) for name, sentences in groups.items() for index in range(len(sentences))]
    flat = [groups[name][index] for name, index in order]
    print(f"[{spec.label}] {args.set_id}: {len(flat)} sentences in {len(groups)} groups", flush=True)

    started = time.time()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    encode, close = make_encoder(spec, args.device, task=args.task)
    passes = [encode(flat) for _ in range(max(1, args.repeat))]
    close()
    seconds = time.time() - started
    peak_mb = (
        float(torch.cuda.max_memory_allocated() / 2**20) if args.device.startswith("cuda") else None
    )
    identical = all(np.array_equal(passes[0], other) for other in passes[1:]) if len(passes) > 1 else None

    matrix = passes[0]
    arrays: dict[str, np.ndarray] = {}
    cursor = 0
    for name, sentences in groups.items():
        arrays[name] = matrix[cursor : cursor + len(sentences)].astype(np.float32)
        cursor += len(sentences)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        **arrays,
        group_names=np.asarray(list(groups)),
        source_sha256=np.asarray(source_sha),
        encoder_label=np.asarray(spec.label),
        template=np.asarray(spec.template),
    )

    info = {
        "set_id": args.set_id,
        "schema": schema,
        "n_sentences": len(flat),
        "groups": {name: len(sentences) for name, sentences in groups.items()},
        "output": str(out_path.resolve().relative_to(ROOT.resolve())),
        "encoder_spec": spec.as_dict(),
        "resolved_model_revision": getattr(encode, "resolved_revision", None),
        "task": args.task if "{task}" in spec.template else None,
        "example_rendered_input": render_text(spec, flat[0], args.task),
        "device": args.device,
        "gpu_name": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "seconds": round(seconds, 2),
        "peak_vram_mb": None if peak_mb is None else round(peak_mb, 1),
        "repeat": args.repeat,
        "repeat_byte_identical": identical,
    }
    write_run_manifest(
        out_dir,
        script="scripts/encode_sentences.py",
        inputs={"sentences": args.sentences},
        encoder_id=spec.model_id,
        extra=info,
        filename=f"{args.set_id}__{source_sha[:8]}.manifest.json",
    )
    print(json.dumps(info, indent=2), flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
