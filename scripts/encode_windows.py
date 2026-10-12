#!/usr/bin/env python3
"""Encode benchmark documents as sentence windows with a frozen encoder (GPU, pvgpu1).

Writes raw pooled embeddings only; centering, whitening, axes and projection
all happen on the CPU side from these files. Output directory:
`<out-root>/<encoder label>/spw<n>_cap<N>/` with, per dataset,
`embeddings_<dataset>.npz` (fp16 `embeddings`, int32 `doc_index`, int32
`n_words`, int32 `n_tokens` = untruncated token count incl. special tokens),
`windows_<dataset>.parquet`, plus `window_stats.json` and a run manifest.
EU-OOD is refused unless `--allow-eu` is passed (final holdout step only).
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

from populism_axis.contextual_axis import split_windows  # noqa: E402
from populism_axis.run_manifest import write_run_manifest  # noqa: E402
from populism_axis.encoders import DEFAULT_TASK, get_spec, make_encoder, render_text  # noqa: E402
from populism_axis.text import CLEANING_VERSION, clean_speech_text  # noqa: E402
from benchmarks import load_documents  # noqa: E402

DATASET_FILES = {
    "english": "english_gpd_speeches.csv",
    "agora": "agora_paragraphs.csv",
    "eu": "eu_sentences.csv",
}


def build_windows(texts: list[str], spw: int, stride: int, max_words: int | None) -> tuple[list[str], list[int]]:
    windows: list[str] = []
    doc_index: list[int] = []
    for index, text in enumerate(texts):
        parts = split_windows(text, spw, stride, max_words=max_words)
        windows.extend(parts)
        doc_index.extend([index] * len(parts))
    return windows, doc_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", default="e5-large-v2")
    parser.add_argument("--datasets", default="english,agora")
    parser.add_argument("--bench-dir", type=Path, default=ROOT / "data" / "benchmarks")
    parser.add_argument("--out-root", type=Path, default=ROOT / "results" / "embeddings")
    parser.add_argument("--sentences-per-window", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-words", type=int, default=60, help="0 = uncapped splitter (no cap)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-eu", action="store_true")
    parser.add_argument("--limit-docs", type=int, default=0, help="smoke testing only")
    parser.add_argument("--clean-text", action="store_true", help=f"apply {CLEANING_VERSION} to English speeches")
    parser.add_argument("--custom-csv", type=Path, default=None,
                        help="encode a custom CSV (external benchmark) instead of the core benchmark datasets; text in --text-column")
    parser.add_argument("--custom-name", default=None, help="dataset name for the output files with --custom-csv")
    parser.add_argument("--text-column", default="text")
    args = parser.parse_args()
    if args.custom_csv and not args.custom_csv.is_absolute():
        args.custom_csv = Path.cwd() / args.custom_csv
    if not args.out_root.is_absolute():
        args.out_root = Path.cwd() / args.out_root

    wanted = [item.strip() for item in args.datasets.split(",") if item.strip()]
    if args.custom_csv:
        if not args.custom_name:
            raise SystemExit("--custom-name is required with --custom-csv")
        wanted = [args.custom_name]
    if "eu" in wanted and not args.allow_eu:
        raise SystemExit("EU-OOD is held out; pass --allow-eu only in the final holdout step")

    import torch
    from transformers import AutoTokenizer

    spec = get_spec(args.encoder, batch_size=args.batch_size, max_length=args.max_length)
    max_words = args.max_words if args.max_words > 0 else None
    setting = f"spw{args.sentences_per_window}_cap{args.max_words}" + ("_clean" if args.clean_text else "")
    out_dir = args.out_root / spec.label / setting
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.custom_csv:
        custom = pd.read_csv(args.custom_csv)
        documents = {args.custom_name: {"frame": custom, "texts": custom[args.text_column].astype(str).tolist()}}
        DATASET_FILES[args.custom_name] = args.custom_csv
    else:
        documents = load_documents(args.bench_dir)
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id, trust_remote_code=spec.trust_remote_code, revision=spec.revision)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    encode, close = make_encoder(spec, args.device, task=args.task)
    effective_max_length = int(getattr(encode, "effective_max_length", spec.max_length))

    stats: dict = {
        "setting": setting,
        "cleaning": CLEANING_VERSION if args.clean_text else None,
        "encoder_spec": spec.as_dict(),
        "datasets": {},
    }
    inputs = {}
    for name in wanted:
        texts = documents[name]["texts"]
        if args.limit_docs:
            texts = texts[: args.limit_docs]
        if args.clean_text and name == "english":
            texts = [clean_speech_text(text)[0] for text in texts]
        windows, doc_index = build_windows(texts, args.sentences_per_window, args.stride, max_words)
        rendered = [render_text(spec, window, args.task) for window in windows]
        n_tokens = np.asarray(
            [len(ids) for ids in tokenizer(rendered, add_special_tokens=True, truncation=False)["input_ids"]],
            dtype=np.int32,
        )
        n_words = np.asarray([len(window.split()) for window in windows], dtype=np.int32)
        print(f"[{spec.label} {setting}] {name}: {len(texts)} docs -> {len(windows)} windows", flush=True)
        dataset_started = time.time()
        matrix = encode(windows)
        np.savez_compressed(
            out_dir / f"embeddings_{name}.npz",
            embeddings=matrix.astype(np.float16),
            doc_index=np.asarray(doc_index, dtype=np.int32),
            n_words=n_words,
            n_tokens=n_tokens,
        )
        pd.DataFrame({"doc_index": doc_index, "window": windows}).to_parquet(
            out_dir / f"windows_{name}.parquet", index=False
        )
        counts = np.bincount(np.asarray(doc_index), minlength=len(texts))
        stats["datasets"][name] = {
            "n_documents": len(texts),
            "n_windows": len(windows),
            "windows_per_doc_quantiles": {str(q): float(np.quantile(counts, q)) for q in (0.05, 0.5, 0.95)},
            "max_window_words": int(n_words.max()),
            "windows_over_60_words": int((n_words > 60).sum()),
            "windows_truncated": int((n_tokens > effective_max_length).sum()),
            "words_in_truncated_windows_share": float(n_words[n_tokens > effective_max_length].sum() / n_words.sum()),
            "effective_max_length": effective_max_length,
            "encode_seconds": round(time.time() - dataset_started, 1),
        }
        inputs[f"bench_{name}"] = DATASET_FILES[name] if args.custom_csv else args.bench_dir / DATASET_FILES[name]
        print(json.dumps(stats["datasets"][name]), flush=True)
    close()
    stats["seconds_total"] = round(time.time() - started, 1)
    stats["peak_vram_mb"] = (
        round(torch.cuda.max_memory_allocated() / 2**20, 1) if args.device.startswith("cuda") else None
    )
    stats["gpu_name"] = torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None
    (out_dir / "window_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_run_manifest(
        out_dir,
        script="scripts/encode_windows.py",
        inputs=inputs,
        encoder_id=spec.model_id,
        extra={"setting": setting, "cleaning": CLEANING_VERSION if args.clean_text else None, "task": args.task if "{task}" in spec.template else None,
               "model_revision": spec.revision, "resolved_model_revision": getattr(encode, "resolved_revision", None), **{
            k: stats[k] for k in ("seconds_total", "peak_vram_mb", "gpu_name")
        }},
    )
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
