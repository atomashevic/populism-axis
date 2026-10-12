#!/usr/bin/env python3
"""Encode corrected independent-label RoBERTa scores on existing one-sentence windows.

Requires pinned model/tokenizer revisions. Creates sigmoid_<name>.npz with raw
logits, independent probabilities and versioned metadata; retains legacy caches.
No change to the historical splitter, token limit or label mapping.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from populism_axis.contextual_axis import split_windows  # noqa: E402
from populism_axis.run_manifest import write_run_manifest  # noqa: E402
from populism_axis.roberta_comparator import pinned_revision, validate_config, save_scores  # noqa: E402

MODEL_ID = "coastalcph/roberta-large-ft-trump-populism"
TOKENIZER_ID = "FacebookAI/roberta-large"


def load_texts(csv: Path, text_column: str, text_path_column: str | None) -> tuple[list[str], dict[str, Path]]:
    frame = pd.read_csv(csv)
    inputs = {"csv": csv}
    if text_path_column is None:
        return frame[text_column].astype(str).tolist(), inputs
    # Same path resolution and decoding as scripts/benchmarks.py for English.
    texts = []
    for index, row in frame.iterrows():
        raw = str(row[text_path_column])
        path = ROOT / raw if raw.startswith("data") else ROOT / "data" / raw
        if not path.exists():
            path = ROOT / "data" / "speeches_20251120" / str(row["merging_variable"])
        texts.append(path.read_text(encoding="utf-8", errors="ignore"))
        inputs[f"text_{index:05d}"] = path
    return texts, inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--text-path-column", help="English benchmark column containing speech file paths")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-revision", type=pinned_revision, required=True)
    parser.add_argument("--tokenizer-revision", type=pinned_revision, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    for attr in ("csv", "out_dir"):
        value = getattr(args, attr)
        if not value.is_absolute():
            setattr(args, attr, Path.cwd() / value)

    if not args.name or Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("--name must be a single filename component")
    cache_path = args.out_dir / f"sigmoid_{args.name}.npz"
    manifest_name = f"roberta_sigmoid_{args.name}.manifest.json"
    if cache_path.exists() or (args.out_dir / manifest_name).exists():
        parser.error("Corrected output already exists; choose a new run directory")

    texts, input_files = load_texts(args.csv, args.text_column, args.text_path_column)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(2)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=args.tokenizer_revision)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, revision=args.model_revision).eval().to(args.device)
    validate_config(model.config)
    if args.device.startswith("cuda"):
        model = model.half()

    windows: list[str] = []
    doc_index: list[int] = []
    for index, text in enumerate(texts):
        parts = split_windows(text, 1)
        windows.extend(parts)
        doc_index.extend([index] * len(parts))
    print(f"[roberta] {args.name}: {len(texts)} docs -> {len(windows)} windows", flush=True)

    logits = np.zeros((len(windows), model.config.num_labels), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(windows), args.batch_size):
            chunk = windows[start: start + args.batch_size]
            batch = tokenizer(chunk, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt").to(args.device)
            logits[start: start + len(chunk)] = model(**batch).logits.float().cpu().numpy()
    window_hash = hashlib.sha256()
    for index, text in zip(doc_index, windows):
        raw = text.encode("utf-8")
        window_hash.update(f"{index}:{len(raw)}:".encode("ascii"))
        window_hash.update(raw)
    metadata = {"name": args.name, "model_id": MODEL_ID, "model_revision": args.model_revision,
                "tokenizer_id": TOKENIZER_ID, "tokenizer_revision": args.tokenizer_revision,
                "resolved_model_revision": getattr(model.config, "_commit_hash", None),
                "max_length": args.max_length, "sentences_per_window": 1, "max_words": 0,
                "text_column": args.text_column, "text_path_column": args.text_path_column, "ordered_windows_sha256": window_hash.hexdigest(),
                "n_documents": len(texts), "n_windows": len(windows)}
    save_scores(cache_path, np.asarray(doc_index, dtype=np.int32), logits, metadata)
    write_run_manifest(args.out_dir, script="scripts/external/encode_roberta_custom.py",
                       inputs=dict(input_files, scores=cache_path), encoder_id=MODEL_ID,
                       extra=dict(metadata, activation="sigmoid", schema="populism_axis.roberta.sigmoid.v1",
                                  id2label={str(k): v for k, v in model.config.id2label.items()}),
                       filename=manifest_name)
    print(f"[roberta] {args.name}: wrote probabilities", flush=True)


if __name__ == "__main__":
    main()
