#!/usr/bin/env python3
"""Translate a CSV text column to English with a Marian OPUS-MT model (GPU, pvgpu1).

Paragraphs are split into sentences (Portuguese/Romance abbreviations protected), each
sentence is translated with beam search, and the translations are rejoined in order.
Writes the input CSV plus `<text-column>_en` (default column name `text_en`) and a run
manifest. Deterministic (beam search, no sampling).
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from populism_axis.run_manifest import write_run_manifest  # noqa: E402

ABBREVIATIONS = ("Sr", "Sra", "Srs", "Sras", "Dr", "Dra", "Drs", "Prof", "Profs", "Exmo", "Exma", "Ex", "art", "arts",
                 "n", "N", "nº", "Nº", "p", "pp", "pág", "págs", "séc", "cf", "etc", "vs", "Eng", "Engº", "Dep", "Deps",
                 "Pres", "Min", "St", "Sta", "Av", "Ilmo", "Ilma", "V", "Exas", "Exas", "S", "Ex.ª", "Ex.mo", "Ex.ma")
_ABBREV = {a.lower() for a in ABBREVIATIONS}
SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[«\"“(\[A-ZÁÉÍÓÚÂÊÔÃÕÇ0-9])")


def split_sentences(text: str) -> list[str]:
    parts = SPLIT.split(text.strip())
    merged: list[str] = []
    for part in parts:
        if merged:
            last_word = merged[-1].rstrip(".!?…").split()[-1] if merged[-1].split() else ""
            if last_word.lower().rstrip(".") in _ABBREV or len(last_word) == 1:
                merged[-1] = merged[-1] + " " + part
                continue
        merged.append(part)
    return [m for m in merged if m.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--text-column", default="text_pt")
    parser.add_argument("--out-column", default="text_en")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--model", default="Helsinki-NLP/opus-mt-ROMANCE-en")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    for attr in ("csv", "out_csv"):
        value = getattr(args, attr)
        if not value.is_absolute():
            setattr(args, attr, Path.cwd() / value)

    import torch
    from transformers import MarianMTModel, MarianTokenizer

    torch.set_num_threads(2)
    frame = pd.read_csv(args.csv)
    if args.limit:
        frame = frame.head(args.limit).copy()
    texts = frame[args.text_column].astype(str).tolist()
    sentences: list[str] = []
    owner: list[int] = []
    for index, text in enumerate(texts):
        parts = split_sentences(text)
        sentences.extend(parts)
        owner.extend([index] * len(parts))
    print(f"[marian] {len(texts)} paragraphs -> {len(sentences)} sentences", flush=True)

    tokenizer = MarianTokenizer.from_pretrained(args.model)
    model = MarianMTModel.from_pretrained(args.model).eval().to(args.device)
    if args.device.startswith("cuda"):
        model = model.half()
        torch.cuda.reset_peak_memory_stats()
    order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
    translated = [""] * len(sentences)
    started = time.time()
    n_truncated_inputs = 0
    with torch.inference_mode():
        for start in range(0, len(order), args.batch_size):
            idx = order[start: start + args.batch_size]
            batch = tokenizer([sentences[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=512)
            n_truncated_inputs += int((batch["attention_mask"].sum(1) >= 512).sum())
            batch = {k: v.to(args.device) for k, v in batch.items()}
            generated = model.generate(**batch, num_beams=args.num_beams, max_new_tokens=args.max_new_tokens, do_sample=False)
            for i, text in zip(idx, tokenizer.batch_decode(generated, skip_special_tokens=True)):
                translated[i] = text.strip()
            if (start // args.batch_size) % 50 == 0:
                print(f"  {min(start + args.batch_size, len(order))}/{len(order)} sentences", flush=True)
    joined = [[] for _ in texts]
    for i, doc in enumerate(owner):
        joined[doc].append(translated[i])
    frame[args.out_column] = [" ".join(parts) for parts in joined]
    frame[f"{args.out_column}_n_sentences"] = [len(parts) for parts in joined]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_csv, index=False)
    seconds = time.time() - started
    peak = float(torch.cuda.max_memory_allocated() / 2**20) if args.device.startswith("cuda") else None
    write_run_manifest(args.out_csv.parent, script="scripts/external/translate_marian.py", inputs={"csv": args.csv},
                       encoder_id=args.model,
                       extra={"n_paragraphs": len(texts), "n_sentences": len(sentences), "num_beams": args.num_beams,
                              "n_truncated_inputs": n_truncated_inputs, "seconds": round(seconds, 1), "peak_vram_mb": peak,
                              "output": args.out_csv.name},
                       filename=f"translate_{args.out_csv.stem}.manifest.json")
    print(f"[marian] wrote {args.out_csv} in {seconds:.0f}s (truncated inputs: {n_truncated_inputs})", flush=True)


if __name__ == "__main__":
    main()
