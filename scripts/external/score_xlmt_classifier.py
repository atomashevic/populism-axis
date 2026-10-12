#!/usr/bin/env python3
"""Run the released XLM-T elite-criticism classifier (Licht et al. 2025) on tweets (GPU, pvgpu1).

The classifier is `cardiffnlp/twitter-xlm-roberta-base` fine-tuned by the authors; the
package ships its weights and tokenizer. Input follows the authors' notebook: the
original-language text with URLs replaced by "http" and handles by "@user". Writes
`xlmt_probabilities.csv` (id, prob_general) and a manifest.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from populism_axis.run_manifest import write_run_manifest  # noqa: E402


def preprocess(text: str) -> str:
    """The authors' preprocessing for XLM-T: handles -> @user, URLs -> http, whitespace collapsed."""
    out = []
    for token in re.sub(r"\s+", " ", str(text)).split(" "):
        if token.startswith("@") and len(token) > 1:
            token = "@user"
        elif token.startswith("http"):
            token = "http"
        out.append(token)
    return " ".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--text-column", default="text_original")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    for attr in ("csv", "model_dir", "out_dir"):
        value = getattr(args, attr)
        if not value.is_absolute():
            setattr(args, attr, Path.cwd() / value)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).eval().to(args.device)
    frame = pd.read_csv(args.csv)
    texts = [preprocess(t) for t in frame[args.text_column].astype(str)]
    probs = np.zeros((len(texts), model.config.num_labels), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(texts), args.batch_size):
            batch = tokenizer(texts[start: start + args.batch_size], padding=True, truncation=True,
                              max_length=args.max_length, return_tensors="pt").to(args.device)
            probs[start: start + args.batch_size] = torch.softmax(model(**batch).logits.float(), dim=-1).cpu().numpy()
    positive = 1 if model.config.num_labels == 2 else int(np.argmax(model.config.num_labels))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": frame["id"], "prob_general": probs[:, positive]}).to_csv(args.out_dir / "xlmt_probabilities.csv", index=False)
    write_run_manifest(args.out_dir, script="scripts/external/score_xlmt_classifier.py", inputs={"csv": args.csv, "config": args.model_dir / "config.json"},
                       encoder_id=str(args.model_dir), extra={"id2label": {str(k): v for k, v in model.config.id2label.items()}, "positive_index": positive,
                                                              "n": int(len(texts)), "example_input": texts[0][:200]},
                       filename="xlmt.manifest.json")
    print("id2label", model.config.id2label, "| wrote", args.out_dir / "xlmt_probabilities.csv")


if __name__ == "__main__":
    main()
