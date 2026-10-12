#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and summarize the Hugging Face dataset coastalcph/populism-eu-speeches.",
    )
    parser.add_argument(
        "--repo-id",
        default="coastalcph/populism-eu-speeches",
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("data/huggingface/coastalcph_populism_eu_speeches"),
        help="Directory where the dataset snapshot should be stored.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("results/populism_eu_speeches_hf_summary.md"),
        help="Markdown summary output path.",
    )
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=Path("data/huggingface/coastalcph_populism_eu_speeches/train.parquet"),
        help="Optional parquet export path.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    args.local_dir.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.parquet_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot_path = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=str(args.local_dir),
        )
    )

    train_csv = snapshot_path / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Expected train.csv under {snapshot_path}")

    frame = pd.read_csv(train_csv)
    frame.to_parquet(args.parquet_path, index=False)

    pop_counts = frame["pop_code"].value_counts(dropna=False).sort_index()
    speaker_counts = frame["speaker"].value_counts(dropna=False).sort_index()
    speech_counts = frame["speech_id"].value_counts(dropna=False).sort_index()

    lines = [
        "# Hugging Face Dataset: coastalcph/populism-eu-speeches",
        "",
        f"- Repo: `https://huggingface.co/datasets/{args.repo_id}`",
        f"- Local snapshot: `{snapshot_path.resolve()}`",
        f"- CSV: `{train_csv.resolve()}`",
        f"- Parquet export: `{args.parquet_path.resolve()}`",
        f"- Rows: `{len(frame)}`",
        f"- Columns: `{len(frame.columns)}`",
        f"- SHA256(train.csv): `{sha256(train_csv)}`",
        "",
        "## Columns",
    ]
    lines.extend(f"- `{column}`: `{str(dtype)}`" for column, dtype in frame.dtypes.items())
    lines.extend(
        [
            "",
            "## pop_code Distribution",
        ]
    )
    lines.extend(f"- `{int(code)}`: `{int(count)}`" for code, count in pop_counts.items())
    lines.extend(
        [
            "",
            "## Speakers",
        ]
    )
    lines.extend(f"- `{speaker}`: `{int(count)}` sentences" for speaker, count in speaker_counts.items())
    lines.extend(
        [
            "",
            "## Speech IDs",
        ]
    )
    lines.extend(f"- `{speech_id}`: `{int(count)}` sentences" for speech_id, count in speech_counts.items())
    sample = frame.head(5).fillna("").astype(str)
    header = "| " + " | ".join(sample.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(sample.columns)) + " |"
    sample_rows = [
        "| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |"
        for row in sample.itertuples(index=False, name=None)
    ]
    lines.extend(
        [
            "",
            "## Sample Rows",
            "",
            header,
            divider,
            *sample_rows,
            "",
        ]
    )

    args.summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Downloaded snapshot to {snapshot_path}")
    print(f"Wrote summary to {args.summary_path}")
    print(f"Wrote parquet to {args.parquet_path}")


if __name__ == "__main__":
    main()
