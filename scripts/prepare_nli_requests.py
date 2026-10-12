#!/usr/bin/env python3
"""stage 6, stage 1 (CPU): choose NLI premises and hypotheses, label-free.

Premises: the top-K windows of every document under the carried configuration
(window-level PE soft conjunction scores, exactly as the carried document score
pools them). Hypotheses: the populist anchor sentences of the carried anchor set,
the `--per-channel` closest to their own channel axis (P and E).

Writes results/step6_nli/requests_<dataset>.parquet and hypotheses.json.
English holdout and EU requests require --final (built in the final step).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from populism_axis.contextual_axis import CHANNELS, WindowProjections, window_conjunction_scores  # noqa: E402
from populism_axis.run_manifest import write_run_manifest  # noqa: E402
from populism_axis.geometry import center_normalize, pole_mean_axes, unit_rows  # noqa: E402
from populism_axis.sentences import load_sentence_groups  # noqa: E402
from run_config import BENCH, FRAMES, SPLIT_PATH, load_anchor_groups, load_embeddings, select_documents  # noqa: E402

STEP = ROOT / "results" / "step6_nli"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carried", type=Path, default=ROOT / "results" / "step5_encoders" / "carried_config.json")
    parser.add_argument("--datasets", default="english,agora")
    parser.add_argument("--english-split", choices=("dev", "holdout"), default="dev")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--per-channel", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=STEP)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--dataset-dir", action="append", default=[],
                        help="embeddings override, e.g. eu=results/embeddings_eu/e5-large-v2/spw3_cap60")
    args = parser.parse_args()
    overrides = {k.strip(): Path(v.strip()) for k, _, v in (item.partition("=") for item in args.dataset_dir)}
    datasets = [d for d in args.datasets.split(",") if d]
    if (args.english_split == "holdout" or "eu" in datasets) and not args.final:
        raise SystemExit("holdout or EU requests require --final")

    carried = json.loads(args.carried.read_text(encoding="utf-8"))
    anchors_npz = ROOT / carried["anchors"]
    groups = load_anchor_groups(anchors_npz)
    axes = pole_mean_axes(groups)
    _, sentences = load_sentence_groups(ROOT / carried["anchor_file"])

    hypotheses = []
    for channel in ("P", "E"):
        cosines = unit_rows(groups[f"{channel}__populist"]) @ axes[channel]
        for index in np.argsort(-cosines)[: args.per_channel]:
            hypotheses.append({"hypothesis_id": f"{channel}{index:02d}", "channel": channel,
                               "text": sentences[f"{channel}__populist"][index], "cosine_to_axis": float(cosines[index])})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    hypotheses_path = args.out_dir / "hypotheses.json"
    if hypotheses_path.exists():
        existing = json.loads(hypotheses_path.read_text(encoding="utf-8"))
        if existing != hypotheses:
            raise SystemExit("hypotheses differ from the frozen step-6 set")
    hypotheses_path.write_text(json.dumps(hypotheses, indent=2) + "\n", encoding="utf-8")

    summary = {}
    for dataset in datasets:
        frame = pd.read_csv(BENCH / FRAMES[dataset])
        embeddings_dir = overrides.get(dataset) or ROOT / carried["embeddings_dirs"][dataset]
        if not embeddings_dir.is_absolute():
            embeddings_dir = ROOT / embeddings_dir
        matrix, doc_index, windows = load_embeddings(embeddings_dir, dataset)
        if dataset == "english":
            split = pd.read_csv(SPLIT_PATH).set_index("id")["split"]
            keep = frame["id"].map(split).eq(args.english_split).to_numpy()
            frame, matrix, doc_index, windows = select_documents(frame, matrix, doc_index, windows, keep)
        normalized, _ = center_normalize(matrix)
        projections = WindowProjections(doc_index=doc_index, windows=[],
                                        projections={c: normalized @ axes[c] for c in CHANNELS})
        scores = window_conjunction_scores(projections, {"P": 1.0, "E": 1.0, "M": 0.0}, conjunction="soft")
        id_column = "id" if dataset == "english" else "row_id"
        rows = []
        order = np.lexsort((-scores, doc_index))
        for doc in range(len(frame)):
            positions = order[np.searchsorted(doc_index[order], doc): np.searchsorted(doc_index[order], doc, side="right")]
            for rank, position in enumerate(positions[: args.top_k]):
                rows.append({"doc_index": doc, "doc_id": frame.loc[doc, id_column], "window_rank": rank,
                             "window_position": int(position), "window_score": float(scores[position]),
                             "premise": windows.loc[position, "window"]})
        requests = pd.DataFrame(rows)
        requests.insert(0, "request_id", np.arange(len(requests)))
        name = f"{dataset}_{args.english_split}" if dataset == "english" else dataset
        requests.to_parquet(args.out_dir / f"requests_{name}.parquet", index=False)
        summary[name] = {"documents": int(len(frame)), "requests": int(len(requests)),
                         "pairs": int(len(requests) * len(hypotheses)),
                         "premise_words_median": float(requests["premise"].str.split().str.len().median())}
        print(name, summary[name], flush=True)
    summary_name = "requests_summary.json" if not args.final else "requests_summary_final.json"
    (args.out_dir / summary_name).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_run_manifest(args.out_dir, script="scripts/prepare_nli_requests.py",
                       inputs={"carried": args.carried, "anchors": anchors_npz},
                       extra={"top_k": args.top_k, "per_channel": args.per_channel, "summary": summary},
                       filename="requests.manifest.json" if not args.final else "requests_final.manifest.json")


if __name__ == "__main__":
    main()
