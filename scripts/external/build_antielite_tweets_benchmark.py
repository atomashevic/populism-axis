#!/usr/bin/env python3
"""Build the party anti-elitism tweets external benchmark (CPU).

Source: replication package of Licht, Abou-Chadi, Barberá & Hua (2025), "Measuring and
understanding parties' anti-elite strategies", Journal of Politics, Harvard Dataverse
doi:10.7910/DVN/T0UKCN (CC0). 5,401 crowd-coded party tweets from 20 countries
(2008–2021) with Dawid-Skene labels `no`, `yes-general`, `yes-specific` (the paper drops
`yes-unsure`), a fixed test split (`test_`, n=1,059), the original text and an English
machine translation (`text_en`).

Tweet texts are kept out of git (Twitter terms; the package itself distributes them):
  tracked   data/benchmark_antielite_tweets/tweets_labels.csv  (ids, country, party, split, labels)
  local     data/external/licht_antielite/tweets_text.csv       (adds text_original, text_en, text_en_clean)
Cleaning of `text_en` for the English encoder, label-free and mirroring the paper's own
preprocessing for XLM-T: URLs removed, @handles replaced by "@user", whitespace collapsed.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from populism_axis.run_manifest import file_sha256, write_run_manifest  # noqa: E402

SRC = ROOT / "data" / "external" / "licht_antielite"
TRACKED = ROOT / "data" / "benchmark_antielite_tweets"
RESULTS = ROOT / "results" / "external" / "antielite_tweets"
URL = re.compile(r"(?:https?://|www\.)\S+")
HANDLE = re.compile(r"@\w+")


def clean_tweet(text: str) -> str:
    text = URL.sub(" ", str(text))
    text = HANDLE.sub("@user", text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    training = pd.read_csv(SRC / "training_data_pooled_samples.tab")
    labels = pd.read_csv(SRC / "dawidskene_labelings_pooled_samples.tab")
    frame = training.merge(labels[["item_id", "country_iso3c", "party_id", "party_name_short", "n_judgments", "judgment_entropy"]],
                           on="item_id", how="left", validate="many_to_one")
    if frame["country_iso3c"].isna().any():
        raise SystemExit("some training tweets have no label metadata")
    # The shipped training file repeats 48 rows verbatim (44 item_ids, 4 of them in the test split); the
    # paper's test metrics were computed on the 1,059 rows as shipped, so rows are kept and the count recorded.
    n_duplicate_rows = int(frame.duplicated(subset=["item_id"]).sum())
    frame["is_duplicate_row"] = frame.duplicated(subset=["item_id"]).astype(int)
    frame["split"] = frame["test_"].map({True: "test", False: "train"})
    frame["y_general"] = (frame["labeling"] == "yes-general").astype(int)
    frame["y_any"] = frame["labeling"].isin(["yes-general", "yes-specific"]).astype(int)
    frame["party"] = frame["country_iso3c"] + "|" + frame["party_name_short"].astype(str)
    frame["text_original"] = frame["text"].astype(str)
    frame["text_en"] = frame["text_en"].astype(str)
    frame["text_en_clean"] = frame["text_en"].map(clean_tweet)
    frame["is_english_source"] = (frame["text"] == frame["text_en"]).astype(int)
    frame = frame.rename(columns={"item_id": "id", "country_iso3c": "country"})
    frame = frame.sort_values(["split", "id"]).reset_index(drop=True)

    tracked_cols = ["id", "status_id", "country", "party", "party_id", "split", "labeling", "y_general", "y_any",
                    "is_english_source", "is_duplicate_row", "n_judgments", "judgment_entropy"]
    frame.insert(0, "row_id", [f"tw_{i:05d}" for i in range(len(frame))])
    tracked_cols = ["row_id"] + tracked_cols
    TRACKED.mkdir(parents=True, exist_ok=True)
    frame[tracked_cols].to_csv(TRACKED / "tweets_labels.csv", index=False)
    frame[tracked_cols + ["text_original", "text_en", "text_en_clean"]].to_csv(SRC / "tweets_text.csv", index=False)

    xlmt = pd.read_csv(SRC / "xlmt_test_results.tab", sep="\t").set_index("Unnamed: 0")
    best = pd.read_csv(SRC / "best_classifier_results.tab", sep="\t")
    published = {"xlmt_test": {row: {c: float(xlmt.loc[row, c]) for c in ("precision", "recall", "f1-score")} for row in ("neg", "pos", "macro avg")},
                 "other_best_test": {f"{r.approach}/{r.learner}/{r.what}": {"precision": float(r.precision), "recall": float(r.recall), "f1": float(r._7)}
                                     for r in best.itertuples()}}
    summary = {
        "schema": "populism_axis.v4.external_benchmark_freeze.v1",
        "freeze_id": "pa_v4_external_antielite_tweets_r1",
        "source": {"paper": "Licht, Abou-Chadi, Barberá & Hua (2025), Journal of Politics, doi:10.1086/730711",
                   "dataverse": "doi:10.7910/DVN/T0UKCN", "license": "CC0-1.0",
                   "files": ["training_data_pooled_samples.tab", "dawidskene_labelings_pooled_samples.tab", "xlmt_test_results.tab", "best_classifier_results.tab"]},
        "positive_class": "yes-general (general elite criticism), as in the paper; yes-specific and no are negatives. y_any = yes-general or yes-specific.",
        "text_used": "text_en (the package's English machine translation), URLs removed, @handles -> @user",
        "counts": {split: {"n": int(len(part)), "y_general": int(part["y_general"].sum()), "y_any": int(part["y_any"].sum()),
                           "countries": int(part["country"].nunique()), "parties": int(part["party"].nunique()),
                           "english_source_share": float(part["is_english_source"].mean())}
                   for split, part in frame.groupby("split")},
        "duplicate_rows_kept": n_duplicate_rows,
        "labels_file_sha256": file_sha256(TRACKED / "tweets_labels.csv"),
        "text_file_sha256_local_only": file_sha256(SRC / "tweets_text.csv"),
        "published_results": published,
        "notes": [
            "The paper's classifier is XLM-T (cardiffnlp/twitter-xlm-roberta-base) fine-tuned on the original-language text of the 4,342 training tweets; its test metrics are on the fixed 1,059-tweet test split.",
            "The fine-tuned classifier is released in the package (data/fits/xlmt-elitecriticism-classifier) and is re-run here on the same test tweets for per-tweet comparison.",
            "The axis is scored on the English translations; 28.5% of tweets are from English-language countries and untranslated.",
        ],
    }
    summary["canonical_sha256"] = hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "benchmark_freeze.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_run_manifest(RESULTS, script="scripts/external/build_antielite_tweets_benchmark.py",
                       inputs={"training": SRC / "training_data_pooled_samples.tab", "labels": SRC / "dawidskene_labelings_pooled_samples.tab"},
                       extra={"freeze_id": summary["freeze_id"]}, filename="build.manifest.json")
    print(json.dumps(summary["counts"], indent=2))
    print("example cleaned:", frame.loc[0, "text_en_clean"][:160])


if __name__ == "__main__":
    main()
