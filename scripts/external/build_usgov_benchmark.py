#!/usr/bin/env python3
"""Build the US governors / 2016 presidential candidates external benchmark (CPU).

Source: replication package of van der Veen, Dzebo, Littvay, Hawkins & Dar (2024),
"Classifying Populist Language in American Presidential and Governor Speeches Using
Automatic Text Analysis", Harvard Dataverse doi:10.7910/DVN/ZOHMVO (CC0), unpacked
under data/external/usgov/. Speeches carry holistic 0-2 populism scores in the GPD
style; sentences carry populist / pluralist / neutral labels drawn from coder rubrics.

Speech texts are re-extracted from the original .docx files with paragraph breaks
(the package CSV joins paragraphs with spaces), then passed through the text
cleaner (metadata lines, URLs) plus a small dataset-specific rule that drops leading
title/byline paragraphs of the presidential transcripts. When a .docx cannot be
parsed, the package CSV text is used and flagged. The package's own per-speech
classifier outputs are kept as the published comparator.

Outputs (tracked): data/benchmark_usgov/speeches.csv, sentences.csv,
comparator_predictions.csv, and results/external/usgov/benchmark_freeze.json.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from populism_axis.run_manifest import file_sha256, write_run_manifest  # noqa: E402
from populism_axis.text import CLEANING_VERSION, clean_speech_text  # noqa: E402

PKG = ROOT / "data" / "external" / "usgov" / "Code + Output 19.2.26"
IMPORT = PKG / "Importing"
OUT_CSV = IMPORT / "Output"
ANALYSE = PKG / "Analysing"
DATA_OUT = ROOT / "data" / "benchmark_usgov"
RESULTS_OUT = ROOT / "results" / "external" / "usgov"
BINARY_CUT = 0.499  # the paper's make_binary_score threshold

# Dataset-specific metadata keys seen at the top of governor files, beyond the standard list.
EXTRA_META = re.compile(
    r"^\s*(state|term|year|date of speech|type of speech|name of speaker|speaker|words|word counts?|source|"
    r"transcribed from|note)\s*(--|:)", re.I
)
TITLE_HINT = re.compile(
    r"https?://|www\.|@\w+|\b(transcript|transcribed|remarks|speech|announcement|address|rally|victory|concession|"
    r"caucus|primary|read |full text|annotated|touts|attacking|journal|press release|inaugural|inauguration|keynote|"
    r"summit|prepared text|highlights|part \d)\b", re.I
)
# A short line that reads as a sentence (ends with punctuation, 4+ words) is speech, not a title, unless it carries a URL.
SENTENCE_LIKE = re.compile(r"[.!?:]\s*[\"”’']?$")
HARD_TITLE = re.compile(r"https?://|www\.|@\w+|prepared text|transcri", re.I)
LEADING_URLS = re.compile(r"^\s*(?:(?:https?://|www\.)\S+\s*)+", re.I)
DATE_HINT = re.compile(r"\b(20\d\d|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", re.I)


def docx_paragraphs(path: Path) -> list[str] | None:
    try:
        archive = zipfile.ZipFile(path)
        names = [n for n in archive.namelist() if re.match(r"word/document\d*\.xml$", n)]
        if not names:
            return None
        xml = archive.read(names[0]).decode("utf-8", errors="ignore")
    except (zipfile.BadZipFile, KeyError, OSError):
        return None
    paragraphs = []
    for block in re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S):
        block = re.sub(r"<w:tab/>", " ", block)
        text = html.unescape("".join(re.findall(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", block, flags=re.S)))
        if text.strip():
            paragraphs.append(text.strip())
    return paragraphs or None


def resolve_path(path: Path) -> Path | None:
    """Exact path, else a sibling whose normalized stem matches (the package renamed a few files)."""
    if path.exists():
        return path
    if not path.parent.exists():
        return None
    key = re.sub(r"[^a-z0-9]", "", path.stem.lower())
    for candidate in path.parent.iterdir():
        if candidate.is_file() and re.sub(r"[^a-z0-9]", "", candidate.stem.lower()) == key:
            return candidate
    return None


def strip_extra_headers(lines: list[str], subset: str) -> tuple[list[str], list[str]]:
    """Dataset-specific: metadata keys in the first 8 lines; title/byline lead-ins for presidential files."""
    removed: list[str] = []
    out = list(lines)
    # metadata keys (governor files: "State: Michigan", "Term: 1", "Words: 447", "Note -- ...")
    kept = []
    for index, line in enumerate(out):
        if index < 8 and len(line) <= 400 and EXTRA_META.match(line):
            removed.append(line)
            continue
        kept.append(line)
    out = kept
    # Leading short paragraphs that look like a title, byline, URL, journal heading or date, before the first
    # paragraph of at least 30 words. Applied to every subset; ordinary openings ("Thank you.") do not match.
    while out and len(out[0].split()) < 30 and (TITLE_HINT.search(out[0]) or DATE_HINT.search(out[0])):
        head = out[0]
        if SENTENCE_LIKE.search(head) and len(head.split()) >= 4 and not HARD_TITLE.search(head):
            break
        removed.append(out.pop(0))
    # Package-CSV fallback texts have no paragraph breaks: strip leading URL tokens only.
    if out:
        stripped = LEADING_URLS.sub("", out[0])
        if stripped != out[0]:
            removed.append(out[0][: len(out[0]) - len(stripped)].strip())
            out[0] = stripped.strip()
    return out, removed


def build_speeches() -> tuple[pd.DataFrame, list[dict]]:
    rows: list[dict] = []
    audit: list[dict] = []
    specs = [
        ("governors_2010_2018", OUT_CSV / "gov_2010_2018.csv", "full path"),
        ("governors_2018_2022", OUT_CSV / "gov_2018_2022.csv", "full path"),
        ("presidential_2016", OUT_CSV / "pres_data.csv", "full path"),
    ]
    for subset, csv_path, path_column in specs:
        frame = pd.read_csv(csv_path)
        for position, row in frame.iterrows():
            relative = str(row[path_column]).replace("\\", "/")
            docx = resolve_path(IMPORT / relative)
            paragraphs = docx_paragraphs(docx) if docx is not None else None
            source = "docx"
            if paragraphs is None:
                source = "package_csv"
                paragraphs = [p for p in str(row["text"]).split("\n") if p.strip()]
            raw_text = "\n".join(paragraphs)
            cleaned, log = clean_speech_text(raw_text)
            lines, extra_removed = strip_extra_headers([l for l in cleaned.split("\n") if l.strip()], subset)
            text = "\n".join(lines)
            if subset == "governors_2010_2018":
                speaker = f"{row['state']}|term{int(row['term num'])}"
                group = row["state + term"]
                speech_type = str(row["speech type"])
                state = row["state"]
            elif subset == "governors_2018_2022":
                speaker = str(row["governor"])
                group = str(row["governor"])
                speech_type = str(row["speech type"])
                state = row["state"]
            else:
                speaker = str(row["speaker"])
                group = str(row["speaker"])
                speech_type = "campaign"
                state = "US"
            speech_id = f"{subset}__{position:03d}"
            rows.append({
                "id": speech_id, "subset": subset, "file_name": row["file name"], "speaker": speaker, "group": group,
                "state": state, "speech_type": speech_type, "score": float(row["score"]),
                "binary_populist": int(float(row["score"]) >= BINARY_CUT), "n_words": len(text.split()), "text": text,
                "text_source": source,
            })
            audit.append({
                "id": speech_id, "file_name": row["file name"], "source": source, "n_paragraphs": len(paragraphs),
                "removed_v4": log.removed_metadata + log.removed_url, "removed_extra": extra_removed,
                "words_raw": len(raw_text.split()), "words_clean": len(text.split()),
                "first_line": text.split("\n")[0][:120] if text else "",
            })
    return pd.DataFrame(rows), audit


def build_sentences() -> pd.DataFrame:
    gov = pd.read_csv(OUT_CSV / "sentences_2010_2018.csv")
    gov = pd.DataFrame({"subset": "governors_2010_2018", "text": gov["text"].astype(str).str.strip(), "label": gov["label"].str.lower()})
    pres = pd.read_csv(OUT_CSV / "pres_sentences.csv")
    pres = pd.DataFrame({"subset": "presidential_2016", "text": pres["text"].astype(str).str.strip(), "label": pres["type"].str.lower(),
                         "speaker": pres["speaker"].astype(str)})
    out = pd.concat([gov, pres], ignore_index=True)
    out = out[out["text"].str.split().str.len() >= 2].reset_index(drop=True)
    out.insert(0, "id", [f"sent_{i:05d}" for i in range(len(out))])
    return out


def _text_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()[:1500]


def build_comparator() -> pd.DataFrame:
    """The package's released per-speech classifier outputs (share of sentences predicted populist).

    The analysis CSVs come from an earlier import with a different row order and some
    different file names, so rows are matched on normalized speech text (first 1,500
    characters). Speeches whose text appears twice in the source get the same share.
    """
    parts = []
    specs = [
        ("governors_2010_2018", OUT_CSV / "gov_2010_2018.csv",
         [ANALYSE / "Speech data 2.csv", ANALYSE / "Speech data 3.csv"], ["pop prop"]),
        ("governors_2018_2022", OUT_CSV / "gov_2018_2022.csv", [ANALYSE / "gov_2018_2022 30x.csv"], None),
        ("presidential_2016", OUT_CSV / "pres_data.csv", [ANALYSE / "finetuned presidential data 27-5.csv"], ["pop prop"]),
    ]
    for subset, source_path, analysis_paths, columns in specs:
        source = pd.read_csv(source_path)
        lookup: dict[str, list[float]] = {}
        for path in analysis_paths:
            frame = pd.read_csv(path)
            cols = columns or [c for c in frame.columns if re.fullmatch(r"pop prop\d+", c)]
            for text, *values in frame[["text", *cols]].itertuples(index=False):
                lookup.setdefault(_text_key(text), []).extend(float(v) for v in values)
        rows = []
        for position, text in enumerate(source["text"]):
            values = lookup.get(_text_key(text))
            rows.append({"id": f"{subset}__{position:03d}", "subset": subset, "file_name": source.loc[position, "file name"],
                         "setfit_pop_share": float(np.mean(values)) if values else np.nan, "n_runs": len(values) if values else 0})
        matched = sum(1 for r in rows if r["n_runs"])
        print(f"comparator {subset}: matched {matched} of {len(rows)} speeches by text", flush=True)
        parts.append(pd.DataFrame(rows))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    speeches, audit = build_speeches()
    sentences = build_sentences()
    comparator = build_comparator()
    merged = speeches.merge(comparator.drop(columns=["subset", "file_name"]), on="id", how="left", validate="one_to_one")
    print("speeches:", speeches.groupby("subset")["id"].count().to_dict(), "| text from csv fallback:",
          int((speeches["text_source"] == "package_csv").sum()), "| comparator matched:", int(merged["setfit_pop_share"].notna().sum()))
    print("sentences:", sentences.groupby(["subset", "label"])["id"].count().to_dict())
    n_removed = sum(len(a["removed_v4"]) + len(a["removed_extra"]) for a in audit)
    print(f"removed lines: {n_removed}; words {sum(a['words_raw'] for a in audit)} -> {sum(a['words_clean'] for a in audit)}")
    if args.audit_only:
        for a in audit:
            if a["source"] != "docx" or a["removed_extra"] or re.search(r"http|@|Transcript|Remarks|Note --|Words:|Term:|State:", a["first_line"]):
                print(f"  {a['id']} [{a['source']}] extra={a['removed_extra'][:3]} first={a['first_line']!r}")
        return

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    RESULTS_OUT.mkdir(parents=True, exist_ok=True)
    speeches.to_csv(DATA_OUT / "speeches.csv", index=False)
    sentences.to_csv(DATA_OUT / "sentences.csv", index=False)
    merged[["id", "subset", "file_name", "setfit_pop_share", "n_runs"]].to_csv(DATA_OUT / "comparator_predictions.csv", index=False)
    (RESULTS_OUT / "cleaning_audit.json").write_text(json.dumps(audit, indent=1) + "\n", encoding="utf-8")
    freeze = {
        "schema": "populism_axis.v4.external_benchmark_freeze.v1",
        "freeze_id": "pa_v4_external_usgov_r1",
        "source": {"paper": "van der Veen, Dzebo, Littvay, Hawkins & Dar (2024), arXiv:2408.15213",
                   "dataverse": "doi:10.7910/DVN/ZOHMVO", "license": "CC0-1.0", "package_file_id": 13676815},
        "cleaning": {"v4": CLEANING_VERSION, "extra": "dataset-specific metadata keys in first 8 lines; presidential title/byline lead-ins"},
        "binary_cut": BINARY_CUT,
        "subsets": {
            subset: {
                "n_speeches": int(len(part)), "n_groups": int(part["group"].nunique()),
                "n_binary_populist": int(part["binary_populist"].sum()),
                "score_mean": float(part["score"].mean()), "median_words": float(part["n_words"].median()),
                "text_from_csv_fallback": int((part["text_source"] == "package_csv").sum()),
            } for subset, part in speeches.groupby("subset")
        },
        "sentences": {f"{s}|{l}": int(n) for (s, l), n in sentences.groupby(["subset", "label"])["id"].count().items()},
        "files": {name: file_sha256(DATA_OUT / name) for name in ("speeches.csv", "sentences.csv", "comparator_predictions.csv")},
        "published_results": {
            "speech_level_table2": {
                "governors_2010_2018": {"n": 288, "accuracy": .85, "precision": .80, "recall": .60, "f1": .69, "auroc_as_reported": .77, "mcc": .70, "r2_share_vs_score": .39},
                "governors_2018_2022": {"n": 26, "accuracy": .92, "precision": 1.0, "recall": .82, "f1": .90, "auroc_as_reported": .91, "mcc": .85, "r2_share_vs_score": .49},
                "presidential_2016": {"n": 45, "accuracy": .89, "precision": .85, "recall": .89, "f1": .87, "auroc_as_reported": .89, "mcc": .77, "r2_share_vs_score": .58},
            },
            "sentence_level_table1_setfit_75_25_split": {
                "governors": {"accuracy": .66, "precision": .73, "recall": .69, "f1": .71, "mcc": .50},
                "presidential": {"accuracy": .70, "precision": .77, "recall": .74, "f1": .75, "mcc": .55},
            },
            "notes": [
                "Speech-level metrics use the share of sentences a SetFit classifier labels populist, thresholded at the F1-optimal cutoff chosen on the same data (in-sample).",
                "The paper's 'AuROC' equals the balanced accuracy of the thresholded prediction ((recall + specificity)/2), not a ranking AUROC; for governors 2010-2018 the released continuous shares give a ranking AUROC of about .84.",
                "One presidential speech (us.2016.Trump.6-22) is identical to GPD US_Trump_Campaign_1, which is in the English holdout (GPD score .85).",
                "The fine-tuned RoBERTa comparator (coastalcph) was trained on Trump 2016 campaign sentences, so it is in-domain for the presidential subset.",
            ],
        },
    }
    freeze["canonical_sha256"] = hashlib.sha256(json.dumps(freeze, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (RESULTS_OUT / "benchmark_freeze.json").write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_run_manifest(RESULTS_OUT, script="scripts/external/build_usgov_benchmark.py",
                       inputs={"gov_2010_2018": OUT_CSV / "gov_2010_2018.csv", "gov_2018_2022": OUT_CSV / "gov_2018_2022.csv",
                               "pres_data": OUT_CSV / "pres_data.csv", "sentences": OUT_CSV / "sentences_2010_2018.csv"},
                       extra={"freeze_id": freeze["freeze_id"]}, filename="build.manifest.json")
    print(json.dumps(freeze["subsets"], indent=2))
    print(f"wrote {DATA_OUT} and {RESULTS_OUT / 'benchmark_freeze.json'}")


if __name__ == "__main__":
    main()
