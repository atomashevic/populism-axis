#!/usr/bin/env python3
"""Build the Portuguese parliament populist-rhetoric external benchmark (CPU).

Source: Moniz, Brás, Santana Pereira, Belchior & Martins (2025), "Dataset on Parties' Use of
Populist Rhetoric in Portugal During the COVID-19 Pandemic (2020–2021)", Zenodo
doi:10.5281/zenodo.14626861 (CC BY 4.0); analysed in Moniz et al. (2024), Análise Social,
doi:10.31447/2022112. 6,214 paragraphs from 59 Assembleia da República debates (Oct 2019 –
Apr 2021) on populism-prone issues, hand-coded as binary indicators of people-centrism,
anti-elitism, popular sovereignty and exclusionism (Krippendorff's alpha .66 / .77 / .80 / .73
on a 10% double-coded sample). No model baseline is published for it.

Writes data/benchmark_portugal/paragraphs.csv (Portuguese text, tracked under CC BY) with
one row per paragraph in source order, and the freeze record under results/external/portugal/.
Text normalisation is label-free: hard line breaks inside paragraphs joined, whitespace collapsed.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from populism_axis.run_manifest import file_sha256, write_run_manifest  # noqa: E402

SRC = ROOT / "data" / "external" / "moniz_portugal" / "Dataset_Populist_Rhetoric_Portugal_Covid19.csv"
TRACKED = ROOT / "data" / "benchmark_portugal"
RESULTS = ROOT / "results" / "external" / "portugal"
PARTIES = {11: "BE", 12: "CDS-PP", 13: "CH", 14: "IL", 15: "L", 16: "PAN", 17: "PCP", 18: "PEV", 19: "PS", 110: "PSD"}
ISSUES = {1: "banking_finance", 2: "corruption", 3: "ethnic_minorities", 4: "immigration", 5: "law_order",
          6: "morality", 7: "political_system", 8: "welfare"}
DIMENSIONS = {"Pop1_PeopleCentrism": "centr", "Pop2_AntiElitism": "elite", "Pop3_PopularSovereignty": "sovereignty",
              "Pop4_Exclusionism": "exclusion"}
ALPHA = {"centr": 0.66, "elite": 0.77, "sovereignty": 0.80, "exclusion": 0.73}


def main() -> None:
    raw = pd.read_csv(SRC, sep=";", encoding="latin-1")
    raw.columns = [c.strip().replace(" ", "") for c in raw.columns]
    frame = pd.DataFrame({"id": [f"pt{i:04d}" for i in range(len(raw))]})
    frame["date"] = pd.to_datetime(raw["A_Date_Claim"], format="%d/%m/%Y").dt.strftime("%Y-%m-%d")
    frame["actor"] = raw["C_Name_Actor"].astype(str).str.strip()
    frame["government_actor"] = raw["D_Actor_Gov"].astype(int)
    frame["party"] = raw["E_Party_Name"].map(lambda v: PARTIES.get(int(v)) if pd.notna(v) else "unknown")
    frame["issue"] = raw["Pop5_Issue"].map(lambda v: ISSUES.get(int(v)) if pd.notna(v) else "unknown")
    for source, target in DIMENSIONS.items():
        values = raw[source].astype(int)
        if not set(values.unique()) <= {0, 1}:
            raise SystemExit(f"{source} is not binary: {sorted(values.unique())}")
        frame[f"{target}_gold"] = values
    frame["text_pt"] = raw["B_Claim"].astype(str).str.replace(r"\s*\n\s*", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    frame["words_pt"] = frame["text_pt"].str.split().str.len()
    frame["is_duplicate_text"] = frame["text_pt"].duplicated().astype(int)
    if (frame["words_pt"] == 0).any():
        raise SystemExit("empty paragraph")
    TRACKED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = TRACKED / "paragraphs.csv"
    frame.to_csv(out, index=False)

    freeze = {
        "schema": "populism_axis.v4.external_benchmark_freeze.v1",
        "freeze_id": "pa_v4_external_portugal_r1",
        "source": {
            "dataset": "Moniz et al. (2025), Zenodo doi:10.5281/zenodo.14626861, CC BY 4.0",
            "paper": "Moniz, Brás, Santana Pereira & Belchior (2024), Análise Social, doi:10.31447/2022112",
            "file": SRC.name, "sha256": file_sha256(SRC), "language": "pt", "unit": "paragraph",
        },
        "n_paragraphs": int(len(frame)), "n_actors": int(frame["actor"].nunique()), "n_sessions": int(frame["date"].nunique()),
        "date_range": [frame["date"].min(), frame["date"].max()],
        "positives": {t: int(frame[f"{t}_gold"].sum()) for t in DIMENSIONS.values()},
        "krippendorff_alpha_published": ALPHA,
        "party_counts": frame["party"].value_counts().to_dict(),
        "words_pt_quantiles": {q: float(frame["words_pt"].quantile(float(q))) for q in ("0.05", "0.5", "0.95")},
        "n_duplicate_texts": int(frame["is_duplicate_text"].sum()),
        "published_model_baseline": None,
        "channel_mapping": {"centr": "P", "elite": "E"},
        "note": "The codebook lists 2/1 and 1/2 codings for anti-elitism and popular sovereignty; the released file uses 0/1 for all four dimensions.",
        "output_sha256": file_sha256(out),
    }
    (RESULTS / "benchmark_freeze.json").write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_run_manifest(RESULTS, script="scripts/external/build_portugal_benchmark.py", inputs={"source": SRC},
                       extra={"n_paragraphs": len(frame)}, filename="build.manifest.json")
    print(json.dumps({k: freeze[k] for k in ("n_paragraphs", "n_actors", "n_sessions", "positives", "n_duplicate_texts")}))


if __name__ == "__main__":
    main()
