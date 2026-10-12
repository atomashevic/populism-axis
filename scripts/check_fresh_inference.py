#!/usr/bin/env python3
"""Fresh-checkout inference check: re-score English-dev with the pinned models and
compare with the stored frozen-reference scores (CPU for prepare/compare; GPU for scoring).

The frozen reference statistics come from English-dev, so on English-dev the frozen
scores equal the transductive ones that the development log reports. Re-scoring
those 174 speeches with scripts/score_texts.py from a clean checkout therefore
tests the whole public path at once: text cleaning, windowing, the pinned e5 and
BART checkpoints, the stored reference assets and the combination. English-dev is
development data; the spent English holdout and EU-OOD are not read.

  prepare  writes <dir>/english_dev_input.csv (id, text: raw speech files in benchmark order)
  compare  reads <dir>/scored/{scores.csv,run_manifest.json}, writes <dir>/fresh_inference_check.json

Between the two, on the GPU host (see docs/fresh-inference-check.md):
  scripts/remote_job.sh fresh_check -- python scripts/score_texts.py --input <dir>/english_dev_input.csv \\
      --text-column text --id-column id --unit speech --reference frozen --clean \\
      --device cuda --nli-batch-size 64 --out-dir <dir>/scored

The acceptance thresholds below were set before any run. Exact equality is not
expected: the historical window embeddings and entailment probabilities were stored
as float16, the GPU differs, and a near-tie can swap a top-8 window.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from populism_axis.model_revisions import BART_LARGE_MNLI, E5_LARGE_V2, PINNED_REVISIONS  # noqa: E402
from populism_axis.run_manifest import git_commit, write_run_manifest  # noqa: E402

BENCH = ROOT / "data" / "benchmarks"
STORED = ROOT / "results" / "frozen_reference" / "scores_english_dev.csv"
REFERENCE_RESULTS = ROOT / "results" / "frozen_reference" / "results.json"
EXPECTED_DOCUMENTS = 174
EXPECTED_WINDOWS = 20284  # results/frozen_reference/runtime_manifest.json, reference.n_windows

# score_texts.py column -> stored frozen column in scores_english_dev.csv.
COLUMNS = {
    "axis": "PE_axis__frozen",
    "axis_P": "P_axis__frozen",
    "axis_E": "E_axis__frozen",
    "claim_check": "PE_nli__frozen",
    "claim_check_P": "P_nli__frozen",
    "claim_check_E": "E_nli__frozen",
    "populism_axis_plus_claim_check": "PE_axis_plus_nli__frozen",
}
AXIS_COLUMNS = ("axis", "axis_P", "axis_E")

# Pre-declared acceptance thresholds.
MIN_RANK_AGREEMENT_AXIS = 0.999      # Spearman, fresh vs stored, axis columns
MIN_RANK_AGREEMENT_CLAIM = 0.99      # claim-check and combined columns (top-window swaps move premises)
MAX_METRIC_SHIFT_AXIS = 0.005        # |rho(fresh, rubric) - stored| for the axis
MAX_METRIC_SHIFT_COMBINED = 0.010    # same, for axis + claim check
# Value tolerance, in SDs of the stored column, so a rescaled or shifted score cannot pass
# on rank agreement alone. Axis columns: every document. Claim-check and combined columns:
# at least 95% of documents, since one swapped top window can move a speech's mean
# entailment by many SDs.
MAX_DIFF_SD = 0.05
MIN_SHARE_WITHIN_CLAIM = 0.95


def english_dev() -> pd.DataFrame:
    frame = pd.read_csv(BENCH / "english_gpd_speeches.csv")
    split = pd.read_csv(BENCH / "english_split.csv").set_index("id")["split"]
    return frame[frame["id"].map(split).eq("dev").to_numpy()].reset_index(drop=True)


def read_speech(row) -> str:
    """Same path resolution and decoding as scripts/benchmarks.py::load_documents."""
    path = ROOT / "data" / str(row.path) if not str(row.path).startswith("data") else ROOT / str(row.path)
    if not path.exists():
        path = ROOT / "data" / "speeches_20251120" / str(row.merging_variable)
    return path.read_text(encoding="utf-8", errors="ignore")


def prepare(directory: Path) -> Path:
    frame = english_dev()
    if len(frame) != EXPECTED_DOCUMENTS:
        raise SystemExit(f"English-dev has {len(frame)} speeches, expected {EXPECTED_DOCUMENTS}")
    out = directory / "english_dev_input.csv"
    directory.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f"{out} exists; use a new directory")
    pd.DataFrame({"id": frame["id"], "text": [read_speech(row) for row in frame.itertuples(index=False)]}).to_csv(out, index=False)
    write_run_manifest(directory, script="scripts/check_fresh_inference.py", argv=["prepare", str(directory)],
                       inputs={"speeches": BENCH / "english_gpd_speeches.csv", "split": BENCH / "english_split.csv"},
                       extra={"step": "prepare", "n_documents": len(frame)}, filename="prepare.manifest.json")
    print(f"wrote {out} ({len(frame)} speeches)")
    return out


def rho(left, right) -> float:
    return float(spearmanr(left, right).statistic)


def compare(directory: Path) -> dict:
    scored_dir = directory / "scored"
    fresh = pd.read_csv(scored_dir / "scores.csv")
    manifest = json.loads((scored_dir / "run_manifest.json").read_text(encoding="utf-8"))
    stored = pd.read_csv(STORED)
    frame = english_dev()
    if list(fresh["id"].astype(str)) != list(frame["id"].astype(str)) or len(stored) != len(frame):
        raise SystemExit("fresh scores do not follow the English-dev benchmark order")
    extra = manifest["extra"]
    rubric = frame["source_score"].to_numpy(float)
    reference = json.loads(REFERENCE_RESULTS.read_text(encoding="utf-8"))["corpora"]["english_dev"]["evaluations"]
    stored_metric = {item["system"]: item["frozen"] for item in reference}

    checks: dict[str, dict] = {}

    def check(name: str, passed: bool, **detail) -> None:
        checks[name] = {"passed": bool(passed), **detail}

    # The scores must come from a clean tree at the commit this comparison runs from
    # (not a stale output copied in from another revision), and the comparison's own
    # tree must be clean too.
    here = git_commit()
    check("clean_checkout", manifest["git"].get("dirty") is False and here["dirty"] is False
          and manifest["git"].get("commit") is not None and manifest["git"].get("commit") == here["commit"],
          scored=manifest["git"], checkout=here)
    check("frozen_reference_mode", extra.get("reference") == "frozen" and extra.get("top_rule") == "k:8"
          and manifest.get("encoder_id") == E5_LARGE_V2 and extra.get("nli_model") == BART_LARGE_MNLI,
          reference=extra.get("reference"), top_rule=extra.get("top_rule"))
    resolved = extra.get("resolved_revisions", {})
    check("pinned_revisions",
          extra.get("encoder_revision") == PINNED_REVISIONS[E5_LARGE_V2] and extra.get("nli_revision") == PINNED_REVISIONS[BART_LARGE_MNLI]
          and resolved.get("encoder") in (None, PINNED_REVISIONS[E5_LARGE_V2])
          and resolved.get("nli_model") in (None, PINNED_REVISIONS[BART_LARGE_MNLI]),
          requested={"encoder": extra.get("encoder_revision"), "nli_model": extra.get("nli_revision")}, resolved=resolved)
    check("window_count", extra.get("n_windows") == EXPECTED_WINDOWS and extra.get("n_documents") == EXPECTED_DOCUMENTS,
          n_windows=extra.get("n_windows"), expected=EXPECTED_WINDOWS)

    columns = {}
    for new, old in COLUMNS.items():
        a, b = fresh[new].to_numpy(float), stored[old].to_numpy(float)
        agreement = rho(a, b)
        threshold = MIN_RANK_AGREEMENT_AXIS if new in AXIS_COLUMNS else MIN_RANK_AGREEMENT_CLAIM
        diff_sd = np.abs(a - b) / max(float(b.std()), 1e-12)
        within = float(np.mean(diff_sd <= MAX_DIFF_SD))
        needed = 1.0 if new in AXIS_COLUMNS else MIN_SHARE_WITHIN_CLAIM
        columns[new] = {"stored_column": old, "rank_agreement": agreement, "rank_threshold": threshold,
                        "max_abs_diff": float(np.max(np.abs(a - b))),
                        "max_rel_diff": float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12))),
                        "max_diff_sd": float(diff_sd.max()), "share_within_tolerance": within,
                        "tolerance_sd": MAX_DIFF_SD, "share_needed": needed,
                        "passed": bool(agreement >= threshold and within >= needed)}
    check("score_agreement", all(c["passed"] for c in columns.values()), columns=columns)

    for system, column, limit in (("PE_axis", "axis", MAX_METRIC_SHIFT_AXIS),
                                  ("PE_axis_plus_nli", "populism_axis_plus_claim_check", MAX_METRIC_SHIFT_COMBINED)):
        value = rho(fresh[column].to_numpy(float), rubric)
        check(f"dev_metric_{system}", abs(value - stored_metric[system]) <= limit,
              fresh=value, stored=stored_metric[system], max_shift=limit)

    report = {"passed": all(c["passed"] for c in checks.values()), "checks": checks,
              "scored_run_hash": manifest.get("run_hash"), "scored_commit": manifest["git"].get("commit"),
              "hostname": manifest.get("hostname"), "packages": manifest.get("packages")}
    out = directory / "fresh_inference_check.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_run_manifest(directory, script="scripts/check_fresh_inference.py", argv=["compare", str(directory)],
                       inputs={"fresh_scores": scored_dir / "scores.csv", "fresh_manifest": scored_dir / "run_manifest.json",
                               "stored_scores": STORED, "stored_results": REFERENCE_RESULTS},
                       extra={"step": "compare", "passed": report["passed"]}, filename="compare.manifest.json")
    for name, item in checks.items():
        print(f"{'PASS' if item['passed'] else 'FAIL'}  {name}")
    print(f"{'PASS' if report['passed'] else 'FAIL'}: wrote {out}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("prepare", "compare"))
    parser.add_argument("directory", type=Path, help="working directory for this check (outside results/)")
    args = parser.parse_args()
    directory = args.directory if args.directory.is_absolute() else Path.cwd() / args.directory
    if args.step == "prepare":
        prepare(directory)
        return 0
    return 0 if compare(directory)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
