#!/usr/bin/env python3
"""Refresh the RoBERTa entries of the stored result files with the corrected sigmoid comparator.

Comparator-only. The committed scoring code (scripts/final_holdout.py main, the `final` stage of
scripts/external/score_*.py) is executed unchanged in a temporary sandbox, twice:

  replay   with the legacy softmax caches (probabilities_*.npz) substituted for load_scores; every
           output file must reproduce the stored file byte for byte.
  sigmoid  with the corrected caches from the two rc-20261002 runs, read through the strict
           load_scores contract.

Every value that differs between the stored file and the sigmoid output must sit under a key,
column or table row naming RoBERTa; axis, claim-check, NLI, SetFit, XLM-T and supervised entries
must be identical. The staged files take the sigmoid values in place and keep the old ones:
JSON under `superseded_softmax_roberta` (keyed by path), CSV as `<column>__superseded_softmax`
columns, Markdown as a `(superseded softmax)` row after each corrected row.

No axis subprocess runs and nothing is encoded: final_holdout.run_axis is redirected to sandbox
copies of the stored B0/B4 run directories and log_access is disabled (the comparator re-evaluation
reads the spent holdout labels, as compare_roberta.py did, and makes no configuration choice).

  build (default)  write staged files, verification.json and refresh.manifest.json (clean tree)
  --apply          move the staged files over results/final and results/external after checking
                   that the targets still hold the bytes the build verified
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from populism_axis.roberta_comparator import load_scores  # noqa: E402
from populism_axis.run_manifest import file_sha256, write_run_manifest  # noqa: E402

FINAL = ROOT / "results" / "final"
EXT = ROOT / "results" / "external"
RUN_MAIN = ROOT / "results" / "roberta-correction" / "rc-20261002T1415Z-sigmoid"
RUN_SENT = ROOT / "results" / "roberta-correction" / "rc-20261002T1530Z-sigmoid-usgov-sentences"
STAGED = HERE / "staged"
SEED, MODEL_LABEL = 20260923, "bart-large-mnli"
NOTE = ("RoBERTa entries use the corrected comparator: independent sigmoid outputs of the multi-label checkpoint "
        "(coastalcph/roberta-large-ft-trump-populism@9b5badfa), same windows, pooling, splits and seeds. The softmax values "
        "first reported are kept under superseded_softmax_roberta. Axis and claim-check entries are unchanged.")

# Cache name -> (corrected run directory, legacy softmax cache).
CACHES = {
    "english": (RUN_MAIN, FINAL / "roberta"), "eu": (RUN_MAIN, FINAL / "roberta"),
    "usgov_speeches": (RUN_MAIN, EXT / "usgov" / "roberta"), "usgov_sentences": (RUN_SENT, EXT / "usgov" / "roberta"),
    "antielite_tweets": (RUN_MAIN, EXT / "antielite_tweets" / "roberta"),
    "popbert_en_train": (RUN_MAIN, EXT / "popbert_en" / "roberta"), "popbert_en_test": (RUN_MAIN, EXT / "popbert_en" / "roberta"),
    "portugal_en": (RUN_MAIN, EXT / "portugal" / "roberta"),
}
# Stage name -> (script, stored directory, refreshed output files, final-stage draws).
STAGES = {
    "final": ("scripts/final_holdout.py", FINAL, ("final_results.json", "eu_paper_protocol.json", "final_results.md"), 4000),
    "usgov": ("scripts/external/score_usgov.py", EXT / "usgov",
              ("final_results.json", "final_results.md", "final_scores_speeches.csv", "final_scores_sentences.csv"), 4000),
    "antielite_tweets": ("scripts/external/score_antielite_tweets.py", EXT / "antielite_tweets",
                         ("final_results.json", "final_results.md", "final_scores.csv"), 2000),
    "popbert_en": ("scripts/external/score_popbert_en.py", EXT / "popbert_en", ("final_results.json", "final_results.md", "final_scores.csv"), 2000),
    "portugal": ("scripts/external/score_portugal.py", EXT / "portugal", ("final_results.json", "final_results.md", "final_scores.csv"), 2000),
}
SUPERSEDED = " (superseded softmax)"


def is_roberta(text: str) -> bool:
    return "roberta" in text.lower()


def cache_name(path: Path) -> str:
    name = Path(path).name
    if not (name.startswith("sigmoid_") and name.endswith(".npz")):
        raise ValueError(f"unexpected RoBERTa cache request: {path}")
    return name[len("sigmoid_"):-len(".npz")]


def sigmoid_loader(path: Path) -> dict:
    run_dir, _ = CACHES[cache_name(path)]
    return load_scores(run_dir / Path(path).name)


def softmax_replay_loader(path: Path) -> dict:
    # Reproduction only: hands the legacy softmax probabilities to the unchanged code; nothing is written.
    name = cache_name(path)
    with np.load(CACHES[name][1] / f"probabilities_{name}.npz") as payload:
        return {"doc_index": payload["doc_index"], "probabilities": payload["probabilities"]}


def load_module(script: str):
    path = ROOT / script
    spec = importlib.util.spec_from_file_location(f"refresh_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_stage(stage: str, loader, sandbox: Path) -> Path:
    script, stored, outputs, draws = STAGES[stage]
    module = load_module(script)
    module.load_scores = loader
    module.write_run_manifest = lambda *a, **k: None  # sandbox manifests are discarded
    out = sandbox / stage
    out.mkdir(parents=True)
    if stage == "final":
        for label in ("B0_v3_config", "B4_axis"):
            shutil.copytree(stored / label, out / label)
        module.FINAL = out
        module.run_axis = lambda label, *_: out / label
        module.log_access = lambda entry: None
        argv, sys.argv = sys.argv, ["final_holdout.py", "--final", "--draws", str(draws), "--seed", str(SEED)]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                module.main()
        finally:
            sys.argv = argv
    else:
        for item in stored.iterdir():
            if item.is_file() and item.name not in outputs and item.name != "final.manifest.json":
                shutil.copy2(item, out / item.name)
        for sub in ("xlmt",):
            if (stored / sub).is_dir():
                shutil.copytree(stored / sub, out / sub)
        module.OUT = out
        with contextlib.redirect_stdout(io.StringIO()):
            module.stage_final(MODEL_LABEL, draws, SEED)
    return out


def json_leaves(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from json_leaves(item, path + (str(key),))
    elif isinstance(value, list) and any(isinstance(x, (dict, list)) for x in value):
        for i, item in enumerate(value):
            yield from json_leaves(item, path + (str(i),))
    else:
        yield path, value


def get_path(value, path):
    for key in path:
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def merge_json(old_text: str, new_text: str, sort_keys: bool) -> tuple[str, dict]:
    old, new = json.loads(old_text), json.loads(new_text)
    old_leaves, new_leaves = dict(json_leaves(old)), dict(json_leaves(new))
    if old_leaves.keys() != new_leaves.keys():
        raise SystemExit("sigmoid output changes the JSON structure")
    roots = set()
    for path, value in old_leaves.items():
        if value == new_leaves[path]:
            continue
        cut = next((i for i, key in enumerate(path) if is_roberta(key)), None)
        if cut is None:
            raise SystemExit(f"non-RoBERTa JSON value changed: {'/'.join(path)}")
        roots.add(path[:cut + 1])
    superseded = {"/" + "/".join(root): get_path(old, root) for root in sorted(roots)}
    merged = dict(new)
    merged["roberta_correction"] = {"note": NOTE, "corrected_runs": [str(RUN_MAIN.relative_to(ROOT)), str(RUN_SENT.relative_to(ROOT))],
                                    "refresh": str(HERE.relative_to(ROOT)), "date": "2026-10-02"}
    merged["superseded_softmax_roberta"] = superseded
    text = json.dumps(merged, indent=2, sort_keys=sort_keys) + "\n"
    changed = sum(1 for p, v in old_leaves.items() if v != new_leaves[p])
    return text, {"changed_leaves": changed, "superseded_subtrees": sorted(superseded)}


def merge_csv(old_text: str, new_text: str) -> tuple[str, dict]:
    old, new = list(csv.reader(io.StringIO(old_text))), list(csv.reader(io.StringIO(new_text)))
    if old[0] != new[0] or len(old) != len(new):
        raise SystemExit("sigmoid output changes the CSV layout")
    changed = [j for j, name in enumerate(old[0]) if any(o[j] != n[j] for o, n in zip(old[1:], new[1:]))]
    for j in changed:
        if not is_roberta(old[0][j]):
            raise SystemExit(f"non-RoBERTa CSV column changed: {old[0][j]}")
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(new[0] + [f"{old[0][j]}__superseded_softmax" for j in changed])
    for o, n in zip(old[1:], new[1:]):
        writer.writerow(n + [o[j] for j in changed])
    text = buffer.getvalue()
    # The merged file minus the appended columns must be the sigmoid output itself.
    check = io.StringIO()
    csv.writer(check, lineterminator="\n").writerows(row[:len(new[0])] for row in csv.reader(io.StringIO(text)))
    if check.getvalue() != new_text:
        raise SystemExit("CSV merge altered the corrected columns")
    return text, {"changed_columns": [old[0][j] for j in changed]}


def merge_md(old_text: str, new_text: str) -> tuple[str, dict]:
    old, new = old_text.rstrip("\n").split("\n"), new_text.rstrip("\n").split("\n")
    if len(old) != len(new):
        raise SystemExit("sigmoid output changes the Markdown layout")
    lines = ["> RoBERTa rows use the corrected comparator (independent sigmoid outputs; 2026-10-02, "
             "`results/roberta-correction/`). Rows marked *superseded softmax* are the values first reported, kept for the record. "
             "Axis and claim-check rows are unchanged.", ""]
    changed = 0
    for o, n in zip(old, new):
        lines.append(n)
        if o == n:
            continue
        cells = o.split("|")
        if not (o.startswith("| ") and is_roberta(cells[1])):
            raise SystemExit(f"non-RoBERTa Markdown line changed: {o}")
        cells[1] = cells[1].rstrip() + SUPERSEDED + " "
        lines.append("|".join(cells))
        changed += 1
    return "\n".join(lines) + "\n", {"changed_rows": changed}


def build() -> None:
    verification: dict = {"seed": SEED, "stages": {}}
    staged_hashes, stored_hashes = {}, {}
    # The sandbox sits under ROOT because final_holdout.py records paths relative to it.
    with tempfile.TemporaryDirectory(prefix=".sandbox-", dir=HERE) as tmp:
        tmp = Path(tmp)
        for stage, (script, stored, outputs, draws) in STAGES.items():
            replay = run_stage(stage, softmax_replay_loader, tmp / "replay")
            corrected = run_stage(stage, sigmoid_loader, tmp / "sigmoid")
            report = {"script": script, "draws": draws, "files": {}}
            for name in outputs:
                target = stored / name
                old_text = target.read_text(encoding="utf-8")
                if (replay / name).read_text(encoding="utf-8") != old_text:
                    raise SystemExit(f"softmax replay does not reproduce {target.relative_to(ROOT)}")
                new_text = (corrected / name).read_text(encoding="utf-8")
                if name.endswith(".json"):
                    text, detail = merge_json(old_text, new_text, sort_keys=name != "eu_paper_protocol.json")
                elif name.endswith(".csv"):
                    text, detail = merge_csv(old_text, new_text)
                else:
                    text, detail = merge_md(old_text, new_text)
                rel = str(target.relative_to(ROOT))
                (STAGED / rel).parent.mkdir(parents=True, exist_ok=True)
                (STAGED / rel).write_text(text, encoding="utf-8")
                stored_hashes[rel], staged_hashes[rel] = file_sha256(target), file_sha256(STAGED / rel)
                report["files"][rel] = {"softmax_replay_byte_identical": True, **detail}
            verification["stages"][stage] = report
    (HERE / "verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    inputs = {f"sigmoid_{n}": run / f"sigmoid_{n}.npz" for n, (run, _) in CACHES.items()}
    inputs |= {f"legacy_{n}": legacy / f"probabilities_{n}.npz" for n, (_, legacy) in CACHES.items()}
    inputs |= {f"stored::{rel}": ROOT / rel for rel in stored_hashes}
    write_run_manifest(HERE, script=str((HERE / "refresh_result_files.py").relative_to(ROOT)), inputs=inputs,
                       extra={"seed": SEED, "nli_model_label": MODEL_LABEL, "outputs_sha256": staged_hashes,
                              "verification": "verification.json"}, filename="refresh.manifest.json")
    for rel, payload in verification["stages"].items():
        for name, detail in payload["files"].items():
            print(name, {k: v for k, v in detail.items() if k != "superseded_subtrees"})


def apply() -> None:
    manifest = json.loads((HERE / "refresh.manifest.json").read_text(encoding="utf-8"))
    outputs = manifest["extra"]["outputs_sha256"]
    for rel, digest in outputs.items():
        if file_sha256(ROOT / rel) != manifest["inputs"][f"stored::{rel}"]["sha256"]:
            raise SystemExit(f"{rel} changed since the build; rebuild")
        if file_sha256(STAGED / rel) != digest:
            raise SystemExit(f"staged {rel} does not match the manifest")
    for rel in outputs:
        shutil.move(STAGED / rel, ROOT / rel)
        print("applied", rel)
    shutil.rmtree(STAGED)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true")
    apply() if parser.parse_args().apply else build()
