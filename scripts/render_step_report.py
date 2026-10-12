#!/usr/bin/env python3
"""Render the generated sections of a stage report from JSON (CPU).

Prints Markdown for "Results on development sets", "Paired difference vs
previous step" and "Provenance", so numbers in reports are never hand-typed.

  python scripts/render_step_report.py \
      --row "B0 uncapped splitter=results/step1_splitter/B0_v3_splitter" \
      --row "B1 cap 60=results/step1_splitter/B1_cap60" \
      --paired results/step1_splitter/paired_vs_previous.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(value, signed: bool = False) -> str:
    if value is None:
        return "–"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def _ci(pair) -> str:
    return f"[{pair[0]:+.3f}, {pair[1]:+.3f}]"


def results_table(rows: list[tuple[str, Path]], system: str) -> str:
    lines = [
        f"System `{system}`. English = development split; CIs are leader-clustered (English) and politician-clustered (Agora), 95%.",
        "",
        "| Configuration | English ρ | English 95% CI | Extreme AUROC | Leader-term ρ | Agora ρ | Agora 95% CI | Agora high-vs-low AUROC |",
        "|---|---:|---|---:|---:|---:|---|---:|",
    ]
    for label, directory in rows:
        evaluation = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
        payload = evaluation["systems"][system]
        e = payload.get("english", {})
        a = payload.get("agora", {})
        lines.append(
            f"| {label} | {_fmt(e.get('spearman'))} | {_ci(e['spearman_ci95']) if e else '–'} | "
            f"{_fmt(e.get('extreme_auroc'))} | {_fmt(e.get('leader_term_spearman'))} | "
            f"{_fmt(a.get('spearman'))} | {_ci(a['spearman_ci95']) if a else '–'} | {_fmt(a.get('high_vs_low_auroc'))} |"
        )
    return "\n".join(lines)


def all_systems_table(label: str, directory: Path) -> str:
    evaluation = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
    lines = [
        f"All systems for {label} (diagnostic; only the carried system decides):",
        "",
        "| System | English ρ | Agora ρ |",
        "|---|---:|---:|",
    ]
    for system, payload in sorted(evaluation["systems"].items()):
        lines.append(
            f"| `{system}` | {_fmt(payload.get('english', {}).get('spearman'))} | "
            f"{_fmt(payload.get('agora', {}).get('spearman'))} |"
        )
    return "\n".join(lines)


def paired_table(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    lines = [
        f"Candidate `{payload['candidate_dir']}` minus baseline `{payload['baseline_dir']}`, "
        f"system `{payload['system']}`, {payload['draws']} paired cluster-bootstrap draws, seed {payload['seed']}.",
        "",
        "| Dataset | Δρ | 95% CI | p | Documents | Clusters |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for dataset, row in payload["paired"].items():
        lines.append(
            f"| {dataset} ({row['cluster']}-clustered) | {_fmt(row['delta_spearman'], True)} | {_ci(row['ci95'])} | "
            f"{row['p_two_sided']:.3f} | {row['n_documents']} | {row['n_clusters']} |"
        )
    decision = payload["decision"]
    lines += ["", f"Rule checks ({decision['rule']}):", ""]
    lines += [f"- {name}: {'pass' if ok else 'fail'}" for name, ok in decision["checks"].items()]
    lines += ["", f"Outcome: **{decision['outcome']}**"]
    return "\n".join(lines)


def provenance_table(rows: list[tuple[str, Path]], extra: list[Path]) -> str:
    lines = ["| Artifact | Commit | Dirty | Run hash | Host |", "|---|---|---|---|---|"]
    paths = [(label, directory / "run_manifest.json") for label, directory in rows]
    paths += [(path.parent.name + "/" + path.name, path) for path in extra]
    for label, path in paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        git = manifest.get("git", {})
        lines.append(
            f"| {label} | `{(git.get('commit') or '')[:7]}` | {git.get('dirty')} | "
            f"`{manifest['run_hash'][:12]}` | {manifest.get('hostname')} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", action="append", required=True, help="label=result_dir")
    parser.add_argument("--paired", type=Path, action="append", default=[])
    parser.add_argument("--system", default="PE__frozen__soft")
    parser.add_argument("--manifest", type=Path, action="append", default=[], help="extra manifests to list")
    parser.add_argument("--all-systems", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = [(item.split("=", 1)[0], Path(item.split("=", 1)[1])) for item in args.row]
    blocks = ["### Results on development sets", "", results_table(rows, args.system)]
    if args.all_systems:
        blocks += ["", all_systems_table(rows[-1][0], rows[-1][1])]
    for path in args.paired:
        blocks += ["", "### Paired difference vs previous step", "", paired_table(path)]
    blocks += ["", "### Provenance", "", provenance_table(rows, args.manifest)]
    text = "\n".join(blocks) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
