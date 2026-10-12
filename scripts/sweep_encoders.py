#!/usr/bin/env python3
"""stage 5: encoder comparison (CPU; encodings come from pvgpu1).

For each encoder, three cells with the proposition anchors and pole-mean axis rule:
  spw3_k8   the carried window/pooling setting (primary comparison)
  spw1_k5   sensitivity row carried from step 4
  spw1_k8   sensitivity row carried from step 4
Every cell is compared with the carried configuration (e5-large-v2, spw3, k:8).
Every encoder is run with two templates where one exists: the model card's
instruction applied to anchors and windows alike, and plain text. The
instruction was pre-declared as a defect fix (the initial run omitted it); the step-5 data
show it lowers agreement in this symmetric axis method, so both templates are
eligible and reported. The grid rule of docs/development-protocol.md selects
among passing cells; ties go to the smaller encoder, then the simpler cell.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from populism_axis.run_manifest import write_run_manifest  # noqa: E402

STEP = ROOT / "results" / "step5_encoders"
EMB = ROOT / "results" / "embeddings"
SYSTEM = "PE__frozen__soft"
PARAMS_M = {"e5-large-v2": 335, "qwen3-emb-0.6b": 596, "qwen3-emb-0.6b-noinstr": 596,
            "me5-large-instruct": 560, "me5-large-instruct-noinstr": 560, "qwen3-emb-4b": 4022,
            "qwen3-emb-4b-noinstr": 4022, "qwen3-emb-8b": 7567, "qwen3-emb-8b-noinstr": 7567}
ABLATIONS: set[str] = set()
CELLS = (("spw3_k8", 3, "k:8"), ("spw1_k5", 1, "k:5"), ("spw1_k8", 1, "k:8"))


def dirs_for(encoder: str, spw: int) -> tuple[Path, Path]:
    if encoder == "e5-large-v2" and spw == 1:
        return EMB / encoder / "spw1_cap60_clean", EMB / encoder / "spw1_cap60"
    return EMB / encoder / f"spw{spw}_cap60_clean", EMB / encoder / f"spw{spw}_cap60_clean"


def anchors_for(encoder: str) -> Path:
    matches = sorted((EMB / encoder / "sentences").glob("v3_proposition__*.npz"))
    if len(matches) != 1:
        raise SystemExit(f"{encoder}: expected one v3_proposition anchor file, found {matches}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoders", default=(
        "qwen3-emb-0.6b,qwen3-emb-0.6b-noinstr,me5-large-instruct,me5-large-instruct-noinstr,"
        "qwen3-emb-4b-noinstr,qwen3-emb-8b-noinstr"))
    parser.add_argument("--carried", type=Path, default=ROOT / "results" / "step4_windows_topk" / "carried_config.json")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--reuse", action="store_true", help="skip scoring cells whose evaluation.json exists")
    args = parser.parse_args()
    carried = json.loads(args.carried.read_text(encoding="utf-8"))
    baseline = ROOT / carried["result_dir"]

    rows = []
    for encoder in ["e5-large-v2"] + [e for e in args.encoders.split(",") if e]:
        for cell, spw, rule in CELLS:
            label = f"{encoder}__{cell}"
            if encoder == "e5-large-v2":
                out_dir = ROOT / "results" / "step4_windows_topk" / "grid" / cell
            else:
                out_dir = STEP / "grid" / label
                english_dir, agora_dir = dirs_for(encoder, spw)
                if not (english_dir / "embeddings_english.npz").exists():
                    print(f"skip {label}: no embeddings at {english_dir}", flush=True)
                    continue
                if not (args.reuse and (out_dir / "evaluation.json").exists()):
                    subprocess.run(
                        [sys.executable, str(ROOT / "scripts" / "run_config.py"), "--label", label,
                         "--embeddings-dir", str(english_dir), "--dataset-dir", f"agora={agora_dir}",
                         "--anchors", str(anchors_for(encoder)), "--out-dir", str(out_dir), "--top-rule", rule,
                         "--draws", str(args.draws)],
                        check=True,
                    )
            paired_path = out_dir / "paired_vs_step4.json"
            if not (args.reuse and paired_path.exists()):
                subprocess.run(
                    [sys.executable, str(ROOT / "scripts" / "compare_steps.py"), "--baseline-dir", str(baseline),
                     "--candidate-dir", str(out_dir), "--out", str(paired_path)],
                    check=True,
                )
            systems = json.loads((out_dir / "evaluation.json").read_text(encoding="utf-8"))["systems"]
            paired = json.loads(paired_path.read_text(encoding="utf-8"))
            row = {
                "encoder": encoder, "cell": cell, "label": label, "params_m": PARAMS_M.get(encoder),
                "eligible": encoder not in ABLATIONS,
                "english_rho": systems[SYSTEM]["english"]["spearman"],
                "english_ci95": systems[SYSTEM]["english"]["spearman_ci95"],
                "english_extreme_auroc": systems[SYSTEM]["english"]["extreme_auroc"],
                "english_leader_term_rho": systems[SYSTEM]["english"]["leader_term_spearman"],
                "agora_rho": systems[SYSTEM]["agora"]["spearman"],
                "agora_high_low_auroc": systems[SYSTEM]["agora"]["high_vs_low_auroc"],
                "single_channels": {s.split("__")[0]: {d: systems[s][d]["spearman"] for d in ("english", "agora")}
                                    for s in ("P__frozen__soft", "E__frozen__soft", "M__frozen__soft", "PEM__frozen__soft")},
                "delta_english": paired["paired"]["english"]["delta_spearman"],
                "delta_english_ci95": paired["paired"]["english"]["ci95"],
                "delta_agora": paired["paired"]["agora"]["delta_spearman"],
                "delta_agora_ci95": paired["paired"]["agora"]["ci95"],
                "rule_passed": paired["decision"]["rule_passed"],
                "result_dir": str(out_dir.relative_to(ROOT)),
            }
            rows.append(row)
            print(f"{label:34s} ENG {row['english_rho']:.4f} (Δ {row['delta_english']:+.4f})  "
                  f"AG {row['agora_rho']:.4f} (Δ {row['delta_agora']:+.4f})  rule={row['rule_passed']}", flush=True)

    passing = [r for r in rows if r["rule_passed"] and r["eligible"] and r["encoder"] != "e5-large-v2"]
    selected = None
    if passing:
        best = max((r["delta_english"] + r["delta_agora"]) / 2 for r in passing)
        tied = [r for r in passing if (r["delta_english"] + r["delta_agora"]) / 2 >= best - 0.005]
        order = {"spw1_k5": 0, "spw1_k8": 1, "spw3_k8": 2}
        selected = min(tied, key=lambda r: (r["params_m"], order[r["cell"]]))["label"]
    summary = {"baseline": carried["result_dir"], "selected": selected, "rows": rows}
    STEP.mkdir(parents=True, exist_ok=True)
    (STEP / "grid.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# stage 5 encoder grid", "",
             f"Baseline `{carried['result_dir']}` (e5-large-v2, spw3, k:8). System `{SYSTEM}`. Selected: `{selected}`.", "",
             "| Encoder | Cell | English-dev ρ | Δ English [95% CI] | Agora ρ | Δ Agora [95% CI] | P / E English ρ | Rule |",
             "|---|---|---:|---|---:|---|---|---|"]
    for r in rows:
        tag = " (ablation)" if not r["eligible"] else ""
        lines.append(
            f"| {r['encoder']}{tag} | {r['cell']} | {r['english_rho']:.3f} | {r['delta_english']:+.3f} "
            f"[{r['delta_english_ci95'][0]:+.3f}, {r['delta_english_ci95'][1]:+.3f}] | {r['agora_rho']:.3f} | "
            f"{r['delta_agora']:+.3f} [{r['delta_agora_ci95'][0]:+.3f}, {r['delta_agora_ci95'][1]:+.3f}] | "
            f"{r['single_channels']['P']['english']:.3f} / {r['single_channels']['E']['english']:.3f} | "
            f"{'pass' if r['rule_passed'] else 'fail'} |"
        )
    (STEP / "grid.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_run_manifest(STEP, script="scripts/sweep_encoders.py", inputs={"carried": args.carried},
                       extra={"encoders": args.encoders, "selected": selected}, filename="grid.manifest.json")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
