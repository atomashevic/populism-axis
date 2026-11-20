#!/usr/bin/env python3
"""Claim-check arms D and E (GPU host): P(yes) of two decision models for window x claim questions.

Plan: research/development-log/claim-check-arms-plan.md. Each of the ten claim hypotheses
becomes one yes/no question with the fixed template in claim_check_arms.QUESTION_TEMPLATE,
asked of every stored window without criteria. The output has the layout of nli_stage.py
(`probs`, n_requests x n_hypotheses x 3, float32, columns [1 - P(yes), 0, P(yes)]), so the
evaluator reads P(yes) where it reads P(entailment) for the NLI arms. `margin` holds the raw
yes-minus-no logit before any prior correction or temperature.

Both SDKs round the probability they return to four decimals, and Von's default further maps
it outside a .2-.8 band. The claim check ranks windows by small probabilities, so this script
does not read the SDK answers. It runs each SDK's own packing, forward pass, prior correction
and temperature, and takes the probability before rounding. The first --check windows are also
sent through the public SDK call and must agree with the unrounded values to the rounding grid.

  von   von-sdk 1.3.7, OptionMarkerBackend.evaluate_noul: one sequence per question, batch 1,
        plus one state-free sequence per question for the SDK's criteria-less prior correction
        (cached, ten in total).
  laya  laya 0.4.37, Agent.predict_batch: one sequence per question; the ten questions of
        --batch-size // 10 windows are collated into one forward call, so a call holds about
        as many sequences as a batch of the NLI arms.

Neither SDK is in uv.lock; install it into the GPU checkout's environment
(docs/environment.md). The manifest records the installed SDK version and the
torch and transformers versions.

--probe N scores only the first N windows of the first corpus and writes probe__<label>.json:
the SDK agreement, truncation and the tail counts of the values as stored. No .npz is written.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from claim_check_arms import ARMS, QUESTION_TEMPLATE, tail_counts  # noqa: E402
from populism_axis.model_revisions import pinned_revision  # noqa: E402
from populism_axis.run_manifest import file_sha256, write_run_manifest  # noqa: E402

STORED_LABELS = ("not_yes", "unused", "yes")
ARM_BY_NAME = {"von": "D", "laya": "E"}
# Largest allowed gap between the SDK's rounded answer and this script's unrounded value.
SDK_TOLERANCE = {"von": 1.5e-4, "laya": 5e-3}


def softmax_pair(logit_yes: float, logit_no: float, temperature: float) -> float:
    """P(yes) of a two-option softmax, in float64."""
    z = (float(logit_yes) - float(logit_no)) / max(float(temperature), 1e-4)
    return float(1.0 / (1.0 + np.exp(-z)))


def snapshot(model: str, revision: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, revision=revision))


class VonArm:
    """Unrounded replica of von.backends.option_marker_backend.OptionMarkerBackend.evaluate_noul."""

    name = "von"
    sequences_per_question = 1

    def __init__(self, model: str, revision: str, device: str):
        # Read by the backend constructor: raw posterior, no silent state truncation, no chain library.
        os.environ["VON_NOUL_DECISION"] = "raw"
        os.environ["VON_ON_OVERFLOW"] = "refuse"
        os.environ["VON_CHAINS_DIR"] = "off"
        import torch
        from von.backends import option_marker_backend as backend_module
        from von.types import Noul

        self.torch, self.Noul = torch, Noul
        self.checkpoint = snapshot(model, revision)
        self.backend = backend_module.OptionMarkerBackend(checkpoint_dir=str(self.checkpoint), device=device)
        self.format_state = backend_module._format_state
        self.model = self.backend._get_model()
        self.descriptions = ["Yes, condition holds true.", "No, condition is false."]  # the SDK's criteria-less pair
        self.null: dict[str, object] = {}
        self.sequences = 0
        self.forward_calls = 0
        self.settings = {"independent_options": bool(self.backend._independent_options),
                         "default_temperature": float(self.backend._default_temp),
                         "calibration_map": self.backend._calib_map is not None,
                         "noul_prior": self.backend._noul_prior,
                         "descriptions": self.descriptions, "device": str(self.backend.device)}

    def _logits(self, state: str, question: str):
        packed = self.model.pack_sequence(state, question, self.descriptions)
        inputs = self.model.tokenizer(packed, return_tensors="pt").to(self.backend.device)
        positions = (inputs["input_ids"][0] == self.model.mask_token_id).nonzero(as_tuple=True)[0].tolist()
        if len(positions) != 2:
            raise RuntimeError(f"von: {len(positions)} option markers for a yes/no question")
        self.sequences += 1
        self.forward_calls += 1
        return self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                          mask_positions=[positions], independent_options=self.backend._independent_options)[0].float()

    def score(self, premises: list[str], questions: list[str]) -> tuple[np.ndarray, np.ndarray]:
        p = np.zeros((len(premises), len(questions)))
        margin = np.zeros_like(p)
        with self.torch.no_grad():
            for i, premise in enumerate(premises):
                for j, question in enumerate(questions):
                    state = self.backend._fit_state(self.model, self.format_state(premise), question, self.descriptions)
                    logits = self._logits(state, question)
                    if question not in self.null:
                        self.null[question] = self._logits("", question)
                    null = self.null[question]
                    bias = null[0] - null[1]
                    prior = self.backend._noul_prior
                    correction = prior["a"] * bias + prior["b"] if prior is not None else 0.7 * bias
                    corrected = self.torch.stack([logits[0] - correction, logits[1]])
                    temperature = self.backend._effective_temperature(corrected, state, 2, self.model.tokenizer, None)
                    p[i, j] = softmax_pair(corrected[0], corrected[1], temperature)
                    margin[i, j] = float(logits[0] - logits[1])
        return p, margin

    def sdk(self, premises: list[str], questions: list[str]) -> np.ndarray:
        out = np.zeros((len(premises), len(questions)))
        for i, premise in enumerate(premises):
            for j, question in enumerate(questions):
                answer = self.backend.evaluate_noul("q", self.format_state(premise), self.Noul(instructions=question))
                out[i, j] = float(answer.noul_raw)
        return out


class LayaArm:
    """Unrounded replica of laya.Agent.predict_batch for criteria-less yes/no questions."""

    name = "laya"
    sequences_per_question = 1

    def __init__(self, model: str, revision: str, device: str):
        os.environ.setdefault("USE_TF", "0")  # the model card reports a load hang when TensorFlow is importable
        import laya

        from huggingface_hub import hf_hub_download

        # The repository bundles three checkpoints; only the English one at the root is used and hashed.
        self.checkpoint = Path(hf_hub_download(model, "model.safetensors", revision=revision)).parent
        self.agent = laya.load(model, device=device, revision=revision)
        module = sys.modules[type(self.agent).__module__]
        self.collate, self.temp_bucket = module.collate_items, module.temp_bucket
        self.option_logits, self.qtype = module._option_logits, module.QTYPES["noul"]
        self.sequences = 0
        self.forward_calls = 0
        self.settings = {"max_len": self.agent.cfg.get("max_len", 512), "head_max_len": self.agent.cfg.get("head_max_len", 192),
                         "dtype": str(getattr(self.agent, "dtype", None)), "device": str(getattr(self.agent, "device", None)),
                         "resolved_revision": getattr(self.agent, "revision", None)}

    def _questions(self, questions: list[str]) -> dict:
        return {f"q{j:02d}": {"type": "noul", "instructions": question} for j, question in enumerate(questions)}

    def score(self, premises: list[str], questions: list[str]) -> tuple[np.ndarray, np.ndarray]:
        definitions = self._questions(questions)
        ids = list(definitions)
        for qid in ids:
            self.agent._check_question(qid, definitions[qid])
        internal = {qid: self.agent._to_internal(definitions[qid]) for qid in ids}
        encoded = [self.agent._encode_state(premise, ids, internal) for premise in premises]
        for items in encoded:
            if any(item["state_stats"]["truncated"] for item in items):
                raise RuntimeError("laya: a window was truncated to fit the token budget")
        import torch

        with torch.no_grad():
            logits, _ = self.agent._forward(self.collate(encoded, self.agent.tok.pad_token_id))
        self.sequences += sum(len(items) for items in encoded)
        self.forward_calls += 1
        p = np.zeros((len(premises), len(questions)))
        margin = np.zeros_like(p)
        row = 0
        for i, items in enumerate(encoded):
            for j, option_row in enumerate(self.option_logits(logits, items, row)):
                if len(option_row) != 2:
                    raise RuntimeError(f"laya: {len(option_row)} options for a yes/no question")
                temperature = self.agent.temperature_by_options.get(self.temp_bucket(self.qtype, 2), self.agent.temperature[self.qtype])
                p[i, j] = softmax_pair(option_row[1], option_row[0], temperature)  # index 1 is "true"
                margin[i, j] = float(option_row[1]) - float(option_row[0])
            row += len(items)
        return p, margin

    def sdk(self, premises: list[str], questions: list[str]) -> np.ndarray:
        definitions = self._questions(questions)
        results = self.agent.predict_batch(premises, definitions, batch_size=len(premises))
        return np.asarray([[result["answers"][qid]["noul"] for qid in definitions] for result in results], dtype=float)


def sdk_version(distribution: str) -> str:
    from importlib.metadata import version

    return version(distribution)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=tuple(ARM_BY_NAME), required=True)
    parser.add_argument("--in-dir", type=Path, default=ROOT / "results" / "step6_nli")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "claim_check_arms" / "nli")
    parser.add_argument("--names", default="english_dev,agora")
    parser.add_argument("--revision", default=None, help="exact 40-character Hub commit (default: the arm's recorded commit)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="sequences per forward call, as for the NLI arms (laya: batch-size // 10 windows); von runs one sequence per call")
    parser.add_argument("--check", type=int, default=16, help="windows compared with the public SDK call")
    parser.add_argument("--probe", type=int, default=0, help="score only the first N windows of the first corpus; write no .npz")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    for attribute in ("in_dir", "out_dir"):
        path = getattr(args, attribute)
        if not path.is_absolute():
            setattr(args, attribute, Path.cwd() / path)
    if args.batch_size < 1 or args.check < 1:
        parser.error("--batch-size and --check must be positive")

    arm = ARMS[ARM_BY_NAME[args.arm]]
    try:
        revision = pinned_revision(args.revision or arm["revision"])
    except ValueError as error:
        raise SystemExit(f"{error} (--revision)")

    import torch

    torch.set_num_threads(2)
    runner = (VonArm if args.arm == "von" else LayaArm)(arm["model"], revision, args.device)
    hypotheses = json.loads((args.in_dir / "hypotheses.json").read_text(encoding="utf-8"))
    questions = [QUESTION_TEMPLATE.format(claim=h["text"]) for h in hypotheses]
    label = arm["label"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cuda = args.device.startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    info: dict = {}
    names = [n for n in args.names.split(",") if n]
    if args.probe:
        names = names[:1]
    for name in names:
        requests = pd.read_parquet(args.in_dir / f"requests_{name}.parquet")
        if args.probe:
            requests = requests.iloc[: args.probe]
        premises = requests["premise"].tolist()

        checked = premises[: args.check]
        own, _ = runner.score(checked, questions)
        gap = float(np.abs(np.round(own, 4) - runner.sdk(checked, questions)).max())
        if gap > SDK_TOLERANCE[args.arm]:
            raise SystemExit(f"{name}: unrounded values differ from the SDK answers by {gap:.5f} (limit {SDK_TOLERANCE[args.arm]})")
        runner.sequences = runner.forward_calls = 0

        order = np.argsort([len(p) for p in premises], kind="stable")
        step = max(1, args.batch_size // len(questions))
        p_yes = np.zeros((len(premises), len(questions)), dtype=np.float64)
        margin = np.zeros_like(p_yes)
        started = time.time()
        for batch_number, start in enumerate(range(0, len(order), step)):
            idx = order[start: start + step]
            p_yes[idx], margin[idx] = runner.score([premises[i] for i in idx], questions)
            if batch_number % 500 == 0:
                print(f"  {name}: {start + len(idx)}/{len(order)} windows", flush=True)
        seconds = round(time.time() - started, 1)
        if not (np.isfinite(p_yes).all() and np.isfinite(margin).all()):
            raise FloatingPointError(f"{name}: non-finite decision-model output")
        stored = np.zeros((len(premises), len(questions), 3), dtype=np.float32)
        stored[:, :, 2] = p_yes
        stored[:, :, 0] = 1.0 - p_yes
        info[name] = {"windows": len(premises), "pairs": int(p_yes.size), "sequences": int(runner.sequences),
                      "forward_calls": int(runner.forward_calls), "seconds": seconds, "sdk_max_gap": gap,
                      "stored_yes": tail_counts(stored[:, :, 2]),
                      "yes_quantiles": {q: float(np.quantile(p_yes, v)) for q, v in (("p01", .01), ("median", .5), ("p99", .99))},
                      "share_yes_gt_0.5": float((p_yes > 0.5).mean()), "sd_yes": float(p_yes.std())}
        if not args.probe:
            out = args.out_dir / f"nli_{name}__{label}.npz"
            np.savez_compressed(out, request_id=requests["request_id"].to_numpy(), probs=stored, margin=margin.astype(np.float32),
                                hypothesis_ids=np.asarray([h["hypothesis_id"] for h in hypotheses]), labels=np.asarray(STORED_LABELS))
            info[name]["output"] = out.name
        print(name, json.dumps(info[name]), flush=True)

    weights = {f.name: file_sha256(f) for f in (runner.checkpoint / "model.safetensors", runner.checkpoint / "option_marker.pt") if f.exists()}
    info.update({"arm": ARM_BY_NAME[args.arm], "model": arm["model"], "model_revision": revision, "weights_sha256": weights,
                 "sdk": {"distribution": arm["sdk"].split("==")[0], "version": sdk_version(arm["sdk"].split("==")[0])},
                 "question_template": QUESTION_TEMPLATE, "question_order": [h["hypothesis_id"] for h in hypotheses],
                 "criteria": None, "settings": runner.settings, "batch_size": args.batch_size, "probe": args.probe,
                 "device_after_run": str(getattr(getattr(runner, "agent", None) or runner.backend, "device", None)),
                 "libraries": {"torch": torch.__version__, "transformers": sdk_version("transformers")},
                 "stored_labels": list(STORED_LABELS), "probs_dtype": "float32",
                 "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else None,
                 "gpu": torch.cuda.get_device_name(0) if cuda else None})
    if args.probe:
        (args.out_dir / f"probe__{label}.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    else:
        write_run_manifest(args.out_dir, script="scripts/decision_model_stage.py", inputs={"hypotheses": args.in_dir / "hypotheses.json"},
                           encoder_id=arm["model"], extra=info, filename=f"nli__{label}.manifest.json")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
