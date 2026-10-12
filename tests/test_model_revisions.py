"""Pinned model revisions and the fresh-inference comparison; no model loading or inference."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import check_fresh_inference  # noqa: E402
import score_texts  # noqa: E402
from populism_axis.encoders import get_spec  # noqa: E402
from populism_axis.model_revisions import (BART_LARGE_MNLI, E5_LARGE_V2, PINNED_REVISIONS,  # noqa: E402
                                           pinned_revision, require_revision, revision_for)
from test_review_guards import run_main  # noqa: E402


class PinnedRevisionTests(unittest.TestCase):
    def test_frozen_models_have_exact_commits(self):
        for model_id in (E5_LARGE_V2, BART_LARGE_MNLI):
            self.assertEqual(pinned_revision(PINNED_REVISIONS[model_id]), PINNED_REVISIONS[model_id])
        self.assertEqual(get_spec("e5-large-v2").revision, PINNED_REVISIONS[E5_LARGE_V2])
        self.assertEqual(get_spec("e5-large-v2", batch_size=16).revision, PINNED_REVISIONS[E5_LARGE_V2])

    def test_moving_refs_are_rejected_and_unpinned_models_need_a_commit(self):
        for value in ("main", "v1.0", "f169b11", "F" * 40, ""):
            with self.assertRaises(ValueError):
                revision_for(E5_LARGE_V2, value)
        self.assertIsNone(revision_for("Qwen/Qwen3-Embedding-0.6B"))
        with self.assertRaisesRegex(ValueError, "no pinned revision"):
            require_revision("Qwen/Qwen3-Embedding-0.6B")
        self.assertEqual(require_revision("Qwen/Qwen3-Embedding-0.6B", "a" * 40), "a" * 40)

    def test_runtime_manifest_declares_the_pinned_snapshots(self):
        manifest = json.loads((ROOT / "results/frozen_reference/runtime_manifest.json").read_text())
        declared = {model["model_id"]: model["pinned_revision"] for model in manifest["models"]}
        self.assertEqual(declared, PINNED_REVISIONS)

    def test_scorer_refuses_unpinned_models_before_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.csv"
            pd.DataFrame({"text": ["The people against the elite."]}).to_csv(path, index=False)
            for extra, message in ((["--nli-model", "cross-encoder/nli-deberta-v3-small"], "no pinned revision"),
                                   (["--nli-revision", "main"], "40-character"),
                                   (["--encoder-revision", "main"], "40-character"),
                                   (["--encoder-revision", "a" * 40], "pinned e5-large-v2 commit only"),
                                   (["--nli-revision", "a" * 40], "pinned facebook/bart-large-mnli commit only")):
                with patch.object(score_texts, "make_encoder") as encoder, patch.object(score_texts, "load_reference") as reference:
                    with self.assertRaisesRegex(SystemExit, message):
                        run_main(score_texts, ["--input", str(path), "--unit", "sentence", "--reference", "frozen",
                                               "--out-dir", str(Path(temp) / "out"), *extra])
                encoder.assert_not_called()
                reference.assert_not_called()


class FreshInferenceCompareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        scored = self.directory / "scored"
        scored.mkdir()
        stored = pd.read_csv(check_fresh_inference.STORED)
        ids = check_fresh_inference.english_dev()["id"]
        self.fresh = pd.DataFrame({"id": ids, **{new: stored[old] for new, old in check_fresh_inference.COLUMNS.items()}})
        self.checkout = {"commit": "c" * 40, "branch": "under-test", "dirty": False}
        patcher = patch.object(check_fresh_inference, "git_commit", lambda: dict(self.checkout))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.manifest = {"git": {"commit": "c" * 40, "dirty": False}, "encoder_id": E5_LARGE_V2, "run_hash": "x",
                         "extra": {"reference": "frozen", "top_rule": "k:8", "nli_model": BART_LARGE_MNLI,
                                   "encoder_revision": PINNED_REVISIONS[E5_LARGE_V2], "nli_revision": PINNED_REVISIONS[BART_LARGE_MNLI],
                                   "resolved_revisions": {"encoder": PINNED_REVISIONS[E5_LARGE_V2], "nli_model": PINNED_REVISIONS[BART_LARGE_MNLI]},
                                   "n_windows": check_fresh_inference.EXPECTED_WINDOWS, "n_documents": check_fresh_inference.EXPECTED_DOCUMENTS}}

    def run_compare(self):
        self.fresh.to_csv(self.directory / "scored/scores.csv", index=False)
        (self.directory / "scored/run_manifest.json").write_text(json.dumps(self.manifest))
        return check_fresh_inference.compare(self.directory)

    def test_stored_scores_with_float16_noise_pass(self):
        noise = np.random.default_rng(0).normal(scale=1e-4, size=len(self.fresh))
        self.fresh["axis"] = self.fresh["axis"] * (1 + noise)
        report = self.run_compare()
        self.assertTrue(report["passed"], json.dumps(report["checks"], indent=1))
        self.assertTrue((self.directory / "fresh_inference_check.json").exists())

    def test_shuffled_scores_wrong_revision_and_dirty_tree_fail(self):
        self.fresh["claim_check"] = self.fresh["claim_check"].sample(frac=1, random_state=1).to_numpy()
        self.manifest["git"]["dirty"] = True
        self.manifest["extra"]["resolved_revisions"]["encoder"] = "b" * 40
        report = self.run_compare()
        self.assertFalse(report["passed"])
        for name in ("clean_checkout", "pinned_revisions", "score_agreement"):
            self.assertFalse(report["checks"][name]["passed"], name)
        self.assertTrue(report["checks"]["window_count"]["passed"])

    def test_rescaled_scores_fail_despite_identical_ranks(self):
        for column in check_fresh_inference.COLUMNS:
            self.fresh[column] = self.fresh[column] * 100
        report = self.run_compare()
        self.assertFalse(report["checks"]["score_agreement"]["passed"])
        self.assertAlmostEqual(report["checks"]["score_agreement"]["columns"]["axis"]["rank_agreement"], 1.0)

    def test_scores_from_another_commit_or_a_dirty_comparison_fail(self):
        self.manifest["git"]["commit"] = "d" * 40
        self.assertFalse(self.run_compare()["checks"]["clean_checkout"]["passed"])
        self.manifest["git"]["commit"] = "c" * 40
        self.checkout["dirty"] = True
        report = self.run_compare()
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["clean_checkout"]["passed"])


if __name__ == "__main__":
    unittest.main()
