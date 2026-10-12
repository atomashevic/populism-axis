"""No model loading or inference: held-out gating, degenerate corpora and frozen-mode encoder checks."""
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_benchmarks  # noqa: E402
import frozen_reference  # noqa: E402
import score_texts  # noqa: E402
from populism_axis.contextual_axis import CHANNELS, WindowProjections  # noqa: E402


def run_main(module, argv):
    with patch.object(sys, "argv", [module.__name__, *argv]), warnings.catch_warnings():
        warnings.simplefilter("ignore")  # toy benchmarks leave some bootstrap metrics undefined
        module.main()


class EvaluatorGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bench = self.root / "bench"
        self.bench.mkdir()
        self.log = self.root / "holdout_access_log.json"
        patcher = patch.object(evaluate_benchmarks, "ACCESS_LOG", self.log)
        patcher.start()
        self.addCleanup(patcher.stop)

        # Eight speeches, alternating dev/holdout; holdout rows carry labels that would flip the result.
        n = 8
        split = np.array(["dev", "holdout"] * (n // 2))
        ids = [f"s{i}" for i in range(n)]
        pd.DataFrame({"id": ids, "split": split}).to_csv(self.bench / "english_split.csv", index=False)
        pd.DataFrame({
            "id": ids,
            "source_score": np.where(split == "dev", np.arange(n), -np.arange(n)).astype(float),
            "leader": [f"L{i % 4}" for i in range(n)],
            "extreme_label": [i % 2 if s == "dev" else -1 for i, s in enumerate(split)],
            "leader_term_id": [f"T{i % 4}" for i in range(n)],
            "leader_term_score": [float(i % 4) for i in range(n)],
        }).to_csv(self.bench / "english_gpd_speeches.csv", index=False)
        self.split = split
        self.n = n

        rng = np.random.default_rng(0)
        doc_index = np.repeat(np.arange(n), 3)
        payload = {"doc_index": doc_index}
        for channel in CHANNELS:
            payload[channel] = rng.normal(size=len(doc_index)) + np.repeat(np.arange(n), 3) * 0.1
        self.contextual = self.root / "contextual"
        (self.contextual / "toy").mkdir(parents=True)
        np.savez(self.contextual / "toy" / "projections_english.npz", **payload)

    def args(self, *extra):
        return ["--label", "toy", "--contextual-dir", str(self.contextual), "--bench-dir", str(self.bench),
                "--out-dir", str(self.contextual), "--draws", "5", "--datasets", "english", *extra]

    def test_eu_requires_final(self):
        with self.assertRaisesRegex(SystemExit, "requires --final"):
            run_main(evaluate_benchmarks, self.args("--datasets", "english,eu"))
        self.assertFalse(self.log.exists())

    def test_holdout_requires_final(self):
        with self.assertRaisesRegex(SystemExit, "requires --final"):
            run_main(evaluate_benchmarks, self.args("--english-split", "holdout"))

    def test_all_split_requires_reason(self):
        with self.assertRaisesRegex(SystemExit, "access-reason"):
            run_main(evaluate_benchmarks, self.args("--english-split", "all"))

    def test_default_scores_dev_rows_only_and_logs_nothing(self):
        with patch("builtins.print"):
            run_main(evaluate_benchmarks, self.args())
        scores = pd.read_csv(self.contextual / "toy" / "scores_english.csv")
        self.assertEqual(len(scores), int((self.split == "dev").sum()))
        evaluation = json.loads((self.contextual / "toy" / "evaluation.json").read_text())
        self.assertEqual(evaluation["english_split"], "dev")
        self.assertFalse(self.log.exists())

    def test_declared_all_split_is_logged(self):
        with patch("builtins.print"):
            run_main(evaluate_benchmarks, self.args("--english-split", "all", "--access-reason", "test"))
        scores = pd.read_csv(self.contextual / "toy" / "scores_english.csv")
        self.assertEqual(len(scores), self.n)
        log = json.loads(self.log.read_text())
        self.assertEqual([entry["english_split"] for entry in log], ["all"])
        self.assertEqual(log[0]["reason"], "test")

    def test_select_documents_reindexes_windows(self):
        projections = WindowProjections(np.array([0, 0, 1, 2, 2]), [], {"P": np.arange(5.0)})
        frame = pd.DataFrame({"id": ["a", "b", "c"]})
        selected, kept = evaluate_benchmarks.select_documents(projections, frame, np.array([True, False, True]))
        np.testing.assert_array_equal(selected.doc_index, [0, 0, 1, 1])
        np.testing.assert_array_equal(selected.projections["P"], [0, 1, 3, 4])
        self.assertEqual(list(kept["id"]), ["a", "c"])


class FrozenReferenceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.axes = {c: v / np.linalg.norm(v) for c, v in zip(frozen_reference.CH, rng.normal(size=(2, 6)))}

    def test_identical_windows_stay_finite_in_corpus_mode(self):
        matrix = np.tile(np.arange(1.0, 7.0), (40, 1))
        quantities, used = frozen_reference.window_quantities(matrix, self.axes, None)
        for name, values in quantities.items():
            self.assertTrue(np.isfinite(values).all(), name)
        np.testing.assert_array_equal(quantities["P_axis"], 0.0)

    def test_varied_windows_match_plain_z_scores(self):
        matrix = np.random.default_rng(2).normal(size=(50, 6))
        quantities, used = frozen_reference.window_quantities(matrix, self.axes, None)
        for c in frozen_reference.CH:
            normalized = frozen_reference.unit_rows(matrix - matrix.mean(axis=0))
            proj = normalized @ self.axes[c]
            np.testing.assert_allclose(quantities[f"{c}_axis"], (proj - proj.mean()) / proj.std())


class FrozenEncoderTests(unittest.TestCase):
    def test_frozen_mode_refuses_other_encoders_before_loading_models(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.csv"
            pd.DataFrame({"text": ["The people against the elite."]}).to_csv(path, index=False)
            with patch.object(score_texts, "make_encoder") as encoder, patch.object(score_texts, "load_reference") as reference:
                with self.assertRaisesRegex(SystemExit, "e5-large-v2"):
                    run_main(score_texts, ["--input", str(path), "--text-column", "text", "--unit", "sentence",
                                           "--reference", "frozen", "--encoder", "e5-small-v2",
                                           "--out-dir", str(Path(temp) / "out")])
            encoder.assert_not_called()
            reference.assert_not_called()


if __name__ == "__main__":
    unittest.main()
