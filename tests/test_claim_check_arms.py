"""Same-size claim-check arms: label layouts, stored-value checks and the partial correlation. No model loading."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import decision_model_stage  # noqa: E402
import evaluate_claim_check_arms as evaluator  # noqa: E402
import nli_stage  # noqa: E402
from claim_check_arms import ARMS, CANDIDATES, CONTRASTS, QUESTION_TEMPLATE, tail_counts  # noqa: E402
from populism_axis.model_revisions import PINNED_REVISIONS, pinned_revision  # noqa: E402


class ArmTableTests(unittest.TestCase):
    def test_every_arm_has_an_exact_commit_and_a_distinct_label(self):
        for arm in ARMS.values():
            self.assertEqual(pinned_revision(arm["revision"]), arm["revision"])
        self.assertEqual(len({arm["label"] for arm in ARMS.values()}), len(ARMS))
        self.assertEqual(ARMS["A"]["revision"], PINNED_REVISIONS[ARMS["A"]["model"]])

    def test_candidates_stay_out_of_the_frozen_pins(self):
        for arm in CANDIDATES:
            self.assertNotIn(ARMS[arm]["model"], PINNED_REVISIONS)

    def test_contrasts_and_template_refer_to_known_arms(self):
        for left, right, _ in CONTRASTS:
            self.assertIn(left, ARMS)
            self.assertIn(right, ARMS)
        self.assertEqual(QUESTION_TEMPLATE.format(claim="X."), "Does this passage assert the following claim? X.")
        self.assertEqual({ARMS[a]["kind"] for a in decision_model_stage.ARM_BY_NAME.values()}, {"decision"})


class LabelLayoutTests(unittest.TestCase):
    def test_three_class_heads_are_reordered(self):
        self.assertEqual(nli_stage.label_layout({0: "entailment", 1: "neutral", 2: "contradiction"}), ([2, 1, 0], nli_stage.TARGET_LABELS))
        self.assertEqual(nli_stage.label_layout({"0": "CONTRADICTION", "1": "NEUTRAL", "2": "ENTAILMENT"})[0], [0, 1, 2])

    def test_binary_head_keeps_entailment_in_the_last_column(self):
        columns, labels = nli_stage.label_layout({0: "entailment", 1: "not_entailment"})
        self.assertEqual(columns, [1, None, 0])
        self.assertEqual(labels, nli_stage.BINARY_LABELS)

    def test_unknown_label_sets_are_refused(self):
        with self.assertRaisesRegex(ValueError, "unsupported label set"):
            nli_stage.label_layout({0: "LABEL_0", 1: "LABEL_1"})


class StoredValueTests(unittest.TestCase):
    def test_rounded_and_saturated_values_are_counted(self):
        rounded = tail_counts(np.asarray([0.0, 0.0012, 1.0, 0.5], dtype=np.float32))
        self.assertEqual((rounded["exact_zero"], rounded["exact_one"], rounded["distinct"]), (1, 1, 4))
        self.assertEqual(rounded["on_4_decimal_grid"], 1.0)
        fine = tail_counts(np.asarray([0.00123456, 0.0345678], dtype=np.float32))
        self.assertEqual(fine["on_4_decimal_grid"], 0.0)

    def test_two_option_probability(self):
        self.assertAlmostEqual(decision_model_stage.softmax_pair(0.0, 0.0, 1.0), 0.5)
        self.assertAlmostEqual(decision_model_stage.softmax_pair(2.0, 0.0, 2.0), 1 / (1 + np.exp(-1.0)))
        self.assertGreater(decision_model_stage.softmax_pair(-20.0, 0.0, 1.0), 0.0)


class PartialSpearmanTests(unittest.TestCase):
    def test_removes_what_the_control_explains(self):
        rng = np.random.default_rng(0)
        z = rng.normal(size=2000)
        target = z + rng.normal(size=2000)
        copy = z + 0.01 * rng.normal(size=2000)
        self.assertLess(abs(evaluator.partial_spearman(copy, target, z)), 0.05)
        informative = target + 0.5 * rng.normal(size=2000)
        self.assertGreater(evaluator.partial_spearman(informative, target, z), 0.5)

    def test_cluster_interval_brackets_the_value(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=300)
        y = x + rng.normal(size=300)
        groups = np.repeat(np.arange(30), 10).astype(str)
        out = evaluator.cluster_interval(lambda a, b: float(np.corrcoef(a, b)[0, 1]), (x, y), groups, 200, 3)
        self.assertLess(out["ci95"][0], out["value"])
        self.assertGreater(out["ci95"][1], out["value"])


if __name__ == "__main__":
    unittest.main()
