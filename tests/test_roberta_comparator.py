"""No model loading or inference: activation, cache safety, and consumer guards."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from populism_axis.roberta_comparator import (LABELS, SCHEMA, independent_probabilities,
    load_scores, pinned_revision, save_scores, validate_config)


class RoBERTaComparatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "sigmoid_fixture.npz"
        self.logits = np.array([[-3, 2, 3], [4, -2, -3]], dtype=np.float32)
        self.index = np.array([0, 1], dtype=np.int32)
        self.metadata = {"model_revision": "a" * 40, "tokenizer_revision": "b" * 40}

    def test_joint_positive_labels_do_not_compete(self):
        p = independent_probabilities(self.logits)
        np.testing.assert_allclose(p[0], [0.047425873, 0.880797078, 0.952574127], rtol=1e-6)
        self.assertGreater(p[0, 1], .5)
        self.assertGreater(p[0, 2], .5)
        self.assertGreater(float(p[0].sum()), 1.8)

    def test_extreme_logits_are_finite_and_independent(self):
        with np.errstate(over="raise", invalid="raise"):
            p = independent_probabilities(np.array([[-1000, 0, 1000]], np.float32))
        np.testing.assert_array_equal(p, [[0, .5, 1]])
        with self.assertRaises(ValueError):
            independent_probabilities(np.array([[0, float("nan"), 1]]))
        with self.assertRaises(ValueError):
            independent_probabilities(np.zeros((2, 2)))

    def test_corrected_cache_round_trip_keeps_raw_logits(self):
        save_scores(self.path, self.index, self.logits, self.metadata)
        p = load_scores(self.path)
        np.testing.assert_array_equal(p["logits"], self.logits)
        np.testing.assert_allclose(p["probabilities"], independent_probabilities(self.logits))
        self.assertEqual(json.loads(str(p["metadata_json"]))["schema"], SCHEMA)

    def test_new_cache_never_overwrites_existing_bytes(self):
        self.path.write_bytes(b"historical data")
        with self.assertRaises(FileExistsError):
            save_scores(self.path, self.index, self.logits, self.metadata)
        self.assertEqual(self.path.read_bytes(), b"historical data")

    def test_legacy_cache_rejected_even_under_corrected_filename(self):
        np.savez(self.path, doc_index=self.index, probabilities=np.ones((2, 3)) / 3)
        with self.assertRaisesRegex(ValueError, "Legacy/untagged"):
            load_scores(self.path)

    def test_tagging_softmax_as_sigmoid_is_rejected(self):
        exp = np.exp(self.logits); p = exp / exp.sum(axis=1, keepdims=True)
        metadata = dict(self.metadata, schema=SCHEMA, activation="sigmoid", id2label=LABELS)
        np.savez(self.path, doc_index=self.index, logits=self.logits, probabilities=p,
                 metadata_json=np.asarray(json.dumps(metadata)))
        with self.assertRaisesRegex(ValueError, "disagree"):
            load_scores(self.path)

    def test_checkpoint_semantics_and_label_order_are_required(self):
        config = SimpleNamespace(problem_type="multi_label_classification", num_labels=3,
                                 id2label={int(k): v for k, v in LABELS.items()})
        validate_config(config)
        config.problem_type = "single_label_classification"
        with self.assertRaises(ValueError): validate_config(config)
        config.problem_type = "multi_label_classification"
        config.id2label[1], config.id2label[2] = config.id2label[2], config.id2label[1]
        with self.assertRaises(ValueError): validate_config(config)

    def test_moving_revision_and_invalid_document_indices_rejected(self):
        for value in ["main", "", "abc123"]:
            with self.assertRaises(ValueError): pinned_revision(value)
        self.assertEqual(pinned_revision("a" * 40), "a" * 40)
        with self.assertRaises(ValueError):
            save_scores(self.path, np.array([-1, 0]), self.logits, self.metadata)
        self.assertFalse(self.path.exists())

    def test_final_holdout_preflights_before_result_writes(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        spec = importlib.util.spec_from_file_location("final_holdout_guard_test", ROOT / "scripts/final_holdout.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        missing = Path(self.temp.name) / "absent_final"
        with patch.object(mod, "FINAL", missing), patch.object(mod, "run_axis") as axis, patch.object(sys, "argv", ["final_holdout.py", "--final"]):
            with self.assertRaisesRegex(FileNotFoundError, "legacy softmax"):
                mod.main()
            axis.assert_not_called()
        self.assertFalse(missing.exists())

    def test_english_file_inputs_use_existing_loader_semantics(self):
        spec = importlib.util.spec_from_file_location("roberta_encoder_text_test", ROOT / "scripts/external/encode_roberta_custom.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        root = Path(self.temp.name); (root / "data").mkdir()
        text = root / "data/speech.txt"; text.write_bytes(b"Original speech.\xff")
        csv = root / "input.csv"; csv.write_text("path,merging_variable\ndata/speech.txt,speech.txt\n")
        with patch.object(mod, "ROOT", root):
            texts, inputs = mod.load_texts(csv, "text", "path")
        self.assertEqual(texts, ["Original speech."])
        self.assertEqual(inputs, {"csv": csv, "text_00000": text})

    def test_encoder_refuses_existing_outputs_before_model_loading(self):
        spec = importlib.util.spec_from_file_location("roberta_encoder_guard_test", ROOT / "scripts/external/encode_roberta_custom.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        out = Path(self.temp.name)
        existing = out / "sigmoid_fixture.npz"; existing.write_bytes(b"preserve me")
        argv = ["encode_roberta_custom.py", "--csv", "unused.csv", "--name", "fixture", "--out-dir", str(out),
                "--model-revision", "a" * 40, "--tokenizer-revision", "b" * 40]
        with patch.object(sys, "argv", argv), patch.dict(sys.modules, {"torch": None, "transformers": None}), patch("sys.stderr"):
            with self.assertRaises(SystemExit) as caught: mod.main()
        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(existing.read_bytes(), b"preserve me")

    def test_external_roberta_helpers_reject_renamed_legacy_outputs(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        for name, filename, invoke in [
            ("score_portugal", "sigmoid_portugal_en.npz", lambda m: m.roberta_scores(2)),
            ("score_usgov", "sigmoid_fixture.npz", lambda m: m.roberta_scores("fixture", 2, "k:1", 1)),
            ("score_popbert_en", "sigmoid_popbert_en_train.npz", lambda m: m.roberta_scores({"train": 2, "test": 2}, 1)),
        ]:
            with self.subTest(script=name):
                spec = importlib.util.spec_from_file_location(name + "_guard_test", ROOT / "scripts/external" / (name + ".py"))
                mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
                out = Path(self.temp.name) / name; (out / "roberta").mkdir(parents=True)
                np.savez(out / "roberta" / filename, doc_index=self.index, probabilities=np.ones((2, 3)) / 3)
                with patch.object(mod, "OUT", out), self.assertRaisesRegex(ValueError, "Legacy/untagged"):
                    invoke(mod)

if __name__ == "__main__":
    unittest.main()
