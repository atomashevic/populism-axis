"""Behavior checks for literal CSV, explicit output writes and CLI adapters."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from populism_axis.assets import load_preset_assets, preset_asset_hashes, verify_assets
from populism_axis.cli import main
from populism_axis.io import ordered_input_hash, read_corpus_csv, write_scores


class PackageIOTests(unittest.TestCase):
    def test_literal_csv_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "texts.csv"
            path.write_text('id,text\n001,"Quoted, text"\nNA,Јавне установе\n', encoding="utf-8")
            texts, ids = read_corpus_csv(path, id_column="id")
            self.assertEqual(ids, ["001", "NA"])
            self.assertEqual(texts, ["Quoted, text", "Јавне установе"])
            self.assertNotEqual(ordered_input_hash(ids, texts), ordered_input_hash(ids[::-1], texts[::-1]))

    def test_invalid_csv_cells_headers_and_rows(self):
        cases = ["id,text\na,\n", "id,text\n,text\n", "id,text\na,one\na,two\n", "id,text\na\n", "id,text\na,b,c\n", "id,id,text\na,b,c\n", "id,other\na,b\n", "id,text\n"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            for content in cases:
                with self.subTest(content=content):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        read_corpus_csv(path, id_column="id")

    def test_output_collisions_preflight_all_paths(self):
        result = SimpleNamespace(scores=pd.DataFrame({"id": ["001"], "s": [1.0]}), evidence=[{"id": "001"}], provenance={"schema": "test"})
        for collision in ("scores.csv", "evidence.json", "run_manifest.json"):
            with self.subTest(collision=collision), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / collision).write_text("original")
                (root / "unrelated.txt").write_text("preserved")
                with self.assertRaises(FileExistsError):
                    write_scores(result, root)
                self.assertEqual((root / collision).read_text(), "original")
                self.assertEqual(len(list(root.iterdir())), 2)
                write_scores(result, root, overwrite=True)
                self.assertEqual((root / "unrelated.txt").read_text(), "preserved")
                self.assertEqual(json.loads((root / "evidence.json").read_text()), result.evidence)
                self.assertTrue((root / "scores.csv").read_text().startswith("id,s\n001,"))

    def test_serialization_error_does_not_create_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "output"
            result = SimpleNamespace(scores=pd.DataFrame(), evidence=[float("nan")], provenance={})
            with self.assertRaises(ValueError):
                write_scores(result, out)
            self.assertFalse(out.exists())

    def test_assets_match_source_bytes(self):
        root = Path(__file__).resolve().parents[1]
        sources = {"anchors": "anchors/proposition_anchors.json", "hypotheses": "anchors/claim_hypotheses.json", "reference_stats": "results/frozen_reference/reference_stats.json", "reference_mean": "results/frozen_reference/reference_mean_embedding.npy"}
        import hashlib
        self.assertEqual(preset_asset_hashes(), {name: hashlib.sha256((root / path).read_bytes()).hexdigest() for name, path in sources.items()})
        self.assertEqual(load_preset_assets()["reference_mean"].shape, (1024,))
        self.assertEqual(verify_assets()["reference_mean_dtype"], "float32")

    def test_cli_template_and_no_implicit_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "template.csv"
            self.assertEqual(main(["template", "--out", str(path)]), 0)
            self.assertTrue(path.read_text().startswith("component,pole,proposition\n"))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                main(["template", "--out", str(path)])
            self.assertEqual(exc.exception.code, 2)

    def test_cli_define_from_template_without_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            main(["template", "--out", str(root / "scale.csv")])
            with contextlib.redirect_stdout(io.StringIO()):
                main(["define", "--input", str(root / "scale.csv"), "--name", "trust", "--out", str(root / "scale.json")])
            payload = json.loads((root / "scale.json").read_text())
            self.assertEqual(payload["name"], "trust")
            self.assertEqual(len(payload["components"]), 1)

    def test_cli_reference_score_and_threshold_workflow_with_mock_encoder(self):
        import hashlib
        import numpy as np
        import populism_axis.scoring

        def fake_factory(*args, **kwargs):
            def encode(texts):
                vectors = []
                for text in texts:
                    digest = hashlib.sha256(text.encode("utf-8")).digest()
                    vector = np.frombuffer(digest, dtype=np.uint8).astype(float) - 127.5
                    vectors.append(vector / np.linalg.norm(vector))
                return np.array(vectors)
            return encode, lambda: None

        source = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp, patch("populism_axis.scoring.make_encoder", fake_factory), patch("populism_axis.scoring.preflight_models"), contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            main(["template", "--out", str(root / "scale.csv")])
            main(["define", "--input", str(root / "scale.csv"), "--name", "trust", "--out", str(root / "scale.json")])
            main(["fit-reference", "--input", str(source / "examples/custom-scale-reference.csv"), "--id-column", "id", "--unit", "sentence", "--scale", str(root / "scale.json"), "--out", str(root / "reference.json")])
            texts = root / "texts.csv"
            texts.write_text("id,text\n001,Institutions act fairly.\nNA,Institutions betray citizens.\n", encoding="utf-8")
            main(["score", "--input", str(texts), "--id-column", "id", "--reference", str(root / "reference.json"), "--out-dir", str(root / "scores")])
            main(["make-classifier", "--reference", str(root / "reference.json"), "--threshold", "0", "--below", "low", "--at-or-above", "high", "--out", str(root / "rule.json")])
            main(["apply-classifier", "--classifier", str(root / "rule.json"), "--input-dir", str(root / "scores"), "--out-dir", str(root / "labels")])
            scores = pd.read_csv(root / "labels/scores.csv", dtype={"id": str}, keep_default_na=False)
            self.assertEqual(scores["id"].tolist(), ["001", "NA"])
            self.assertIn("trust__label", scores)
            manifest = json.loads((root / "scores/run_manifest.json").read_text())
            self.assertIn("sha256", manifest["inputs"]["input"])
            self.assertEqual(manifest["reference"], "fitted")

    def test_cli_rejects_classifier_incompatibility_before_scoring(self):
        from populism_axis import define_scale, FittedReference, ThresholdRule
        from populism_axis.scoring import scoring_settings
        scale = define_scale("trust", positive=["Institutions are fair."], contrast=["Institutions are unfair."])
        reference = FittedReference({
            "schema": "populism_axis.reference.v1", "scale": scale.to_dict(),
            "settings": scoring_settings(scale), "axes": {"trust": [1., 0.]},
            "window": {"mu": [0., 0.], "proj": {"trust": [0., 1.]}, "act": {}},
            "document": {"axis": [0., 1.]}, "hypotheses": [],
            "inputs": {"n_documents": 30, "n_windows": 30, "sha256": "fixture"},
            "execution": {},
        })
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference.save(root / "reference.json")
            rules = [
                ("corpus", ThresholdRule(scale.identity, reference.identity, "trust", 0, "low", "high")),
                ("frozen", ThresholdRule(scale.identity, reference.identity, "trust", 0, "low", "high")),
                (str(root / "reference.json"), ThresholdRule(scale.identity, "other-reference", "trust", 0, "low", "high")),
                (str(root / "reference.json"), ThresholdRule("other-definition", reference.identity, "trust", 0, "low", "high")),
                (str(root / "reference.json"), ThresholdRule(scale.identity, reference.identity, "unavailable", 0, "low", "high")),
            ]
            for index, (ref, rule) in enumerate(rules):
                with self.subTest(reference=ref, score=rule.score):
                    rule_path = root / f"rule{index}.json"
                    rule.save(rule_path)
                    with patch("populism_axis.io.score_csv") as scoring, patch("populism_axis.scoring.make_encoder") as encoder, contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                        main(["score", "--input", "absent.csv", "--reference", ref, "--classifier", str(rule_path), "--out-dir", str(root / "outputs")])
                    self.assertEqual(error.exception.code, 2)
                    scoring.assert_not_called()
                    encoder.assert_not_called()
                    self.assertFalse((root / "outputs").exists())

    def test_cli_assets_no_model_inference(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["verify-assets"]), 0)
        self.assertEqual(len(json.loads(output.getvalue())["hashes"]), 4)

    def test_cli_collision_rejects_before_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run_manifest.json").write_text("existing")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                main(["score", "--input", "absent.csv", "--reference", "frozen", "--out-dir", str(root)])
            self.assertEqual(exc.exception.code, 2)
            self.assertEqual((root / "run_manifest.json").read_text(), "existing")


if __name__ == "__main__":
    unittest.main()
