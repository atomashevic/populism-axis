"""Tests for the pipeline step-0 infrastructure: provenance, holdout split, encoder specs, sentence loader."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from populism_axis.run_manifest import build_run_manifest, file_sha256, write_run_manifest  # noqa: E402
from populism_axis.encoders import ENCODER_SPECS, get_spec, render_text  # noqa: E402
from populism_axis.sentences import load_sentence_groups  # noqa: E402
from populism_axis.split import SplitRule, choose_country_split, split_frame  # noqa: E402


class RunManifestTests(unittest.TestCase):
    def test_sha_and_deterministic_run_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "input.txt"
            source.write_text("anchor text\n", encoding="utf-8")
            self.assertEqual(file_sha256(source), hashlib.sha256(b"anchor text\n").hexdigest())
            kwargs = dict(script="x.py", argv=["--a", "1"], inputs={"src": source}, encoder_id="enc", extra={"k": 3})
            first = build_run_manifest(**kwargs, timestamp="t0")
            second = build_run_manifest(**kwargs, timestamp="t1")
            self.assertEqual(first["run_hash"], second["run_hash"])
            changed = build_run_manifest(**{**kwargs, "extra": {"k": 5}}, timestamp="t0")
            self.assertNotEqual(first["run_hash"], changed["run_hash"])

    def test_written_manifest_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "input.txt"
            source.write_text("x", encoding="utf-8")
            path = write_run_manifest(Path(tmpdir) / "out", script="x.py", argv=[], inputs={"src": source})
            payload = json.loads(path.read_text(encoding="utf-8"))
            for key in ("schema", "timestamp_utc", "hostname", "git", "packages", "inputs", "run_hash"):
                self.assertIn(key, payload)
            self.assertEqual(payload["inputs"]["src"]["sha256"], file_sha256(source))
            self.assertIn("commit", payload["git"])


def _synthetic_speeches(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for country_index in range(30):
        for leader_index in range(2):
            for speech in range(3):
                score = float(rng.choice([0.0, 0.0, 0.2, 0.5, 1.2]))
                rows.append(
                    {
                        "id": f"c{country_index}_l{leader_index}_s{speech}",
                        "country": f"C{country_index:02d}",
                        "leader": f"C{country_index:02d}_L{leader_index}",
                        "source_score": score,
                        "extreme_label": 1 if score >= 1.0 else (0 if score == 0.0 else -1),
                    }
                )
    return pd.DataFrame(rows)


class HoldoutSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = _synthetic_speeches()
        self.rule = SplitRule(n_candidates=400, min_holdout_extreme_pos=2, min_dev_extreme_pos=4,
                              min_holdout_nonzero=5, min_holdout_leaders=5)

    def test_split_is_country_disjoint_and_meets_rule(self) -> None:
        choice = choose_country_split(self.frame, self.rule)
        split = split_frame(self.frame, choice["holdout_countries"])
        by_country = split.groupby("country")["split"].nunique()
        self.assertTrue((by_country == 1).all())
        stats = choice["stats"]
        self.assertGreaterEqual(stats["holdout_share"], self.rule.holdout_share_min)
        self.assertLessEqual(stats["holdout_share"], self.rule.holdout_share_max)
        self.assertGreaterEqual(stats["holdout_extreme_pos"], self.rule.min_holdout_extreme_pos)
        self.assertTrue(all(isinstance(item, str) for item in choice["holdout_countries"]))

    def test_split_is_deterministic(self) -> None:
        first = choose_country_split(self.frame, self.rule)
        second = choose_country_split(self.frame, self.rule)
        self.assertEqual(first["holdout_countries"], second["holdout_countries"])

    def test_impossible_rule_raises(self) -> None:
        rule = SplitRule(n_candidates=50, min_holdout_extreme_pos=10_000)
        with self.assertRaisesRegex(ValueError, "no country partition"):
            choose_country_split(self.frame, rule)


class EncoderSpecTests(unittest.TestCase):
    def test_templates(self) -> None:
        self.assertEqual(render_text(get_spec("e5-large-v2"), "hello"), "query: hello")
        self.assertEqual(render_text(get_spec("qwen3-emb-0.6b"), "hello", task="T"), "Instruct: T\nQuery:hello")
        self.assertEqual(render_text(get_spec("qwen3-emb-0.6b-noinstr"), "hello", task="T"), "hello")
        self.assertEqual(render_text(get_spec("me5-large-instruct"), "hello", task="T"), "Instruct: T\nQuery: hello")
        self.assertEqual(render_text(get_spec("gte-large-en-v1.5"), "hello"), "hello")

    def test_qwen_uses_last_token_pooling(self) -> None:
        for label, spec in ENCODER_SPECS.items():
            if spec.model_id.startswith("Qwen/"):
                self.assertEqual(spec.pooling, "last", label)

    def test_overrides_and_unknown(self) -> None:
        self.assertEqual(get_spec("e5-large-v2", batch_size=16).batch_size, 16)
        self.assertEqual(get_spec("e5-large-v2", batch_size=None).batch_size, 64)
        with self.assertRaises(KeyError):
            get_spec("not-a-model")


class SentenceLoaderTests(unittest.TestCase):
    def test_v3_proposition_file(self) -> None:
        schema, groups = load_sentence_groups(ROOT / "anchors" / "proposition_anchors.json")
        self.assertTrue(schema.startswith("populism_axis.v3.proposition_anchors"))
        self.assertEqual(sorted(groups), sorted(f"{c}__{p}" for c in "PEM" for p in ("populist", "mainstream")))
        self.assertTrue(all(len(items) == 10 for items in groups.values()))

    def test_v4_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pairs = Path(tmpdir) / "pairs.json"
            pairs.write_text(json.dumps({
                "schema": "populism_axis.v4.pair_anchors.v1",
                "channels": {"E": {"pairs": [{"populist": "a", "matched_mainstream": "b"}]}},
            }), encoding="utf-8")
            _, groups = load_sentence_groups(pairs)
            self.assertEqual(groups, {"E__populist": ["a"], "E__mainstream": ["b"]})
            nuisance = Path(tmpdir) / "nuisance.json"
            nuisance.write_text(json.dumps({
                "schema": "populism_axis.v4.nuisance_sets.v1",
                "sets": {"incivility": {"pairs": [{"nuisance": "x", "neutral": "y"}]}},
            }), encoding="utf-8")
            _, groups = load_sentence_groups(nuisance)
            self.assertEqual(groups, {"incivility__nuisance": ["x"], "incivility__neutral": ["y"]})
            bad = Path(tmpdir) / "bad.json"
            bad.write_text(json.dumps({"schema": "unknown"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_sentence_groups(bad)


if __name__ == "__main__":
    unittest.main()
