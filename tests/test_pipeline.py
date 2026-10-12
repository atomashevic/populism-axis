"""Tests for stage 1: splitter cap, pole-mean axis rule, aggregation, document selection, adoption rule."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

from populism_axis.contextual_axis import (  # noqa: E402
    WindowProjections,
    _chunk_long_window,
    aggregate_scores,
    split_windows,
)
from populism_axis.geometry import center_normalize, difference_parallelism, pole_mean_axes, unit_rows  # noqa: E402


def _words(n: int, prefix: str = "w") -> list[str]:
    return [f"{prefix}{i}" for i in range(n)]


class SplitterTests(unittest.TestCase):
    def test_no_cap_matches_v3(self) -> None:
        text = "First sentence here. Second one! " + " ".join(_words(150)) + ". Short end?"
        v3_parts = ["First sentence here.", "Second one!", " ".join(_words(150)) + ".", "Short end?"]
        self.assertEqual(split_windows(text), v3_parts)
        self.assertEqual(split_windows(text, max_words=None), v3_parts)
        self.assertEqual(split_windows(text, max_words=0), v3_parts)

    def test_comma_rich_sentence_is_chunked_on_clauses(self) -> None:
        clauses = [" ".join(_words(12, f"c{k}_")) + "," for k in range(15)]
        sentence = " ".join(clauses)[:-1] + "."
        chunks = _chunk_long_window(sentence, 60)
        self.assertTrue(all(len(chunk.split()) <= 60 for chunk in chunks))
        self.assertEqual(" ".join(chunks).split(), sentence.split())
        # Clause packing: 12-word clauses pack five to a chunk.
        self.assertEqual([len(chunk.split()) for chunk in chunks], [60, 60, 60])

    def test_punctuation_free_run_is_cut_into_spans(self) -> None:
        run = " ".join(_words(130))
        chunks = _chunk_long_window(run, 60)
        self.assertEqual([len(chunk.split()) for chunk in chunks], [60, 60, 10])
        self.assertEqual(" ".join(chunks), run)

    def test_newlines_are_clause_boundaries_only_inside_long_segments(self) -> None:
        short = "Line one\nline two."
        self.assertEqual(split_windows(short, max_words=60), ["Line one\nline two."])
        long_text = "\n".join(" ".join(_words(40, f"l{k}_")) for k in range(3))
        chunks = split_windows(long_text, max_words=60)
        self.assertEqual([len(chunk.split()) for chunk in chunks], [40, 40, 40])

    def test_cap_applies_before_multi_sentence_grouping(self) -> None:
        text = "A b c. " + " ".join(_words(100)) + ". D e f."
        windows = split_windows(text, sentences_per_window=2, max_words=60)
        self.assertTrue(all(len(window.split()) <= 120 for window in windows))
        self.assertEqual(len(windows), 3)  # parts: [A b c.], [60 words], [40 words.], [D e f.] -> 3 pairs

    @unittest.skipUnless((ROOT / "data" / "speeches_20251120").is_dir(),
                         "GPD speech texts are not redistributed (data/README.md)")
    def test_known_single_window_speeches_are_split(self) -> None:
        frame = pd.read_csv(ROOT / "data" / "benchmarks" / "english_gpd_speeches.csv").set_index("id")
        for speech_id in ("UK_Brown_Famous_1", "Canada_Harper_Campaign_2", "Canada_Harper_Ribbon_2"):
            text = (ROOT / frame.loc[speech_id, "path"]).read_text(encoding="utf-8", errors="ignore")
            self.assertEqual(len(split_windows(text)), 1, speech_id)
            capped = split_windows(text, max_words=60)
            self.assertGreaterEqual(len(capped), 9, speech_id)
            self.assertTrue(all(len(window.split()) <= 60 for window in capped))
            self.assertEqual(" ".join(capped).split(), text.split())


class GeometryTests(unittest.TestCase):
    def test_pole_mean_axes_matches_v3_formula(self) -> None:
        rng = np.random.default_rng(0)
        groups = {f"{c}__{p}": rng.normal(size=(4, 8)) for c in "PEM" for p in ("populist", "mainstream")}
        axes = pole_mean_axes(groups)
        for channel in "PEM":
            pop = groups[f"{channel}__populist"]
            main = groups[f"{channel}__mainstream"]
            pop = pop / np.linalg.norm(pop, axis=1, keepdims=True)
            main = main / np.linalg.norm(main, axis=1, keepdims=True)
            expected = pop.mean(0) - main.mean(0)
            np.testing.assert_allclose(axes[channel], expected / np.linalg.norm(expected))

    def test_center_normalize(self) -> None:
        matrix = np.array([[3.0, 0.0], [1.0, 0.0], [2.0, 2.0]])
        normalized, mean = center_normalize(matrix)
        np.testing.assert_allclose(mean, [2.0, 2.0 / 3.0])
        np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1.0)

    def test_parallelism_of_identical_differences_is_one(self) -> None:
        main = np.eye(3)
        pop = main + np.array([0.0, 0.0, 5.0])
        self.assertAlmostEqual(difference_parallelism(pop, main), difference_parallelism(pop, main))
        self.assertTrue(np.all(np.isfinite(unit_rows(pop))))


class AggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        # Document 0 has one window strong on both channels; document 1 is strong on P only.
        self.projections = WindowProjections(
            doc_index=np.array([0, 0, 1, 1]),
            windows=["a", "b", "c", "d"],
            projections={
                "P": np.array([2.0, -1.0, 2.0, -1.0]),
                "E": np.array([2.0, -1.0, -1.0, 0.0]),
                "M": np.zeros(4),
            },
        )

    def test_soft_conjunction_ranks_joint_activation_first(self) -> None:
        scores = aggregate_scores(self.projections, 2, {"P": 1.0, "E": 1.0, "M": 0.0}, top_k=1, conjunction="soft")
        self.assertGreater(scores[0], scores[1])

    def test_hard_conjunction_gates_documents_without_both_channels(self) -> None:
        scores = aggregate_scores(self.projections, 2, {"P": 1.0, "E": 1.0, "M": 0.0}, top_k=2, conjunction="hard")
        self.assertGreater(scores[0], 0.0)
        self.assertEqual(scores[1], 0.0)


class RunConfigTests(unittest.TestCase):
    def test_select_documents_reindexes_windows(self) -> None:
        from run_config import select_documents

        frame = pd.DataFrame({"id": ["a", "b", "c"]})
        matrix = np.arange(12, dtype=float).reshape(6, 2)
        doc_index = np.array([0, 0, 1, 2, 2, 2])
        windows = pd.DataFrame({"window": list("uvwxyz")})
        kept_frame, kept_matrix, kept_index, kept_windows = select_documents(
            frame, matrix, doc_index, windows, np.array([True, False, True])
        )
        self.assertEqual(kept_frame["id"].tolist(), ["a", "c"])
        self.assertEqual(kept_index.tolist(), [0, 0, 1, 1, 1])
        self.assertEqual(kept_windows["window"].tolist(), list("uvxyz"))
        np.testing.assert_array_equal(kept_matrix, matrix[[0, 1, 3, 4, 5]])

    def test_adoption_rule(self) -> None:
        from compare_steps import decide

        good = {"delta_spearman": 0.02, "ci95": [-0.05, 0.09]}
        flat = {"delta_spearman": 0.0, "ci95": [-0.03, 0.03]}
        harmful = {"delta_spearman": -0.05, "ci95": [-0.09, -0.01]}
        self.assertEqual(decide(good, flat, "standard")["outcome"], "ADOPT")
        self.assertEqual(decide(flat, flat, "standard")["outcome"], "KEEP PREVIOUS")
        self.assertEqual(decide(good, harmful, "standard")["outcome"], "KEEP PREVIOUS")
        self.assertTrue(decide(harmful, harmful, "defect_fix")["outcome"].startswith("ADOPT"))


class TextCleaningTests(unittest.TestCase):
    def test_header_block_and_urls_removed(self) -> None:
        from populism_axis.text import clean_speech_text

        raw = (
            "﻿Name of politician: A Leader\nTitle of Speech: Opening\nDate of Speech: 1 May 2010\n"
            "Links: https://example.org/x\n\n\nFriends, thank you for coming.\nWe will build roads.\n\n"
            "https://example.org/y\nCategory: Ribbon"
        )
        cleaned, log = clean_speech_text(raw)
        self.assertEqual(cleaned, "Friends, thank you for coming.\nWe will build roads.")
        self.assertEqual(len(log.removed_metadata), 5)
        self.assertEqual(len(log.removed_url), 1)

    def test_speaker_labels_and_prose_kept(self) -> None:
        from populism_axis.text import clean_speech_text

        raw = "THE PRESIDENT: Thank you all.\nAUDIENCE: Yes we can!\nFirst of all: it is good to be here."
        cleaned, log = clean_speech_text(raw)
        self.assertEqual(cleaned, raw)
        self.assertEqual(log.removed_metadata, [])

    def test_date_key_only_removed_near_edges(self) -> None:
        from populism_axis.text import clean_speech_text

        body = "\n".join(f"Sentence number {i} of the speech." for i in range(30))
        middle = "Opening line.\n" + "\n".join(["x."] * 20) + "\nDate: we act now.\n" + body
        self.assertIn("Date: we act now.", clean_speech_text(middle)[0])
        self.assertNotIn("Date:", clean_speech_text("Date: 5 May\nHello there.")[0])

    def test_chinese_lines_removed_and_stray_ideographs_stripped(self) -> None:
        from populism_axis.text import clean_speech_text

        raw = "Good morning.\n令人遺憾的是，中華民國積極參與國際活動。\nThe Republic of China (中華民國) welcomes you."
        cleaned, log = clean_speech_text(raw)
        self.assertEqual(log.removed_cjk_lines, 1)
        self.assertNotRegex(cleaned, r"[一-鿿]")
        self.assertIn("Good morning.", cleaned)
        self.assertIn("The Republic of China", cleaned)

    def test_long_source_line_with_url_removed(self) -> None:
        from populism_axis.text import clean_speech_text

        line = "Source: " + "https://example.org/" + "a" * 300
        self.assertEqual(clean_speech_text("Hello.\n" + "\n".join(["Body."] * 30) + "\n" + line + "\n" + "\n".join(["More."] * 20))[1].removed_metadata, [line])


if __name__ == "__main__":
    unittest.main()


class Step2GeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(1)
        base = rng.normal(size=(500, 8))
        base[:, 0] *= 10.0  # one dominant direction
        self.matrix = base

    def test_pc_removal_zeroes_dominant_direction(self) -> None:
        from populism_axis.geometry import apply_transform, fit_reference_transform

        transform = fit_reference_transform(self.matrix, n_pcs=1)
        self.assertGreater(abs(transform.pcs[0, 0]), 0.99)
        out = apply_transform(self.matrix, transform)
        self.assertLess(float(np.abs(out @ transform.pcs[0]).max()), 1e-9)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0)

    def test_zero_pcs_no_whitening_is_center_normalize(self) -> None:
        from populism_axis.geometry import apply_transform, fit_reference_transform

        transform = fit_reference_transform(self.matrix, n_pcs=0)
        expected, _ = center_normalize(self.matrix)
        np.testing.assert_allclose(apply_transform(self.matrix, transform), expected)

    def test_full_whitening_gives_identity_covariance(self) -> None:
        from populism_axis.geometry import fit_reference_transform

        transform = fit_reference_transform(self.matrix, n_pcs=0, whiten_shrinkage=0.0)
        white = (self.matrix - transform.mean) @ transform.whitener
        np.testing.assert_allclose(white.T @ white / len(white), np.eye(8), atol=1e-8)

    def test_residualized_axes_orthogonal_and_unit(self) -> None:
        from populism_axis.geometry import residualize_axes

        rng = np.random.default_rng(2)
        axes = {c: rng.normal(size=8) for c in "PE"}
        axes = {c: v / np.linalg.norm(v) for c, v in axes.items()}
        directions = {f"n{i}": rng.normal(size=8) for i in range(3)}
        directions = {k: v / np.linalg.norm(v) for k, v in directions.items()}
        out, diagnostics = residualize_axes(axes, directions)
        for channel, axis in out.items():
            self.assertAlmostEqual(float(np.linalg.norm(axis)), 1.0)
            for direction in directions.values():
                self.assertLess(abs(float(axis @ direction)), 1e-9)
            self.assertGreater(diagnostics["removed_norm_share"][channel], 0.0)
        with self.assertRaises(ValueError):
            residualize_axes({"P": directions["n0"]}, directions)

    def test_transformed_pairs_equal_mean_of_pair_differences(self) -> None:
        from populism_axis.geometry import apply_transform, fit_reference_transform, transformed_pole_axes

        rng = np.random.default_rng(3)
        groups = {"P__populist": rng.normal(size=(5, 8)), "P__mainstream": rng.normal(size=(5, 8))}
        transform = fit_reference_transform(self.matrix, n_pcs=2)
        axis = transformed_pole_axes(groups, transform, channels=("P",))["P"]
        diffs = apply_transform(groups["P__populist"], transform) - apply_transform(groups["P__mainstream"], transform)
        expected = diffs.mean(axis=0)
        np.testing.assert_allclose(axis, expected / np.linalg.norm(expected))


class PoolingRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scores = np.array([5.0, 1.0, 3.0, 2.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.doc_index = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    def test_k_frac_mean(self) -> None:
        from populism_axis.contextual_axis import aggregate_from_window_scores as pool

        np.testing.assert_allclose(pool(self.scores, self.doc_index, 2, "k:2"), [4.0, 5.0])
        np.testing.assert_allclose(pool(self.scores, self.doc_index, 2, "frac:0.5"), [4.0, 10.0 / 3.0])
        np.testing.assert_allclose(pool(self.scores, self.doc_index, 2, "frac:0.01"), [5.0, 10.0])
        np.testing.assert_allclose(pool(self.scores, self.doc_index, 2, "mean"), [2.75, 10.0 / 6.0])

    def test_softcount_is_share_above_global_quantile(self) -> None:
        from populism_axis.contextual_axis import aggregate_from_window_scores as pool

        # 0.5-quantile of all ten scores is 0.5; doc 0 has 4/4 above, doc 1 has 1/6.
        np.testing.assert_allclose(pool(self.scores, self.doc_index, 2, "softcount:0.5"), [1.0, 1.0 / 6.0])

    def test_empty_document_and_unknown_rule(self) -> None:
        from populism_axis.contextual_axis import aggregate_from_window_scores as pool

        self.assertEqual(pool(self.scores, self.doc_index, 3, "k:3")[2], 0.0)
        with self.assertRaises(ValueError):
            pool(self.scores, self.doc_index, 2, "median")
