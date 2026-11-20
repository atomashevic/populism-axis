"""Check APSI abstention boundaries, weighting, and declared chunk pooling."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('apsi_comparator', ROOT / 'scripts/apsi_comparator.py')
apsi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(apsi)


class APSIScoringTests(unittest.TestCase):
    def setUp(self):
        self.config = apsi.load_config()

    def test_pinned_inventory(self):
        self.assertEqual(len(self.config['topic_hypotheses']), 25)
        self.assertEqual(sum(h['direction'] == 'populist' for h in self.config['hypotheses']), 12)
        self.assertEqual(sum(h['direction'] == 'pluralist' for h in self.config['hypotheses']), 11)

    def test_topic_boundary_and_no_hypothesis(self):
        self.assertEqual(apsi.score_probabilities([.599999], [], self.config)['abstention_reason'], 'topic')
        self.assertEqual(apsi.score_probabilities([.6], np.full(23, .15), self.config)['abstention_reason'],
                         'no_hypothesis_above_threshold')

    def test_floor_preserves_abstention(self):
        self.assertEqual(apsi.score_probabilities([.8], np.full(23, .249999), self.config)['abstention_reason'],
                         'weak_top_two')
        self.assertTrue(np.isfinite(apsi.score_probabilities([.8], np.full(23, .25), self.config)['score']))

    def test_adaptive_k_and_direction(self):
        # One fully entailed unit-weight populist claim, other probabilities zero.
        probs = np.zeros(23)
        probs[0] = 1
        self.assertAlmostEqual(apsi.score_probabilities([1], probs, self.config)['score'], 6.25)
        probs[0] = 0
        probs[2] = 1  # pluralist hypothesis weight .60, k=4
        self.assertAlmostEqual(apsi.score_probabilities([1], probs, self.config)['score'], 4.25)

    def test_full_side_averages_when_k_exceeds_side_count(self):
        # Expected value obtained independently from pinned upstream scorer.
        self.assertAlmostEqual(apsi.score_probabilities([1], np.ones(23), self.config)['score'],
                               5.297348484848484)

    def test_chunk_abstention_and_mean(self):
        results = [{'score': float('nan'), 'topic_probability': .2},
                   {'score': 8., 'topic_probability': .9}, {'score': 4., 'topic_probability': .7}]
        result = apsi.aggregate_chunks(results)
        self.assertEqual(result['score'], 6.)
        self.assertEqual((result['n_chunks'], result['n_scored_chunks']), (3, 2))
        self.assertTrue(np.isnan(apsi.aggregate_chunks(results[:1])['score']))


if __name__ == '__main__':
    unittest.main()
