"""Behaviour checks for comparator matching and unchanged NLI pooling."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_apsi_comparison import aligned, comparison, loso_predictions, pool_nli


class ComparatorEvaluationTests(unittest.TestCase):
    def test_nli_conjunction_precedes_document_mean(self):
        req = pd.DataFrame({'doc_index': [0, 0, 0, 1], 'window_rank': [0, 1, 2, 0]})
        hyp = [{'channel': 'P'}, {'channel': 'P'}, {'channel': 'E'}]
        probs = np.array([[.25, .1, 1], [1, .2, .25], [1, 1, 1], [.04, .01, .25]])
        scores = pool_nli(req, probs, hyp, 2, 2)
        np.testing.assert_allclose(scores['PE'], [.5, .1])
        np.testing.assert_allclose(scores['P'], [.625, .04])

    def test_abstention_removed_from_both_sides(self):
        y = np.array([0, 0, 1, 1]); groups = np.arange(4)
        result = comparison(y, np.array([0., np.nan, 1., 2.]),
                            np.array([0., 100., 1., 2.]), groups, 'auroc', 50, 1)
        self.assertEqual(result['n_matched'], 3)
        self.assertEqual(result['delta'], 0)
        self.assertEqual(result['delta_ci95'], [0, 0])

    def test_no_scored_rows_is_explicit(self):
        result = comparison(np.array([0, 1]), np.array([np.nan, np.nan]),
                            np.array([0., 1.]), np.array(['a', 'b']), 'auroc', 10, 1)
        self.assertEqual(result['n_matched'], 0)
        self.assertEqual(result['valid_draws'], 0)
        self.assertTrue(np.isnan(result['delta']))

    def test_scalar_loso_can_have_distinct_component_thresholds(self):
        f = pd.DataFrame({'speaker': ['a'] * 4 + ['b'] * 4,
                          'pop_code': [0, 0, 1, 3] * 2})
        score = np.array([0., 1., 2., 3.] * 2)
        truth, pred = loso_predictions(f, score, score)
        np.testing.assert_array_equal(pred, truth)
        score[0] = np.nan
        _, pred = loso_predictions(f, score, score)
        self.assertTrue(np.isnan(pred[0]).all())

    def test_alignment_does_not_accept_reordered_rows(self):
        with self.assertRaises(ValueError):
            aligned(pd.DataFrame({'id': ['a', 'b']}), pd.DataFrame({'id': ['b', 'a']}))


if __name__ == '__main__':
    unittest.main()
