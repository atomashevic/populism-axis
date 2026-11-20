"""Check that bootstrap resamples keep every stratum and its size, or draw whole strata."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from populism_axis.resampling import sampler  # noqa: E402


class ResamplingTests(unittest.TestCase):
    strata = np.array(['b', 'a', 'b', 'c', 'a', 'b', 'c', 'c', 'c'])

    def test_within_keeps_strata_and_sizes(self):
        draw, rng = sampler(self.strata, 'within'), np.random.default_rng(0)
        for _ in range(50):
            ix = draw(rng)
            self.assertEqual(len(ix), len(self.strata))
            for s in np.unique(self.strata):
                self.assertEqual((self.strata[ix] == s).sum(), (self.strata == s).sum())

    def test_within_draws_with_replacement(self):
        draw, rng = sampler(self.strata, 'within'), np.random.default_rng(1)
        self.assertTrue(any(len(np.unique(draw(rng))) < len(self.strata) for _ in range(50)))

    def test_one_stratum_is_an_ordinary_bootstrap(self):
        draw, rng = sampler(np.full(1000, 'all'), 'within'), np.random.default_rng(2)
        ix = draw(rng)
        self.assertEqual(len(ix), 1000)
        self.assertLess(len(np.unique(ix)), 750)

    def test_strata_draws_whole_strata(self):
        draw, rng = sampler(self.strata, 'strata'), np.random.default_rng(3)
        members = {s: set(np.flatnonzero(self.strata == s)) for s in np.unique(self.strata)}
        for _ in range(50):
            ix = list(draw(rng))
            while ix:
                s = self.strata[ix[0]]
                block = ix[:len(members[s])]
                self.assertEqual(set(block), members[s])
                ix = ix[len(block):]

    def test_unknown_scheme(self):
        with self.assertRaises(ValueError):
            sampler(self.strata, 'speakers')


if __name__ == '__main__':
    unittest.main()
