"""Bootstrap resampling of texts that share a speaker, leader, state-term, politician or country.

The benchmark intervals resample texts within these strata: every stratum keeps its texts' number,
and the texts are drawn with replacement inside it. The intervals therefore describe uncertainty
about the texts given the speakers of a benchmark. Resampling whole strata instead treats the
speakers as sampled; with the five EU-OOD speakers or the six AgoraSpeech politicians that interval
rests on few distinct resamples. A benchmark without speaker identifiers forms one stratum, which
makes the within-strata draw an ordinary bootstrap of its texts.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

SCHEMES = ("within", "strata")


def sampler(strata: np.ndarray, scheme: str = "within") -> Callable[[np.random.Generator], np.ndarray]:
    """A function that returns the indices of one bootstrap resample from a random generator.

    ``within`` draws texts with replacement inside each stratum, with one generator call per
    resample; ``strata`` draws whole strata with replacement, as many as there are strata.
    """
    _, inverse = np.unique(np.asarray(strata), return_inverse=True)
    if scheme == "within":
        order = np.argsort(inverse, kind="stable")
        sizes = np.bincount(inverse)
        offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        of_position = inverse[order]
        start, size = offsets[of_position], sizes[of_position]
        return lambda rng: order[start + (rng.random(len(order)) * size).astype(np.int64)]
    if scheme == "strata":
        members = [np.flatnonzero(inverse == i) for i in range(inverse.max() + 1)]
        return lambda rng: np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
    raise ValueError(f"unknown resampling scheme {scheme!r}; expected one of {SCHEMES}")
