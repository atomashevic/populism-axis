#!/usr/bin/env python3
"""Paired leader-clustered comparison of candidate measures.

Reports the difference in Spearman correlation against the GPD rubric between
each pair of methods, resampling leaders (not speeches) so the interval respects
the clustering in the corpus.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]


def paired_delta(
    left: np.ndarray,
    right: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    draws: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    index_by_group = {group: np.flatnonzero(groups == group) for group in unique}
    observed = float(spearmanr(left, target).statistic - spearmanr(right, target).statistic)
    samples = []
    for _ in range(draws):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([index_by_group[group] for group in drawn])
        if len(np.unique(target[indices])) < 2:
            continue
        try:
            delta = spearmanr(left[indices], target[indices]).statistic - spearmanr(
                right[indices], target[indices]
            ).statistic
        except ValueError:
            continue
        if np.isfinite(delta):
            samples.append(delta)
    array = np.asarray(samples)
    return {
        "delta_spearman": observed,
        "ci95": [float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))],
        "p_two_sided": float(2.0 * min((array <= 0).mean(), (array >= 0).mean())),
        "n_draws": int(len(array)),
    }
