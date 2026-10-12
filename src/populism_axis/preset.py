"""Exact numerical helpers extracted from the historical frozen-reference scorer."""
import numpy as np
from .geometry import unit_rows
CH = ('P', 'E')
SOFTNESS = 0.25


def nonzero_median(values):
    positive = values[values > 0.0]
    return float(np.median(positive)) if len(positive) else 1.0


def standardize(values, ref):
    mean, sd = (float(values.mean()), float(values.std())) if ref is None else ref
    return (values - mean) / sd if sd > 1e-12 else np.zeros_like(values)


def window_quantities(matrix, axes, stats):
    used = {}
    mu = matrix.mean(axis=0) if stats is None else np.asarray(stats['mu'])
    used['mu'] = mu
    normalized = unit_rows(matrix - mu)
    z, act, used['proj'], used['act'] = {}, {}, {}, {}
    for c in CH:
        proj = normalized @ axes[c]
        mean, sd = (float(proj.mean()), float(proj.std())) if stats is None else stats['proj'][c]
        z[c] = standardize(proj, (mean, sd))
        value = np.log1p(np.exp(z[c] / SOFTNESS)) ** 2
        scale = nonzero_median(value) if stats is None else stats['act'][c]
        act[c] = value / scale
        used['proj'][c], used['act'][c] = (mean, sd), scale
    pe = np.sqrt(act['P'] * act['E'])
    return {'PE_axis': pe, 'P_axis': z['P'], 'E_axis': z['E']}, used


def top_windows(window_score, doc_index, n, k):
    order = np.lexsort((-window_score, doc_index))
    sorted_docs = doc_index[order]
    return {doc: order[np.searchsorted(sorted_docs, doc):np.searchsorted(sorted_docs, doc, side='right')][:k] for doc in range(n)}
