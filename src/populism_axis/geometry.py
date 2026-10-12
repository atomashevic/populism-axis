"""Axis construction and embedding-space transforms for the pipeline, on stored embeddings.

All functions are pure numpy and run on the CPU from raw pooled embeddings
written by `scripts/encode_windows.py` and `encode_sentences.py`.

The pole-mean rule (`transform="v3"`) is reproduced exactly: anchor sentences are
L2-normalized without centering, each channel axis is
normalize(mean(populist) - mean(mainstream)), and window embeddings are
centered on their own dataset mean and then L2-normalized.

The step-2 transform (`transform="ref"`) instead passes windows, anchors and
nuisance sentences through one label-free `ReferenceTransform` fitted on the
scored dataset's windows: centering, removal of top principal directions,
optional shrinkage whitening, L2 normalization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

CHANNELS = ("P", "E", "M")


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms <= 1e-12] = 1.0
    return matrix / norms


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("cannot normalize a zero vector")
    return vector / norm


def center_normalize(matrix: np.ndarray, mean: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Subtract `mean` (default: the matrix's own mean) and L2-normalize rows."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if mean is None:
        mean = matrix.mean(axis=0)
    return unit_rows(matrix - mean), mean


def pole_mean_axes(
    groups: Mapping[str, np.ndarray],
    channels: tuple[str, ...] = CHANNELS,
) -> dict[str, np.ndarray]:
    """pole-mean axis rule: normalize(mean(unit populist) - mean(unit mainstream)) per channel."""
    axes: dict[str, np.ndarray] = {}
    for channel in channels:
        populist = unit_rows(groups[f"{channel}__populist"])
        mainstream = unit_rows(groups[f"{channel}__mainstream"])
        axes[channel] = normalize(populist.mean(axis=0) - mainstream.mean(axis=0))
    return axes


def axis_gram(axes: Mapping[str, np.ndarray]) -> dict[str, dict[str, float]]:
    keys = list(axes)
    return {a: {b: float(axes[a] @ axes[b]) for b in keys} for a in keys}


def difference_parallelism(populist: np.ndarray, mainstream: np.ndarray) -> float:
    """Mean pairwise cosine among index-paired difference vectors (anchor QA, not a selection rule)."""
    differences = unit_rows(unit_rows(populist) - unit_rows(mainstream))
    gram = differences @ differences.T
    n = len(differences)
    return float((gram.sum() - n) / (n * (n - 1))) if n > 1 else float("nan")


# ---------------------------------------------------------------------------
# Step 2: label-free reference transforms and nuisance residualization.
# ---------------------------------------------------------------------------


@dataclass
class ReferenceTransform:
    """Label-free transform fitted on one dataset's window embeddings.

    Applied identically to windows, anchor sentences and nuisance sentences:
    subtract `mean`, remove the top principal directions `pcs` (rows, unit
    norm), optionally multiply by a shrinkage whitener, then L2-normalize.
    """

    mean: np.ndarray
    pcs: np.ndarray
    whitener: np.ndarray | None = None
    info: dict = field(default_factory=dict)


def fit_reference_transform(
    matrix: np.ndarray, n_pcs: int = 0, whiten_shrinkage: float | None = None
) -> ReferenceTransform:
    """Fit mean, top-`n_pcs` principal directions and an optional ZCA whitener.

    Whitening uses Sigma_a = (1 - a) Sigma + a (tr(Sigma)/D) I on the centered,
    PC-removed data and W = Sigma_a^{-1/2}; `whiten_shrinkage=None` skips it.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    dim = centered.shape[1]
    pcs = np.zeros((0, dim))
    explained: list[float] = []
    if n_pcs > 0:
        covariance = centered.T @ centered / len(centered)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        pcs = eigenvectors[:, order[:n_pcs]].T
        explained = (eigenvalues[order[:n_pcs]] / eigenvalues.sum()).tolist()
        centered = centered - (centered @ pcs.T) @ pcs
    whitener = None
    if whiten_shrinkage is not None:
        covariance = centered.T @ centered / len(centered)
        target = np.trace(covariance) / dim
        shrunk = (1.0 - whiten_shrinkage) * covariance + whiten_shrinkage * target * np.eye(dim)
        eigenvalues, eigenvectors = np.linalg.eigh(shrunk)
        eigenvalues = np.clip(eigenvalues, 1e-12, None)
        whitener = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    return ReferenceTransform(
        mean=mean,
        pcs=pcs,
        whitener=whitener,
        info={"n_pcs": int(n_pcs), "explained_variance_of_removed_pcs": explained,
              "whiten_shrinkage": whiten_shrinkage, "n_fit_rows": int(len(matrix))},
    )


def apply_transform(matrix: np.ndarray, transform: ReferenceTransform) -> np.ndarray:
    centered = np.asarray(matrix, dtype=np.float64) - transform.mean
    if len(transform.pcs):
        centered = centered - (centered @ transform.pcs.T) @ transform.pcs
    if transform.whitener is not None:
        centered = centered @ transform.whitener
    return unit_rows(centered)


def transformed_pole_axes(
    groups: Mapping[str, np.ndarray],
    transform: ReferenceTransform,
    channels: tuple[str, ...] = CHANNELS,
) -> dict[str, np.ndarray]:
    """normalize(mean(T(populist)) - mean(T(mainstream))) with anchors in the window space.

    For index-paired anchor sets this equals the normalized mean of pair differences.
    """
    return {
        channel: normalize(
            apply_transform(groups[f"{channel}__populist"], transform).mean(axis=0)
            - apply_transform(groups[f"{channel}__mainstream"], transform).mean(axis=0)
        )
        for channel in channels
    }


def nuisance_directions(
    groups: Mapping[str, np.ndarray], transform: ReferenceTransform
) -> dict[str, np.ndarray]:
    """One direction per nuisance set: normalize(mean(T(nuisance)) - mean(T(neutral)))."""
    names = sorted({name.rsplit("__", 1)[0] for name in groups if name.endswith("__nuisance")})
    return {
        name: normalize(
            apply_transform(groups[f"{name}__nuisance"], transform).mean(axis=0)
            - apply_transform(groups[f"{name}__neutral"], transform).mean(axis=0)
        )
        for name in names
    }


def residualize_axes(
    axes: Mapping[str, np.ndarray], directions: Mapping[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict]:
    """Project the span of the nuisance directions out of every axis (QR), re-normalize.

    Mirrors `v21_geometry.residualize_against_controls`. Returns the new axes and
    a diagnostics dict with each axis's cosine to each nuisance direction before
    and after, and the share of each axis's norm that was removed.
    """
    names = sorted(directions)
    basis, _ = np.linalg.qr(np.vstack([directions[name] for name in names]).T)
    out: dict[str, np.ndarray] = {}
    diagnostics: dict = {"cosine_before": {}, "cosine_after": {}, "removed_norm_share": {}}
    for channel, axis in axes.items():
        residual = axis - basis @ (basis.T @ axis)
        norm = float(np.linalg.norm(residual))
        if norm <= 1e-9:
            raise ValueError(f"axis {channel} lies inside the nuisance span")
        out[channel] = residual / norm
        diagnostics["cosine_before"][channel] = {n: float(axis @ directions[n]) for n in names}
        diagnostics["cosine_after"][channel] = {n: float(out[channel] @ directions[n]) for n in names}
        diagnostics["removed_norm_share"][channel] = float(1.0 - norm**2)
    return out, diagnostics
