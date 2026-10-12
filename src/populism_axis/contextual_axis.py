"""Contextual realization of the frozen V2.1 P/E/M axis geometry.

The construct, the anchor families, the channel structure and the local
conjunction aggregation are all inherited unchanged from the V2.1 freeze. The
only thing that changes is the representation: instead of averaging static
type-level vectors over the tokens of a window, each window is encoded once by
a frozen contextual sentence encoder and projected onto the same kind of
pole-difference direction.

This keeps the explainability surface intact. An axis is still a named list of
anchor pairs, a score is still a linear projection, and a document score is
still traceable to the specific sentences that activated each channel.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

CHANNELS = ("P", "E", "M")

# Carrier sentences place each anchor phrase in a short, politically neutral
# frame so that phrase encodings live in the same region of the space as the
# speech sentences they are projected against.
CARRIER_TEMPLATES = (
    "{phrase}",
    "This is about {phrase}.",
    "The real question here is {phrase}.",
    "What we are seeing in our politics is {phrase}.",
)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("cannot normalize a zero vector")
    return vector / norm


@dataclass(frozen=True)
class AnchorPair:
    family_id: str
    channel: str
    variant_id: str
    positive: str
    negative: str


@dataclass(frozen=True)
class ContextualAxes:
    encoder_id: str
    channel_axes: Mapping[str, np.ndarray]
    family_vectors: Mapping[str, Mapping[str, np.ndarray]]
    pairs_by_channel: Mapping[str, tuple[AnchorPair, ...]]
    anchor_sha256: str

    def channel_gram(self) -> dict[str, dict[str, float]]:
        return {
            left: {
                right: float(np.dot(self.channel_axes[left], self.channel_axes[right]))
                for right in CHANNELS
            }
            for left in CHANNELS
        }


def load_frozen_anchor_pairs(
    freeze_path: Path,
    inventory_path: Path,
    track: str = "raw",
    all_variants: bool = True,
) -> dict[str, tuple[AnchorPair, ...]]:
    """Read the exact anchor families frozen for the V2.1 R1 benchmark."""
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    families = {item["pair_family_id"]: item for item in inventory["pair_families"]}
    variants = {item["variant_id"]: item for item in inventory["pair_variants"]}
    terms = {item["term_id"]: item for item in inventory["terms"]}

    selected = freeze["selected_family_ids"][track]
    out: dict[str, tuple[AnchorPair, ...]] = {}
    for channel in CHANNELS:
        pairs: list[AnchorPair] = []
        for family_id in selected[channel]:
            family = families[family_id]
            variant_ids = family["variant_ids"] if all_variants else [family["canonical_variant_id"]]
            for variant_id in variant_ids:
                variant = variants[variant_id]
                pairs.append(
                    AnchorPair(
                        family_id=family_id,
                        channel=channel,
                        variant_id=variant_id,
                        positive=terms[variant["positive_term_id"]]["surface_form"],
                        negative=terms[variant["negative_term_id"]]["surface_form"],
                    )
                )
        out[channel] = tuple(pairs)
    return out


def _anchor_hash(pairs_by_channel: Mapping[str, Sequence[AnchorPair]]) -> str:
    payload = {
        channel: sorted(f"{pair.variant_id}|{pair.positive}|{pair.negative}" for pair in pairs)
        for channel, pairs in pairs_by_channel.items()
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def build_contextual_axes(
    pairs_by_channel: Mapping[str, Sequence[AnchorPair]],
    encode: Callable[[Sequence[str]], np.ndarray],
    encoder_id: str,
    templates: Sequence[str] = CARRIER_TEMPLATES,
) -> ContextualAxes:
    """Embed every anchor phrase in carrier sentences and average pole differences.

    Family vector = mean over that family's surface variants of
    (positive centroid - negative centroid). Channel axis = normalized mean of
    family vectors. This mirrors the V2.1 static construction one-for-one.
    """
    phrases: list[str] = []
    for pairs in pairs_by_channel.values():
        for pair in pairs:
            phrases.extend([pair.positive, pair.negative])
    unique_phrases = sorted(set(phrases))

    rendered: list[str] = []
    for phrase in unique_phrases:
        rendered.extend(template.format(phrase=phrase) for template in templates)
    matrix = encode(rendered)
    per_template = matrix.reshape(len(unique_phrases), len(templates), -1)
    centroid_by_phrase = {
        phrase: per_template[index].mean(axis=0)
        for index, phrase in enumerate(unique_phrases)
    }

    channel_axes: dict[str, np.ndarray] = {}
    family_vectors: dict[str, dict[str, np.ndarray]] = {}
    for channel, pairs in pairs_by_channel.items():
        grouped: dict[str, list[np.ndarray]] = {}
        for pair in pairs:
            difference = centroid_by_phrase[pair.positive] - centroid_by_phrase[pair.negative]
            grouped.setdefault(pair.family_id, []).append(difference)
        vectors: dict[str, np.ndarray] = {}
        for family_id, differences in grouped.items():
            vectors[family_id] = _normalize(np.vstack(differences).mean(axis=0))
        family_vectors[channel] = vectors
        stacked = np.vstack([vectors[key] for key in sorted(vectors)])
        channel_axes[channel] = _normalize(stacked.mean(axis=0))

    return ContextualAxes(
        encoder_id=encoder_id,
        channel_axes=channel_axes,
        family_vectors=family_vectors,
        pairs_by_channel={key: tuple(value) for key, value in pairs_by_channel.items()},
        anchor_sha256=_anchor_hash(pairs_by_channel),
    )


def channel_parallelism(axes: ContextualAxes, channel: str) -> float:
    """Mean pairwise cosine among a channel's family vectors (V2.1's quality metric)."""
    vectors = axes.family_vectors[channel]
    matrix = np.vstack([vectors[key] for key in sorted(vectors)])
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    gram = matrix @ matrix.T
    n = len(matrix)
    return float((gram.sum() - n) / (n * (n - 1)))


CLAUSE_SPLIT = re.compile(r"(?<=[;:,])\s+|\s*\n+\s*")


def _chunk_long_window(text: str, max_words: int) -> list[str]:
    """Split a segment longer than `max_words` words into pieces of at most `max_words`.

    Clause boundaries (`;`, `:`, `,` followed by whitespace, and line breaks) are
    preferred: clauses are packed greedily into chunks up to the cap. A single
    clause that is still over the cap is cut into fixed spans of `max_words` words.
    Joining the output with single spaces reproduces the input's words in order.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text] if words else []
    pieces: list[list[str]] = []
    for clause in CLAUSE_SPLIT.split(text):
        clause_words = clause.split()
        if not clause_words:
            continue
        if len(clause_words) > max_words:
            for start in range(0, len(clause_words), max_words):
                pieces.append(clause_words[start : start + max_words])
        else:
            pieces.append(clause_words)
    chunks: list[list[str]] = []
    for piece in pieces:
        if chunks and len(chunks[-1]) + len(piece) <= max_words:
            chunks[-1] = chunks[-1] + piece
        else:
            chunks.append(list(piece))
    return [" ".join(chunk) for chunk in chunks]


def split_windows(
    text: str,
    sentences_per_window: int = 1,
    stride: int | None = None,
    max_words: int | None = None,
) -> list[str]:
    """Split text into sentence windows.

    `max_words=None` reproduces the uncapped splitter exactly. With a cap, any sentence
    longer than `max_words` words is first chunked by `_chunk_long_window`, so a
    speech without sentence punctuation no longer becomes one window that the
    encoder truncates. The cap applies to sentences before multi-sentence grouping.
    """
    parts = [item.strip() for item in SENTENCE_SPLIT.split(str(text)) if item.strip()]
    if max_words is not None and max_words > 0:
        parts = [chunk for part in parts for chunk in _chunk_long_window(part, max_words)]
    if not parts:
        return [str(text)] if str(text).strip() else []
    if sentences_per_window <= 1:
        return parts
    step = stride if stride is not None else 1
    windows = []
    for start in range(0, max(1, len(parts) - sentences_per_window + 1), step):
        windows.append(" ".join(parts[start : start + sentences_per_window]))
    return windows or [" ".join(parts)]


@dataclass
class WindowProjections:
    """Per-document window projections onto each channel axis."""

    doc_index: np.ndarray
    windows: list[str]
    projections: dict[str, np.ndarray]


def project_windows(
    documents: Sequence[str],
    axes: ContextualAxes,
    encode: Callable[[Sequence[str]], np.ndarray],
    sentences_per_window: int = 1,
    center: bool = True,
) -> WindowProjections:
    """Encode every window once and project onto all three channel axes.

    Centering removes the dataset mean embedding before projection. Contextual
    spaces are strongly anisotropic; without this the projection is dominated by
    a common component shared by every sentence.
    """
    windows: list[str] = []
    doc_index: list[int] = []
    for index, document in enumerate(documents):
        parts = split_windows(document, sentences_per_window)
        windows.extend(parts)
        doc_index.extend([index] * len(parts))

    matrix = encode(windows) if windows else np.zeros((0, 1), dtype=float)
    if center and len(matrix):
        matrix = matrix - matrix.mean(axis=0, keepdims=True)
    if len(matrix):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms <= 1e-12] = 1.0
        matrix = matrix / norms

    projections = {
        channel: (matrix @ axes.channel_axes[channel] if len(matrix) else np.zeros(0, dtype=float))
        for channel in CHANNELS
    }
    return WindowProjections(
        doc_index=np.asarray(doc_index, dtype=int),
        windows=windows,
        projections=projections,
    )


def _standardize(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    std = float(values.std())
    if std <= 1e-12:
        return np.zeros_like(values)
    return (values - float(values.mean())) / std


def _nonzero_median(values: np.ndarray) -> float:
    positive = values[values > 0.0]
    return float(np.median(positive)) if len(positive) else 1.0


def window_conjunction_scores(
    projections: WindowProjections,
    weights: Mapping[str, float],
    tau: float = 0.0,
    conjunction: str = "hard",
    softness: float = 0.25,
    signs: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Window-level local-conjunction scores (the quantity `aggregate_scores` pools)."""
    active = [channel for channel in CHANNELS if weights.get(channel, 0.0) > 0.0]
    total = sum(weights[channel] for channel in active)
    normalized_weights = {channel: weights[channel] / total for channel in active}
    sign_map = {channel: 1.0 for channel in CHANNELS}
    if signs:
        sign_map.update({key: float(value) for key, value in signs.items()})

    activation: dict[str, np.ndarray] = {}
    for channel in active:
        z = sign_map[channel] * _standardize(projections.projections[channel])
        if conjunction == "hard":
            value = np.maximum(0.0, z - tau) ** 2
        else:
            value = np.log1p(np.exp((z - tau) / softness)) ** 2
        scale = _nonzero_median(value)
        activation[channel] = value / (scale if scale > 0 else 1.0)

    if activation:
        stacked = np.vstack([activation[channel] for channel in active])
        weight_vector = np.asarray([normalized_weights[channel] for channel in active], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            logs = np.log(np.where(stacked > 0.0, stacked, 1.0))
            window_scores = np.exp(weight_vector @ logs)
        window_scores = np.where((stacked <= 0.0).any(axis=0), 0.0, window_scores)
    else:
        window_scores = np.zeros(len(projections.doc_index), dtype=float)

    return window_scores


def aggregate_scores(
    projections: WindowProjections,
    n_documents: int,
    weights: Mapping[str, float],
    top_k: int = 3,
    tau: float = 0.0,
    conjunction: str = "hard",
    softness: float = 0.25,
    signs: Mapping[str, float] | None = None,
    top_rule: str | None = None,
) -> np.ndarray:
    """V2.1 local-conjunction aggregation applied to contextual projections.

    `top_rule` overrides `top_k` for document pooling (see `aggregate_from_window_scores`).

    Projections are z-scored within the dataset (label-free) so that `tau` is on
    a common scale across encoders. `conjunction='hard'` reproduces the frozen
    V2.1 rule where a window scores zero if any required channel is inactive.
    `conjunction='soft'` replaces the hinge with a softplus so that near-misses
    and cross-sentence splits keep contributing to the ranking.

    `signs` orients each channel toward the populist pole. The V2.1 inventory
    writes P and M with the virtue term as the positive pole but E with the
    pathology term as the positive pole, so a construct-faithful conjunction
    needs an explicit per-channel sign rather than the implicit +1 everywhere.
    """
    window_scores = window_conjunction_scores(projections, weights, tau, conjunction, softness, signs)
    return aggregate_from_window_scores(
        window_scores, projections.doc_index, n_documents, top_rule or f"k:{top_k}"
    )


def aggregate_from_window_scores(
    window_scores: np.ndarray, doc_index: np.ndarray, n_documents: int, rule: str = "k:3"
) -> np.ndarray:
    """Pool window scores into document scores.

    Rules:
      k:<n>          mean of the n highest windows (initial configuration: k:3)
      frac:<f>       mean of the ceil(f * n_windows) highest windows, at least one
      softcount:<q>  share of the document's windows above the dataset-wide
                     q-quantile of all window scores (a label-free pervasiveness readout)
      mean           mean of all windows
    """
    kind, _, value = rule.partition(":")
    threshold = None
    if kind == "softcount":
        threshold = float(np.quantile(window_scores, float(value))) if len(window_scores) else 0.0
    elif kind not in {"k", "frac", "mean"}:
        raise ValueError(f"unknown pooling rule {rule!r}")
    out = np.zeros(n_documents, dtype=float)
    order = np.argsort(doc_index, kind="stable")
    boundaries = np.searchsorted(doc_index[order], np.arange(n_documents + 1))
    for index in range(n_documents):
        rows = order[boundaries[index] : boundaries[index + 1]]
        if not len(rows):
            continue
        values = window_scores[rows]
        if kind == "softcount":
            out[index] = float((values > threshold).mean())
            continue
        if kind == "mean":
            out[index] = float(values.mean())
            continue
        values = np.sort(values)[::-1]
        take = int(value) if kind == "k" else max(1, int(np.ceil(float(value) * len(values))))
        out[index] = float(values[: min(take, len(values))].mean())
    return out


def aggregate_single_channel(
    projections: WindowProjections,
    n_documents: int,
    channel: str,
    top_k: int = 3,
    mode: str = "topk",
) -> np.ndarray:
    """Diagnostic aggregation for one channel with no conjunction gate."""
    values = _standardize(projections.projections[channel])
    out = np.zeros(n_documents, dtype=float)
    for index in range(n_documents):
        mask = projections.doc_index == index
        if not mask.any():
            continue
        subset = values[mask]
        if mode == "mean":
            out[index] = float(subset.mean())
        elif mode == "max":
            out[index] = float(subset.max())
        else:
            ordered = np.sort(subset)[::-1]
            out[index] = float(ordered[: min(top_k, len(ordered))].mean())
    return out


def top_evidence_windows(
    projections: WindowProjections,
    doc_index: int,
    weights: Mapping[str, float],
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Return the highest-scoring windows for one document, with channel detail.

    This is the explainability hook: every document score can be traced back to
    the specific sentences that drove it and the per-channel projection of each.
    """
    active = [channel for channel in CHANNELS if weights.get(channel, 0.0) > 0.0]
    standardized = {channel: _standardize(projections.projections[channel]) for channel in CHANNELS}
    mask = np.flatnonzero(projections.doc_index == doc_index)
    rows = []
    for position in mask:
        channel_values = {channel: float(standardized[channel][position]) for channel in CHANNELS}
        gate = min(channel_values[channel] for channel in active) if active else 0.0
        rows.append(
            {
                "window": projections.windows[position],
                "channel_z": channel_values,
                "min_active_channel_z": gate,
            }
        )
    rows.sort(key=lambda row: row["min_active_channel_z"], reverse=True)
    return rows[:limit]
