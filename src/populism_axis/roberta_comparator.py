"""Independent-label RoBERTa scores and a strict, versioned cache contract."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model_revisions import pinned_revision

SCHEMA = "populism_axis.roberta.sigmoid.v1"
LABELS = {"0": "(a) No populism", "1": "(b) Anti-elitism populism", "2": "(c) People-centrism populism"}


def validate_config(config) -> None:
    labels = {str(k): v for k, v in config.id2label.items()}
    if config.problem_type != "multi_label_classification" or config.num_labels != 3 or labels != LABELS:
        raise ValueError("RoBERTa comparator requires the verified three independent labels in their original order")


def independent_probabilities(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("Expected finite RoBERTa logits of shape (windows, 3)")
    return np.exp(-np.logaddexp(np.float32(0), -values))


def save_scores(path: Path, doc_index: np.ndarray, logits: np.ndarray, metadata: dict) -> None:
    """Create a new cache exclusively; never replace historical predictions."""
    probabilities = independent_probabilities(logits)
    metadata = dict(metadata, schema=SCHEMA, activation="sigmoid", id2label=LABELS)
    payload = dict(doc_index=np.asarray(doc_index, dtype=np.int32), logits=np.asarray(logits, dtype=np.float32),
                   probabilities=probabilities, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    _validate_payload(payload)
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **payload)


def _validate_payload(payload) -> dict:
    required = {"doc_index", "logits", "probabilities", "metadata_json"}
    if not required.issubset(payload):
        raise ValueError("Legacy/untagged RoBERTa cache rejected: rerun the comparator with independent sigmoid; do not relabel softmax files")
    metadata = json.loads(str(payload["metadata_json"].item()))
    if metadata.get("schema") != SCHEMA or metadata.get("activation") != "sigmoid" or metadata.get("id2label") != LABELS:
        raise ValueError("RoBERTa cache has incompatible activation/schema/labels")
    for field in ("model_revision", "tokenizer_revision"):
        pinned_revision(metadata.get(field, ""))
    expected = independent_probabilities(payload["logits"])
    probs = payload["probabilities"]
    index = payload["doc_index"]
    if index.shape != (len(expected),) or not np.issubdtype(index.dtype, np.integer) or (index < 0).any():
        raise ValueError("RoBERTa cache has invalid document indices")
    if probs.shape != expected.shape or not np.allclose(probs, expected, rtol=1e-6, atol=1e-7):
        raise ValueError("RoBERTa probabilities disagree with sigmoid(logits)")
    return metadata


def load_scores(path: Path) -> dict:
    """Reject old softmax caches even if copied under a corrected filename."""
    try:
        with np.load(path, allow_pickle=False) as cache:
            payload = {key: cache[key] for key in cache.files}
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Corrected RoBERTa cache missing: {path}; retained probabilities_*.npz are legacy softmax") from exc
    _validate_payload(payload)
    return payload
