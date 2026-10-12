"""Exact Hugging Face revisions for the frozen configuration's two models.

Loading `intfloat/e5-large-v2` or `facebook/bart-large-mnli` by name alone follows
the Hub's moving `main` branch. These commit SHAs fix the bytes instead. They are
the snapshots inspected on the laptop and recorded in
results/frozen_reference/runtime_manifest.json, and they are the current `main` of
both repositories. Which revisions the historical pvgpu1 runs loaded is not
recorded; see docs/model-revisions.md.
"""
from __future__ import annotations

import re

E5_LARGE_V2 = "intfloat/e5-large-v2"
BART_LARGE_MNLI = "facebook/bart-large-mnli"

PINNED_REVISIONS = {
    E5_LARGE_V2: "f169b11e22de13617baa190a028a32f3493550b6",
    BART_LARGE_MNLI: "d7645e127eaf1aefc7862fd59a17a5aa8558b8ce",
}


def pinned_revision(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value or ""):
        raise ValueError("Use an exact 40-character checkpoint/tokenizer commit SHA, not a moving ref")
    return value


def revision_for(model_id: str, override: str | None = None) -> str | None:
    """The exact revision to load: a validated override, else the pinned one, else None (floating)."""
    if override is not None:
        return pinned_revision(override)
    return PINNED_REVISIONS.get(model_id)


def require_revision(model_id: str, override: str | None = None) -> str:
    """Like revision_for, but refuse a model with no exact revision."""
    revision = revision_for(model_id, override)
    if revision is None:
        raise ValueError(f"{model_id} has no pinned revision; pass an exact 40-character commit SHA")
    return revision
