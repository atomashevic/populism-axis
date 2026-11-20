"""Arms of the same-size claim-check comparison (research/development-log/claim-check-arms-plan.md).

One table read by the GPU scripts (nli_stage.py, decision_model_stage.py) and by the CPU
evaluator (evaluate_claim_check_arms.py). The revisions are the Hub `main` commits read on
2026-11-19. They are candidates of a follow-up analysis, not part of the frozen configuration,
so they are kept out of populism_axis.model_revisions.PINNED_REVISIONS.
"""
from __future__ import annotations

REFERENCE = "bart-large-mnli"

# Fixed question for the decision models; `{claim}` is one of the ten claim hypotheses.
QUESTION_TEMPLATE = "Does this passage assert the following claim? {claim}"

ARMS = {
    "A": {"label": "bart-large-mnli", "model": "facebook/bart-large-mnli",
          "revision": "d7645e127eaf1aefc7862fd59a17a5aa8558b8ce", "kind": "nli", "backbone": "BART-large",
          "head": "3-class NLI"},
    "B": {"label": "ModernBERT-large-nli", "model": "tasksource/ModernBERT-large-nli",
          "revision": "ca476cb923a8637073d4ceb0f19f7fc236e260d4", "kind": "nli", "backbone": "ModernBERT-large",
          "head": "3-class NLI"},
    "C": {"label": "ModernBERT-large-zeroshot-v2.0", "model": "MoritzLaurer/ModernBERT-large-zeroshot-v2.0",
          "revision": "a51e07b524299e309dd2b88d48b0cfa2bd9ec598", "kind": "nli", "backbone": "ModernBERT-large",
          "head": "binary entailment"},
    "D": {"label": "von", "model": "wfzyx/von",
          "revision": "3a34cbdfaa1e90f2a9eb6b3bfc5ac5c92a18cb01", "kind": "decision", "backbone": "ModernBERT-large",
          "head": "Option-Marker decision head", "sdk": "von-sdk==1.3.7"},
    "E": {"label": "laya", "model": "convaiinnovations/laya",
          "revision": "e333cb7c823cd1787276228260f56d02fbf5180b", "kind": "decision", "backbone": "ModernBERT-large",
          "head": "decision head (RLCD)", "sdk": "laya==0.4.37"},
}

CANDIDATES = ("B", "C", "D", "E")
# (left, right, what the contrast isolates); deltas are left minus right.
CONTRASTS = (
    ("B", "A", "backbone, NLI task held fixed"),
    ("C", "B", "broad zero-shot mix against NLI-only training"),
    ("D", "B", "Von decision head against the 3-class NLI head"),
    ("D", "C", "Von decision head against the binary entailment head"),
    ("E", "B", "Laya decision head against the 3-class NLI head"),
    ("E", "C", "Laya decision head against the binary entailment head"),
)


def tail_counts(entailment) -> dict:
    """Exact zeros, exact ones and distinct values of a stored probability array.

    Rounded or saturated outputs show up here as a small distinct count or as mass at 0 and 1;
    the claim check ranks windows by small probabilities, so both would erase its signal.
    """
    import numpy as np

    values = np.asarray(entailment).ravel().astype(np.float64)
    return {"n": int(values.size), "exact_zero": int((values == 0).sum()), "exact_one": int((values == 1).sum()),
            "distinct": int(np.unique(values).size),
            "on_4_decimal_grid": float(np.mean(np.abs(values - np.round(values, 4)) < 1e-7))}
