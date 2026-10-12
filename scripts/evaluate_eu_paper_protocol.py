#!/usr/bin/env python3
"""Evaluate our axes on EU-OOD under the Chalkidis et al. (2025) protocol.

The paper scores the EU-OOD set as 3-way multi-label sentence classification
(neutral / anti-elitism / people-centrism) and reports macro-F1. Our AUROC
numbers are not comparable to that, so this script reproduces their protocol.

Channel-to-class mapping is fixed in advance by construct, not by search:
    E channel -> anti-elitism
    P channel -> people-centrism
    neutral   -> neither predicted

Two thresholding regimes are reported:
  prevalence  - fully label-free on EU. Thresholds are set so predicted
                positive rates match the TRUMP-2016 training prevalence, which
                is the same information the paper hands its LLMs in the
                Distribution-Aware prompt.
  loso_f1     - leave-one-speaker-out; thresholds maximize F1 on the other
                four speakers. Grouped so no speaker informs its own threshold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]

# Prevalence of each class in the paper's TRUMP-2016 training data (dataset card).
TRUMP_PREVALENCE = {"AE": 0.055, "PC": 0.034}

# Published EU-OOD macro-F1 (Table 12, translated) for reference.
PAPER_EU_OOD = {
    "RoBERTa (paper)": {
        "Le Pen": 0.487, "Weidel": 0.329, "Zemmour": 0.388, "Kickl": 0.408,
        "Tsipras": 0.550, "Overall": 0.508,
    },
    "Qwen3-14B (paper)": {
        "Le Pen": 0.609, "Weidel": 0.437, "Zemmour": 0.583, "Kickl": 0.473,
        "Tsipras": 0.658, "Overall": 0.613,
    },
    "Gemini 2.5 Flash (paper)": {
        "Le Pen": 0.650, "Weidel": 0.314, "Zemmour": 0.554, "Kickl": 0.558,
        "Tsipras": 0.586, "Overall": 0.590,
    },
}
SURNAME = {
    "Marine Le Pen": "Le Pen",
    "Alice Weidel": "Weidel",
    "Eric Zemmour": "Zemmour",
    "Herbert Kickl": "Kickl",
    "Alexis Tsipras": "Tsipras",
}


def window_max(doc_index: np.ndarray, values: np.ndarray, n_documents: int) -> np.ndarray:
    out = np.full(n_documents, -np.inf, dtype=float)
    np.maximum.at(out, doc_index, values)
    out[~np.isfinite(out)] = 0.0
    return out


def macro_f1(truth: dict[str, np.ndarray], predicted: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    neutral_true = ((truth["AE"] == 0) & (truth["PC"] == 0)).astype(int)
    neutral_pred = ((predicted["AE"] == 0) & (predicted["PC"] == 0)).astype(int)
    scores = {
        "N": f1_score(neutral_true[mask], neutral_pred[mask], zero_division=0),
        "AE": f1_score(truth["AE"][mask], predicted["AE"][mask], zero_division=0),
        "PC": f1_score(truth["PC"][mask], predicted["PC"][mask], zero_division=0),
    }
    scores["macro"] = float(np.mean([scores["N"], scores["AE"], scores["PC"]]))
    return {key: float(value) for key, value in scores.items()}


def prevalence_threshold(values: np.ndarray, rate: float) -> float:
    return float(np.quantile(values, 1.0 - rate))


def best_f1_threshold(values: np.ndarray, labels: np.ndarray) -> float:
    order = np.unique(np.quantile(values, np.linspace(0.5, 0.999, 120)))
    best, best_score = order[0], -1.0
    for threshold in order:
        score = f1_score(labels, (values > threshold).astype(int), zero_division=0)
        if score > best_score:
            best, best_score = threshold, score
    return float(best)
