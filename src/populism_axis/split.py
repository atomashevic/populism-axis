"""Country-level development/holdout split of the English GPD benchmark.

Countries, not speeches, are the sampling unit, so no leader or country
appears on both sides. Labels are used only to keep the two halves comparable
(share of speeches, extreme positives, score distribution); they are never used
to choose a method.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


@dataclass(frozen=True)
class SplitRule:
    seed: int = 20260923
    n_candidates: int = 5000
    holdout_share_min: float = 0.27
    holdout_share_max: float = 0.33
    min_holdout_extreme_pos: int = 4
    min_dev_extreme_pos: int = 8
    min_holdout_nonzero: int = 20
    min_holdout_leaders: int = 20
    inclusion_probability: float = 0.30

    def as_dict(self) -> dict:
        return asdict(self)


def _split_stats(frame: pd.DataFrame, holdout_mask: np.ndarray) -> dict:
    hold = frame[holdout_mask]
    dev = frame[~holdout_mask]
    return {
        "holdout_share": float(holdout_mask.mean()),
        "holdout_extreme_pos": int((hold["extreme_label"] == 1).sum()),
        "dev_extreme_pos": int((dev["extreme_label"] == 1).sum()),
        "holdout_nonzero": int((hold["source_score"] > 0).sum()),
        "holdout_leaders": int(hold["leader"].nunique()),
    }


def _passes(stats: dict, rule: SplitRule) -> bool:
    return (
        rule.holdout_share_min <= stats["holdout_share"] <= rule.holdout_share_max
        and stats["holdout_extreme_pos"] >= rule.min_holdout_extreme_pos
        and stats["dev_extreme_pos"] >= rule.min_dev_extreme_pos
        and stats["holdout_nonzero"] >= rule.min_holdout_nonzero
        and stats["holdout_leaders"] >= rule.min_holdout_leaders
    )


def balance_distance(frame: pd.DataFrame, holdout_mask: np.ndarray) -> float:
    """|mean difference| + KS distance of source_score between the two halves."""
    dev = frame.loc[~holdout_mask, "source_score"].to_numpy(dtype=float)
    hold = frame.loc[holdout_mask, "source_score"].to_numpy(dtype=float)
    return float(abs(dev.mean() - hold.mean()) + ks_2samp(dev, hold).statistic)


def choose_country_split(frame: pd.DataFrame, rule: SplitRule = SplitRule()) -> dict:
    """Sample country partitions, keep those meeting the rule, return the most balanced.

    Returns a dict with the holdout country list, the per-split statistics, the
    balance distance, and how many candidates passed. Deterministic for a seed.
    """
    countries = np.array(sorted(frame["country"].astype(str).unique()))
    country_of_row = frame["country"].astype(str).to_numpy()
    rng = np.random.default_rng(rule.seed)
    best: dict | None = None
    n_passed = 0
    seen: set[tuple[str, ...]] = set()
    for _ in range(rule.n_candidates):
        chosen = countries[rng.random(len(countries)) < rule.inclusion_probability]
        key = tuple(chosen)
        if not key or key in seen:
            continue
        seen.add(key)
        mask = np.isin(country_of_row, chosen)
        stats = _split_stats(frame, mask)
        if not _passes(stats, rule):
            continue
        n_passed += 1
        distance = balance_distance(frame, mask)
        if best is None or distance < best["balance_distance"] - 1e-12:
            best = {"holdout_countries": [str(item) for item in key], "stats": stats, "balance_distance": distance}
    if best is None:
        raise ValueError("no country partition satisfied the split rule")
    best["n_candidates_passed"] = n_passed
    best["n_candidates_unique"] = len(seen)
    return best


def split_frame(frame: pd.DataFrame, holdout_countries: list[str]) -> pd.DataFrame:
    out = frame[["id", "country", "leader"]].copy()
    out["split"] = np.where(out["country"].astype(str).isin(holdout_countries), "holdout", "dev")
    return out.sort_values("id").reset_index(drop=True)
