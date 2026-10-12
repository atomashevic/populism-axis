"""Benchmark loading helpers shared by the encode and holdout-freeze scripts.

`load_documents` reads the three core benchmark tables under `data/benchmarks/`
and the English speech texts they point to. `english_null_baselines` computes
the text-free reference predictors (word count, country and speech-type means)
and the human-coder ceiling that the holdout freeze records.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def load_documents(bench_dir: Path) -> dict[str, dict]:
    english = pd.read_csv(bench_dir / "english_gpd_speeches.csv")
    texts = []
    for row in english.itertuples(index=False):
        path = ROOT / "data" / str(row.path) if not str(row.path).startswith("data") else ROOT / str(row.path)
        if not path.exists():
            path = ROOT / "data" / "speeches_20251120" / str(row.merging_variable)
        texts.append(path.read_text(encoding="utf-8", errors="ignore"))
    eu = pd.read_csv(bench_dir / "eu_sentences.csv")
    agora = pd.read_csv(bench_dir / "agora_paragraphs.csv")
    return {
        "english": {"frame": english, "texts": texts},
        "eu": {"frame": eu, "texts": eu["text"].astype(str).tolist()},
        "agora": {"frame": agora, "texts": agora["text"].astype(str).tolist()},
    }


def _cluster_bootstrap_spearman(values: np.ndarray, target: np.ndarray, groups: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    index_by_group = {group: np.flatnonzero(groups == group) for group in unique}
    samples = []
    for _ in range(draws):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([index_by_group[group] for group in drawn])
        if len(np.unique(target[indices])) < 2 or len(np.unique(values[indices])) < 2:
            continue
        samples.append(spearmanr(values[indices], target[indices]).statistic)
    if not samples:
        return float("nan"), float("nan")
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def english_null_baselines(english: pd.DataFrame, gpd_csv: Path, draws: int, seed: int) -> dict:
    target = english["source_score"].to_numpy(dtype=float)
    groups = english["leader"].to_numpy()
    baselines: dict[str, dict] = {}
    for name, series in [
        ("word_count", english["word_count"].astype(float)),
        ("country_mean_rubric", english.groupby("country")["source_score"].transform("mean")),
        ("speechtype_mean_rubric", english.groupby("speechtype")["source_score"].transform("mean")),
        ("leader_mean_rubric_in_sample", english.groupby("leader")["source_score"].transform("mean")),
    ]:
        values = series.to_numpy(dtype=float)
        rho = float(spearmanr(values, target).statistic)
        low, high = _cluster_bootstrap_spearman(values, target, groups, draws, seed)
        entry = {"spearman": rho, "spearman_ci95": [low, high]}
        extreme = english[english["extreme_label"] >= 0]
        if extreme["extreme_label"].nunique() == 2:
            mask = english["extreme_label"].to_numpy() >= 0
            entry["extreme_auroc"] = float(roc_auc_score(english.loc[mask, "extreme_label"].to_numpy(dtype=int), values[mask]))
        baselines[name] = entry

    # Human ceiling: pairwise coder agreement restricted to the English pool.
    raw = pd.read_csv(gpd_csv)
    raw["rubricgrade"] = pd.to_numeric(raw["rubricgrade"], errors="coerce")
    raw = raw[raw["merging_variable"].isin(set(english["merging_variable"]))]
    raw = raw.dropna(subset=["rubricgrade"])
    pivot = raw.pivot_table(index="merging_variable", columns="codernum", values="rubricgrade", aggfunc="mean")
    pairs = []
    coders = list(pivot.columns)
    for left_index in range(len(coders)):
        for right_index in range(left_index + 1, len(coders)):
            both = pivot[[coders[left_index], coders[right_index]]].dropna()
            if len(both) >= 30:
                pairs.append({"coder_a": int(coders[left_index]), "coder_b": int(coders[right_index]), "n": int(len(both)),
                              "spearman": float(spearmanr(both.iloc[:, 0], both.iloc[:, 1]).statistic),
                              "pearson": float(both.iloc[:, 0].corr(both.iloc[:, 1]))})
    ceiling = float(np.mean([item["spearman"] for item in pairs])) if pairs else float("nan")
    return {"confound_baselines": baselines, "human_coder_pairs": pairs, "human_ceiling_spearman": ceiling}
