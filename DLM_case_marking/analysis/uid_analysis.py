"""
UID Analysis — Uniform Information Density

Tests whether case-marking languages distribute information more evenly
across sentences (lower surprisal variance).
"""

import pandas as pd
import numpy as np
from scipy import stats

from src.utils import (
    logger, load_dataframe, save_dataframe,
    get_all_languages, get_language_config,
)


def compute_uid_per_sentence(surprisal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-sentence UID metrics:
      - surprisal_mean: mean word surprisal
      - surprisal_var: variance of word surprisal
      - surprisal_std: std dev of word surprisal
      - surprisal_cv: coefficient of variation (std / mean)
      - surprisal_range: max - min surprisal

    Lower variance/CV = more uniform = better UID.
    """
    grouped = surprisal_df.groupby(["lang", "sent_id"])

    uid = grouped["surprisal"].agg(
        surprisal_mean="mean",
        surprisal_var="var",
        surprisal_std="std",
        surprisal_min="min",
        surprisal_max="max",
        n_words="count",
    ).reset_index()

    uid["surprisal_range"] = uid["surprisal_max"] - uid["surprisal_min"]
    uid["surprisal_cv"] = uid["surprisal_std"] / uid["surprisal_mean"]

    # Also compute entropy stats
    if "entropy" in surprisal_df.columns:
        entropy_stats = grouped["entropy"].agg(
            entropy_mean="mean",
            entropy_var="var",
        ).reset_index()
        uid = uid.merge(entropy_stats, on=["lang", "sent_id"])

    # Add language type
    uid["lang_type"] = uid["lang"].apply(
        lambda x: get_language_config(x)["type"]
    )

    # Filter out very short sentences (< 4 words)
    uid = uid[uid["n_words"] >= 4]

    return uid


def compare_uid_by_lang_type(uid_df: pd.DataFrame) -> pd.DataFrame:
    """Compare UID metrics between case-marking and non-case-marking groups."""
    results = []

    for metric in ["surprisal_var", "surprisal_cv", "surprisal_range"]:
        case_vals = uid_df[uid_df["lang_type"] == "case_marking"][metric].dropna()
        non_case_vals = uid_df[uid_df["lang_type"] == "non_case_marking"][metric].dropna()

        t_stat, p_val = stats.ttest_ind(case_vals, non_case_vals)
        cohens_d = (case_vals.mean() - non_case_vals.mean()) / np.sqrt(
            (case_vals.var() + non_case_vals.var()) / 2
        )

        results.append({
            "metric": metric,
            "case_marking_mean": case_vals.mean(),
            "non_case_marking_mean": non_case_vals.mean(),
            "difference": case_vals.mean() - non_case_vals.mean(),
            "t_statistic": t_stat,
            "p_value": p_val,
            "cohens_d": cohens_d,
        })

    return pd.DataFrame(results)


def run_uid_analysis():
    """Run the full UID analysis."""
    logger.info("=" * 60)
    logger.info("UID ANALYSIS: Uniform Information Density")
    logger.info("=" * 60)

    surprisal = load_dataframe("surprisal_all")
    uid = compute_uid_per_sentence(surprisal)
    save_dataframe(uid, "uid_per_sentence", subdir="results")

    comparison = compare_uid_by_lang_type(uid)
    save_dataframe(comparison, "uid_comparison", subdir="results")

    logger.info("\nUID Comparison (case-marking vs non-case-marking):")
    for _, row in comparison.iterrows():
        logger.info(
            f"  {row['metric']}: "
            f"case={row['case_marking_mean']:.3f}, "
            f"non-case={row['non_case_marking_mean']:.3f}, "
            f"p={row['p_value']:.4f}, d={row['cohens_d']:.3f}"
        )

    # Per-language summary
    lang_uid = uid.groupby(["lang", "lang_type"]).agg(
        mean_cv=("surprisal_cv", "mean"),
        mean_var=("surprisal_var", "mean"),
    ).reset_index()
    save_dataframe(lang_uid, "uid_per_language", subdir="results")
    logger.info(f"\n{lang_uid.to_string()}")

    return uid, comparison


if __name__ == "__main__":
    run_uid_analysis()
