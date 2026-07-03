"""
Critical Region Analysis

Examines surprisal specifically at dependency "reunion points" — where a
head word appears after a distant dependent.

Tests whether case-marking languages show lower "reunion cost" when a
long-distance dependency is finally resolved.
"""

import pandas as pd
import numpy as np
from scipy import stats

from src.utils import (
    logger, load_config, load_dataframe, save_dataframe,
    get_language_config, get_all_languages,
)


def identify_reunion_points(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify reunion points: head words that resolve a dependency with
    a distant dependent (dep_length >= threshold).

    Enriches the data with distance bins for stratified analysis.
    """
    config = load_config()
    threshold = config["analysis"]["long_distance_threshold"]
    bins = config["analysis"]["dep_length_bins"]

    # Filter to long-distance deps only
    long_deps = merged_df[merged_df["dep_length"] >= threshold].copy()

    # Create distance bins
    def assign_bin(length):
        for low, high in bins:
            if low <= length <= high:
                return f"{low}-{high}" if high < 999 else f"{low}+"
        return "other"

    long_deps["distance_bin"] = long_deps["dep_length"].apply(assign_bin)

    return long_deps


def analyze_reunion_cost(reunion_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare reunion cost (head surprisal at reunion point) across
    language types and distance bins.
    """
    results = []

    for lang in reunion_df["lang"].unique():
        lang_config = get_language_config(lang)
        lang_data = reunion_df[reunion_df["lang"] == lang]

        for dist_bin in lang_data["distance_bin"].unique():
            bin_data = lang_data[lang_data["distance_bin"] == dist_bin]
            valid = bin_data["head_surprisal"].dropna()

            if len(valid) < 5:
                continue

            results.append({
                "lang": lang,
                "lang_type": lang_config["type"],
                "distance_bin": dist_bin,
                "mean_head_surprisal": valid.mean(),
                "std_head_surprisal": valid.std(),
                "median_head_surprisal": valid.median(),
                "n_observations": len(valid),
            })

    return pd.DataFrame(results)


def analyze_case_effect_at_reunion(reunion_df: pd.DataFrame) -> pd.DataFrame:
    """
    Within case-marking languages: compare reunion cost when the dependent
    WAS case-marked vs when it was NOT.
    """
    if "dep_is_case_marked" not in reunion_df.columns:
        logger.warning("Case annotation missing. Run case annotation first.")
        return pd.DataFrame()

    from src.utils import get_case_marking_languages
    case_langs = get_case_marking_languages()
    case_data = reunion_df[reunion_df["lang"].isin(case_langs)]

    results = []
    for lang in case_langs:
        lang_data = case_data[case_data["lang"] == lang]

        for marked in [True, False]:
            subset = lang_data[lang_data["dep_is_case_marked"] == marked]
            valid = subset["head_surprisal"].dropna()
            if len(valid) < 10:
                continue

            results.append({
                "lang": lang,
                "dep_case_marked": marked,
                "mean_reunion_surprisal": valid.mean(),
                "std": valid.std(),
                "n": len(valid),
                "mean_dep_length": subset["dep_length"].mean(),
            })

    results_df = pd.DataFrame(results)

    # Compute within-language effects
    for lang in case_langs:
        lr = results_df[results_df["lang"] == lang]
        if len(lr) == 2:
            marked_s = lr[lr["dep_case_marked"] == True]["mean_reunion_surprisal"].values[0]
            unmarked_s = lr[lr["dep_case_marked"] == False]["mean_reunion_surprisal"].values[0]
            logger.info(
                f"  {lang}: case-marked dep → head surprisal = {marked_s:.2f}, "
                f"unmarked dep → {unmarked_s:.2f}, "
                f"diff = {unmarked_s - marked_s:.3f}"
            )

    return results_df


def run_critical_region_analysis():
    """Run the full critical region analysis."""
    logger.info("=" * 60)
    logger.info("CRITICAL REGION ANALYSIS: Reunion Points")
    logger.info("=" * 60)

    merged = load_dataframe("merged_arcs_surprisal_all")
    reunion = identify_reunion_points(merged)
    save_dataframe(reunion, "reunion_points", subdir="results")

    logger.info(f"Total reunion points (long deps): {len(reunion)}")

    # Analysis by language type and distance bin
    reunion_cost = analyze_reunion_cost(reunion)
    save_dataframe(reunion_cost, "reunion_cost_by_bin", subdir="results")

    logger.info("\nReunion cost by language type and distance:")
    for lang_type in ["case_marking", "non_case_marking"]:
        type_data = reunion_cost[reunion_cost["lang_type"] == lang_type]
        logger.info(f"\n  {lang_type}:")
        for _, row in type_data.groupby("distance_bin").agg(
            mean_surprisal=("mean_head_surprisal", "mean")
        ).reset_index().iterrows():
            logger.info(f"    distance {row['distance_bin']}: {row['mean_surprisal']:.2f}")

    # Case effect at reunion (within case-marking langs)
    case_effect = analyze_case_effect_at_reunion(reunion)
    if len(case_effect) > 0:
        save_dataframe(case_effect, "reunion_case_effect", subdir="results")

    logger.info("\n✓ Critical region analysis complete.")
    return reunion_cost, case_effect


if __name__ == "__main__":
    run_critical_region_analysis()
