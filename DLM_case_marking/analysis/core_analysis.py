"""
Core Analysis — Surprisal × Dependency Length regression across languages.

Main analyses:
  1. Surprisal vs dep length slopes per language
  2. Case-marking vs non-case-marking group comparison
  3. Argument deps vs all deps
"""

import pandas as pd
import numpy as np
from scipy import stats

from src.utils import (
    logger, load_dataframe, save_dataframe, get_all_languages,
    get_language_config, get_case_marking_languages,
    get_non_case_marking_languages,
)


def merge_surprisal_with_arcs(lang: str) -> pd.DataFrame:
    """Merge surprisal values with dependency arc data for a language."""
    arcs = load_dataframe(f"arcs_{lang}")
    surprisal = load_dataframe(f"surprisal_{lang}")

    # Merge surprisal at the HEAD position (processing cost at reunion)
    head_surp = surprisal[["lang", "sent_id", "word_pos", "surprisal", "entropy"]].rename(
        columns={"word_pos": "head_pos", "surprisal": "head_surprisal", "entropy": "head_entropy"}
    )
    merged = arcs.merge(head_surp, on=["lang", "sent_id", "head_pos"], how="inner")

    # Also merge surprisal at the DEPENDENT position
    dep_surp = surprisal[["lang", "sent_id", "word_pos", "surprisal", "entropy"]].rename(
        columns={"word_pos": "dep_pos", "surprisal": "dep_surprisal", "entropy": "dep_entropy"}
    )
    merged = merged.merge(dep_surp, on=["lang", "sent_id", "dep_pos"], how="left")

    return merged


def compute_slopes(merged_df: pd.DataFrame, dep_col="dep_length", surp_col="head_surprisal"):
    """Compute regression slope of surprisal ~ dep_length for each language."""
    results = []
    for lang, group in merged_df.groupby("lang"):
        valid = group.dropna(subset=[dep_col, surp_col])
        if len(valid) < 20:
            continue
        slope, intercept, r, p, se = stats.linregress(valid[dep_col], valid[surp_col])
        lang_config = get_language_config(lang)
        results.append({
            "lang": lang,
            "lang_type": lang_config["type"],
            "slope": slope,
            "intercept": intercept,
            "r_squared": r ** 2,
            "p_value": p,
            "std_error": se,
            "n_observations": len(valid),
        })
    return pd.DataFrame(results)


def analysis_1_slopes_by_language():
    """
    Analysis 1: Compute surprisal-distance slopes for ALL deps per language.
    Key prediction: case-marking languages have flatter slopes.
    """
    logger.info("=" * 60)
    logger.info("ANALYSIS 1: Surprisal × Dependency Length Slopes")
    logger.info("=" * 60)

    all_merged = []
    for lang in get_all_languages():
        try:
            merged = merge_surprisal_with_arcs(lang)
            all_merged.append(merged)
        except FileNotFoundError:
            logger.warning(f"Skipping {lang}: data not found")

    if not all_merged:
        raise RuntimeError("No data available for analysis.")

    combined = pd.concat(all_merged, ignore_index=True)
    save_dataframe(combined, "merged_arcs_surprisal_all")

    # All deps slopes
    all_slopes = compute_slopes(combined)
    save_dataframe(all_slopes, "slopes_all_deps", subdir="results")

    # Group comparison
    case_slopes = all_slopes[all_slopes["lang_type"] == "case_marking"]["slope"]
    non_case_slopes = all_slopes[all_slopes["lang_type"] == "non_case_marking"]["slope"]

    logger.info("\nSlopes (all dependencies):")
    logger.info(f"  Case-marking languages:     mean slope = {case_slopes.mean():.4f}")
    logger.info(f"  Non-case-marking languages:  mean slope = {non_case_slopes.mean():.4f}")

    if len(case_slopes) > 1 and len(non_case_slopes) > 1:
        t_stat, p_val = stats.ttest_ind(case_slopes, non_case_slopes)
        logger.info(f"  t-test: t = {t_stat:.3f}, p = {p_val:.4f}")

    return all_slopes, combined


def analysis_2_argument_deps():
    """
    Analysis 2: Same as Analysis 1 but restricted to argument deps.
    Prediction: slope difference is LARGER for argument deps.
    """
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 2: Argument Dependencies Only")
    logger.info("=" * 60)

    combined = load_dataframe("merged_arcs_surprisal_all")
    arg_only = combined[combined["is_argument"]]

    arg_slopes = compute_slopes(arg_only)
    save_dataframe(arg_slopes, "slopes_argument_deps", subdir="results")

    case_slopes = arg_slopes[arg_slopes["lang_type"] == "case_marking"]["slope"]
    non_case_slopes = arg_slopes[arg_slopes["lang_type"] == "non_case_marking"]["slope"]

    logger.info("\nSlopes (argument dependencies only):")
    logger.info(f"  Case-marking:     mean slope = {case_slopes.mean():.4f}")
    logger.info(f"  Non-case-marking: mean slope = {non_case_slopes.mean():.4f}")

    if len(case_slopes) > 1 and len(non_case_slopes) > 1:
        t_stat, p_val = stats.ttest_ind(case_slopes, non_case_slopes)
        logger.info(f"  t-test: t = {t_stat:.3f}, p = {p_val:.4f}")

    return arg_slopes


def analysis_3_within_case_marking():
    """
    Analysis 3: Within case-marking languages, compare slopes for
    arcs where dependent IS case-marked vs NOT.
    """
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 3: Case-Marked vs Unmarked (within case-marking langs)")
    logger.info("=" * 60)

    combined = load_dataframe("merged_arcs_surprisal_all")
    case_langs = get_case_marking_languages()
    case_data = combined[combined["lang"].isin(case_langs)]

    if "dep_is_case_marked" not in case_data.columns:
        # Need to merge case annotations
        from src.case_marker_detector import annotate_arcs_with_case
        tokens_case = load_dataframe("tokens_case_all")
        case_data = annotate_arcs_with_case(case_data, tokens_case)

    results = []
    for lang in case_langs:
        lang_data = case_data[case_data["lang"] == lang]

        for marked_status, label in [(True, "case_marked"), (False, "unmarked")]:
            subset = lang_data[lang_data["dep_is_case_marked"] == marked_status]
            valid = subset.dropna(subset=["dep_length", "head_surprisal"])
            if len(valid) < 20:
                continue
            slope, _, r, p, se = stats.linregress(valid["dep_length"], valid["head_surprisal"])
            results.append({
                "lang": lang, "case_status": label,
                "slope": slope, "r_squared": r**2,
                "p_value": p, "n": len(valid),
            })

    results_df = pd.DataFrame(results)
    save_dataframe(results_df, "slopes_case_vs_unmarked", subdir="results")

    logger.info("\nWithin case-marking languages:")
    for lang in case_langs:
        lang_res = results_df[results_df["lang"] == lang]
        if len(lang_res) == 2:
            marked_slope = lang_res[lang_res["case_status"] == "case_marked"]["slope"].values[0]
            unmarked_slope = lang_res[lang_res["case_status"] == "unmarked"]["slope"].values[0]
            logger.info(f"  {lang}: marked slope = {marked_slope:.4f}, unmarked slope = {unmarked_slope:.4f}")

    return results_df


def run_core_analysis():
    """Run all core analyses."""
    slopes_all, combined = analysis_1_slopes_by_language()
    slopes_arg = analysis_2_argument_deps()
    slopes_case = analysis_3_within_case_marking()
    logger.info("\n✓ Core analysis complete. Results saved to results/")
    return slopes_all, slopes_arg, slopes_case


if __name__ == "__main__":
    run_core_analysis()
