"""
Statistical Models — Mixed-effects and OLS regression for the DLM analysis.

Models:
  1. surprisal ~ dep_length * lang_type + controls
  2. head_surprisal ~ dep_length * case_marked + controls (within case langs)
  3. delta_surprisal ~ dep_length (ablation analysis)
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

from src.utils import (
    logger, load_dataframe, save_dataframe,
    get_case_marking_languages, get_tables_dir,
)


def model_1_cross_linguistic(merged_df: pd.DataFrame) -> dict:
    """
    Model 1: Cross-linguistic comparison.
    head_surprisal ~ dep_length * lang_type + dep_log_freq + sent_length

    Tests whether the effect of dependency length on surprisal differs
    between case-marking and non-case-marking languages.
    """
    logger.info("MODEL 1: Cross-linguistic surprisal ~ dep_length * lang_type")

    df = merged_df.dropna(subset=["head_surprisal", "dep_length"]).copy()
    df["lang_type_binary"] = (df["lang_type"] == "case_marking").astype(int) if "lang_type" in df.columns else 0

    # Add lang_type if not present
    if "lang_type" not in df.columns:
        from src.utils import get_language_config
        df["lang_type"] = df["lang"].apply(lambda x: get_language_config(x)["type"])

    df["is_case_marking"] = (df["lang_type"] == "case_marking").astype(int)

    # Ensure numeric
    for col in ["dep_length", "head_surprisal", "sent_length"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["dep_length", "head_surprisal", "sent_length"])

    # OLS regression (mixed effects would need pymer4/R)
    formula = "head_surprisal ~ dep_length * is_case_marking + sent_length"

    if "dep_log_freq" in df.columns:
        formula += " + dep_log_freq"

    try:
        model = smf.ols(formula, data=df).fit()
        logger.info(f"\n{model.summary()}")

        results = {
            "formula": formula,
            "r_squared": model.rsquared,
            "adj_r_squared": model.rsquared_adj,
            "n_obs": int(model.nobs),
            "coefficients": model.params.to_dict(),
            "p_values": model.pvalues.to_dict(),
            "conf_intervals": model.conf_int().to_dict(),
        }

        # Save summary to text file
        tables_dir = get_tables_dir()
        with open(tables_dir / "model1_summary.txt", "w") as f:
            f.write(str(model.summary()))

        return results
    except Exception as e:
        logger.error(f"Model 1 failed: {e}")
        return {}


def model_2_within_case_marking(merged_df: pd.DataFrame) -> dict:
    """
    Model 2: Within case-marking languages.
    head_surprisal ~ dep_length * dep_is_case_marked + controls

    Tests whether case marking flattens the surprisal-distance curve.
    """
    logger.info("\nMODEL 2: Within case-marking — surprisal ~ dep_length * case_marked")

    case_langs = get_case_marking_languages()
    df = merged_df[merged_df["lang"].isin(case_langs)].copy()

    if "dep_is_case_marked" not in df.columns:
        logger.info("dep_is_case_marked not found — annotating now...")
        try:
            from src.case_marker_detector import annotate_arcs_with_case
            from src.utils import load_dataframe
            tokens_case = load_dataframe("tokens_case_all")
            df = annotate_arcs_with_case(df, tokens_case)
        except Exception as e:
            logger.warning(f"Could not annotate case marking: {e}")
            return {}

    df["case_marked_int"] = df["dep_is_case_marked"].astype(int)
    df = df.dropna(subset=["head_surprisal", "dep_length"])

    formula = "head_surprisal ~ dep_length * case_marked_int + sent_length"
    if "dep_log_freq" in df.columns:
        formula += " + dep_log_freq"

    try:
        model = smf.ols(formula, data=df).fit()
        logger.info(f"\n{model.summary()}")

        tables_dir = get_tables_dir()
        with open(tables_dir / "model2_summary.txt", "w") as f:
            f.write(str(model.summary()))

        return {
            "formula": formula,
            "r_squared": model.rsquared,
            "coefficients": model.params.to_dict(),
            "p_values": model.pvalues.to_dict(),
        }
    except Exception as e:
        logger.error(f"Model 2 failed: {e}")
        return {}


def model_3_ablation(ablation_df: pd.DataFrame, arcs_df: pd.DataFrame) -> dict:
    """
    Model 3: Ablation analysis.
    delta_surprisal ~ dep_length

    Tests whether case markers contribute MORE information at longer distances.
    """
    logger.info("\nMODEL 3: Ablation — delta_surprisal ~ dep_length")

    # Merge ablation with dependency info
    merged = ablation_df.merge(
        arcs_df[["lang", "sent_id", "head_pos", "dep_length", "is_argument"]],
        left_on=["lang", "sent_id", "word_pos"],
        right_on=["lang", "sent_id", "head_pos"],
        how="inner",
    )

    merged = merged.dropna(subset=["delta_surprisal", "dep_length"])
    if len(merged) < 20:
        logger.warning("Too few data points for ablation model.")
        return {}

    formula = "delta_surprisal ~ dep_length"
    try:
        model = smf.ols(formula, data=merged).fit()
        logger.info(f"\n{model.summary()}")

        tables_dir = get_tables_dir()
        with open(tables_dir / "model3_ablation_summary.txt", "w") as f:
            f.write(str(model.summary()))

        return {
            "formula": formula,
            "r_squared": model.rsquared,
            "coefficients": model.params.to_dict(),
            "p_values": model.pvalues.to_dict(),
        }
    except Exception as e:
        logger.error(f"Model 3 failed: {e}")
        return {}


def run_all_models():
    """Run all statistical models."""
    merged = load_dataframe("merged_arcs_surprisal_all")
    r1 = model_1_cross_linguistic(merged)
    r2 = model_2_within_case_marking(merged)

    try:
        ablation = load_dataframe("ablation_all")
        arcs = load_dataframe("arcs_all")
        r3 = model_3_ablation(ablation, arcs)
    except FileNotFoundError:
        logger.warning("Ablation data not found, skipping Model 3")
        r3 = {}

    logger.info("\n✓ All statistical models complete.")
    return r1, r2, r3


if __name__ == "__main__":
    run_all_models()
