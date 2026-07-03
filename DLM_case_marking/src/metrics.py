"""
Metrics — Aggregated measures per language for the DLM case marking analysis.
"""

import pandas as pd
import numpy as np
from scipy import stats

from src.utils import (
    logger, load_config, load_dataframe, save_dataframe,
    get_all_languages, get_language_config,
)


def compute_language_summary(lang: str) -> dict:
    """Compute summary metrics for a single language."""
    config = get_language_config(lang)
    lang_type = config["type"]

    try:
        arcs = load_dataframe(f"arcs_{lang}")
    except FileNotFoundError:
        return {}

    try:
        surprisal = load_dataframe(f"surprisal_{lang}")
    except FileNotFoundError:
        surprisal = None

    # Basic dependency length stats
    metrics = {
        "lang": lang,
        "lang_type": lang_type,
        "n_sentences": arcs["sent_id"].nunique(),
        "n_arcs": len(arcs),
        "mean_dep_length": arcs["dep_length"].mean(),
        "median_dep_length": arcs["dep_length"].median(),
        "std_dep_length": arcs["dep_length"].std(),
        "mean_sent_length": arcs.groupby("sent_id")["sent_length"].first().mean(),
    }

    # Argument dependency stats
    arg_arcs = arcs[arcs["is_argument"]]
    metrics["n_argument_arcs"] = len(arg_arcs)
    metrics["mean_arg_dep_length"] = arg_arcs["dep_length"].mean() if len(arg_arcs) > 0 else np.nan

    # Surprisal stats
    if surprisal is not None and len(surprisal) > 0:
        metrics["mean_surprisal"] = surprisal["surprisal"].mean()
        metrics["std_surprisal"] = surprisal["surprisal"].std()
        metrics["mean_entropy"] = surprisal["entropy"].mean()

    # Surprisal-distance slope (key metric!)
    if surprisal is not None and len(surprisal) > 0:
        try:
            merged = arcs.merge(
                surprisal[["lang", "sent_id", "word_pos", "surprisal"]].rename(
                    columns={"word_pos": "head_pos", "surprisal": "head_surprisal"}
                ),
                on=["lang", "sent_id", "head_pos"],
                how="inner",
            )
            if len(merged) > 10:
                slope, intercept, r, p, se = stats.linregress(
                    merged["dep_length"], merged["head_surprisal"]
                )
                metrics["surprisal_slope"] = slope
                metrics["surprisal_slope_r2"] = r ** 2
                metrics["surprisal_slope_p"] = p

                # Same for argument deps only
                arg_merged = merged[merged["is_argument"]]
                if len(arg_merged) > 10:
                    s2, _, r2, p2, _ = stats.linregress(
                        arg_merged["dep_length"], arg_merged["head_surprisal"]
                    )
                    metrics["arg_surprisal_slope"] = s2
                    metrics["arg_surprisal_slope_r2"] = r2 ** 2
        except Exception as e:
            logger.warning(f"Slope computation failed for {lang}: {e}")

    return metrics


def compute_all_summaries() -> pd.DataFrame:
    """Compute summary metrics for all languages."""
    all_langs = get_all_languages()
    summaries = []
    for lang in all_langs:
        m = compute_language_summary(lang)
        if m:
            summaries.append(m)
            logger.info(f"✓ {lang}: mean dep length = {m.get('mean_dep_length', '?'):.2f}")
    df = pd.DataFrame(summaries)
    save_dataframe(df, "language_summaries", subdir="results")
    return df


if __name__ == "__main__":
    compute_all_summaries()
