"""
Dependency Extractor

Extracts dependency arcs from parsed UD token DataFrames and computes
dependency lengths, arc types, and argument classification.
"""

import pandas as pd
import numpy as np
from typing import List, Optional

from src.utils import (
    logger,
    load_config,
    load_dataframe,
    save_dataframe,
    get_all_languages,
)


def extract_dependency_arcs(tokens_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all dependency arcs from a token-level DataFrame.

    For each token that has a head (head != 0, i.e. not ROOT), creates
    a row describing the dependency arc.

    Returns DataFrame with columns:
        lang, sent_id, dep_pos, dep_form, dep_lemma, dep_upos, dep_feats,
        head_pos, head_form, head_lemma, head_upos, head_feats,
        deprel, dep_length, is_argument, sent_length
    """
    config = load_config()
    argument_deprels = set(config["analysis"]["argument_deprels"])

    arcs = []

    # Group by (lang, sent_id) for efficient per-sentence processing
    grouped = tokens_df.groupby(["lang", "sent_id"])

    for (lang, sent_id), sent_tokens in grouped:
        # Build a lookup: token_id → token row
        token_lookup = {}
        for _, row in sent_tokens.iterrows():
            token_lookup[row["token_id"]] = row

        sent_length = sent_tokens.iloc[0]["sent_length"]

        for _, dep_row in sent_tokens.iterrows():
            head_id = dep_row["head"]

            # Skip ROOT arcs (head == 0)
            if head_id == 0:
                continue

            # Look up the head token
            if head_id not in token_lookup:
                continue  # Malformed data, skip

            head_row = token_lookup[head_id]

            dep_pos = dep_row["token_id"]
            head_pos = head_row["token_id"]
            dep_length = abs(head_pos - dep_pos)
            deprel = dep_row["deprel"]

            # Normalise deprel: strip subtypes for argument classification
            # e.g. "nsubj:pass" → base is "nsubj:pass" (already in config)
            is_argument = deprel in argument_deprels

            arcs.append({
                "lang": lang,
                "sent_id": sent_id,
                "dep_pos": dep_pos,
                "dep_form": dep_row["form"],
                "dep_lemma": dep_row["lemma"],
                "dep_upos": dep_row["upos"],
                "dep_feats": dep_row["feats_str"],
                "head_pos": head_pos,
                "head_form": head_row["form"],
                "head_lemma": head_row["lemma"],
                "head_upos": head_row["upos"],
                "head_feats": head_row["feats_str"],
                "deprel": deprel,
                "dep_length": dep_length,
                "is_argument": is_argument,
                "sent_length": sent_length,
                # Direction: positive = head is to the right of dependent
                "direction": "right" if head_pos > dep_pos else "left",
            })

    arcs_df = pd.DataFrame(arcs)

    if len(arcs_df) > 0:
        logger.info(
            f"Extracted {len(arcs_df)} dependency arcs "
            f"(mean length: {arcs_df['dep_length'].mean():.2f})"
        )
    else:
        logger.warning("No dependency arcs extracted!")

    return arcs_df


def compute_word_frequencies(tokens_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute word frequencies per language from the token DataFrame.

    Returns DataFrame with columns: lang, form, frequency, log_frequency
    """
    freq = (
        tokens_df.groupby(["lang", "form"])
        .size()
        .reset_index(name="frequency")
    )

    # Total tokens per language for relative frequency
    totals = tokens_df.groupby("lang").size().reset_index(name="total")
    freq = freq.merge(totals, on="lang")
    freq["rel_frequency"] = freq["frequency"] / freq["total"]
    freq["log_frequency"] = np.log10(freq["frequency"] + 1)

    freq = freq.drop(columns=["total"])

    return freq


def enrich_arcs_with_frequency(
    arcs_df: pd.DataFrame, freq_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add word frequency information to dependency arcs (for controls in regression).
    """
    # Merge dep word frequency
    dep_freq = freq_df[["lang", "form", "log_frequency"]].rename(
        columns={"form": "dep_form", "log_frequency": "dep_log_freq"}
    )
    arcs_df = arcs_df.merge(dep_freq, on=["lang", "dep_form"], how="left")

    # Merge head word frequency
    head_freq = freq_df[["lang", "form", "log_frequency"]].rename(
        columns={"form": "head_form", "log_frequency": "head_log_freq"}
    )
    arcs_df = arcs_df.merge(head_freq, on=["lang", "head_form"], how="left")

    # Fill NaNs (unseen words) with 0
    arcs_df["dep_log_freq"] = arcs_df["dep_log_freq"].fillna(0)
    arcs_df["head_log_freq"] = arcs_df["head_log_freq"].fillna(0)

    return arcs_df


def extract_all_languages() -> pd.DataFrame:
    """
    Extract dependency arcs for all languages from saved token DataFrames.
    """
    all_langs = get_all_languages()
    all_arcs = []
    all_freqs = []

    for lang in all_langs:
        try:
            tokens_df = load_dataframe(f"tokens_{lang}")
        except FileNotFoundError:
            logger.warning(f"Skipping {lang}: token data not found")
            continue

        # Extract arcs
        arcs = extract_dependency_arcs(tokens_df)
        if len(arcs) == 0:
            continue

        # Compute frequencies
        freq = compute_word_frequencies(tokens_df)

        # Enrich arcs with frequency
        arcs = enrich_arcs_with_frequency(arcs, freq)

        save_dataframe(arcs, f"arcs_{lang}")
        save_dataframe(freq, f"freq_{lang}")

        all_arcs.append(arcs)
        all_freqs.append(freq)

        logger.info(
            f"✓ {lang}: {len(arcs)} arcs, "
            f"mean dep length = {arcs['dep_length'].mean():.2f}, "
            f"argument arcs = {arcs['is_argument'].sum()}"
        )

    if not all_arcs:
        raise RuntimeError("No arcs extracted for any language.")

    combined_arcs = pd.concat(all_arcs, ignore_index=True)
    combined_freq = pd.concat(all_freqs, ignore_index=True)

    save_dataframe(combined_arcs, "arcs_all")
    save_dataframe(combined_freq, "freq_all")

    # Print summary
    summary = (
        combined_arcs.groupby("lang")
        .agg(
            n_arcs=("dep_length", "count"),
            mean_dep_length=("dep_length", "mean"),
            median_dep_length=("dep_length", "median"),
            n_argument_arcs=("is_argument", "sum"),
        )
        .round(2)
    )
    logger.info(f"\nDependency Arc Summary:\n{summary.to_string()}")

    return combined_arcs


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    extract_all_languages()
