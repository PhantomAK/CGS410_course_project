"""
Case Marker Detector

Identifies case-marked tokens in Universal Dependencies data using both
morphological features and language-specific heuristics.

Strategy per language:
  - Hindi/Japanese/Korean: Case markers are separate tokens (postpositions/particles)
    linked via deprel="case". We flag both the marker AND its head as case-marked.
  - German/Turkish: Case is inflectional (morphological), detected via
    the Case= feature in the UD morphological features column.
"""

import re
from typing import Dict, List, Set, Optional, Tuple

import pandas as pd
import numpy as np

from src.utils import (
    logger,
    load_config,
    get_language_config,
    get_case_marking_languages,
    load_dataframe,
    save_dataframe,
)




def _has_case_feature(feats_str: str) -> bool:
    """Check if the UD feature string contains a Case= feature."""
    if not feats_str or pd.isna(feats_str):
        return False
    return bool(re.search(r"Case=\w+", feats_str))


def _extract_case_value(feats_str: str) -> str:
    """Extract the Case value from a UD feature string (e.g. 'Nom', 'Acc')."""
    if not feats_str or pd.isna(feats_str):
        return ""
    match = re.search(r"Case=(\w+)", feats_str)
    return match.group(1) if match else ""


def detect_case_by_deprel(
    tokens_df: pd.DataFrame,
    case_marker_forms: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    Detect case marking via the deprel='case' relation (for particle/postposition langs).
    Optimized version using vectorized operations.
    """
    df = tokens_df.copy()
    df["is_case_marker"] = False
    df["is_case_marked"] = False
    df["case_type"] = ""

    
    case_mask = df["deprel"].str.startswith("case", na=False)


    if case_marker_forms:
        form_mask = df["form"].isin(case_marker_forms)
        case_mask = case_mask & form_mask

    
    df.loc[case_mask, "is_case_marker"] = True
    df.loc[case_mask, "case_type"] = df.loc[case_mask, "form"]

    
    case_info = df[case_mask][["lang", "sent_id", "head", "form"]].copy()
    case_info = case_info.rename(columns={"head": "token_id", "form": "extracted_case"})

    
    case_info = (
        case_info.groupby(["lang", "sent_id", "token_id"])["extracted_case"]
        .apply(lambda x: "+".join(x))
        .reset_index()
    )

    
    df = df.merge(case_info, on=["lang", "sent_id", "token_id"], how="left")

    
    merged_mask = df["extracted_case"].notna()
    df.loc[merged_mask, "is_case_marked"] = True
    df.loc[merged_mask, "case_type"] = df.loc[merged_mask, "extracted_case"]

    
    df = df.drop(columns=["extracted_case"])

    return df


def detect_case_by_morphology(tokens_df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect case marking via morphological features (for inflectional langs like German, Turkish).

    Uses the Case= feature from the UD feats column.
    """
    df = tokens_df.copy()

    
    df["is_case_marker"] = False


    df["is_case_marked"] = df["feats_str"].apply(_has_case_feature)
    df["case_type"] = df["feats_str"].apply(_extract_case_value)

    return df




def detect_hindi(tokens_df: pd.DataFrame) -> pd.DataFrame:
    """Detect case marking in Hindi (postpositions via deprel='case')."""
    config = get_language_config("hindi")
    known_markers = set(config.get("case_markers", {}).get("postpositions", []))
    df = detect_case_by_deprel(tokens_df, case_marker_forms=known_markers)
    
    morph_case = tokens_df["feats_str"].apply(_has_case_feature)
    df.loc[morph_case & ~df["is_case_marked"], "is_case_marked"] = True
    df.loc[morph_case & (df["case_type"] == ""), "case_type"] = (
        tokens_df.loc[morph_case & (df["case_type"] == ""), "feats_str"]
        .apply(_extract_case_value)
    )
    return df


def detect_japanese(tokens_df: pd.DataFrame) -> pd.DataFrame:
    """Detect case marking in Japanese (particles via deprel='case')."""
    config = get_language_config("japanese")
    known_markers = set(config.get("case_markers", {}).get("particles", []))
    return detect_case_by_deprel(tokens_df, case_marker_forms=known_markers)


def detect_korean(tokens_df: pd.DataFrame) -> pd.DataFrame:
    """Detect case marking in Korean (particles via deprel='case')."""
    config = get_language_config("korean")
    known_markers = set(config.get("case_markers", {}).get("particles", []))
    return detect_case_by_deprel(tokens_df, case_marker_forms=known_markers)


def detect_german(tokens_df: pd.DataFrame) -> pd.DataFrame:
    """Detect case marking in German (morphological inflection on determiners/nouns)."""
    return detect_case_by_morphology(tokens_df)


def detect_turkish(tokens_df: pd.DataFrame) -> pd.DataFrame:
    """Detect case marking in Turkish (agglutinative suffixes, morphological features)."""
    return detect_case_by_morphology(tokens_df)



# Dispatcher


# Map language keys to their detector functions
_DETECTORS = {
    "hindi": detect_hindi,
    "japanese": detect_japanese,
    "korean": detect_korean,
    "german": detect_german,
    "turkish": detect_turkish,
}


def detect_case_marking(lang: str, tokens_df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect case marking for a given language.

    For non-case-marking languages (English, Chinese, etc.), simply adds
    False columns for consistency.
    """
    config = get_language_config(lang)

    if config["type"] == "non_case_marking":
        df = tokens_df.copy()
        df["is_case_marker"] = False
        df["is_case_marked"] = False
        df["case_type"] = ""
        return df

    if lang not in _DETECTORS:
        logger.warning(
            f"No case detector for '{lang}', using morphological fallback"
        )
        return detect_case_by_morphology(tokens_df)

    detector = _DETECTORS[lang]
    return detector(tokens_df)


def detect_all_languages() -> pd.DataFrame:
    """
    Run case detection for all configured languages.
    Loads per-language token data, annotates case marking, and saves results.
    """
    from src.utils import get_all_languages

    all_langs = get_all_languages()
    dfs = []

    for lang in all_langs:
        try:
            tokens_df = load_dataframe(f"tokens_{lang}")
        except FileNotFoundError:
            logger.warning(f"Skipping {lang}: token data not found")
            continue

        # Run case detection
        annotated = detect_case_marking(lang, tokens_df)

        # Statistics
        n_tokens = len(annotated)
        n_case_markers = annotated["is_case_marker"].sum()
        n_case_marked = annotated["is_case_marked"].sum()

        config = get_language_config(lang)
        lang_type = config["type"]

        logger.info(
            f"✓ {lang} ({lang_type}): "
            f"{n_tokens} tokens, "
            f"{n_case_markers} case markers ({100*n_case_markers/n_tokens:.1f}%), "
            f"{n_case_marked} case-marked tokens ({100*n_case_marked/n_tokens:.1f}%)"
        )

        if lang_type == "case_marking":
            # Show top case markers
            top_cases = (
                annotated[annotated["is_case_marker"]]
                ["form"]
                .value_counts()
                .head(10)
            )
            if len(top_cases) > 0:
                logger.info(f"  Top case markers: {dict(top_cases)}")

        save_dataframe(annotated, f"tokens_case_{lang}")
        dfs.append(annotated)

    if not dfs:
        raise RuntimeError("No languages processed.")

    combined = pd.concat(dfs, ignore_index=True)
    save_dataframe(combined, "tokens_case_all")

    return combined



# Merge case info into dependency arcs


def annotate_arcs_with_case(
    arcs_df: pd.DataFrame, tokens_case_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add case-marking information to dependency arcs.

    For each arc, determines:
      - dep_is_case_marked: whether the dependent has case marking
      - dep_case_type: the case type on the dependent
      - head_is_case_marked: whether the head has case marking
      - head_case_type: the case type on the head
    """
    # Prepare dependent-side case info
    dep_case = tokens_case_df[
        ["lang", "sent_id", "token_id", "is_case_marked", "case_type"]
    ].rename(columns={
        "token_id": "dep_pos",
        "is_case_marked": "dep_is_case_marked",
        "case_type": "dep_case_type",
    })

    # Prepare head-side case info
    head_case = tokens_case_df[
        ["lang", "sent_id", "token_id", "is_case_marked", "case_type"]
    ].rename(columns={
        "token_id": "head_pos",
        "is_case_marked": "head_is_case_marked",
        "case_type": "head_case_type",
    })

    # Merge
    enriched = arcs_df.merge(
        dep_case, on=["lang", "sent_id", "dep_pos"], how="left"
    )
    enriched = enriched.merge(
        head_case, on=["lang", "sent_id", "head_pos"], how="left"
    )

    # Fill NaNs
    for col in ["dep_is_case_marked", "head_is_case_marked"]:
        enriched[col] = enriched[col].fillna(False)
    for col in ["dep_case_type", "head_case_type"]:
        enriched[col] = enriched[col].fillna("")

    # Add convenience column: either end is case-marked
    enriched["arc_has_case"] = (
        enriched["dep_is_case_marked"] | enriched["head_is_case_marked"]
    )

    return enriched




if __name__ == "__main__":
    detect_all_languages()
