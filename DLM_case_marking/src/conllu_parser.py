"""
Universal Dependencies CoNLL-U parser.

Reads UD treebank files and converts them into structured pandas DataFrames
suitable for dependency analysis.
"""

import re
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import pandas as pd
from conllu import parse_incr
from tqdm import tqdm

from src.utils import (
    logger,
    load_config,
    get_raw_data_dir,
    get_processed_data_dir,
    save_dataframe,
    get_all_languages,
    get_language_config,
)


def _parse_feats(feats: Optional[Dict]) -> Dict[str, str]:
    """
    Normalise the morphological features dictionary.
    conllu library returns None when there are no features.
    """
    if feats is None:
        return {}
    return {k: v for k, v in feats.items()}


def parse_conllu_file(filepath: Path) -> List[Dict]:
    """
    Parse a single CoNLL-U file and return a list of sentence dicts.

    Each sentence dict contains:
        - sent_id: str
        - text: str (original sentence text)
        - tokens: list of token dicts with keys:
            id, form, lemma, upos, xpos, feats, head, deprel
    """
    sentences = []

    with open(filepath, "r", encoding="utf-8") as f:
        for sent in parse_incr(f):
            # Extract sentence metadata
            sent_id = sent.metadata.get("sent_id", "")
            text = sent.metadata.get("text", "")

            tokens = []
            for token in sent:
                # Skip multi-word tokens (e.g., "del" = "de" + "el")
                # They have tuple IDs like (1, 2) — we want individual words
                if isinstance(token["id"], tuple):
                    continue
                # Skip empty nodes (decimal IDs like 1.1)
                if not isinstance(token["id"], int):
                    continue

                feats = _parse_feats(token.get("feats"))

                tokens.append({
                    "id": token["id"],
                    "form": token["form"],
                    "lemma": token.get("lemma", ""),
                    "upos": token.get("upos", ""),
                    "xpos": token.get("xpos", ""),
                    "feats": feats,
                    "head": token.get("head", 0),
                    "deprel": token.get("deprel", ""),
                })

            sentences.append({
                "sent_id": sent_id,
                "text": text,
                "tokens": tokens,
            })

    return sentences


def sentences_to_dataframe(
    sentences: List[Dict], lang: str
) -> pd.DataFrame:
    """
    Convert parsed sentences to a flat DataFrame with one row per token.

    Columns:
        lang, sent_id, sent_text, token_id, form, lemma, upos, xpos,
        feats_str, head, deprel, sent_length
    """
    rows = []
    for sent in sentences:
        sent_len = len(sent["tokens"])
        for tok in sent["tokens"]:
            # Serialise feats dict to a string for storage
            feats_str = "|".join(
                f"{k}={v}" for k, v in sorted(tok["feats"].items())
            ) if tok["feats"] else ""

            rows.append({
                "lang": lang,
                "sent_id": sent["sent_id"],
                "sent_text": sent["text"],
                "token_id": tok["id"],
                "form": tok["form"],
                "lemma": tok["lemma"],
                "upos": tok["upos"],
                "xpos": tok["xpos"],
                "feats_str": feats_str,
                "head": tok["head"],
                "deprel": tok["deprel"],
                "sent_length": sent_len,
            })

    return pd.DataFrame(rows)


def find_conllu_files(treebank_dir: Path) -> List[Path]:
    """
    Find all .conllu files in a treebank directory.
    Returns files sorted by split: train, dev, test.
    """
    files = [
        p for p in treebank_dir.glob("*.conllu")
        if any(split in p.name.lower() for split in ["train", "dev", "test"])
    ]
    if not files:
        raise FileNotFoundError(
            f"No standard .conllu files (train, dev, test) found in {treebank_dir}"
        )

    # Sort so train comes first, then dev, then test
    def sort_key(p: Path) -> int:
        name = p.stem.lower()
        if "train" in name:
            return 0
        elif "dev" in name:
            return 1
        elif "test" in name:
            return 2
        return 3

    return sorted(files, key=sort_key)


def parse_language(lang: str) -> pd.DataFrame:
    """
    Parse all CoNLL-U files for a given language and return a unified DataFrame.

    The function looks for the treebank directory under data/raw/<treebank_name>/.
    """
    lang_config = get_language_config(lang)
    treebank = lang_config["treebank"]
    raw_dir = get_raw_data_dir()
    treebank_dir = raw_dir / treebank

    if not treebank_dir.exists():
        raise FileNotFoundError(
            f"Treebank directory not found: {treebank_dir}\n"
            f"Run `python download_data.py` first."
        )

    conllu_files = find_conllu_files(treebank_dir)
    logger.info(
        f"Parsing {lang} ({treebank}): {len(conllu_files)} file(s)"
    )

    all_sentences = []
    for fpath in conllu_files:
        logger.info(f"  Reading {fpath.name}...")
        sents = parse_conllu_file(fpath)
        all_sentences.extend(sents)

    logger.info(f"  Total sentences for {lang}: {len(all_sentences)}")

    df = sentences_to_dataframe(all_sentences, lang)
    return df


def parse_all_languages() -> pd.DataFrame:
    """
    Parse treebanks for all configured languages.
    Returns a single concatenated DataFrame and saves per-language files.
    """
    all_langs = get_all_languages()
    dfs = []

    for lang in all_langs:
        try:
            df = parse_language(lang)
            save_dataframe(df, f"tokens_{lang}")
            dfs.append(df)
            logger.info(
                f"✓ {lang}: {df['sent_id'].nunique()} sentences, "
                f"{len(df)} tokens"
            )
        except FileNotFoundError as e:
            logger.warning(f"✗ {lang}: {e}")
            continue

    if not dfs:
        raise RuntimeError("No treebanks were parsed. Check data/raw/.")

    combined = pd.concat(dfs, ignore_index=True)
    save_dataframe(combined, "tokens_all")

    logger.info(
        f"\n{'='*50}\n"
        f"Total: {combined['lang'].nunique()} languages, "
        f"{combined['sent_id'].nunique()} sentences, "
        f"{len(combined)} tokens\n"
        f"{'='*50}"
    )

    return combined


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parse_all_languages()
