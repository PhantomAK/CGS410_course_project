"""
DLM Case Marking — Surprisal & Entropy Engine
Collaborative Research Project

Computes word-level surprisal and entropy using a multilingual autoregressive
language model (XGLM by default).

NOTE: This module is computationally expensive. Running on GPU (e.g., Kaggle or Colab) 
is highly recommended for full dataset processing.

Formulae:
  Surprisal: S(w_i) = -log₂ P(w_i | context)
  Entropy:   H(i)  = -Σ P(w) log₂ P(w)
"""

import os
import math
from typing import List, Dict, Tuple, Optional

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import (
    logger,
    load_config,
    load_dataframe,
    save_dataframe,
    get_all_languages,
    get_cache_dir,
)


class SurprisalCalculator:
    """
    Compute token-level and word-level surprisal using a causal LM.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
    ):
        config = load_config()
        model_config = config["model"]

        self.model_name = model_name or model_config["name"]
        self.batch_size = batch_size or model_config.get("batch_size", 16)

        # Resolve device
        if device is None:
            device = model_config.get("device", "auto")
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading model: {self.model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            f"Model loaded: {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M params"
        )

    @torch.no_grad()
    def compute_token_surprisal(
        self, text: str
    ) -> List[Dict]:
        """
        Compute per-token surprisal and entropy for a single text.

        Returns a list of dicts, one per token (excluding the first token):
            {
                "token_id": int,  (position in the BPE sequence, 0-indexed)
                "token": str,
                "surprisal": float,  (in bits, base 2)
                "entropy": float,    (in bits, base 2)
            }

        Note: The first token has no preceding context, so we skip it.
        """
        # Tokenize
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=load_config()["model"].get("max_length", 512),
        )
        input_ids = inputs["input_ids"].to(self.device)

        if input_ids.shape[1] <= 1:
            return []

        # Forward pass
        outputs = self.model(input_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

        # Shift: logits[i] predicts token[i+1]
        shift_logits = logits[:, :-1, :]   # (1, seq_len-1, vocab_size)
        shift_labels = input_ids[:, 1:]     # (1, seq_len-1)

        # Log probabilities
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        probs = torch.exp(log_probs)

        # Per-token surprisal: -log₂ P(actual_token)
        token_log_probs = torch.gather(
            log_probs, 2, shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # (1, seq_len-1)

        # Convert from nats to bits
        token_surprisal = (-token_log_probs / math.log(2)).squeeze(0)

        # Per-position entropy: -Σ P(w) log₂ P(w)
        token_entropy = (
            -(probs * log_probs / math.log(2)).sum(dim=-1).squeeze(0)
        )

        # Convert to list of dicts
        results = []
        all_token_ids = input_ids.squeeze(0).tolist()

        for i in range(len(token_surprisal)):
            token_str = self.tokenizer.decode(
                [all_token_ids[i + 1]], clean_up_tokenization_spaces=False
            )
            results.append({
                "token_idx": i + 1,  # Position in BPE sequence (0-indexed)
                "token": token_str,
                "bpe_id": all_token_ids[i + 1],
                "surprisal": token_surprisal[i].item(),
                "entropy": token_entropy[i].item(),
            })

        return results

    def aggregate_to_words(
        self,
        text: str,
        words: List[str],
        token_results: List[Dict],
    ) -> List[Dict]:
        """
        Aggregate BPE-level surprisal to word-level.

        Strategy:
          - Surprisal: sum of subword log-probs, then negate and convert to bits
            (equivalent to: sum of subword surprisals in bits)
          - Entropy: use the entropy at the FIRST subword of each word
            (the point of maximum uncertainty about the word identity)

        Args:
            text: The original sentence text
            words: List of word forms from the UD tokenisation
            token_results: BPE-level results from compute_token_surprisal

        Returns:
            List of dicts, one per UD word:
                {
                    "word_pos": int (1-indexed, matching UD token_id),
                    "word": str,
                    "surprisal": float (bits),
                    "entropy": float (bits),
                    "n_subwords": int,
                }
        """
        if not token_results:
            # Return NaN for all words if no BPE tokens
            return [
                {
                    "word_pos": i + 1,
                    "word": w,
                    "surprisal": float("nan"),
                    "entropy": float("nan"),
                    "n_subwords": 0,
                }
                for i, w in enumerate(words)
            ]

        # Reconstruct the full BPE token string and try to align with words
        bpe_tokens = [r["token"] for r in token_results]

        # Add the first token (which we skipped for surprisal)
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=load_config()["model"].get("max_length", 512),
        )
        first_token = self.tokenizer.decode(
            [inputs["input_ids"][0, 0].item()],
            clean_up_tokenization_spaces=False,
        )
        all_bpe = [first_token] + bpe_tokens

        # Greedy alignment: match BPE tokens to UD words
        word_results = []
        bpe_idx = 0  # Current position in BPE sequence (0 = first token)

        for word_pos, word in enumerate(words, start=1):
            # Accumulate BPE tokens until they cover this word
            accumulated = ""
            subword_surprisals = []
            first_entropy = None
            start_bpe = bpe_idx

            while bpe_idx < len(all_bpe):
                bpe_str = all_bpe[bpe_idx].strip()
                accumulated += bpe_str

                # Get surprisal for this BPE token
                # token_results is offset by 1 (no entry for first token)
                if bpe_idx > 0 and (bpe_idx - 1) < len(token_results):
                    subword_surprisals.append(
                        token_results[bpe_idx - 1]["surprisal"]
                    )
                    if first_entropy is None:
                        first_entropy = token_results[bpe_idx - 1]["entropy"]

                bpe_idx += 1

                # Check if accumulated text covers the current word
                # Use flexible matching: strip and normalise
                clean_acc = accumulated.replace(" ", "").replace("▁", "").replace("Ġ", "")
                clean_word = word.replace(" ", "")

                if clean_word in clean_acc or len(clean_acc) >= len(clean_word):
                    break

            # Compute word-level surprisal (sum of subword surprisals)
            word_surprisal = sum(subword_surprisals) if subword_surprisals else float("nan")
            word_entropy = first_entropy if first_entropy is not None else float("nan")

            word_results.append({
                "word_pos": word_pos,
                "word": word,
                "surprisal": word_surprisal,
                "entropy": word_entropy,
                "n_subwords": len(subword_surprisals),
            })

        return word_results

    def process_sentence(
        self,
        text: str,
        words: List[str],
        lang: str,
        sent_id: str,
    ) -> pd.DataFrame:
        """
        Compute word-level surprisal for a single sentence.

        Returns DataFrame with columns:
            lang, sent_id, word_pos, word, surprisal, entropy, n_subwords
        """
        token_results = self.compute_token_surprisal(text)
        word_results = self.aggregate_to_words(text, words, token_results)

        for r in word_results:
            r["lang"] = lang
            r["sent_id"] = sent_id

        return pd.DataFrame(word_results)


def compute_surprisal_for_language(
    lang: str,
    calculator: SurprisalCalculator,
    max_sentences: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute surprisal for all sentences in a language.

    Args:
        lang: Language key
        calculator: Initialised SurprisalCalculator
        max_sentences: If set, process only this many sentences (for testing)

    Returns:
        DataFrame with word-level surprisal values
    """
    # Load token data
    tokens_df = load_dataframe(f"tokens_{lang}")

    # Get unique sentences
    sentences = (
        tokens_df.groupby("sent_id")
        .agg(
            text=("sent_text", "first"),
            words=("form", list),
        )
        .reset_index()
    )

    if max_sentences:
        sentences = sentences.head(max_sentences)

    logger.info(f"Computing surprisal for {lang}: {len(sentences)} sentences")

    results = []
    for _, row in tqdm(
        sentences.iterrows(),
        total=len(sentences),
        desc=f"Surprisal [{lang}]",
    ):
        try:
            sent_df = calculator.process_sentence(
                text=row["text"],
                words=row["words"],
                lang=lang,
                sent_id=row["sent_id"],
            )
            results.append(sent_df)
        except Exception as e:
            logger.warning(
                f"Error processing {lang}/{row['sent_id']}: {e}"
            )
            continue

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)
    return combined


def compute_all_languages(max_sentences=None):
    """Compute surprisal for all languages defined in config with robust saving."""
    logger.info("Initializing SurprisalCalculator for surprisal phase...")
    calculator = SurprisalCalculator()
    
    all_langs = get_all_languages()
    all_results = []
    
    for lang in all_langs:
        logger.info(f"============================================================")
        logger.info(f"Processing {lang}...")
        logger.info(f"============================================================")
        
        try:
            df = compute_surprisal_for_language(lang, calculator, max_sentences=max_sentences)
            
            if df is not None and not df.empty:
                save_dataframe(df, f"surprisal_{lang}")
                all_results.append(df)
                logger.info(f"✓ Saved surprisal_{lang}")
            else:
                logger.warning(f"No surprisal data generated for {lang}")
                
        except Exception as e:
            logger.error(f"Error during surprisal computation for {lang}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        save_dataframe(combined, "surprisal_all")
        logger.info("✅ Phase 4 complete: All surprisal files saved.")
        return combined
    
    return pd.DataFrame()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute word-level surprisal")
    parser.add_argument("--lang", type=str, default=None, help="Compute for a specific language")
    parser.add_argument("--max-sentences", type=int, default=None, help="Limit sentences for testing")
    args = parser.parse_args()

    from src.utils import load_config
    load_config()

    if args.lang:
        calc = SurprisalCalculator()
        df = compute_surprisal_for_language(args.lang, calc, max_sentences=args.max_sentences)
        save_dataframe(df, f"surprisal_{args.lang}")
    else:
        compute_all_languages(max_sentences=args.max_sentences)
