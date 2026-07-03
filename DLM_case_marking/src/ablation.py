

from typing import List, Optional, Set
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.utils import (
    logger, load_config, get_language_config,
    get_case_marking_languages, load_dataframe, save_dataframe,
)
from src.surprisal_calculator import SurprisalCalculator


def ablate_particle_language(words, case_marker_positions):
    
    ablated_words = []
    position_mapping = {}
    new_pos = 1
    for i, word in enumerate(words, start=1):
        if i in case_marker_positions:
            position_mapping[i] = None
        else:
            ablated_words.append(word)
            position_mapping[i] = new_pos
            new_pos += 1
    return " ".join(ablated_words), ablated_words, position_mapping


def run_ablation_for_language(lang, calculator, max_sentences=None):
    
    lang_config = get_language_config(lang)
    if lang_config["type"] != "case_marking":
        return pd.DataFrame()

    tokens_df = load_dataframe(f"tokens_case_{lang}")
    original_surp = load_dataframe(f"surprisal_{lang}")
    is_particle_lang = lang in ("hindi", "japanese", "korean")

    sentences = tokens_df.groupby("sent_id")
    results = []
    sent_ids = list(sentences.groups.keys())
    if max_sentences:
        sent_ids = sent_ids[:max_sentences]

    for sent_id in tqdm(sent_ids, desc=f"Ablation [{lang}]"):
        sent_tokens = sentences.get_group(sent_id)
        words = sent_tokens["form"].tolist()

        if is_particle_lang:
            case_positions = set(
                sent_tokens[sent_tokens["is_case_marker"]]["token_id"].tolist()
            )
        else:
            case_positions = set(
                sent_tokens[sent_tokens["is_case_marked"]]["token_id"].tolist()
            )

        if not case_positions:
            continue

       
        if is_particle_lang:
            ablated_text, ablated_words, pos_map = ablate_particle_language(
                words, case_positions
            )
        else:
            ablated_words = []
            pos_map = {}
            new_pos = 1
            for _, tok in sent_tokens.iterrows():
                pos = tok["token_id"]
                if pos in case_positions:
                    ablated_words.append(tok["lemma"])
                else:
                    ablated_words.append(tok["form"])
                pos_map[pos] = new_pos
                new_pos += 1
            ablated_text = " ".join(ablated_words)

        if not ablated_words:
            continue

        try:
            ablated_results = calculator.process_sentence(
                ablated_text, ablated_words, lang, sent_id,
            )
        except Exception as e:
            logger.warning(f"Ablation error {lang}/{sent_id}: {e}")
            continue

        orig_sent = original_surp[original_surp["sent_id"] == sent_id]
        for _, orig_row in orig_sent.iterrows():
            orig_pos = orig_row["word_pos"]
            if orig_pos in pos_map and pos_map[orig_pos] is not None:
                ablated_pos = pos_map[orig_pos]
                ablated_match = ablated_results[
                    ablated_results["word_pos"] == ablated_pos
                ]
                ablated_surp = (
                    ablated_match.iloc[0]["surprisal"]
                    if len(ablated_match) > 0 else float("nan")
                )
            else:
                ablated_surp = float("nan")

            delta = (
                ablated_surp - orig_row["surprisal"]
                if not (np.isnan(ablated_surp) or np.isnan(orig_row["surprisal"]))
                else float("nan")
            )
            results.append({
                "lang": lang, "sent_id": sent_id,
                "word_pos": orig_pos, "word": orig_row["word"],
                "surprisal_original": orig_row["surprisal"],
                "surprisal_ablated": ablated_surp,
                "delta_surprisal": delta,
            })

    return pd.DataFrame(results)


def run_all_ablations(max_sentences=None):
    
    logger.info("Initializing SurprisalCalculator for ablation phase...")
    calculator = SurprisalCalculator()
    
    case_langs = get_case_marking_languages()
    all_results = []
    
    for lang in case_langs:
        logger.info(f"============================================================")
        logger.info(f"Ablating {lang}...")
        logger.info(f"============================================================")
        
        try:
            df = run_ablation_for_language(lang, calculator, max_sentences)
            
            if df is not None and len(df) > 0:
                save_dataframe(df, f"ablation_{lang}")
                all_results.append(df)
            else:
                logger.warning(f"No ablation results for {lang}")
                
        except Exception as e:
            logger.error(f"Error during ablation for {lang}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        save_dataframe(combined, "ablation_all")
        logger.info("✅ Phase 5 complete: Combined ablation results saved.")
        return combined
    
    return pd.DataFrame()


if __name__ == "__main__":
    
    from src.utils import load_config
    load_config()
    run_all_ablations()
