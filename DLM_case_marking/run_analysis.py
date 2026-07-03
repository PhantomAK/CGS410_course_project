

import pandas as pd
import numpy as np
from scipy import stats
import os
from pathlib import Path

from src.utils import (
    logger, load_dataframe, save_dataframe, 
    get_all_languages, get_case_marking_languages, 
    load_config, get_processed_data_dir
)
from src.analysis_utils import calculate_slopes, calculate_delta_surprisal_stats

def main():
    logger.info("============================================================")
    logger.info("PHASE 6: Statistical Analysis")
    logger.info("============================================================")

    load_config()
    all_langs = get_all_languages()
    case_marking_langs = get_case_marking_languages()
    processed_dir = get_processed_data_dir()
    
    logger.info(f"Checking data directory: {processed_dir.absolute()}")
    available_files = os.listdir(processed_dir)

    all_slope_data = []
    all_delta_data = []

    for lang in all_langs:
    
        arcs_file = next((f for f in available_files if f.startswith(f"arcs_{lang}")), None)
        surp_file = next((f for f in available_files if f.startswith(f"surprisal_{lang}")), None)
        
        if not arcs_file or not surp_file:
            continue

        logger.info(f"Analyzing {lang}...")
        try:
            arcs = load_dataframe(f"arcs_{lang}")
            surprisal = load_dataframe(f"surprisal_{lang}")
            
            
            id_col = 'id' if 'id' in arcs.columns else ('token_id' if 'token_id' in arcs.columns else None)
           
            pos_col = 'word_pos' if 'word_pos' in surprisal.columns else ('token_id' if 'token_id' in surprisal.columns else None)
            
            if not id_col or not pos_col:
                logger.error(f"  ✗ Column mismatch for {lang}. Arcs columns: {list(arcs.columns)}, Surp columns: {list(surprisal.columns)}")
                continue

           
            arcs[id_col] = arcs[id_col].astype(str)
            surprisal[pos_col] = surprisal[pos_col].astype(str)
            arcs["sent_id"] = arcs["sent_id"].astype(str)
            surprisal["sent_id"] = surprisal["sent_id"].astype(str)

            merged = pd.merge(
                arcs, 
                surprisal,
                left_on=["lang", "sent_id", id_col],
                right_on=["lang", "sent_id", pos_col],
                how="inner"
            )
            
            if merged.empty:
                logger.warning(f"  ! Merge empty for {lang}. Check if 'sent_id' values match.")
                continue

        
            lang_slope = calculate_slopes(merged)
            if not lang_slope.empty:
                lang_slope["group"] = "Case-Marking" if lang in case_marking_langs else "Non-Case-Marking"
                all_slope_data.append(lang_slope)
                logger.info(f"  ✓ Calculated slope: {lang_slope.iloc[0]['slope']:.6f}")
            
            
            if lang in case_marking_langs:
                try:
                    ablation = load_dataframe(f"ablation_{lang}")
                    lang_delta = calculate_delta_surprisal_stats(ablation)
                    all_delta_data.append(lang_delta)
                except: pass

        except Exception as e:
            logger.error(f"  ✗ Error processing {lang}: {e}")
            continue

   
    if all_slope_data:
        slopes_results = pd.concat(all_slope_data, ignore_index=True)
        save_dataframe(slopes_results, "slope_results", subdir="results")
        
        case_slopes = slopes_results[slopes_results["group"] == "Case-Marking"]["slope"]
        control_slopes = slopes_results[slopes_results["group"] == "Non-Case-Marking"]["slope"]
        
        t_stat, p_val = stats.ttest_ind(case_slopes, control_slopes)
        
        logger.info("\n" + "="*40)
        logger.info("FINAL STATISTICAL RESULTS")
        logger.info("="*40)
        logger.info(f"Mean Slope (Case):    {case_slopes.mean():.6f}")
        logger.info(f"Mean Slope (Control): {control_slopes.mean():.6f}")
        logger.info(f"Mitigation Effect:   {control_slopes.mean() - case_slopes.mean():.6f}")
        logger.info(f"Significance (p):    {p_val:.6f}")
        logger.info("="*40)
    else:
        logger.error("No data could be merged! Check your sent_id formats.")
    
    if all_delta_data:
        delta_results = pd.concat(all_delta_data, ignore_index=True)
        save_dataframe(delta_results, "ablation_summary", subdir="results")

    logger.info("✅ Phase 6 complete!")

if __name__ == "__main__":
    main()
