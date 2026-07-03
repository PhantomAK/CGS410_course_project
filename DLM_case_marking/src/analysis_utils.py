"""
Analysis Utilities for DLM Case Marking.
Contains statistical models and metric calculations.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any

def calculate_slopes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the regression slope of surprisal vs. dependency length.
    df must contain 'surprisal' and 'dep_length'.
    """
    results = []
    
    # We group by language and dependency type (or just language)
    for lang in df['lang'].unique():
        lang_df = df[df['lang'] == lang].dropna(subset=['surprisal', 'dep_length'])
        
        if len(lang_df) < 10:
            continue
            
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            lang_df['dep_length'], 
            lang_df['surprisal']
        )
        
        results.append({
            'lang': lang,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'n': len(lang_df)
        })
        
    return pd.DataFrame(results)

def calculate_delta_surprisal_stats(ablation_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate mean delta surprisal and significance per language.
    """
    results = []
    for lang in ablation_df['lang'].unique():
        lang_df = ablation_df[ablation_df['lang'] == lang].dropna(subset=['delta_surprisal'])
        
        if len(lang_df) < 5:
            continue
            
        mean_delta = lang_df['delta_surprisal'].mean()
        std_delta = lang_df['delta_surprisal'].std()
        
        # One-sample t-test (is delta significantly > 0?)
        t_stat, p_val = stats.ttest_1samp(lang_df['delta_surprisal'], 0)
        
        results.append({
            'lang': lang,
            'mean_delta': mean_delta,
            'std_delta': std_delta,
            't_stat': t_stat,
            'p_value': p_val,
            'n': len(lang_df)
        })
        
    return pd.DataFrame(results)
