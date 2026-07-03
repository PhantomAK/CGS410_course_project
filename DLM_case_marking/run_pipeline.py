

import argparse
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import logger


def phase_download():
    """Phase 1: Download UD treebanks."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Downloading UD treebanks")
    logger.info("=" * 60)
    from download_data import download_all
    download_all()


def phase_parse():
    """Phase 2: Parse CoNLL-U files into DataFrames."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Parsing UD treebanks")
    logger.info("=" * 60)
    from src.conllu_parser import parse_all_languages
    parse_all_languages()


def phase_extract():
    """Phase 3: Extract dependencies and detect case markers."""
    logger.info("=" * 60)
    logger.info("PHASE 3: Extracting dependencies & detecting case markers")
    logger.info("=" * 60)

    from src.dependency_extractor import extract_all_languages
    extract_all_languages()

    from src.case_marker_detector import detect_all_languages, annotate_arcs_with_case
    detect_all_languages()

   
    from src.utils import load_dataframe, save_dataframe
    arcs = load_dataframe("arcs_all")
    tokens_case = load_dataframe("tokens_case_all")
    enriched = annotate_arcs_with_case(arcs, tokens_case)
    save_dataframe(enriched, "arcs_all")
    logger.info(f"✓ Enriched {len(enriched)} arcs with case annotations")


def phase_surprisal():
    """Phase 4: Compute surprisal (needs GPU)."""
    logger.info("=" * 60)
    logger.info("PHASE 4: Computing surprisal & entropy")
    logger.info("=" * 60)
    from src.surprisal_calculator import compute_all_languages
    compute_all_languages()


def phase_ablation():
    """Phase 5: Run case marker ablation (needs GPU)."""
    logger.info("=" * 60)
    logger.info("PHASE 5: Case marker ablation")
    logger.info("=" * 60)
    from src.ablation import run_all_ablations
    run_all_ablations()


def _prepare_analysis_data():
    """Pre-step: Concatenate per-language files needed by analysis modules."""
    from src.utils import (
        load_dataframe, save_dataframe,
        get_all_languages, get_case_marking_languages,
        get_processed_data_dir,
    )

    processed = get_processed_data_dir()

   
    if not (processed / "surprisal_all.parquet").exists():
        logger.info("Creating surprisal_all from per-language files...")
        all_surp = []
        for lang in get_all_languages():
            try:
                all_surp.append(load_dataframe(f"surprisal_{lang}"))
            except FileNotFoundError:
                logger.warning(f"  surprisal_{lang} not found, skipping")
        if all_surp:
            combined = pd.concat(all_surp, ignore_index=True)
            save_dataframe(combined, "surprisal_all")
            logger.info(f"  ✓ surprisal_all: {len(combined)} rows")
    else:
        logger.info("surprisal_all already exists, skipping.")

    
    if not (processed / "ablation_all.parquet").exists():
        logger.info("Creating ablation_all from per-language files...")
        all_abl = []
        for lang in get_case_marking_languages():
            try:
                all_abl.append(load_dataframe(f"ablation_{lang}"))
            except FileNotFoundError:
                logger.warning(f"  ablation_{lang} not found, skipping")
        if all_abl:
            combined = pd.concat(all_abl, ignore_index=True)
            save_dataframe(combined, "ablation_all")
            logger.info(f"  ✓ ablation_all: {len(combined)} rows")
    else:
        logger.info("ablation_all already exists, skipping.")


def phase_analyze():
    """Phase 6: Run all analyses."""
    logger.info("=" * 60)
    logger.info("PHASE 6: Running analyses")
    logger.info("=" * 60)

    
    _prepare_analysis_data()

    from src.metrics import compute_all_summaries
    compute_all_summaries()

    from analysis.core_analysis import run_core_analysis
    run_core_analysis()

    from analysis.statistical_models import run_all_models
    run_all_models()

    from analysis.uid_analysis import run_uid_analysis
    run_uid_analysis()

    from analysis.critical_region_analysis import run_critical_region_analysis
    run_critical_region_analysis()


def phase_visualize():
    """Phase 7: Generate all plots."""
    logger.info("=" * 60)
    logger.info("PHASE 7: Generating visualizations")
    logger.info("=" * 60)
    from analysis.visualizations import generate_all_plots
    generate_all_plots()



PHASES = {
    "download": phase_download,
    "parse": phase_parse,
    "extract": phase_extract,
    "surprisal": phase_surprisal,
    "ablation": phase_ablation,
    "analyze": phase_analyze,
    "visualize": phase_visualize,
}

ALL_PHASES_ORDER = ["download", "parse", "extract", "surprisal", "ablation", "analyze", "visualize"]


def main():
    parser = argparse.ArgumentParser(description="DLM Case Marking Pipeline")
    parser.add_argument(
        "--phase", type=str, default="all",
        choices=list(PHASES.keys()) + ["all"],
        help="Which phase to run (default: all)",
    )
    args = parser.parse_args()

    if args.phase == "all":
        for phase_name in ALL_PHASES_ORDER:
            PHASES[phase_name]()
            logger.info("")
    else:
        PHASES[args.phase]()

    logger.info("\n✅ Pipeline complete!")


if __name__ == "__main__":
    main()
