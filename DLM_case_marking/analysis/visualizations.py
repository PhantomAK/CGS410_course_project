"""
Visualizations — All plots for the DLM Case Marking project.

Generates publication-quality figures using matplotlib and seaborn.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.utils import (
    logger, load_dataframe, get_figures_dir,
    get_language_config, get_case_marking_languages,
    get_non_case_marking_languages,
)

# ===================================================================
# Style setup
# ===================================================================

# Color palette: warm for case-marking, cool for non-case-marking
CASE_COLOR = "#E74C3C"       # Red family
NON_CASE_COLOR = "#3498DB"   # Blue family

CASE_PALETTE = ["#E74C3C", "#E67E22", "#F39C12", "#D35400", "#C0392B"]
NON_CASE_PALETTE = ["#3498DB", "#2ECC71", "#1ABC9C", "#9B59B6", "#34495E"]

def setup_style():
    """Set publication-quality plot style."""
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "font.size": 12,
        "font.family": "sans-serif",
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    sns.set_palette("Set2")


def get_lang_color(lang):
    """Get color for a language based on its type."""
    config = get_language_config(lang)
    case_langs = get_case_marking_languages()
    non_case_langs = get_non_case_marking_languages()

    if config["type"] == "case_marking":
        idx = case_langs.index(lang) % len(CASE_PALETTE)
        return CASE_PALETTE[idx]
    else:
        idx = non_case_langs.index(lang) % len(NON_CASE_PALETTE)
        return NON_CASE_PALETTE[idx]


# ===================================================================
# Plot functions
# ===================================================================

def plot_1_dep_length_by_language():
    """Bar chart of mean dependency length per language, colored by type."""
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    try:
        summary = load_dataframe("language_summaries", subdir="results")
    except FileNotFoundError:
        logger.warning("Language summaries not found. Run metrics first.")
        return

    summary = summary.sort_values("mean_dep_length", ascending=True)
    colors = [
        CASE_COLOR if t == "case_marking" else NON_CASE_COLOR
        for t in summary["lang_type"]
    ]

    bars = ax.barh(summary["lang"], summary["mean_dep_length"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Mean Dependency Length")
    ax.set_title("Mean Dependency Length by Language")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CASE_COLOR, label="Case-marking"),
        Patch(facecolor=NON_CASE_COLOR, label="Non-case-marking"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    fig.savefig(get_figures_dir() / "01_dep_length_by_language.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("✓ Plot 1: Dependency length by language")


def plot_2_surprisal_vs_distance():
    """THE key figure: surprisal × dependency length, one line per language."""
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    try:
        merged = load_dataframe("merged_arcs_surprisal_all")
    except FileNotFoundError:
        logger.warning("Merged data not found.")
        return

    # Bin dependency lengths for cleaner plot
    merged["dep_length_bin"] = merged["dep_length"].clip(upper=10)

    for lang in merged["lang"].unique():
        lang_data = merged[merged["lang"] == lang]
        binned = lang_data.groupby("dep_length_bin")["head_surprisal"].mean().reset_index()

        color = get_lang_color(lang)
        config = get_language_config(lang)
        marker = "o" if config["type"] == "case_marking" else "s"
        linestyle = "-" if config["type"] == "case_marking" else "--"

        ax.plot(
            binned["dep_length_bin"], binned["head_surprisal"],
            marker=marker, color=color, linestyle=linestyle,
            label=lang.capitalize(), linewidth=2, markersize=6, alpha=0.85,
        )

    ax.set_xlabel("Dependency Length")
    ax.set_ylabel("Mean Head Surprisal (bits)")
    ax.set_title("Surprisal at Head Word vs. Dependency Length")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True)

    plt.tight_layout()
    fig.savefig(get_figures_dir() / "02_surprisal_vs_distance.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("✓ Plot 2: Surprisal vs distance")


def plot_3_slope_comparison():
    """Bar chart comparing slopes between case-marking and non-case-marking."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, (title, filename) in enumerate([
        ("All Dependencies", "slopes_all_deps"),
        ("Argument Dependencies", "slopes_argument_deps"),
    ]):
        try:
            slopes = load_dataframe(filename, subdir="results")
        except FileNotFoundError:
            continue

        ax = axes[i]
        slopes = slopes.sort_values("slope")
        colors = [
            CASE_COLOR if t == "case_marking" else NON_CASE_COLOR
            for t in slopes["lang_type"]
        ]

        ax.barh(slopes["lang"], slopes["slope"], color=colors, edgecolor="white")
        ax.set_xlabel("Slope (Δ surprisal / Δ dep length)")
        ax.set_title(title)
        ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CASE_COLOR, label="Case-marking"),
        Patch(facecolor=NON_CASE_COLOR, label="Non-case-marking"),
    ]
    axes[0].legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    fig.savefig(get_figures_dir() / "03_slope_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("✓ Plot 3: Slope comparison")


def plot_4_uid_distribution():
    """Distribution of surprisal variance (UID) by language type."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    try:
        uid = load_dataframe("uid_per_sentence", subdir="results")
    except FileNotFoundError:
        logger.warning("UID data not found.")
        return

    # Surprisal CV distribution
    ax = axes[0]
    for lang_type, color, label in [
        ("case_marking", CASE_COLOR, "Case-marking"),
        ("non_case_marking", NON_CASE_COLOR, "Non-case-marking"),
    ]:
        data = uid[uid["lang_type"] == lang_type]["surprisal_cv"].dropna()
        data = data[data < data.quantile(0.99)]  # Trim outliers
        ax.hist(data, bins=50, alpha=0.6, color=color, label=label, density=True)

    ax.set_xlabel("Surprisal Coefficient of Variation")
    ax.set_ylabel("Density")
    ax.set_title("Information Uniformity (lower = more uniform)")
    ax.legend()

    # Per-language box plot
    ax = axes[1]
    uid_trimmed = uid[uid["surprisal_cv"] < uid["surprisal_cv"].quantile(0.99)]
    order = uid_trimmed.groupby("lang")["surprisal_cv"].median().sort_values().index

    sns.boxplot(
        data=uid_trimmed, x="lang", y="surprisal_cv", ax=ax,
        order=order, palette={
            lang: get_lang_color(lang) for lang in order
        },
    )
    ax.set_xlabel("Language")
    ax.set_ylabel("Surprisal CV")
    ax.set_title("Information Uniformity by Language")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    fig.savefig(get_figures_dir() / "04_uid_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("✓ Plot 4: UID distribution")


def plot_5_critical_region():
    """Surprisal at reunion points by distance bin and language type."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    try:
        reunion = load_dataframe("reunion_cost_by_bin", subdir="results")
    except FileNotFoundError:
        logger.warning("Reunion data not found.")
        return

    # Aggregate by lang_type and distance bin
    agg = reunion.groupby(["lang_type", "distance_bin"]).agg(
        mean_surprisal=("mean_head_surprisal", "mean"),
        se=("mean_head_surprisal", "sem"),
    ).reset_index()

    for lang_type, color, marker in [
        ("case_marking", CASE_COLOR, "o"),
        ("non_case_marking", NON_CASE_COLOR, "s"),
    ]:
        data = agg[agg["lang_type"] == lang_type].sort_values("distance_bin")
        label = lang_type.replace("_", "-").title()
        ax.errorbar(
            data["distance_bin"], data["mean_surprisal"],
            yerr=data["se"], marker=marker, color=color,
            label=label, linewidth=2, markersize=8, capsize=4,
        )

    ax.set_xlabel("Dependency Distance Bin")
    ax.set_ylabel("Mean Head Surprisal at Reunion (bits)")
    ax.set_title("Processing Cost at Dependency Reunion Points")
    ax.legend()

    plt.tight_layout()
    fig.savefig(get_figures_dir() / "05_critical_region.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("✓ Plot 5: Critical region")


def plot_6_ablation_delta():
    """Delta surprisal from ablation vs dependency length."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    try:
        ablation = load_dataframe("ablation_all")
        arcs = load_dataframe("arcs_all")
    except FileNotFoundError:
        logger.warning("Ablation data not found.")
        return

    # Merge with dep length
    merged = ablation.merge(
        arcs[["lang", "sent_id", "head_pos", "dep_length"]],
        left_on=["lang", "sent_id", "word_pos"],
        right_on=["lang", "sent_id", "head_pos"],
        how="inner",
    )
    merged = merged.dropna(subset=["delta_surprisal"])
    merged["dep_length_bin"] = merged["dep_length"].clip(upper=10)

    for lang in merged["lang"].unique():
        lang_data = merged[merged["lang"] == lang]
        binned = lang_data.groupby("dep_length_bin")["delta_surprisal"].mean().reset_index()
        ax.plot(
            binned["dep_length_bin"], binned["delta_surprisal"],
            marker="o", color=get_lang_color(lang),
            label=lang.capitalize(), linewidth=2, markersize=6,
        )

    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Dependency Length")
    ax.set_ylabel("Δ Surprisal (ablated − original, bits)")
    ax.set_title("Information Contributed by Case Markers vs Distance")
    ax.legend()

    plt.tight_layout()
    fig.savefig(get_figures_dir() / "06_ablation_delta.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("✓ Plot 6: Ablation delta")


def generate_all_plots():
    """Generate all figures."""
    logger.info("=" * 60)
    logger.info("GENERATING ALL VISUALIZATIONS")
    logger.info("=" * 60)

    plot_1_dep_length_by_language()
    plot_2_surprisal_vs_distance()
    plot_3_slope_comparison()
    plot_4_uid_distribution()
    plot_5_critical_region()
    plot_6_ablation_delta()

    logger.info(f"\n✓ All plots saved to {get_figures_dir()}")


if __name__ == "__main__":
    generate_all_plots()
