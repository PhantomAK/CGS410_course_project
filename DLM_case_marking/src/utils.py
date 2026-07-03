"""
DLM Case Marking — Research Utility Suite
Collaborative Research Project

Shared utility functions for data I/O, path management, and logging.
"""

import os
import yaml
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the project logger."""
    logger = logging.getLogger("dlm_case_marking")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s — %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


logger = setup_logging()

# ---------------------------------------------------------------------------
# Configuration & Path Management
# ---------------------------------------------------------------------------

_ORIGINAL_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def is_writable(path: Path) -> bool:
    """Check if a directory is writable."""
    if not path.exists():
        try:
            path.mkdir(parents=True, exist_ok=True)
            return True
        except (OSError, PermissionError):
            return False
    return os.access(path, os.W_OK)


def get_project_root() -> Path:
    """Return the root directory. Returns CWD if the original root is read-only."""
    if not is_writable(_ORIGINAL_ROOT):
        return Path.cwd()
    return _ORIGINAL_ROOT


def get_data_input_root() -> Path:
    """Always return the original root for reading raw data."""
    return _ORIGINAL_ROOT


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the project configuration from config.yaml.
    Results are cached after the first call.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    if config_path is None:
        # Check CWD first, then original root
        config_path = Path.cwd() / "config.yaml"
        if not config_path.exists():
            config_path = _ORIGINAL_ROOT / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        _CONFIG_CACHE = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_path}")
    return _CONFIG_CACHE


def get_language_config(lang: str) -> Dict[str, Any]:
    """Get configuration for a specific language."""
    config = load_config()
    if lang not in config["languages"]:
        raise ValueError(
            f"Unknown language '{lang}'. "
            f"Available: {list(config['languages'].keys())}"
        )
    return config["languages"][lang]


def get_case_marking_languages() -> List[str]:
    """Return list of case-marking language keys."""
    config = load_config()
    return [
        lang for lang, info in config["languages"].items()
        if info["type"] == "case_marking"
    ]


def get_non_case_marking_languages() -> List[str]:
    """Return list of non-case-marking language keys."""
    config = load_config()
    return [
        lang for lang, info in config["languages"].items()
        if info["type"] == "non_case_marking"
    ]


def get_all_languages() -> List[str]:
    """Return list of all language keys."""
    config = load_config()
    return list(config["languages"].keys())

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist, return Path object."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_raw_data_dir() -> Path:
    """Return path to raw data directory (usually read-only)."""
    config = load_config()
    return get_data_input_root() / config["paths"]["raw_data"]


def get_processed_data_dir() -> Path:
    """Return path to processed data directory (must be writable)."""
    config = load_config()
    return ensure_dir(get_project_root() / config["paths"]["processed_data"])


def get_results_dir() -> Path:
    """Return path to results directory."""
    config = load_config()
    return ensure_dir(get_project_root() / config["paths"]["results"])


def get_figures_dir() -> Path:
    """Return path to figures directory."""
    config = load_config()
    return ensure_dir(get_project_root() / config["paths"]["figures"])


def get_tables_dir() -> Path:
    """Return path to tables directory."""
    config = load_config()
    return ensure_dir(get_project_root() / config["paths"]["tables"])


def get_cache_dir() -> Path:
    """Return path to cache directory."""
    config = load_config()
    return ensure_dir(get_project_root() / config["paths"]["cache"])

# ---------------------------------------------------------------------------
# Data I/O helpers
# ---------------------------------------------------------------------------

def save_dataframe(df: pd.DataFrame, name: str, subdir: str = "processed") -> Path:
    """
    Save a DataFrame to the processed data directory.
    """
    if subdir == "processed":
        base_dir = get_processed_data_dir()
    elif subdir == "results":
        base_dir = get_results_dir()
    else:
        base_dir = ensure_dir(get_project_root() / subdir)

    parquet_path = base_dir / f"{name}.parquet"
    csv_path = base_dir / f"{name}.csv"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    logger.info(f"Saved {name}: {len(df)} rows → {parquet_path}")
    return parquet_path


def load_dataframe(name: str, subdir: str = "processed") -> pd.DataFrame:
    """
    Load a DataFrame, searching both writable and input directories.
    """
    config = load_config()
    
    if subdir == "processed":
        search_dirs = [
            get_processed_data_dir(), 
            get_data_input_root() / config["paths"]["processed_data"]
        ]
    elif subdir == "results":
        search_dirs = [get_results_dir()]
    else:
        search_dirs = [
            get_project_root() / subdir, 
            get_data_input_root() / subdir
        ]

    for base_dir in search_dirs:
        if not base_dir.exists():
            continue
            
        parquet_path = base_dir / f"{name}.parquet"
        csv_path = base_dir / f"{name}.csv"

        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            logger.info(f"Loaded {name} from Parquet ({base_dir})")
            return df
        elif csv_path.exists():
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {name} from CSV ({base_dir})")
            return df

    raise FileNotFoundError(
        f"Data file '{name}' not found in any of: {[str(d) for d in search_dirs]}"
    )
