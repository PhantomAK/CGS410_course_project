

import os
import sys
import zipfile
import tarfile
import shutil
import requests
from pathlib import Path
from io import BytesIO


sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import (
    logger,
    load_config,
    get_raw_data_dir,
    get_all_languages,
    get_language_config,
)


UD_GITHUB_BASE = "https://github.com/UniversalDependencies"


def download_treebank(lang: str, force: bool = False) -> Path:
    """
    Download a UD treebank for a given language from GitHub.
    Downloads the repository as a zip archive and extracts the CoNLL-U files.
    Args:
        lang: Language key from config (e.g. 'hindi')
        force: If True, re-download even if files already exist
    Returns:
        Path to the extracted treebank directory
    """
    config = get_language_config(lang)
    treebank = config["treebank"]
    raw_dir = get_raw_data_dir()
    treebank_dir = raw_dir / treebank

    
    if treebank_dir.exists() and not force:
        conllu_files = list(treebank_dir.glob("*.conllu"))
        if conllu_files:
            logger.info(
                f"✓ {lang} ({treebank}): already downloaded "
                f"({len(conllu_files)} files)"
            )
            return treebank_dir

   
    zip_url = f"{UD_GITHUB_BASE}/{treebank}/archive/refs/heads/master.zip"
    logger.info(f"Downloading {treebank} from {zip_url}...")

    try:
        response = requests.get(zip_url, timeout=120, stream=True)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to download {treebank}: {e}")
        raise

    
    logger.info(f"Extracting {treebank}...")
    treebank_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(BytesIO(response.content)) as zf:
      
        prefix = f"{treebank}-master/"
        for member in zf.namelist():
            if member.startswith(prefix) and member.endswith(".conllu"):
               
                filename = Path(member).name
                target = treebank_dir / filename
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                logger.info(f"  Extracted: {filename}")

    conllu_files = list(treebank_dir.glob("*.conllu"))
    if not conllu_files:
        logger.warning(
            f"No .conllu files found after extraction for {treebank}!"
        )
    else:
        logger.info(
            f"✓ {lang} ({treebank}): {len(conllu_files)} CoNLL-U files"
        )

    return treebank_dir


def download_all(force: bool = False):
    """Download treebanks for all configured languages."""
    langs = get_all_languages()

    logger.info(f"Downloading UD treebanks for {len(langs)} languages...")
    logger.info("=" * 60)

    success = []
    failed = []

    for lang in langs:
        try:
            download_treebank(lang, force=force)
            success.append(lang)
        except Exception as e:
            logger.error(f"✗ {lang}: {e}")
            failed.append(lang)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Download complete: {len(success)} succeeded, {len(failed)} failed")
    if failed:
        logger.warning(f"Failed languages: {failed}")




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download UD treebanks for the DLM case marking project"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="Download only a specific language (e.g. 'hindi')",
    )

    args = parser.parse_args()

    if args.lang:
        download_treebank(args.lang, force=args.force)
    else:
        download_all(force=args.force)
