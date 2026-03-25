"""
Tool: load_dataset.py

Purpose: Downloads the HS code dataset from GitHub and validates its structure.
         Saves to data/hs_codes.csv in the project directory.

Inputs:  None (downloads from hardcoded URL).
Outputs: data/hs_codes.csv file.
Logging: Logs download progress, validation results, and errors.
Failure: Retries up to 3 times on network failure.
"""

import sys
import time
import urllib.request
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.utils.logger import get_logger

logger = get_logger("load_dataset")

DATASET_URL = (
    "https://raw.githubusercontent.com/datasets/"
    "harmonized-system/master/data/harmonized-system.csv"
)
OUTPUT_PATH = PROJECT_ROOT / "data" / "hs_codes.csv"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


def download_dataset() -> Path:
    """Download the HS dataset with retry logic.

    Returns:
        Path to the downloaded CSV file.

    Raises:
        RuntimeError: If download fails after all retries.
    """
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("Downloading HS dataset (attempt %d/%d)...", attempt, MAX_RETRIES)
            urllib.request.urlretrieve(DATASET_URL, str(OUTPUT_PATH))
            logger.info("Downloaded to %s", OUTPUT_PATH)
            return OUTPUT_PATH
        except Exception as e:
            logger.warning("Download failed: %s", e)
            if attempt < MAX_RETRIES:
                logger.info("Retrying in %d seconds...", RETRY_DELAY)
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(f"Failed to download dataset after {MAX_RETRIES} attempts: {e}")


def validate_dataset(path: Path) -> bool:
    """Validate the downloaded CSV structure.

    Returns:
        True if valid, False otherwise.
    """
    import csv

    logger.info("Validating dataset at %s...", path)

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        required = {"section", "hscode", "description", "parent", "level"}
        missing = required - set(headers)
        if missing:
            logger.error("Missing columns: %s", missing)
            return False

        row_count = sum(1 for _ in reader) + 1  # +1 for header

    logger.info("Dataset valid: %d rows, columns: %s", row_count, headers)
    return True


if __name__ == "__main__":
    if OUTPUT_PATH.exists():
        logger.info("Dataset already exists at %s", OUTPUT_PATH)
        if validate_dataset(OUTPUT_PATH):
            logger.info("Existing dataset is valid.")
        else:
            logger.warning("Existing dataset is invalid. Re-downloading...")
            path = download_dataset()
            validate_dataset(path)
    else:
        path = download_dataset()
        validate_dataset(path)
