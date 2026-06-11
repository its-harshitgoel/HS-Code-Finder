"""Downloads the HS code dataset from GitHub and saves to data/hs_codes.csv."""

import sys
import time
import urllib.request
from pathlib import Path

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
RETRY_DELAY = 2


def download_dataset() -> Path:
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
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(f"Failed to download dataset after {MAX_RETRIES} attempts: {e}")


def validate_dataset(path: Path) -> bool:
    import csv
    logger.info("Validating dataset at %s...", path)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        missing = {"section", "hscode", "description", "parent", "level"} - set(headers)
        if missing:
            logger.error("Missing columns: %s", missing)
            return False
        row_count = sum(1 for _ in reader) + 1
    logger.info("Dataset valid: %d rows", row_count)
    return True


if __name__ == "__main__":
    if OUTPUT_PATH.exists():
        logger.info("Dataset already exists at %s", OUTPUT_PATH)
        if not validate_dataset(OUTPUT_PATH):
            logger.warning("Invalid dataset, re-downloading...")
            validate_dataset(download_dataset())
    else:
        validate_dataset(download_dataset())
