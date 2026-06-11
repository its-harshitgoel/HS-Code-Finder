from pathlib import Path

import pandas as pd

from models.schemas import HSEntry
from utils.logger import get_logger

logger = get_logger("hs_knowledge")


class HSKnowledgeBase:
    def __init__(self) -> None:
        self._entries: list[HSEntry] = []
        self._by_code: dict[str, HSEntry] = {}
        self._by_parent: dict[str, list[HSEntry]] = {}
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def load(self, csv_path: str | Path) -> None:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"HS dataset not found: {csv_path}\n"
                "Run `python tools/load_dataset.py` to download it."
            )

        logger.info("Loading HS dataset from %s", csv_path)
        df = pd.read_csv(csv_path, dtype={"hscode": str, "parent": str})

        required_cols = {"section", "hscode", "description", "parent", "level"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df["hscode"] = df["hscode"].astype(str).str.strip()
        df["parent"] = df["parent"].astype(str).str.strip()

        self._entries = []
        self._by_code = {}
        self._by_parent = {}

        for row in df.itertuples(index=False):
            entry = HSEntry(
                section=str(row.section).strip(),
                hs_code=row.hscode,
                description=str(row.description).strip(),
                parent=row.parent,
                level=int(row.level),
            )
            self._entries.append(entry)
            self._by_code[entry.hs_code] = entry
            self._by_parent.setdefault(entry.parent, []).append(entry)

        self._loaded = True
        logger.info(
            "Loaded %d HS entries (%d chapters, %d headings, %d subheadings)",
            len(self._entries),
            sum(1 for e in self._entries if e.level == 2),
            sum(1 for e in self._entries if e.level == 4),
            sum(1 for e in self._entries if e.level == 6),
        )

    def get_by_code(self, hs_code: str) -> HSEntry | None:
        return self._by_code.get(hs_code)

    def get_hierarchy_path(self, hs_code: str) -> list[HSEntry]:
        path = []
        current = self._by_code.get(hs_code)
        while current:
            path.append(current)
            if current.parent == "TOTAL" or current.parent not in self._by_code:
                break
            current = self._by_code.get(current.parent)
        path.reverse()
        return path

    def get_subheadings(self) -> list[HSEntry]:
        return [e for e in self._entries if e.level == 6]
