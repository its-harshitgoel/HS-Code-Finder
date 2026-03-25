"""
HS Knowledge Base Service.

Purpose: Loads the HS classification dataset and provides lookup, hierarchy
         traversal, and filtering capabilities.
Inputs:  CSV file path (data/hs_codes.csv).
Outputs: HSEntry objects, filtered lists, hierarchy paths.
"""

from pathlib import Path

import pandas as pd

from backend.models.schemas import HSEntry
from backend.utils.logger import get_logger

logger = get_logger("hs_knowledge")


class HSKnowledgeBase:
    """Manages the HS classification dataset in memory."""

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
        """Load HS dataset from CSV file.

        Args:
            csv_path: Path to the HS codes CSV file.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If required columns are missing.
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"HS dataset not found: {csv_path}")

        logger.info("Loading HS dataset from %s", csv_path)

        df = pd.read_csv(csv_path, dtype={"hscode": str, "parent": str})

        required_cols = {"section", "hscode", "description", "parent", "level"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Ensure HS codes are strings with no whitespace
        df["hscode"] = df["hscode"].astype(str).str.strip()
        df["parent"] = df["parent"].astype(str).str.strip()

        self._entries = []
        self._by_code = {}
        self._by_parent = {}

        for _, row in df.iterrows():
            entry = HSEntry(
                section=str(row["section"]).strip(),
                hs_code=row["hscode"],
                description=str(row["description"]).strip(),
                parent=row["parent"],
                level=int(row["level"]),
            )
            self._entries.append(entry)
            self._by_code[entry.hs_code] = entry

            if entry.parent not in self._by_parent:
                self._by_parent[entry.parent] = []
            self._by_parent[entry.parent].append(entry)

        self._loaded = True
        logger.info(
            "Loaded %d HS entries (%d chapters, %d headings, %d subheadings)",
            len(self._entries),
            sum(1 for e in self._entries if e.level == 2),
            sum(1 for e in self._entries if e.level == 4),
            sum(1 for e in self._entries if e.level == 6),
        )

    def get_all_entries(self) -> list[HSEntry]:
        """Return all HS entries."""
        return self._entries

    def get_by_code(self, hs_code: str) -> HSEntry | None:
        """Look up a single HS entry by its code."""
        return self._by_code.get(hs_code)

    def get_children(self, parent_code: str) -> list[HSEntry]:
        """Get all direct children of a given HS code."""
        return self._by_parent.get(parent_code, [])

    def get_hierarchy_path(self, hs_code: str) -> list[HSEntry]:
        """Get the full hierarchy path from chapter down to the given code.

        Returns:
            List of HSEntry from broadest (chapter) to most specific.
        """
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
        """Return only level-6 (subheading) entries for indexing."""
        return [e for e in self._entries if e.level == 6]

    def get_headings(self) -> list[HSEntry]:
        """Return only level-4 (heading) entries."""
        return [e for e in self._entries if e.level == 4]

    def get_chapters(self) -> list[HSEntry]:
        """Return only level-2 (chapter) entries."""
        return [e for e in self._entries if e.level == 2]

    def code_exists(self, hs_code: str) -> bool:
        """Check if an HS code exists in the dataset."""
        return hs_code in self._by_code
