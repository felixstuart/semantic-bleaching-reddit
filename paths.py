"""Locations of large data files that live outside the repo tree.

`wiktionary.jsonl` (22 GB) used to sit at the repo root. It was moved to a
sibling directory so the editor's file watcher, workspace search, Pylance and
Spotlight stop walking it -- it made the workspace 30 GB, of which it was 73%.

Resolution order for the Wiktionary dump:
  1. $WIKTIONARY_PATH                      (explicit override, wins outright)
  2. $SEMANTIC_DRIFT_DATA/wiktionary.jsonl (relocatable data root)
  3. ../SemanticDrift-data/wiktionary.jsonl (default new home)
  4. ./wiktionary.jsonl                    (legacy in-repo location, if present)

Steps 3 and 4 are only used when the file is actually there, so a checkout that
still has the dump at the old path keeps working unchanged.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(
    os.environ.get("SEMANTIC_DRIFT_DATA", REPO_ROOT.parent / "SemanticDrift-data")
)


def wiktionary_path() -> Path:
    """Return the path to wiktionary.jsonl, preferring env overrides."""
    override = os.environ.get("WIKTIONARY_PATH")
    if override:
        return Path(override)

    for candidate in (DATA_ROOT / "wiktionary.jsonl", REPO_ROOT / "wiktionary.jsonl"):
        if candidate.exists():
            return candidate

    # Nothing on disk: return the expected location so the caller's open() raises
    # a FileNotFoundError naming the place the file is supposed to be.
    return DATA_ROOT / "wiktionary.jsonl"


WIKTIONARY_PATH = wiktionary_path()
