from __future__ import annotations

from pathlib import Path


# Repo root = parent of `src/`
REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
CONFIG_DIR = REPO_ROOT / "config"
CONFIG_CASES_DIR = CONFIG_DIR / "cases"
LOGS_DIR = REPO_ROOT / "logs"
TMP_DIR = REPO_ROOT / "tmp"
OUTPUT_DIR = REPO_ROOT / "output"


def ensure_dirs() -> None:
    """Create common output directories if missing."""
    for d in (CONFIG_CASES_DIR, LOGS_DIR, TMP_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def data_path(*parts: str | Path) -> Path:
    return DATA_DIR.joinpath(*map(str, parts))


def cases_path(*parts: str | Path) -> Path:
    return CONFIG_CASES_DIR.joinpath(*map(str, parts))


def logs_path(*parts: str | Path) -> Path:
    return LOGS_DIR.joinpath(*map(str, parts))


def tmp_path(*parts: str | Path) -> Path:
    return TMP_DIR.joinpath(*map(str, parts))


def output_path(*parts: str | Path) -> Path:
    return OUTPUT_DIR.joinpath(*map(str, parts))

