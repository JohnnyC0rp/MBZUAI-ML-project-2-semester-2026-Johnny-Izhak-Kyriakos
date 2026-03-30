from __future__ import annotations

from pathlib import Path


REQUIRED_DATA_FILES = ("credit_train.csv", "credit_test.csv")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def candidate_data_roots() -> list[Path]:
    root = repo_root()
    candidates = [root, root.parent, root.parent.parent]
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def resolve_data_file(filename: str) -> Path:
    for candidate_root in candidate_data_roots():
        candidate = candidate_root / filename
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in candidate_data_roots())
    raise FileNotFoundError(
        f"Could not find {filename}. I searched: {searched}. "
        "Place the Kaggle CSV files in the repo root or one of its parent folders."
    )


def is_data_ready() -> bool:
    return all(_is_resolveable(name) for name in REQUIRED_DATA_FILES)


def _is_resolveable(filename: str) -> bool:
    try:
        resolve_data_file(filename)
    except FileNotFoundError:
        return False
    return True
