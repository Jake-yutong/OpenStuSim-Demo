"""Load or create the sample science questions table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DEFAULT_QUESTIONS, QUESTION_COLUMNS, SAMPLE_QUESTIONS_PATH


def ensure_sample_questions(path: Path = SAMPLE_QUESTIONS_PATH) -> Path:
    """Create the sample questions CSV if it is missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        df = pd.DataFrame(DEFAULT_QUESTIONS, columns=QUESTION_COLUMNS)
        df.to_csv(path, index=False)
    return path


def load_questions(path: Path = SAMPLE_QUESTIONS_PATH) -> pd.DataFrame:
    """Return questions as a pandas DataFrame with the expected columns."""
    ensure_sample_questions(path)
    df = pd.read_csv(path)
    missing = [col for col in QUESTION_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required question columns: {missing}")
    return df[QUESTION_COLUMNS].copy()
