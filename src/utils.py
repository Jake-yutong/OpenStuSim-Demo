"""Small file helpers for the OpenStuSim demo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def load_prompt_template(path: Path) -> str:
    """Load a prompt template from disk."""
    return path.read_text(encoding="utf-8")


def render_prompt(template: str, variables: dict[str, str]) -> str:
    """Render a template using Python's standard format syntax."""
    return template.format(**variables)


def write_jsonl(records: Iterable[dict], path: Path) -> None:
    """Write dictionaries to a UTF-8 JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """Read a UTF-8 JSONL file into a list of dictionaries."""
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
