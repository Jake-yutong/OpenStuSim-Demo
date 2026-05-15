"""Pre-test answer generation for dry-run and real LLM modes."""

from __future__ import annotations

import pandas as pd

from src.config import PRE_ANSWERS_PATH, PROMPT_TEMPLATE_PATHS
from src.llm_client import LLMClient
from src.utils import load_prompt_template, render_prompt, write_jsonl


def generate_pre_answers(
    questions: pd.DataFrame,
    profiles: list[dict[str, str]],
    prompt_modes: list[str],
    client: LLMClient,
    temperature: float,
) -> list[dict]:
    """Create one pre-test answer for each question-profile-prompt tuple."""
    templates = {
        mode: load_prompt_template(PROMPT_TEMPLATE_PATHS[mode])
        for mode in prompt_modes
    }
    records: list[dict] = []
    for _, row in questions.iterrows():
        for profile in profiles:
            for mode in prompt_modes:
                variables = {
                    "profile_name": profile["profile"],
                    "profile_description": profile["profile_description"],
                    "question": str(row["question"]),
                }
                prompt = render_prompt(templates[mode], variables)
                records.append(
                    {
                        "qid": str(row["qid"]),
                        "question": str(row["question"]),
                        "reference_answer": str(row["reference_answer"]),
                        "profile": profile["profile"],
                        "profile_description": profile["profile_description"],
                        "prompt_mode": mode,
                        "prompt": prompt,
                        "pre_answer": client.generate(prompt, temperature=temperature),
                    }
                )
    return records


def run_dryrun(
    questions: pd.DataFrame,
    profiles: list[dict[str, str]],
    prompt_modes: list[str],
    client: LLMClient,
    temperature: float,
) -> int:
    """Generate pre-test answers and save them as JSONL."""
    records = generate_pre_answers(questions, profiles, prompt_modes, client, temperature)
    write_jsonl(records, PRE_ANSWERS_PATH)
    return len(records)
