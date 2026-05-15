"""Generate tutor feedback for pre-test student answers."""

from __future__ import annotations

from src.config import FEEDBACK_PATH, FEEDBACK_PROMPT_PATH, PRE_JUDGED_PATH
from src.llm_client import LLMClient
from src.utils import load_prompt_template, read_jsonl, render_prompt, write_jsonl


def dryrun_feedback(record: dict) -> str:
    """Return deterministic feedback for local dry-run testing."""
    label = record.get("pre_label", "")
    if label == "Correct":
        return "Good start. Try to make the causal mechanism explicit in one clear sentence."
    if label == "Partially correct":
        return "You have part of the idea. Add the missing mechanism and connect it directly to the question."
    if label == "Contradictory":
        return "Recheck the main science concept. Your answer seems to point in the wrong direction."
    return "Focus on the relevant science concept and explain why it causes the observed result."


def run_feedback(client: LLMClient, input_path=PRE_JUDGED_PATH, output_path=FEEDBACK_PATH) -> int:
    """Read judged pre-test answers and write tutor feedback records."""
    template = load_prompt_template(FEEDBACK_PROMPT_PATH)
    records: list[dict] = []
    for record in read_jsonl(input_path):
        prompt = render_prompt(
            template,
            {
                "question": str(record["question"]),
                "reference_answer": str(record["reference_answer"]),
                "pre_answer": str(record["pre_answer"]),
            },
        )
        feedback = dryrun_feedback(record) if client.dry_run else client.generate(prompt, temperature=0.3, max_tokens=128)
        new_record = dict(record)
        new_record["feedback_prompt"] = prompt
        new_record["feedback"] = feedback
        records.append(new_record)

    write_jsonl(records, output_path)
    return len(records)
