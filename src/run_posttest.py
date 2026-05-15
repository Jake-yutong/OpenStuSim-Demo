"""Generate revised post-feedback student answers."""

from __future__ import annotations

from src.config import FEEDBACK_PATH, POST_ANSWER_PROMPT_PATH, POST_ANSWERS_PATH
from src.llm_client import LLMClient
from src.utils import load_prompt_template, read_jsonl, render_prompt, write_jsonl


def dryrun_post_answer(record: dict) -> str:
    """Return deterministic revised answers with profile-specific improvement."""
    profile = record.get("profile", "")
    if profile == "Low":
        return "I think the answer is about the main science idea, but I can only explain part of it."
    if profile == "Medium":
        return "The key process causes the result because it changes how the objects or materials interact."
    if profile == "High":
        return "The mechanism from the reference idea explains the observation directly and should be stated clearly."
    return "This is a revised dry-run student answer."


def run_posttest(
    client: LLMClient,
    input_path=FEEDBACK_PATH,
    output_path=POST_ANSWERS_PATH,
    temperature: float = 0.7,
) -> int:
    """Read feedback records and write revised post-test answers."""
    template = load_prompt_template(POST_ANSWER_PROMPT_PATH)
    records: list[dict] = []
    for record in read_jsonl(input_path):
        prompt = render_prompt(
            template,
            {
                "pre_answer": str(record["pre_answer"]),
                "feedback": str(record["feedback"]),
                "profile_description": str(record["profile_description"]),
                "question": str(record["question"]),
            },
        )
        answer = dryrun_post_answer(record) if client.dry_run else client.generate(prompt, temperature=temperature)
        new_record = dict(record)
        new_record["post_prompt"] = prompt
        new_record["post_answer"] = answer
        records.append(new_record)

    write_jsonl(records, output_path)
    return len(records)
