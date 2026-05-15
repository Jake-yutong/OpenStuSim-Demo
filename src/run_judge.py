"""Judge generated student answers with SciEntsBank-style 5-way labels."""

from __future__ import annotations

import json
import math
import re

from src.config import JUDGE_PROMPT_PATH, PRE_ANSWERS_PATH, PRE_JUDGED_PATH
from src.llm_client import LLMClient
from src.utils import load_prompt_template, read_jsonl, render_prompt, write_jsonl


def parse_judge_json(raw: str) -> dict:
    """Parse judge JSON, with regex fallback for object-shaped output."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {
        "label": "ParseError",
        "score": math.nan,
        "reason": raw,
    }


def _score_to_label(score: float) -> str:
    if score >= 1.0:
        return "Correct"
    if score >= 0.5:
        return "Partially correct"
    return "Contradictory"


def dryrun_judge(record: dict, answer_field: str = "pre_answer") -> dict:
    """Assign deterministic profile-based labels for local dry-run testing."""
    profile = record.get("profile", "")
    qid = str(record.get("qid", ""))
    prompt_mode = str(record.get("prompt_mode", ""))
    bucket = sum(ord(ch) for ch in f"{qid}-{profile}-{prompt_mode}") % 4

    if answer_field == "post_answer":
        pre_score = record.get("pre_score", 0.0)
        if not isinstance(pre_score, int | float) or math.isnan(float(pre_score)):
            score = math.nan
            label = "ParseError"
        elif profile == "Low":
            score = min(0.5, float(pre_score) + 0.5)
            label = _score_to_label(score)
        elif profile == "Medium":
            score = min(1.0, float(pre_score) + 0.5)
            label = _score_to_label(score)
        elif profile == "High":
            score = 1.0 if float(pre_score) < 1.0 else float(pre_score)
            label = _score_to_label(score)
        else:
            score = math.nan
            label = "ParseError"
        return {
            "label": label,
            "score": score,
            "reason": f"[dry_run] Post-test score adjusted from profile={profile}.",
        }

    if profile == "Low":
        label = "Partially correct" if bucket == 0 else "Contradictory"
        score = 0.5 if label == "Partially correct" else 0.0
    elif profile == "Medium":
        label = "Correct" if bucket == 0 else "Partially correct"
        score = 1.0 if label == "Correct" else 0.5
    elif profile == "High":
        label = "Partially correct" if bucket == 0 else "Correct"
        score = 0.5 if label == "Partially correct" else 1.0
    else:
        label = "ParseError"
        score = math.nan

    return {
        "label": label,
        "score": score,
        "reason": f"[dry_run] Assigned from profile={profile}.",
    }


def judge_record(
    record: dict,
    client: LLMClient,
    template: str,
    answer_field: str = "pre_answer",
) -> dict:
    """Return a judge result for one generated answer."""
    if client.dry_run:
        return dryrun_judge(record, answer_field)

    prompt = render_prompt(
        template,
        {
            "question": str(record["question"]),
            "reference_answer": str(record["reference_answer"]),
            "student_answer": str(record[answer_field]),
        },
    )
    raw = client.generate(prompt, temperature=0.0, max_tokens=256)
    parsed = parse_judge_json(raw)
    return {
        "label": str(parsed.get("label", "ParseError")),
        "score": parsed.get("score", math.nan),
        "reason": str(parsed.get("reason", raw)),
    }


def compute_n_gain(pre_score, post_score) -> float | None:
    """Compute normalized gain for one record."""
    try:
        pre = float(pre_score)
        post = float(post_score)
    except (TypeError, ValueError):
        return None
    if math.isnan(pre) or math.isnan(post) or pre >= 1.0:
        return None
    return (post - pre) / (1.0 - pre)


def run_judge(
    client: LLMClient,
    input_path=PRE_ANSWERS_PATH,
    output_path=PRE_JUDGED_PATH,
    answer_field: str = "pre_answer",
    output_prefix: str = "pre",
) -> int:
    """Read answers, judge them, and write JSONL results."""
    template = load_prompt_template(JUDGE_PROMPT_PATH)
    judged_records: list[dict] = []
    for record in read_jsonl(input_path):
        result = judge_record(record, client, template, answer_field)
        judged = dict(record)
        judged[f"{output_prefix}_label"] = result["label"]
        judged[f"{output_prefix}_score"] = result["score"]
        judged[f"{output_prefix}_judge_reason"] = result["reason"]
        if output_prefix == "post":
            judged["n_gain"] = compute_n_gain(judged.get("pre_score"), judged.get("post_score"))
        judged_records.append(judged)

    write_jsonl(judged_records, output_path)
    return len(judged_records)
