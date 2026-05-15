"""Command-line entry point for the OpenStuSim research demo."""

from __future__ import annotations

import argparse

from src.build_profiles import build_default_profiles
from src.config import (
    DEFAULT_PROMPT_MODES,
    FEEDBACK_PATH,
    LABEL_DISTRIBUTION_FIG_PATH,
    METRICS_CSV_PATH,
    METRICS_SUMMARY_PATH,
    N_GAIN_FIG_PATH,
    POST_ANSWERS_PATH,
    POST_JUDGED_PATH,
    PRE_ANSWERS_PATH,
    PRE_JUDGED_PATH,
    PROFILE_SCORES_FIG_PATH,
    PROMPT_TEMPLATE_PATHS,
    SAMPLE_QUESTIONS_PATH,
)
from src.dryrun import run_dryrun
from src.evaluate import run_evaluation
from src.llm_client import LLMClient
from src.load_data import load_questions
from src.run_feedback import run_feedback
from src.run_judge import run_judge
from src.run_posttest import run_posttest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenStuSim demo: simulate student pre-test answers for science questions."
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run without LLM APIs and generate placeholder pre-test answers.",
    )
    parser.add_argument(
        "--prompt_modes",
        nargs="+",
        default=DEFAULT_PROMPT_MODES,
        choices=list(PROMPT_TEMPLATE_PATHS.keys()),
        help="Prompt modes to render for student answer generation.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name for real LLM generation. Defaults to MODEL_NAME or gpt-4o-mini.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for real LLM generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    questions = load_questions(SAMPLE_QUESTIONS_PATH)
    profiles = build_default_profiles()
    try:
        client = LLMClient(model=args.model, dry_run=args.dry_run)
        n_records = run_dryrun(questions, profiles, args.prompt_modes, client, args.temperature)
        n_pre_judged = run_judge(client)
        n_feedback = run_feedback(client)
        n_post_answers = run_posttest(client, temperature=args.temperature)
        n_post_judged = run_judge(
            client,
            input_path=POST_ANSWERS_PATH,
            output_path=POST_JUDGED_PATH,
            answer_field="post_answer",
            output_prefix="post",
        )
        n_metrics = run_evaluation()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    mode = "dry run" if args.dry_run else "LLM run"
    print(f"OpenStuSim {mode} complete.")
    print(f"Questions: {len(questions)}")
    print(f"Profiles: {len(profiles)}")
    print(f"Prompt modes: {', '.join(args.prompt_modes)}")
    print(f"Model: {client.model}")
    print(f"Pre-answer records written: {n_records}")
    print(f"Pre-judged records written: {n_pre_judged}")
    print(f"Feedback records written: {n_feedback}")
    print(f"Post-answer records written: {n_post_answers}")
    print(f"Post-judged records written: {n_post_judged}")
    print(f"Metric rows written: {n_metrics}")
    print(f"Pre answers output: {PRE_ANSWERS_PATH}")
    print(f"Pre judged output: {PRE_JUDGED_PATH}")
    print(f"Feedback output: {FEEDBACK_PATH}")
    print(f"Post answers output: {POST_ANSWERS_PATH}")
    print(f"Post judged output: {POST_JUDGED_PATH}")
    print(f"Metrics CSV: {METRICS_CSV_PATH}")
    print(f"Metrics summary: {METRICS_SUMMARY_PATH}")
    print(f"Profile scores figure: {PROFILE_SCORES_FIG_PATH}")
    print(f"N-gain figure: {N_GAIN_FIG_PATH}")
    print(f"Label distribution figure: {LABEL_DISTRIBUTION_FIG_PATH}")


if __name__ == "__main__":
    main()
