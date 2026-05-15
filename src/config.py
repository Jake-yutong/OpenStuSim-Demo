"""Project paths and small default data for OpenStuSim demo."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

SAMPLE_QUESTIONS_PATH = DATA_DIR / "sample_questions.csv"
PRE_ANSWERS_PATH = OUTPUTS_DIR / "pre_answers.jsonl"
PRE_JUDGED_PATH = OUTPUTS_DIR / "pre_judged.jsonl"
FEEDBACK_PATH = OUTPUTS_DIR / "feedback.jsonl"
POST_ANSWERS_PATH = OUTPUTS_DIR / "post_answers.jsonl"
POST_JUDGED_PATH = OUTPUTS_DIR / "post_judged.jsonl"
METRICS_CSV_PATH = OUTPUTS_DIR / "metrics.csv"
METRICS_SUMMARY_PATH = OUTPUTS_DIR / "metrics_summary.json"
PROFILE_SCORES_FIG_PATH = FIGURES_DIR / "profile_scores.png"
N_GAIN_FIG_PATH = FIGURES_DIR / "n_gain.png"
LABEL_DISTRIBUTION_FIG_PATH = FIGURES_DIR / "label_distribution.png"
JUDGE_PROMPT_PATH = PROMPTS_DIR / "judge_5way.txt"
FEEDBACK_PROMPT_PATH = PROMPTS_DIR / "teacher_feedback.txt"
POST_ANSWER_PROMPT_PATH = PROMPTS_DIR / "post_answer.txt"

DEFAULT_PROMPT_MODES = ["roleplay"]
PROMPT_TEMPLATE_PATHS = {
    "roleplay": PROMPTS_DIR / "student_roleplay.txt",
    "teacher_prediction": PROMPTS_DIR / "student_teacher_prediction.txt",
}

QUESTION_COLUMNS = [
    "qid",
    "question",
    "reference_answer",
    "real_correct",
    "real_partial",
    "real_contradictory",
    "real_irrelevant",
    "real_nondomain",
    "real_mean_score",
]

DEFAULT_QUESTIONS = [
    {
        "qid": "Q001",
        "question": "Why does a metal spoon feel colder than a wooden spoon at the same room temperature?",
        "reference_answer": "Metal conducts heat away from your hand faster than wood, so it feels colder even when both objects have the same temperature.",
        "real_correct": 8,
        "real_partial": 3,
        "real_contradictory": 1,
        "real_irrelevant": 1,
        "real_nondomain": 0,
        "real_mean_score": 0.72,
    },
    {
        "qid": "Q002",
        "question": "Explain why plants need light for photosynthesis.",
        "reference_answer": "Plants use light energy to convert carbon dioxide and water into glucose and oxygen during photosynthesis.",
        "real_correct": 7,
        "real_partial": 4,
        "real_contradictory": 1,
        "real_irrelevant": 1,
        "real_nondomain": 0,
        "real_mean_score": 0.68,
    },
    {
        "qid": "Q003",
        "question": "What causes day and night on Earth?",
        "reference_answer": "Day and night are caused by Earth's rotation, which turns different parts of the planet toward or away from the Sun.",
        "real_correct": 9,
        "real_partial": 2,
        "real_contradictory": 1,
        "real_irrelevant": 0,
        "real_nondomain": 0,
        "real_mean_score": 0.81,
    },
    {
        "qid": "Q004",
        "question": "Why does salt dissolve in water?",
        "reference_answer": "Water molecules are polar and attract the positive and negative ions in salt, pulling them apart into solution.",
        "real_correct": 6,
        "real_partial": 5,
        "real_contradictory": 1,
        "real_irrelevant": 1,
        "real_nondomain": 0,
        "real_mean_score": 0.63,
    },
    {
        "qid": "Q005",
        "question": "Why do objects fall toward Earth when dropped?",
        "reference_answer": "Objects fall toward Earth because Earth's gravity pulls objects with mass toward its center.",
        "real_correct": 8,
        "real_partial": 3,
        "real_contradictory": 1,
        "real_irrelevant": 1,
        "real_nondomain": 0,
        "real_mean_score": 0.74,
    },
]
