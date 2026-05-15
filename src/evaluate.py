"""Compute simple evaluation metrics and figures for OpenStuSim."""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    LABEL_DISTRIBUTION_FIG_PATH,
    METRICS_CSV_PATH,
    METRICS_SUMMARY_PATH,
    N_GAIN_FIG_PATH,
    POST_JUDGED_PATH,
    PROFILE_SCORES_FIG_PATH,
    SAMPLE_QUESTIONS_PATH,
)
from src.utils import read_jsonl


LABELS = ["Correct", "Partially correct", "Contradictory", "Irrelevant", "Non-domain"]
REAL_LABEL_COLUMNS = {
    "Correct": "real_correct",
    "Partially correct": "real_partial",
    "Contradictory": "real_contradictory",
    "Irrelevant": "real_irrelevant",
    "Non-domain": "real_nondomain",
}


def _word_count(text: str) -> int:
    return len(str(text).split())


def _real_label_distribution(questions: pd.DataFrame) -> dict[str, float]:
    counts = {label: float(questions[col].sum()) for label, col in REAL_LABEL_COLUMNS.items()}
    total = sum(counts.values())
    if total == 0:
        return {label: 0.0 for label in LABELS}
    return {label: counts[label] / total for label in LABELS}


def _sim_label_distribution(df: pd.DataFrame) -> dict[str, float]:
    counts = df["pre_label"].value_counts(normalize=True).to_dict()
    return {label: float(counts.get(label, 0.0)) for label in LABELS}


def compute_metrics(records: pd.DataFrame, questions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return long-form metrics and summary data."""
    df = records.copy()
    df["score_gain"] = df["post_score"] - df["pre_score"]
    df["pre_word_count"] = df["pre_answer"].map(_word_count)
    df["pre_char_count"] = df["pre_answer"].astype(str).map(len)
    df["over_improved"] = (df["pre_score"] == 0.0) & (df["post_score"] == 1.0)

    metric_rows: list[dict] = []

    grouped = df.groupby(["prompt_mode", "profile"], dropna=False)
    profile_summary = grouped.agg(
        mean_pre_score=("pre_score", "mean"),
        mean_post_score=("post_score", "mean"),
        mean_score_gain=("score_gain", "mean"),
        mean_n_gain=("n_gain", "mean"),
        over_improvement_rate=("over_improved", "mean"),
        mean_pre_word_count=("pre_word_count", "mean"),
        mean_pre_char_count=("pre_char_count", "mean"),
    ).reset_index()

    for _, row in profile_summary.iterrows():
        for metric_name in [
            "mean_pre_score",
            "mean_post_score",
            "mean_score_gain",
            "mean_n_gain",
            "over_improvement_rate",
            "mean_pre_word_count",
            "mean_pre_char_count",
        ]:
            metric_rows.append(
                {
                    "metric": metric_name,
                    "prompt_mode": row["prompt_mode"],
                    "profile": row["profile"],
                    "value": row[metric_name],
                }
            )

    correlations: dict[str, float | None] = {}
    for prompt_mode, sub in df.groupby("prompt_mode"):
        simulated = sub.groupby("qid", as_index=False)["pre_score"].mean()
        merged = simulated.merge(questions[["qid", "real_mean_score"]], on="qid", how="inner")
        if len(merged) >= 2 and merged["pre_score"].nunique() > 1 and merged["real_mean_score"].nunique() > 1:
            corr = float(merged["pre_score"].corr(merged["real_mean_score"], method="pearson"))
        else:
            corr = None
        correlations[prompt_mode] = corr
        metric_rows.append(
            {
                "metric": "difficulty_pearson_corr",
                "prompt_mode": prompt_mode,
                "profile": "ALL",
                "value": corr,
            }
        )

    real_dist = _real_label_distribution(questions)
    label_alignment: dict[str, dict] = {}
    for prompt_mode, sub in df.groupby("prompt_mode"):
        sim_dist = _sim_label_distribution(sub)
        l1 = sum(abs(real_dist[label] - sim_dist[label]) for label in LABELS)
        label_alignment[prompt_mode] = {
            "l1_distance": l1,
            "simulated_distribution": sim_dist,
        }
        metric_rows.append(
            {
                "metric": "label_distribution_l1",
                "prompt_mode": prompt_mode,
                "profile": "ALL",
                "value": l1,
            }
        )

    metrics_df = pd.DataFrame(metric_rows)
    summary = {
        "profile_metrics": profile_summary.to_dict(orient="records"),
        "difficulty_pearson_corr": correlations,
        "real_label_distribution": real_dist,
        "label_distribution_alignment": label_alignment,
    }
    return metrics_df, summary


def plot_profile_scores(records: pd.DataFrame) -> None:
    summary = records.groupby("profile")[["pre_score", "post_score"]].mean().reindex(["Low", "Medium", "High"])
    ax = summary.plot(kind="bar", figsize=(7, 4), rot=0)
    ax.set_ylabel("Mean score")
    ax.set_xlabel("Profile")
    ax.set_ylim(0, 1.05)
    ax.set_title("Mean Pre/Post Score by Profile")
    ax.legend(["Pre", "Post"])
    plt.tight_layout()
    PROFILE_SCORES_FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PROFILE_SCORES_FIG_PATH, dpi=150)
    plt.close()


def plot_n_gain(records: pd.DataFrame) -> None:
    summary = records.groupby("profile")["n_gain"].mean().reindex(["Low", "Medium", "High"])
    ax = summary.plot(kind="bar", figsize=(7, 4), rot=0)
    ax.set_ylabel("Mean normalized gain")
    ax.set_xlabel("Profile")
    ax.set_ylim(0, 1.05)
    ax.set_title("Mean Normalized Gain by Profile")
    plt.tight_layout()
    N_GAIN_FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(N_GAIN_FIG_PATH, dpi=150)
    plt.close()


def plot_label_distribution(records: pd.DataFrame, questions: pd.DataFrame) -> None:
    real_dist = _real_label_distribution(questions)
    sim_dist = _sim_label_distribution(records)
    plot_df = pd.DataFrame({"Real": real_dist, "Simulated": sim_dist}).reindex(LABELS)
    ax = plot_df.plot(kind="bar", figsize=(9, 4), rot=25)
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Label")
    ax.set_ylim(0, 1.0)
    ax.set_title("Real vs Simulated Pre-label Distribution")
    plt.tight_layout()
    LABEL_DISTRIBUTION_FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(LABEL_DISTRIBUTION_FIG_PATH, dpi=150)
    plt.close()


def run_evaluation(input_path=POST_JUDGED_PATH, questions_path=SAMPLE_QUESTIONS_PATH) -> int:
    """Compute metrics, save summaries, and generate figures."""
    records = pd.DataFrame(read_jsonl(input_path))
    questions = pd.read_csv(questions_path)
    metrics_df, summary = compute_metrics(records, questions)

    METRICS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_CSV_PATH, index=False)
    METRICS_SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_profile_scores(records)
    plot_n_gain(records)
    plot_label_distribution(records, questions)
    return len(metrics_df)
