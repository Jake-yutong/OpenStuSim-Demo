"""Streamlit interface for the OpenStuSim demo."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.build_profiles import build_default_profiles
from src.config import (
    DEFAULT_PROMPT_MODES,
    FEEDBACK_PATH,
    LABEL_DISTRIBUTION_FIG_PATH,
    METRICS_CSV_PATH,
    METRICS_SUMMARY_PATH,
    N_GAIN_FIG_PATH,
    POST_ANSWER_PROMPT_PATH,
    POST_ANSWERS_PATH,
    POST_JUDGED_PATH,
    PRE_ANSWERS_PATH,
    PRE_JUDGED_PATH,
    PROFILE_SCORES_FIG_PATH,
    PROMPT_TEMPLATE_PATHS,
    SAMPLE_QUESTIONS_PATH,
    JUDGE_PROMPT_PATH,
    FEEDBACK_PROMPT_PATH,
)
from src.dryrun import run_dryrun
from src.evaluate import run_evaluation
from src.llm_client import LLMClient
from src.load_data import load_questions
from src.run_feedback import run_feedback
from src.run_judge import run_judge
from src.run_posttest import run_posttest
from src.utils import load_prompt_template, read_jsonl


PROMPT_FILES = {
    "student_roleplay": PROMPT_TEMPLATE_PATHS["roleplay"],
    "student_teacher_prediction": PROMPT_TEMPLATE_PATHS["teacher_prediction"],
    "teacher_feedback": FEEDBACK_PROMPT_PATH,
    "post_answer": POST_ANSWER_PROMPT_PATH,
    "judge_5way": JUDGE_PROMPT_PATH,
}

PROVIDER_PRESETS = {
    "OpenAI": {
        "base_url": "",
        "model": "gpt-4o-mini",
    },
    "DeepSeek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    },
    "Qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
    "Custom": {
        "base_url": os.getenv("OPENAI_BASE_URL", ""),
        "model": os.getenv("MODEL_NAME", "gpt-4o-mini"),
    },
}


st.set_page_config(page_title="OpenStuSim Demo", layout="wide")


def _status(message: str, kind: str = "info") -> None:
    """Render status text without Streamlit alert widgets.

    Some Streamlit builds can fail inside st.error/st.success when optional
    emoji helpers are missing. Plain markdown keeps the app usable.
    """
    colors = {
        "success": "#0f7b0f",
        "error": "#b00020",
        "info": "#345995",
    }
    color = colors.get(kind, colors["info"])
    st.markdown(
        f"<div style='color:{color}; font-weight:600; margin:0.35rem 0;'>{message}</div>",
        unsafe_allow_html=True,
    )


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".jsonl":
        return pd.DataFrame(read_jsonl(path))
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.DataFrame()


def _save_prompts_from_state() -> None:
    for name, path in PROMPT_FILES.items():
        value = st.session_state.get(f"prompt_{name}")
        if value is not None:
            path.write_text(value, encoding="utf-8")


def _apply_api_env(api_key: str, base_url: str) -> None:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    elif "OPENAI_BASE_URL" in os.environ:
        del os.environ["OPENAI_BASE_URL"]


def _test_api(model: str | None, temperature: float, api_key: str, base_url: str) -> str:
    _apply_api_env(api_key, base_url)
    client = LLMClient(model=model or None, dry_run=False)
    return client.generate(
        "Reply with exactly: OK",
        temperature=temperature,
        max_tokens=8,
    ).strip()


def _run_pipeline(
    dry_run: bool,
    prompt_modes: list[str],
    model: str | None,
    temperature: float,
    api_key: str,
    base_url: str,
) -> dict[str, int]:
    _save_prompts_from_state()
    _apply_api_env(api_key, base_url)

    questions = load_questions(SAMPLE_QUESTIONS_PATH)
    profiles = build_default_profiles()
    client = LLMClient(model=model or None, dry_run=dry_run)

    n_pre = run_dryrun(questions, profiles, prompt_modes, client, temperature)
    n_pre_judged = run_judge(client)
    n_feedback = run_feedback(client)
    n_post = run_posttest(client, temperature=temperature)
    n_post_judged = run_judge(
        client,
        input_path=POST_ANSWERS_PATH,
        output_path=POST_JUDGED_PATH,
        answer_field="post_answer",
        output_prefix="post",
    )
    n_metrics = run_evaluation()
    return {
        "pre_answers": n_pre,
        "pre_judged": n_pre_judged,
        "feedback": n_feedback,
        "post_answers": n_post,
        "post_judged": n_post_judged,
        "metrics": n_metrics,
    }


st.title("OpenStuSim Demo")
st.caption("End-to-end student simulation, feedback, revision, judging, and metrics.")

with st.sidebar:
    st.header("Run Settings")
    dry_run = st.toggle("Dry run", value=True)
    prompt_modes = st.multiselect(
        "Prompt modes",
        options=list(PROMPT_TEMPLATE_PATHS.keys()),
        default=DEFAULT_PROMPT_MODES,
    )
    provider = st.selectbox(
        "Provider",
        options=list(PROVIDER_PRESETS.keys()),
        disabled=dry_run,
        help="DeepSeek and Qwen use OpenAI-compatible APIs with provider-specific base URLs.",
    )
    if st.session_state.get("last_provider") != provider:
        preset = PROVIDER_PRESETS[provider]
        st.session_state["model_input"] = preset["model"]
        st.session_state["base_url_input"] = preset["base_url"]
        st.session_state["last_provider"] = provider

    model = st.text_input("Model", key="model_input")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
    api_key = st.text_input("API key", type="password", disabled=dry_run)
    base_url = st.text_input(
        "Base URL",
        key="base_url_input",
        disabled=dry_run,
        help="OpenAI can leave this blank. DeepSeek/Qwen should use the preset URL unless your endpoint differs.",
    )
    if not dry_run and base_url:
        st.caption(f"Requests will be sent to: {base_url}")

    test_clicked = st.button("Test API", use_container_width=True, disabled=dry_run)
    if test_clicked:
        if not api_key:
            _status("Enter an API key first, or keep Dry run enabled.", "error")
        else:
            with st.spinner("Testing API with a tiny request..."):
                try:
                    reply = _test_api(model, temperature, api_key, base_url)
                    _status(f"API test passed. Response: {reply}", "success")
                except Exception as exc:
                    _status(f"API test failed: {exc}", "error")

    run_clicked = st.button("Run Full Demo", type="primary", use_container_width=True)
    if run_clicked:
        if not prompt_modes:
            _status("Select at least one prompt mode.", "error")
        else:
            with st.spinner("Running OpenStuSim pipeline..."):
                try:
                    counts = _run_pipeline(dry_run, prompt_modes, model, temperature, api_key, base_url)
                    st.session_state["last_counts"] = counts
                    _status("Pipeline finished.", "success")
                except Exception as exc:
                    _status(str(exc), "error")

tabs = st.tabs(["Questions", "Prompt Templates", "Process Trace", "Metrics", "Figures"])

with tabs[0]:
    st.subheader("Question Set")
    questions_df = load_questions(SAMPLE_QUESTIONS_PATH)
    st.dataframe(questions_df, use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Editable Prompt Templates")
    st.caption("Edits are saved when you click Save Prompts or Run Full Demo.")
    for name, path in PROMPT_FILES.items():
        if f"prompt_{name}" not in st.session_state:
            st.session_state[f"prompt_{name}"] = load_prompt_template(path)
        st.text_area(name, key=f"prompt_{name}", height=190)
    if st.button("Save Prompts"):
        _save_prompts_from_state()
        _status("Prompt templates saved.", "success")

with tabs[2]:
    st.subheader("Generated Records")
    counts = st.session_state.get("last_counts")
    if counts:
        st.write(counts)

    view_name = st.selectbox(
        "Output stage",
        [
            "pre_answers",
            "pre_judged",
            "feedback",
            "post_answers",
            "post_judged",
        ],
    )
    path_map = {
        "pre_answers": PRE_ANSWERS_PATH,
        "pre_judged": PRE_JUDGED_PATH,
        "feedback": FEEDBACK_PATH,
        "post_answers": POST_ANSWERS_PATH,
        "post_judged": POST_JUDGED_PATH,
    }
    stage_df = _read_table(path_map[view_name])
    if stage_df.empty:
        _status("No records yet. Run the demo first.", "info")
    else:
        display_cols = [
            col
            for col in [
                "qid",
                "profile",
                "prompt_mode",
                "pre_answer",
                "pre_label",
                "pre_score",
                "feedback",
                "post_answer",
                "post_label",
                "post_score",
                "n_gain",
            ]
            if col in stage_df.columns
        ]
        st.dataframe(stage_df[display_cols], use_container_width=True, hide_index=True)
        with st.expander("Full raw records"):
            st.dataframe(stage_df, use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Metrics")
    metrics_df = _read_table(METRICS_CSV_PATH)
    if metrics_df.empty:
        _status("No metrics yet. Run the demo first.", "info")
    else:
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    if METRICS_SUMMARY_PATH.exists():
        st.subheader("Summary JSON")
        st.json(METRICS_SUMMARY_PATH.read_text(encoding="utf-8"))

with tabs[4]:
    st.subheader("Figures")
    fig_paths = [
        ("Profile scores", PROFILE_SCORES_FIG_PATH),
        ("Normalized gain", N_GAIN_FIG_PATH),
        ("Label distribution", LABEL_DISTRIBUTION_FIG_PATH),
    ]
    for title, path in fig_paths:
        st.markdown(f"**{title}**")
        if path.exists():
            st.image(str(path), use_container_width=True)
        else:
            _status(f"{path.name} has not been generated yet.", "info")
