"""Streamlit interface for the OpenStuSim demo."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.build_profiles import build_default_profiles
from src.config import (
    DEFAULT_PROMPT_MODES,
    FEEDBACK_PATH,
    FEEDBACK_PROMPT_PATH,
    JUDGE_PROMPT_PATH,
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
    "OpenAI": {"base_url": "", "model": "gpt-4o-mini"},
    "DeepSeek": {"base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
    "Qwen": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-plus"},
    "Custom": {"base_url": os.getenv("OPENAI_BASE_URL", ""), "model": os.getenv("MODEL_NAME", "gpt-4o-mini")},
}

I18N = {
    "en": {
        "app_caption": "End-to-end student simulation, feedback, revision, judging, and metrics.",
        "run_settings": "Run Settings",
        "dry_run": "Dry run",
        "max_questions": "Max questions",
        "timeout_seconds": "Request timeout seconds",
        "prompt_modes": "Prompt modes",
        "provider_help": "DeepSeek and Qwen use OpenAI-compatible APIs with provider-specific base URLs.",
        "base_url_help": "OpenAI can leave this blank. DeepSeek/Qwen should use the preset URL unless your endpoint differs.",
        "requests_to": "Requests will be sent to",
        "test_api": "Test API",
        "need_key": "Enter an API key first, or keep Dry run enabled.",
        "testing_api": "Testing API with a tiny request...",
        "api_ok": "API test passed. Response",
        "api_failed": "API test failed",
        "run_full_demo": "Run Full Demo",
        "select_prompt_mode": "Select at least one prompt mode.",
        "running_pipeline": "Running OpenStuSim pipeline...",
        "pipeline_finished": "Pipeline finished.",
        "live_process": "Live Process",
        "latest_stage_preview": "Latest Stage Preview",
        "questions": "Questions",
        "prompt_templates": "Prompt Templates",
        "process_trace": "Process Trace",
        "metrics": "Metrics",
        "figures": "Figures",
        "question_set": "Question Set",
        "editable_prompts": "Editable Prompt Templates",
        "prompt_edit_note": "Edits are saved when you click Save Prompts or Run Full Demo.",
        "save_prompts": "Save Prompts",
        "prompts_saved": "Prompt templates saved.",
        "generated_records": "Generated Records",
        "teaching_process": "Teaching Process View",
        "record_selector": "Select one simulated student record",
        "student_pre_answer": "Student pre-test answer",
        "pre_score": "Pre-test score",
        "teacher_feedback": "Teacher feedback",
        "student_revised_answer": "Student revised answer",
        "post_score": "Post-test score",
        "n_gain_label": "Normalized gain",
        "no_trace": "No complete teaching trace yet. Run the demo first.",
        "output_stage": "Output stage",
        "no_records": "No records yet. Run the demo first.",
        "full_raw_records": "Full raw records",
        "key_results": "Key Results",
        "auto_conclusion": "Auto Conclusion",
        "raw_metrics": "Raw Metrics",
        "no_metrics": "No metrics yet. Run the demo first.",
        "summary_json": "Summary JSON",
        "profile_scores": "Profile scores",
        "normalized_gain": "Normalized gain",
        "label_distribution": "Label distribution",
        "figure_missing": "has not been generated yet.",
        "pre_answers": "Pre-test answers",
        "pre_judged": "Pre-test judging",
        "feedback": "Tutor feedback",
        "post_answers": "Post-feedback answers",
        "post_judged": "Post-test judging and n-gain",
        "metrics_stage": "Metrics and figures",
        "records": "records",
    },
    "zh": {
        "app_caption": "\u7aef\u5230\u7aef\u5b66\u751f\u6a21\u62df\u3001\u6559\u5b66\u53cd\u9988\u3001\u4fee\u6b63\u56de\u7b54\u3001\u81ea\u52a8\u8bc4\u5206\u4e0e\u6307\u6807\u8bc4\u4f30\u3002",
        "run_settings": "\u8fd0\u884c\u8bbe\u7f6e",
        "dry_run": "Dry run\uff08\u672c\u5730\u6f14\u793a\uff09",
        "max_questions": "\u6700\u591a\u9898\u76ee\u6570",
        "timeout_seconds": "\u5355\u6b21\u8bf7\u6c42\u8d85\u65f6\u79d2\u6570",
        "prompt_modes": "Prompt \u6a21\u5f0f",
        "provider_help": "DeepSeek \u548c Qwen \u4f7f\u7528 OpenAI-compatible API\uff0c\u5e76\u9700\u8981\u5bf9\u5e94\u7684 base URL\u3002",
        "base_url_help": "OpenAI \u53ef\u7559\u7a7a\u3002DeepSeek/Qwen \u5efa\u8bae\u4f7f\u7528\u9884\u8bbe URL\uff0c\u9664\u975e\u4f60\u7684 endpoint \u4e0d\u540c\u3002",
        "requests_to": "\u8bf7\u6c42\u5c06\u53d1\u9001\u81f3",
        "test_api": "\u6d4b\u8bd5 API",
        "need_key": "\u8bf7\u5148\u8f93\u5165 API key\uff0c\u6216\u4fdd\u6301 Dry run \u5f00\u542f\u3002",
        "testing_api": "\u6b63\u5728\u7528\u6781\u5c0f\u8bf7\u6c42\u6d4b\u8bd5 API...",
        "api_ok": "API \u6d4b\u8bd5\u901a\u8fc7\u3002\u8fd4\u56de",
        "api_failed": "API \u6d4b\u8bd5\u5931\u8d25",
        "run_full_demo": "\u8fd0\u884c\u5b8c\u6574 Demo",
        "select_prompt_mode": "\u8bf7\u81f3\u5c11\u9009\u62e9\u4e00\u4e2a prompt \u6a21\u5f0f\u3002",
        "running_pipeline": "\u6b63\u5728\u8fd0\u884c OpenStuSim pipeline...",
        "pipeline_finished": "Pipeline \u5df2\u5b8c\u6210\u3002",
        "live_process": "\u5b9e\u65f6\u8fc7\u7a0b",
        "latest_stage_preview": "\u6700\u65b0\u9636\u6bb5\u9884\u89c8",
        "questions": "\u95ee\u9898",
        "prompt_templates": "Prompt \u6a21\u677f",
        "process_trace": "\u8fc7\u7a0b\u8ffd\u8e2a",
        "metrics": "\u6307\u6807",
        "figures": "\u56fe\u8868",
        "question_set": "\u95ee\u9898\u96c6",
        "editable_prompts": "\u53ef\u7f16\u8f91 Prompt \u6a21\u677f",
        "prompt_edit_note": "\u70b9\u51fb\u4fdd\u5b58 Prompt \u6216\u8fd0\u884c\u5b8c\u6574 Demo \u65f6\u4f1a\u4fdd\u5b58\u4fee\u6539\u3002",
        "save_prompts": "\u4fdd\u5b58 Prompt",
        "prompts_saved": "Prompt \u6a21\u677f\u5df2\u4fdd\u5b58\u3002",
        "generated_records": "\u751f\u6210\u8bb0\u5f55",
        "teaching_process": "\u6559\u5b66\u8fc7\u7a0b\u89c6\u56fe",
        "record_selector": "\u9009\u62e9\u4e00\u6761\u6a21\u62df\u5b66\u751f\u8bb0\u5f55",
        "student_pre_answer": "\u5b66\u751f\u521d\u6b21\u56de\u7b54",
        "pre_score": "Pre-test \u5206\u6570",
        "teacher_feedback": "\u6559\u5e08\u6559\u5b66\u53cd\u9988",
        "student_revised_answer": "\u5b66\u751f\u4fee\u6b63\u56de\u7b54",
        "post_score": "Post-test \u5206\u6570",
        "n_gain_label": "Normalized gain",
        "no_trace": "\u6682\u65e0\u5b8c\u6574\u6559\u5b66\u8fc7\u7a0b\u3002\u8bf7\u5148\u8fd0\u884c Demo\u3002",
        "output_stage": "\u8f93\u51fa\u9636\u6bb5",
        "no_records": "\u6682\u65e0\u8bb0\u5f55\u3002\u8bf7\u5148\u8fd0\u884c Demo\u3002",
        "full_raw_records": "\u5b8c\u6574\u539f\u59cb\u8bb0\u5f55",
        "key_results": "\u5173\u952e\u7ed3\u679c",
        "auto_conclusion": "\u81ea\u52a8\u7ed3\u8bba",
        "raw_metrics": "\u539f\u59cb\u6307\u6807",
        "no_metrics": "\u6682\u65e0\u6307\u6807\u3002\u8bf7\u5148\u8fd0\u884c Demo\u3002",
        "summary_json": "Summary JSON",
        "profile_scores": "Profile \u5206\u6570",
        "normalized_gain": "Normalized gain",
        "label_distribution": "\u6807\u7b7e\u5206\u5e03",
        "figure_missing": "\u5c1a\u672a\u751f\u6210\u3002",
        "pre_answers": "Pre-test \u56de\u7b54",
        "pre_judged": "Pre-test \u8bc4\u5206",
        "feedback": "\u6559\u5e08\u53cd\u9988",
        "post_answers": "\u53cd\u9988\u540e\u56de\u7b54",
        "post_judged": "Post-test \u8bc4\u5206\u4e0e n-gain",
        "metrics_stage": "\u6307\u6807\u4e0e\u56fe\u8868",
        "records": "\u6761\u8bb0\u5f55",
    },
}


st.set_page_config(page_title="OpenStuSim Demo", layout="wide")

if "language" not in st.session_state:
    st.session_state["language"] = "English"


def _lang_code() -> str:
    return "zh" if st.session_state.get("language") == "\u4e2d\u6587" else "en"


def _t(key: str) -> str:
    return I18N[_lang_code()].get(key, key)


def _status(message: str, kind: str = "info") -> None:
    colors = {"success": "#0f7b0f", "error": "#b00020", "info": "#345995"}
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


def _stage_preview(path: Path) -> pd.DataFrame:
    df = _read_table(path)
    if df.empty:
        return df
    columns = [
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
        if col in df.columns
    ]
    return df[columns].head(6)


def _render_steps(placeholder, steps: list[dict]) -> None:
    with placeholder.container():
        st.subheader(_t("live_process"))
        for step in steps:
            st.markdown(f"- {step['name']}: **{step['count']}** {_t('records')}")
        if steps:
            latest = steps[-1]
            preview = _stage_preview(latest["path"])
            if not preview.empty:
                st.markdown(f"**{_t('latest_stage_preview')}: {latest['name']}**")
                st.dataframe(preview, use_container_width=True, hide_index=True)


def _metric_value(metrics_df: pd.DataFrame, metric: str, profile: str | None = None) -> float | None:
    sub = metrics_df[metrics_df["metric"] == metric]
    if profile is not None:
        sub = sub[sub["profile"] == profile]
    if sub.empty:
        return None
    return float(sub["value"].mean())


def _load_summary() -> dict:
    if not METRICS_SUMMARY_PATH.exists():
        return {}
    return json.loads(METRICS_SUMMARY_PATH.read_text(encoding="utf-8"))


def _build_conclusion(metrics_df: pd.DataFrame, summary: dict) -> str:
    if metrics_df.empty:
        return _t("no_metrics")

    low = _metric_value(metrics_df, "mean_pre_score", "Low")
    medium = _metric_value(metrics_df, "mean_pre_score", "Medium")
    high = _metric_value(metrics_df, "mean_pre_score", "High")
    mean_gain = _metric_value(metrics_df, "mean_n_gain")
    over = _metric_value(metrics_df, "over_improvement_rate")
    l1 = _metric_value(metrics_df, "label_distribution_l1")
    controllable = low is not None and medium is not None and high is not None and low <= medium <= high

    if _lang_code() == "zh":
        parts = [
            "Profile controllability \u57fa\u672c\u6210\u7acb\u3002"
            if controllable
            else "Profile controllability \u4e0d\u591f\u7406\u60f3\uff0cLow/Medium/High \u7684 pre-score \u6392\u5e8f\u4e0d\u5b8c\u5168\u7b26\u5408\u9884\u671f\u3002"
        ]
        if mean_gain is not None:
            parts.append(f"\u5e73\u5747 normalized gain \u4e3a {mean_gain:.2f}\uff0c\u8868\u793a\u53cd\u9988\u540e\u5b58\u5728\u53ef\u89c2\u5bdf\u7684\u63d0\u5347\u3002")
        if over is not None:
            parts.append(f"Over-improvement rate \u4e3a {over:.2%}\uff1b\u8be5\u503c\u8d8a\u9ad8\uff0c\u8d8a\u53ef\u80fd\u8bf4\u660e\u6a21\u578b\u628a\u5b66\u751f\u63d0\u5347\u5f97\u8fc7\u5feb\u3002")
        if l1 is not None:
            parts.append(f"\u6807\u7b7e\u5206\u5e03 L1 distance \u4e3a {l1:.2f}\uff1b\u8be5\u503c\u8d8a\u4f4e\uff0c\u6a21\u62df\u9519\u8bef\u7c7b\u578b\u8d8a\u63a5\u8fd1\u771f\u5b9e\u5b66\u751f\u5206\u5e03\u3002")
        parts.append("\u603b\u4f53\u4e0a\uff0c\u5e94\u540c\u65f6\u67e5\u770b profile \u6392\u5e8f\u3001gain\u3001over-improvement \u548c label alignment\uff0c\u800c\u4e0d\u662f\u53ea\u770b post-score\u3002")
        return " ".join(parts)

    parts = [
        "Profile controllability is broadly supported."
        if controllable
        else "Profile controllability is weak because Low/Medium/High pre-scores are not fully ordered."
    ]
    if mean_gain is not None:
        parts.append(f"The mean normalized gain is {mean_gain:.2f}, indicating observable feedback-driven improvement.")
    if over is not None:
        parts.append(f"The over-improvement rate is {over:.2%}; higher values suggest unrealistically large learning jumps.")
    if l1 is not None:
        parts.append(f"The label-distribution L1 distance is {l1:.2f}; lower values indicate closer alignment with real student error patterns.")
    parts.append("Overall interpretation should combine profile ordering, gain, over-improvement, and label alignment rather than relying only on post-score.")
    return " ".join(parts)


def _show_key_metrics(metrics_df: pd.DataFrame, summary: dict) -> None:
    if metrics_df.empty:
        _status(_t("no_metrics"), "info")
        return

    col1, col2, col3, col4 = st.columns(4)
    mean_pre = _metric_value(metrics_df, "mean_pre_score")
    mean_post = _metric_value(metrics_df, "mean_post_score")
    mean_gain = _metric_value(metrics_df, "mean_n_gain")
    l1 = _metric_value(metrics_df, "label_distribution_l1")
    col1.metric("Mean pre-score", f"{mean_pre:.2f}" if mean_pre is not None else "NA")
    col2.metric("Mean post-score", f"{mean_post:.2f}" if mean_post is not None else "NA")
    col3.metric("Mean n-gain", f"{mean_gain:.2f}" if mean_gain is not None else "NA")
    col4.metric("Label L1", f"{l1:.2f}" if l1 is not None else "NA")

    profile_rows = [row for row in summary.get("profile_metrics", []) if row.get("prompt_mode") and row.get("profile")]
    if profile_rows:
        profile_df = pd.DataFrame(profile_rows)
        keep = [
            "prompt_mode",
            "profile",
            "mean_pre_score",
            "mean_post_score",
            "mean_score_gain",
            "mean_n_gain",
            "over_improvement_rate",
        ]
        st.dataframe(profile_df[keep], use_container_width=True, hide_index=True)


def _render_teaching_trace(df: pd.DataFrame) -> None:
    if df.empty or "post_answer" not in df.columns:
        _status(_t("no_trace"), "info")
        return

    labels = [
        f"{row.qid} | {row.profile} | {row.prompt_mode}"
        for row in df[["qid", "profile", "prompt_mode"]].itertuples(index=False)
    ]
    selected = st.selectbox(_t("record_selector"), labels)
    row = df.iloc[labels.index(selected)]

    st.markdown(f"**Question:** {row['question']}")
    st.markdown(f"**Reference answer:** {row['reference_answer']}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"### {_t('student_pre_answer')}")
        st.write(row.get("pre_answer", ""))
        st.metric(_t("pre_score"), row.get("pre_score", "NA"))
        st.caption(str(row.get("pre_label", "")))
    with c2:
        st.markdown(f"### {_t('teacher_feedback')}")
        st.write(row.get("feedback", ""))

    c3, c4 = st.columns(2)
    with c3:
        st.markdown(f"### {_t('student_revised_answer')}")
        st.write(row.get("post_answer", ""))
    with c4:
        st.metric(_t("post_score"), row.get("post_score", "NA"))
        st.metric(_t("n_gain_label"), row.get("n_gain", "NA"))
        st.caption(str(row.get("post_label", "")))


def _save_prompts_from_state() -> None:
    for name, path in PROMPT_FILES.items():
        value = st.session_state.get(f"prompt_{name}")
        if value is not None:
            path.write_text(value, encoding="utf-8")


def _apply_api_env(api_key: str, base_url: str, timeout_seconds: int) -> None:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    elif "OPENAI_BASE_URL" in os.environ:
        del os.environ["OPENAI_BASE_URL"]
    os.environ["OPENAI_TIMEOUT"] = str(timeout_seconds)


def _test_api(model: str | None, temperature: float, api_key: str, base_url: str, timeout_seconds: int) -> str:
    _apply_api_env(api_key, base_url, timeout_seconds)
    client = LLMClient(model=model or None, dry_run=False)
    return client.generate("Reply with exactly: OK", temperature=temperature, max_tokens=8).strip()


def _run_pipeline(
    dry_run: bool,
    prompt_modes: list[str],
    model: str | None,
    temperature: float,
    api_key: str,
    base_url: str,
    timeout_seconds: int,
    max_questions: int,
    progress_placeholder=None,
) -> dict[str, int]:
    _save_prompts_from_state()
    _apply_api_env(api_key, base_url, timeout_seconds)

    questions = load_questions(SAMPLE_QUESTIONS_PATH).head(max_questions)
    profiles = build_default_profiles()
    client = LLMClient(model=model or None, dry_run=dry_run)
    steps: list[dict] = []

    def add_step(name: str, count: int, path: Path) -> None:
        steps.append({"name": name, "count": count, "path": path})
        st.session_state["run_steps"] = steps
        if progress_placeholder is not None:
            _render_steps(progress_placeholder, steps)

    n_pre = run_dryrun(questions, profiles, prompt_modes, client, temperature)
    add_step(_t("pre_answers"), n_pre, PRE_ANSWERS_PATH)
    n_pre_judged = run_judge(client)
    add_step(_t("pre_judged"), n_pre_judged, PRE_JUDGED_PATH)
    n_feedback = run_feedback(client)
    add_step(_t("feedback"), n_feedback, FEEDBACK_PATH)
    n_post = run_posttest(client, temperature=temperature)
    add_step(_t("post_answers"), n_post, POST_ANSWERS_PATH)
    n_post_judged = run_judge(
        client,
        input_path=POST_ANSWERS_PATH,
        output_path=POST_JUDGED_PATH,
        answer_field="post_answer",
        output_prefix="post",
    )
    add_step(_t("post_judged"), n_post_judged, POST_JUDGED_PATH)
    n_metrics = run_evaluation()
    add_step(_t("metrics_stage"), n_metrics, METRICS_CSV_PATH)
    return {
        "pre_answers": n_pre,
        "pre_judged": n_pre_judged,
        "feedback": n_feedback,
        "post_answers": n_post,
        "post_judged": n_post_judged,
        "metrics": n_metrics,
    }


with st.sidebar:
    st.selectbox("Language / \u8bed\u8a00", ["English", "\u4e2d\u6587"], key="language")

st.title("OpenStuSim Demo")
st.caption(_t("app_caption"))
run_progress = st.empty()

with st.sidebar:
    st.header(_t("run_settings"))
    dry_run = st.toggle(_t("dry_run"), value=True)
    all_questions = load_questions(SAMPLE_QUESTIONS_PATH)
    max_questions = st.number_input(_t("max_questions"), min_value=1, max_value=len(all_questions), value=len(all_questions), step=1)
    timeout_seconds = st.number_input(_t("timeout_seconds"), min_value=10, max_value=300, value=60, step=10)
    prompt_modes = st.multiselect(_t("prompt_modes"), options=list(PROMPT_TEMPLATE_PATHS.keys()), default=DEFAULT_PROMPT_MODES)
    provider = st.selectbox("Provider", options=list(PROVIDER_PRESETS.keys()), disabled=dry_run, help=_t("provider_help"))
    if st.session_state.get("last_provider") != provider:
        preset = PROVIDER_PRESETS[provider]
        st.session_state["model_input"] = preset["model"]
        st.session_state["base_url_input"] = preset["base_url"]
        st.session_state["last_provider"] = provider

    model = st.text_input("Model", key="model_input")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
    api_key = st.text_input("API key", type="password", disabled=dry_run)
    base_url = st.text_input("Base URL", key="base_url_input", disabled=dry_run, help=_t("base_url_help"))
    if not dry_run and base_url:
        st.caption(f"{_t('requests_to')}: {base_url}")

    test_clicked = st.button(_t("test_api"), use_container_width=True, disabled=dry_run)
    if test_clicked:
        if not api_key:
            _status(_t("need_key"), "error")
        else:
            with st.spinner(_t("testing_api")):
                try:
                    reply = _test_api(model, temperature, api_key, base_url, int(timeout_seconds))
                    _status(f"{_t('api_ok')}: {reply}", "success")
                except Exception as exc:
                    _status(f"{_t('api_failed')}: {exc}", "error")

    run_clicked = st.button(_t("run_full_demo"), type="primary", use_container_width=True)
    if run_clicked:
        if not prompt_modes:
            _status(_t("select_prompt_mode"), "error")
        else:
            with st.spinner(_t("running_pipeline")):
                try:
                    counts = _run_pipeline(
                        dry_run,
                        prompt_modes,
                        model,
                        temperature,
                        api_key,
                        base_url,
                        int(timeout_seconds),
                        int(max_questions),
                        progress_placeholder=run_progress,
                    )
                    st.session_state["last_counts"] = counts
                    _status(_t("pipeline_finished"), "success")
                except Exception as exc:
                    _status(str(exc), "error")

tabs = st.tabs([_t("questions"), _t("prompt_templates"), _t("process_trace"), _t("metrics"), _t("figures")])

with tabs[0]:
    st.subheader(_t("question_set"))
    st.dataframe(load_questions(SAMPLE_QUESTIONS_PATH), use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader(_t("editable_prompts"))
    st.caption(_t("prompt_edit_note"))
    for name, path in PROMPT_FILES.items():
        if f"prompt_{name}" not in st.session_state:
            st.session_state[f"prompt_{name}"] = load_prompt_template(path)
        st.text_area(name, key=f"prompt_{name}", height=190)
    if st.button(_t("save_prompts")):
        _save_prompts_from_state()
        _status(_t("prompts_saved"), "success")

with tabs[2]:
    st.subheader(_t("teaching_process"))
    post_df = _read_table(POST_JUDGED_PATH)
    _render_teaching_trace(post_df)

    st.subheader(_t("generated_records"))
    counts = st.session_state.get("last_counts")
    if counts:
        st.write(counts)
    steps = st.session_state.get("run_steps", [])
    if steps:
        _render_steps(st.empty(), steps)

    view_name = st.selectbox(
        _t("output_stage"),
        ["pre_answers", "pre_judged", "feedback", "post_answers", "post_judged"],
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
        _status(_t("no_records"), "info")
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
        with st.expander(_t("full_raw_records")):
            st.dataframe(stage_df, use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader(_t("key_results"))
    metrics_df = _read_table(METRICS_CSV_PATH)
    summary = _load_summary()
    if metrics_df.empty:
        _status(_t("no_metrics"), "info")
    else:
        _show_key_metrics(metrics_df, summary)
        st.subheader(_t("auto_conclusion"))
        st.markdown(_build_conclusion(metrics_df, summary))
        st.subheader(_t("raw_metrics"))
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    if METRICS_SUMMARY_PATH.exists():
        st.subheader(_t("summary_json"))
        st.json(summary)

with tabs[4]:
    st.subheader(_t("figures"))
    fig_paths = [
        (_t("profile_scores"), PROFILE_SCORES_FIG_PATH),
        (_t("normalized_gain"), N_GAIN_FIG_PATH),
        (_t("label_distribution"), LABEL_DISTRIBUTION_FIG_PATH),
    ]
    for title, path in fig_paths:
        st.markdown(f"**{title}**")
        if path.exists():
            st.image(str(path), use_container_width=True)
        else:
            _status(f"{path.name} {_t('figure_missing')}", "info")
