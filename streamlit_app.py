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

I18N = {
    "en": {
        "app_caption": "End-to-end student simulation, feedback, revision, judging, and metrics.",
        "run_settings": "Run Settings",
        "dry_run": "Dry run",
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
        "app_caption": "端到端学生模拟、教学反馈、修正回答、自动评分与指标评估。",
        "run_settings": "运行设置",
        "dry_run": "Dry run（本地演示）",
        "prompt_modes": "Prompt 模式",
        "provider_help": "DeepSeek 和 Qwen 使用 OpenAI-compatible API，并需要对应的 base URL。",
        "base_url_help": "OpenAI 可留空。DeepSeek/Qwen 建议使用预设 URL，除非你的 endpoint 不同。",
        "requests_to": "请求将发送至",
        "test_api": "测试 API",
        "need_key": "请先输入 API key，或保持 Dry run 开启。",
        "testing_api": "正在用极小请求测试 API...",
        "api_ok": "API 测试通过。返回",
        "api_failed": "API 测试失败",
        "run_full_demo": "运行完整 Demo",
        "select_prompt_mode": "请至少选择一个 prompt 模式。",
        "running_pipeline": "正在运行 OpenStuSim pipeline...",
        "pipeline_finished": "Pipeline 已完成。",
        "live_process": "实时过程",
        "latest_stage_preview": "最新阶段预览",
        "questions": "问题",
        "prompt_templates": "Prompt 模板",
        "process_trace": "过程追踪",
        "metrics": "指标",
        "figures": "图表",
        "question_set": "问题集",
        "editable_prompts": "可编辑 Prompt 模板",
        "prompt_edit_note": "点击保存 Prompt 或运行完整 Demo 时会保存修改。",
        "save_prompts": "保存 Prompt",
        "prompts_saved": "Prompt 模板已保存。",
        "generated_records": "生成记录",
        "output_stage": "输出阶段",
        "no_records": "暂无记录。请先运行 Demo。",
        "full_raw_records": "完整原始记录",
        "key_results": "关键结果",
        "auto_conclusion": "自动结论",
        "raw_metrics": "原始指标",
        "no_metrics": "暂无指标。请先运行 Demo。",
        "summary_json": "Summary JSON",
        "profile_scores": "Profile 分数",
        "normalized_gain": "Normalized gain",
        "label_distribution": "标签分布",
        "figure_missing": "尚未生成。",
        "pre_answers": "Pre-test 回答",
        "pre_judged": "Pre-test 评分",
        "feedback": "教师反馈",
        "post_answers": "反馈后回答",
        "post_judged": "Post-test 评分与 n-gain",
        "metrics_stage": "指标与图表",
        "records": "条记录",
    },
}


st.set_page_config(page_title="OpenStuSim Demo", layout="wide")


if "language" not in st.session_state:
    st.session_state["language"] = "English"


def _lang_code() -> str:
    return "zh" if st.session_state.get("language") == "中文" else "en"


def _t(key: str) -> str:
    return I18N[_lang_code()].get(key, key)


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

    controllable = (
        low is not None
        and medium is not None
        and high is not None
        and low <= medium <= high
    )

    if _lang_code() == "zh":
        parts = []
        parts.append("Profile controllability 基本成立。" if controllable else "Profile controllability 不够理想，Low/Medium/High 的 pre-score 排序不完全符合预期。")
        if mean_gain is not None:
            parts.append(f"平均 normalized gain 为 {mean_gain:.2f}，表示反馈后存在可观察的提升。")
        if over is not None:
            parts.append(f"Over-improvement rate 为 {over:.2%}；该值越高，越可能说明模型把学生提升得过快。")
        if l1 is not None:
            parts.append(f"标签分布 L1 distance 为 {l1:.2f}；该值越低，模拟错误类型越接近真实学生分布。")
        parts.append("总体上，应同时查看 profile 排序、gain、over-improvement 与 label alignment，而不是只看 post-score。")
        return " ".join(parts)

    parts = []
    parts.append("Profile controllability is broadly supported." if controllable else "Profile controllability is weak because Low/Medium/High pre-scores are not fully ordered.")
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

    profile_rows = [
        row
        for row in summary.get("profile_metrics", [])
        if row.get("prompt_mode") and row.get("profile")
    ]
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
    progress_placeholder=None,
) -> dict[str, int]:
    _save_prompts_from_state()
    _apply_api_env(api_key, base_url)

    questions = load_questions(SAMPLE_QUESTIONS_PATH)
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
    st.selectbox("Language / 语言", ["English", "中文"], key="language")

st.title("OpenStuSim Demo")
st.caption(_t("app_caption"))
run_progress = st.empty()

with st.sidebar:
    st.header(_t("run_settings"))
    dry_run = st.toggle(_t("dry_run"), value=True)
    prompt_modes = st.multiselect(
        _t("prompt_modes"),
        options=list(PROMPT_TEMPLATE_PATHS.keys()),
        default=DEFAULT_PROMPT_MODES,
    )
    provider = st.selectbox(
        "Provider",
        options=list(PROVIDER_PRESETS.keys()),
        disabled=dry_run,
        help=_t("provider_help"),
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
        help=_t("base_url_help"),
    )
    if not dry_run and base_url:
        st.caption(f"{_t('requests_to')}: {base_url}")

    test_clicked = st.button(_t("test_api"), use_container_width=True, disabled=dry_run)
    if test_clicked:
        if not api_key:
            _status(_t("need_key"), "error")
        else:
            with st.spinner(_t("testing_api")):
                try:
                    reply = _test_api(model, temperature, api_key, base_url)
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
                        progress_placeholder=run_progress,
                    )
                    st.session_state["last_counts"] = counts
                    _status(_t("pipeline_finished"), "success")
                except Exception as exc:
                    _status(str(exc), "error")

tabs = st.tabs([
    _t("questions"),
    _t("prompt_templates"),
    _t("process_trace"),
    _t("metrics"),
    _t("figures"),
])

with tabs[0]:
    st.subheader(_t("question_set"))
    questions_df = load_questions(SAMPLE_QUESTIONS_PATH)
    st.dataframe(questions_df, use_container_width=True, hide_index=True)

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
    st.subheader(_t("generated_records"))
    counts = st.session_state.get("last_counts")
    if counts:
        st.write(counts)

    steps = st.session_state.get("run_steps", [])
    if steps:
        _render_steps(st.empty(), steps)

    view_name = st.selectbox(
        _t("output_stage"),
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
