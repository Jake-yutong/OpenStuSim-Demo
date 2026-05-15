# OpenStuSim Demo

A minimal research demo for testing whether LLMs can simulate students answering open-ended science questions.

The demo can run locally in dry-run mode or call an OpenAI-compatible chat API for answer generation, feedback, revision, and judging.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python run_demo.py --dry_run
```

To render both student simulation prompt modes:

```bash
python run_demo.py --dry_run --prompt_modes roleplay teacher_prediction
```

To call a real OpenAI-compatible API:

```bash
set OPENAI_API_KEY=your_key_here
python run_demo.py --model gpt-4o-mini --prompt_modes roleplay teacher_prediction
```

Optional environment variables:

- `OPENAI_BASE_URL`
- `MODEL_NAME`

If `data/sample_questions.csv` is missing, the script creates a dummy file with five open-ended science questions.

## Outputs

Check:

- `data/sample_questions.csv`
- `outputs/pre_answers.jsonl`
- `outputs/pre_judged.jsonl`
- `outputs/feedback.jsonl`
- `outputs/post_answers.jsonl`
- `outputs/post_judged.jsonl`
- `outputs/metrics.csv`
- `outputs/metrics_summary.json`
- `figures/profile_scores.png`
- `figures/n_gain.png`
- `figures/label_distribution.png`

`post_judged.jsonl` contains pre/post labels, scores, tutor feedback, revised answers, and per-record normalized gain.

## Metrics

`outputs/metrics.csv` stores long-form metric rows. `outputs/metrics_summary.json` stores grouped summaries.

- Profile controllability: mean pre-score, post-score, and score gain for each prompt mode and student profile.
- Normalized gain: mean `n_gain = (post_score - pre_score) / (1 - pre_score)`, excluding records where `pre_score >= 1.0`.
- Over-improvement rate: percentage of records where `pre_score == 0` and `post_score == 1.0`.
- Per-question difficulty correlation: Pearson correlation between simulated mean pre-score per question and `real_mean_score`.
- Label distribution alignment: L1 distance between real SciEntsBank-style label proportions and simulated pre-label proportions.
- Answer length: mean word and character counts of generated pre-test answers.

## Limitations

This is a rapid research demo, not a validated benchmark. Dry-run outputs are deterministic placeholders. Real LLM results depend on the model, prompt mode, API settings, and judge reliability. The sample CSV is tiny; replace it with a larger SciEntsBank-derived table before drawing research conclusions.
