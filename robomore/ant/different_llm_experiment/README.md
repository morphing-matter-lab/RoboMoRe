# Different LLM Experiment (Ant)

This folder contains the multi-model Ant experiment entrypoint:

- `run_ant_multi_llm_experiment.py`

It reuses core logic from:

- `robomore/ant/design.py` (`ant_design`)
- `robomore/ant/utils.py` (`Train`, `Eva`, `compute_ant_volume`)
- `robomore/ant/prompts.py` (prompt templates)

## What this script does

For each configured LLM model:

1. Generate diverse reward functions.
2. Generate diverse morphologies.
3. Run grid combinations (`rewardfunc x morphology`):
   - train with `utils.Train(...)`
   - evaluate with `utils.Eva(...)`
4. Track LLM token usage and estimated cost.
5. Export per-model CSV files.

## Environment variables

Set API keys by provider:

- OpenAI-family models:
  - `OPENAI_API_KEY` (or `LLM_API_KEY`)
  - optional `OPENAI_BASE_URL` (defaults to `https://api.openai.com/v1`)
- DeepSeek models:
  - `DEEPSEEK_API_KEY`
  - optional `DEEPSEEK_BASE_URL` (defaults to `https://api.deepseek.com/v1`)
- Claude models:
  - `ANTHROPIC_API_KEY` (or `Anthropic_KEY`)
  - optional `ANTHROPIC_BASE_URL` (defaults to `https://api.anthropic.com`)

Training device/runtime is controlled in `utils.py` via env vars:

- `ROBOMORE_DEVICE` (`cuda` or `cpu`)
- `ROBOMORE_NUM_ENVS`
- `ROBOMORE_VEC_MODE` (`subproc` or `dummy`)
- `ROBOMORE_BATCH_SIZE`
- `ROBOMORE_TRAIN_FREQ`
- `ROBOMORE_GRADIENT_STEPS`

## Example run

```bash
conda activate robomore

export OPENAI_API_KEY="..."
export DEEPSEEK_API_KEY="..."
export ANTHROPIC_API_KEY="..."

export ROBOMORE_DEVICE=cuda
export ROBOMORE_VEC_MODE=subproc
export ROBOMORE_NUM_ENVS=32
export ROBOMORE_BATCH_SIZE=2048
export ROBOMORE_TRAIN_FREQ=16
export ROBOMORE_GRADIENT_STEPS=8

python run_ant_multi_llm_experiment.py \
  --models gpt-4o-mini o3 gpt-5.5-medium deepseek-v4 claude-opus-4.7 \
  --morphology-nums 2 \
  --rewardfunc-nums 2 \
  --total-timesteps 100000 \
  --eval-runs 5
```

## Outputs

Each model creates one timestamped folder under:

- `robomore/ant/different_llm_experiment/results/<timestamp>_<model_tag>/`

CSV outputs are under `csv/`:

- `<model_tag>_results.csv`
- `<model_tag>_usage.csv`
- `<model_tag>_usage_summary.csv`
- `<model_tag>_coarse_efficiency_matrix.csv`
- `<model_tag>_coarse_fitness_matrix.csv`
- `<model_tag>_fine_efficiency_matrix.csv`
- `<model_tag>_fine_fitness_matrix.csv`

## Resume / skip trained checkpoints

When checkpoints already exist, you can skip retraining and go straight to evaluation:

```bash
python run_ant_multi_llm_experiment.py \
  --models qwen3.7max \
  --morphology-nums 25 \
  --rewardfunc-nums 5 \
  --reuse-generation-from "/path/to/old_run_dir" \
  --reuse-training-from "/path/to/old_run_dir"
```

- If `--reuse-training-from` is omitted but `--reuse-generation-from` is set, the script will also try that directory for checkpoints.
- Use `--force-retrain` to ignore existing checkpoints and retrain all pairs.

## In-place run directory

By default, the script creates a new timestamped run folder.  
If you want to continue and write directly into an existing folder, set `--run-dir`:

```bash
python run_ant_multi_llm_experiment.py \
  --models claude-opus-4.7 \
  --run-dir "results/<timestamp>_claude_opus_4_7" \
  --reuse-generation-from "results/<timestamp>_claude_opus_4_7" \
  --reuse-training-from "results/<timestamp>_claude_opus_4_7"
```

`--run-dir` currently supports exactly one model per run (to avoid output file/checkpoint collisions).

When `--run-dir` points to an existing run directory and `--reuse-generation-from` is not provided, the script will auto-detect `env/` and `assets/` and reuse them (resume mode), so it won't regenerate reward/morphology artifacts.

## Submission-safe recommendation

- Never commit real API keys into source files or notebooks.
- Prefer environment variables and `.env` files ignored by git.
- Use `scripts/prepare_submission_package.py` to export a sanitized release copy before submission.
