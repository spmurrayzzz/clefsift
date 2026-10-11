# Training

Training uses Unsloth `FastDecisionModel` and `DecisionTrainer`. Dependencies live in [pyproject.toml](../pyproject.toml) and [uv.lock](../uv.lock); uv manages the Python environment. There is no required environment directory name or machine-specific launcher.

## Setup

From the repository root, install the training dependencies once:

```bash
uv sync --locked --extra train
```

Use a permitted local dataset and the pinned Cloudflare Clef snapshot. Training accepts a JSONL file with `text` and `label` fields, or a checkout containing exactly one JSONL file. It writes separate train and validation files before loading model weights. It does not require an input manifest or regenerate examples.

The [recipe](../finetune/recipe-general.yaml) supplies training settings and optional default paths:

- `model.path`: local Clef snapshot at revision `2f3de3dd85f379784083b0814d997ab627200f0c`. `--model` overrides it.
- `data.path`: JSONL file or dataset checkout. `--data` overrides it.
- `output_dir`: parent directory for new runs. `--output-dir` overrides it.
- `minutes`: total wall-time limit, including validation and export.
- `train.num_train_epochs`: requested epochs.

Recipe paths resolve from the recipe file. CLI paths resolve from the current directory and can use different disks. Prepared data, checkpoints, reports, and exports stay under `--output-dir`; the source model and dataset remain in their original locations. No recipe edits are needed when CLI paths are supplied. Keep private paths in a local copy outside Git. The `train` section uses Hugging Face `TrainingArguments` names. The `lora` and `decision` sections use Unsloth argument names.

Dependency and cache locations are separate from run output. Set `UV_PROJECT_ENVIRONMENT` and `UV_CACHE_DIR` before installation to choose the uv environment and cache directories. `HF_HOME` selects the Hugging Face cache. `TMPDIR` selects the temporary directory.

## Choose splits at runtime

For genuinely unsplit input, provide train, validation, and test fractions plus a seed:

```bash
uv run --no-sync tools/clefsift train --data /path/to/examples.jsonl \
  --split 0.8 0.1 0.1 --seed 42
```

These values are examples, not defaults. Fractions must sum to one. Train and validation must be positive; test may be zero. Random splitting requires an explicit seed, which also overrides the training seed. Ungrouped rows are stratified by their supplied labels. Each present label needs enough rows for all requested splits. Supplied source groups, author/document identities, and duplicate texts stay together. Grouped splits can differ from the target fractions; the report records actual counts.

For an ordered file with existing split boundaries, supply the counts instead:

```bash
uv run --no-sync tools/clefsift train --data /path/to/ordered-data.jsonl \
  --split-rows "$TRAIN_ROWS" "$VALIDATION_ROWS" "$TEST_ROWS"
```

Counts refer to nonblank JSON records. This writes only the requested train and validation prefix. The test count is recorded as caller-declared; the remaining source is neither read nor verified. Use this mode with the original boundaries of the pooled public export. Do not randomly repartition its existing holdout.

Already separated `train.jsonl` and `validation.jsonl` (or `eval.jsonl`) files need no split arguments. The same applies to rows with explicit `split` or `metadata.split` assignments. Existing test files remain unopened. Split arguments cannot override existing assignments, and mixed assigned/unassigned input fails.

The general recipe requires epochs. It cannot fall back to the historical 128-step limit. The example's 720-minute cap is a limit, not a runtime estimate or approval to start a job.

## Run

Check available GPU memory and disk before starting. Do not stop another job to make room. Keep at least 30 GiB free on the output filesystem in addition to checkpoint and export space.

```bash
uv run --no-sync tools/clefsift train \
  --model /path/to/clef \
  --data /path/to/examples.jsonl \
  --output-dir /path/to/runs \
  --split 0.8 0.1 0.1 --seed 42
```

For a local recipe:

```bash
uv run --no-sync tools/clefsift train --config /path/to/recipe.yaml \
  --split 0.8 0.1 0.1 --seed 42
```

Every invocation creates a new timestamped directory under `output_dir`. Rank zero prepares the selected splits onto disk and records runtime options, actual counts, source locators, and hashes. All workers then read only the prepared train and validation files. Original text and extra source fields remain unchanged. The source files stay untouched.

The run trains, selects a checkpoint on validation, and exports the selected model. With `--output-dir /path/to/runs`, the finished model is:

```text
/path/to/runs/<timestamp>-<id>/release/
```

This directory contains the merged BF16 weights, FP32 decision head, and tokenizer files. Prepared data, checkpoints, and reports stay in the same run directory. The runner generates `<timestamp>-<id>` and uses the fixed `release/` name. `--output-dir` selects the parent for the whole run, not an exact final model directory. The command prints the full destination as `Release:` after export.

The run preserves prior runs and failed artifacts. It does not deploy or restart a server. Existing internal dataset directories with a manifest remain supported.

## Multiple GPUs

Use PyTorch's launcher and standard device selection:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --no-sync torchrun --standalone \
  --nproc_per_node=2 tools/clefsift train --config /path/to/recipe.yaml \
  --split 0.8 0.1 0.1 --seed 42
```

DDP keeps one complete model copy on each selected GPU. It does not combine their memory. Each worker receives different training batches; the framework synchronizes gradients and gathers validation predictions. Only rank zero writes shared reports and exports the model.

The global batch size is:

```text
per_device_train_batch_size × gradient_accumulation_steps × GPU count
```

The default single-GPU recipe uses a global batch of four. To retain that batch with two GPUs, set `gradient_accumulation_steps: 2`. The runner prints the effective batch and optimizer-step count before loading weights. Accelerate can repeat tail examples to make distributed batches equal in size.

## What the run preserves

- The decision question, four labels, explicit splits, and any supplied source groups.
- Complete requests, with no truncation or JSON reinterpretation of state text.
- A BF16 backbone, new LoRA adapters, and the existing FP32 decision head.
- Label-smoothed cross-entropy plus Brier loss, without automatic calibration or augmentation.
- Checkpoint selection by macro F1, then lower Brier, then earlier step.
- The final holdout remains excluded from training, selection, and default validation.

The final optimizer step is saved and evaluated even between save intervals. After selection, the run measures the original reference with adapters disabled and its original head restored. This avoids preparing the model for evaluation before DDP training. Historical held-out and regression panels, when present, are scored only after selection; they are not pooled.

## Results and export

The command prints the run directory, selected checkpoint, and release directory. Each run contains:

| File | Purpose |
|---|---|
| `data/train.jsonl`, `data/validation.jsonl` | Automatically prepared training and validation files |
| `data/test.jsonl` | New test partition when a positive random test fraction was requested; never scored automatically |
| `data/preparation.json` | Runtime split settings, actual counts, source locators, and prepared-file hashes |
| `config.json`, `environment.json` | Configuration, schedule, and dependency versions |
| `train_log.jsonl`, `progress.json` | Training and checkpoint-selection progress |
| `train/checkpoint-*` | Unsloth adapters, FP32 head, and framework training state |
| `step-*-matched/` | Validation predictions and metrics |
| `selection.json` | Selected checkpoint and artifact hashes |
| `summary.json` | Completion or failure, including export status |
| `release/` | Merged BF16 backbone, FP32 head, and original tokenizer/encoder files |

A successful general run reports `selected-final-sealed` and `export_status: complete`. It does not claim final evaluation has occurred. If export fails, the selected checkpoint remains available for the standalone [export command](serving.md#export-a-new-checkpoint).

For historical datasets, select [recipe.yaml](../finetune/recipe.yaml) explicitly. That recipe retains the 128-step schedule. Historical optimizer state is not an Unsloth resume format. This command always starts a fresh run.

Dataset preparation and generation remain separate. See the [general corpus workflow](general-corpus.md). No new accuracy claim follows from syntax or interface checks; GPU validation remains necessary before a full run.
