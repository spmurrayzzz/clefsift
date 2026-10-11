# Training

Training uses Unsloth `FastDecisionModel` and `DecisionTrainer`. Dependencies live in [pyproject.toml](../pyproject.toml) and [uv.lock](../uv.lock); uv manages the Python environment. There is no required environment directory name or machine-specific launcher.

## Setup

From the repository root, install the training dependencies once:

```bash
uv sync --locked --extra train
```

Use a permitted local copy of the frozen dataset and the pinned Cloudflare Clef snapshot. Training never downloads model weights or generates examples. Do not use the pooled public text/label export as training input; it lacks split and provenance information.

Edit [recipe-general.yaml](../finetune/recipe-general.yaml):

- `model.path`: local Clef snapshot at revision `2f3de3dd85f379784083b0814d997ab627200f0c`.
- `data.directory`: frozen dataset directory containing `manifest.json`, `train.jsonl`, and `eval.jsonl`.
- `output_dir`: parent directory for new runs.
- `minutes`: total wall-time limit, including validation and export.
- `train.num_train_epochs`: requested epochs.

Paths resolve from the recipe file. Keep private paths in a local copy outside Git. The `train` section uses Hugging Face `TrainingArguments` names. The `lora` and `decision` sections use Unsloth argument names.

The general recipe requires epochs. It cannot fall back to the historical 128-step limit. The example's 720-minute cap is a limit, not a runtime estimate or approval to start a job.

## Run

Check available GPU memory and disk before starting. Do not stop another job to make room. Keep at least 30 GiB free in addition to checkpoint and export space.

```bash
uv run --no-sync tools/clefsift train
```

For a local recipe:

```bash
uv run --no-sync tools/clefsift train --config /path/to/recipe.yaml
```

Every invocation creates a new timestamped directory under `output_dir`. It trains, selects a checkpoint on validation, and exports the selected model to that run's `release/` directory. It preserves prior runs and failed artifacts. It does not deploy or restart a server.

## Multiple GPUs

Use PyTorch's launcher and standard device selection:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --no-sync torchrun --standalone \
  --nproc_per_node=2 tools/clefsift train --config /path/to/recipe.yaml
```

DDP keeps one complete model copy on each selected GPU. It does not combine their memory. Each worker receives different training batches; the framework synchronizes gradients and gathers validation predictions. Only rank zero writes shared reports and exports the model.

The global batch size is:

```text
per_device_train_batch_size × gradient_accumulation_steps × GPU count
```

The default single-GPU recipe uses a global batch of four. To retain that batch with two GPUs, set `gradient_accumulation_steps: 2`. The runner prints the effective batch and optimizer-step count before loading weights. Accelerate can repeat tail examples to make distributed batches equal in size.

## What the run preserves

- The existing decision question, four labels, source groups, and explicit splits.
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
