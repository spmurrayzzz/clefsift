# Training

Training uses Unsloth `FastDecisionModel` and `DecisionTrainer`. uv installs the dependencies from [pyproject.toml](../pyproject.toml) and [uv.lock](../uv.lock).

## Setup

Use a CUDA machine with [uv](https://docs.astral.sh/uv/getting-started/installation/) installed. Pass paths directly as command arguments. No environment variables or YAML edits are needed.

### 1. Clone and install

```bash
git clone https://github.com/spmurrayzzz/clefsift.git
cd clefsift
uv sync --locked --extra train
```

### 2. Download the dataset to your target path

```bash
uv run --no-sync hf download spmurrayzzz/clefsift-cleaned clefsift-cleaned.jsonl \
  --repo-type dataset --revision 1f891d864e5759fc15f6706a41a16285a92d259f \
  --local-dir /path/to/dataset
```

The command creates the target directory. Replace `/path/to/dataset` with your chosen location.

### 3. Train with explicit paths and splits

Use a complete local Clef reference checkpoint. See [reference checkpoint](#reference-checkpoint) if you need to download one.

For the dataset downloaded above, copy the split values as shown. They preserve its existing splits; you do not need to count rows.

```bash
uv run --no-sync tools/clefsift train \
  --model /path/to/clef \
  --data /path/to/dataset/clefsift-cleaned.jsonl \
  --output-dir /path/to/output \
  --split-rows 73896 8424 8464
```

`--model` is the reference checkpoint directory. `--data` is the dataset file or checkout directory. `--output-dir` is the parent directory for the run's prepared data, checkpoints, reports, and export. These paths can use different disks.

The existing test holdout stays out of training and validation. For your own unsplit dataset, use [fraction-based splits](#choose-splits-at-runtime).

Use an available GPU and keep at least 30 GiB free beyond checkpoint and export space. The default recipe runs one epoch on one GPU with a 720-minute total limit, including validation and export.

The finished fine-tuned model is `/path/to/output/<timestamp>-<id>/release/`. The command prints the full destination as `Release:`. This folder contains the merged weights, FP32 decision head, and tokenizer files. Export is automatic; no second command is needed.

## Reference checkpoint

If you already have the pinned Cloudflare Clef snapshot, pass its directory to `--model`. Otherwise, download it once to your chosen path:

```bash
uv run --no-sync hf download Cloudflare/clef \
  --revision 2f3de3dd85f379784083b0814d997ab627200f0c \
  --local-dir /path/to/clef
```

## Choose splits at runtime

For your own genuinely unsplit dataset, choose fractions. This example uses 80% train, 10% validation, and 10% test:

```bash
uv run --no-sync tools/clefsift train \
  --model /path/to/clef \
  --data /path/to/examples.jsonl \
  --output-dir /path/to/output \
  --split 0.8 0.1 0.1 --seed 42
```

The order is train, validation, test. These fractions are examples, not defaults. They must sum to one; train and validation must be positive, and test may be zero. The required seed makes splitting reproducible and also sets the training seed.

Input can be a JSONL file with `text` and `label` fields or a checkout containing exactly one JSONL file. Supplied groups, author/document identities, and duplicate texts stay together. Each present label needs enough rows for the requested splits. Actual counts can differ from the requested fractions; the preparation report records them.

For ordered data with existing boundaries, use `--split-rows` with its original counts, as in the published-dataset walkthrough. Counts refer to nonblank JSON records. Preparation reads only the train and validation prefix; the declared test count is recorded without reading or verifying the tail. Do not randomly repartition the published dataset's existing holdout.

Already separated `train.jsonl` and `validation.jsonl` (or `eval.jsonl`) files need no split flags. Rows with explicit `split` or `metadata.split` assignments also keep their existing splits. Existing test files remain unopened. Split flags cannot override existing assignments, and mixed assigned/unassigned input fails.

## Training settings

The default [recipe](../finetune/recipe-general.yaml) sets one epoch and a 720-minute total cap, including validation and export. To change training settings, make a local copy and add `--config /path/to/recipe.yaml` to the train command. The general recipe requires positive epochs and cannot fall back to the historical 128-step schedule.

The `train` section uses Hugging Face `TrainingArguments` names. The `lora` and `decision` sections use Unsloth argument names. CLI paths override recipe paths and resolve from the current directory. Recipe paths resolve from the recipe file. Keep private recipe copies outside Git.

## Run behavior

Each invocation creates a new timestamped directory under `--output-dir`. Rank zero prepares train and validation files before loading model weights and records runtime options, actual counts, source locators, and hashes. All workers read only the prepared train and validation files. Original text, extra source fields, and source files stay unchanged.

The run selects a checkpoint on validation and exports automatically to its fixed `release/` subdirectory. `--output-dir` selects the parent for the whole run, not an exact final model directory. Prior runs and failed artifacts are preserved. Training does not deploy or restart a server. Existing internal dataset directories with a manifest remain supported.

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
