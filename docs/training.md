# Training

This guide repeats the ClefSift process, not the exact frozen examples or reported scores.
The repo does not distribute training text or generated examples.
Different generators, package versions, and hardware can change the results.

Run these commands from the repo root.
Each output directory must be new. Keep all previous artifacts.

## 1. Prepare the environment

Reference training used Linux, Python 3.11.12, CUDA 13.0, and one 96 GB RTX PRO 6000 Blackwell GPU.
The toolkit uses one GPU per process. Two GPUs do not combine their memory.
CPU memory and disk must also accommodate the 27B model.
Training stops below 30 GiB of free disk.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) before these commands:

```bash
uv venv --python 3.11.12 .venv
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.14.1 torchvision==0.29.1
uv pip install --python .venv/bin/python -r finetune/requirements.txt
source .venv/bin/activate
```

`requirements.txt` pins the toolkit and main packages from the working environment.
It is not a complete lock of transitive dependencies.
The public toolkit pin is `c83f7bc1acc23c81ad0de838510c59c48bf60c79`.
Historical records identify toolkit version `0.1.0`, not its Git commit.

Download the pinned reference only after you check the available disk and memory:

```bash
export CLEFSIFT_MODEL=$(hf download Cloudflare/clef \
  --revision 2f3de3dd85f379784083b0814d997ab627200f0c)
```

The full model download is approximately 55 GB.
The tools use this local snapshot and do not download weights during training.

## 2. Prepare parent groups

Use English source text that you have permission to use and send to the generator.
Preserve authorship evidence, source rights, labels, and official split assignments.
The tools check structure and provenance fields. They do not establish authorship or legal rights.

Each source group needs one human, one fully AI, and one mixed example.
Keep all three examples in the same explicit `train` or `eval` split.
Do not include final-test text.

For your own corpus, supply one JSON object per line:

```json
{
  "id": "source-1-human",
  "text": "<permitted source text>",
  "label": "human",
  "group": "source-1",
  "split": "train",
  "source": "<source reference>",
  "lang": "eng",
  "label_status": "supplied-provisional",
  "authorship_status": "not-independently-verified",
  "rights_status": "<recorded permission or license>"
}
```

Preserve additional provenance fields when available.
Use `human`, `ai`, and `mixed` as the parent labels.
The preparation tool rejects duplicate text across groups and groups that cross splits.
It excludes a whole group when any complete request exceeds 1,536 tokens.
It does not truncate text.

```bash
python finetune/prepare.py --source data/source.jsonl \
  --model "$CLEFSIFT_MODEL" --out-dir data/parents
```

### Original LLMTrace selection

The original runs used `iitolstykh/LLMTrace_detection` at revision `332053804f8798177ff2ecb1148e1839ed429ba3`.
Authorship and source rights remain unresolved. The upstream Apache-2.0 declaration does not clear every source text.
Do not acquire or reuse this corpus unless your rights review permits it.

`finetune/llmtrace-selection.json` contains only source IDs, line numbers, and file hashes.
Its locators recover the original 300 training and 50 validation parent groups.
The original audit excluded malformed candidates, known duplicate groups, and flagged high-Cyrillic groups.
The locators retain those selection decisions. They do not replace a source-rights review.

If permitted, acquire only the pinned training and validation files:

```bash
hf download iitolstykh/LLMTrace_detection --repo-type dataset \
  --revision 332053804f8798177ff2ecb1148e1839ed429ba3 \
  --include train.jsonl valid.jsonl --local-dir data/llmtrace
python finetune/prepare.py --llmtrace data/llmtrace \
  --model "$CLEFSIFT_MODEL" --out-dir data/parents
```

The importer verifies both file hashes. It never reads the final-test file.
New metadata differs from the historical dataset files, even when request text matches.

## 3. Generate controlled variants

The original run used subscription-backed Luna models through Pi:
`gpt-5.6-luna` for development and `gpt-6-luna` for held-out edits and insertions.
Public access to those exact models is not established.

The public tool keeps the original prompts and acceptance checks but uses an OpenAI-compatible chat-completions API.
It uses provider sampling defaults without a sampling seed.
Use distinct, fixed model IDs for development and held-out generation.
The API must return the requested model ID, JSON output, and a complete assistant response.
An alias that resolves to another response ID will fail the identity check.
Development and holdout tasks for the same validation parent share one identical prompt.
A real API call sends that prompt to two different models, which is intended.
Key any replay or cache of saved generations on both the prompt and the model ID.
A prompt-only cache serves the development output to the holdout task and records the wrong generator.

Set these environment variables:

| Variable | Value |
|---|---|
| `CLEFSIFT_GENERATOR_URL` | API base URL, normally ending in `/v1` |
| `CLEFSIFT_DEVELOPMENT_MODEL` | Exact development model ID |
| `CLEFSIFT_HOLDOUT_MODEL` | Exact held-out model ID |
| `CLEFSIFT_GENERATOR_KEY` | API key, if required |

Use HTTPS for remote APIs. HTTP is permitted only on loopback addresses.
Do not put keys in source files or command arguments.

Create the plan without API calls:

```bash
python finetune/generate.py plan --parents data/parents \
  --out-dir data/generation --base-url "$CLEFSIFT_GENERATOR_URL" \
  --development-model "$CLEFSIFT_DEVELOPMENT_MODEL" \
  --holdout-model "$CLEFSIFT_HOLDOUT_MODEL"
```

For 300 training and 50 validation parents, the plan has 400 tasks and 600 requested variants.
Every parent receives a whole-text copy edit for the AI-assisted class.
Stable ranking assigns new mixed insertions to 125 training and 25 matched-validation groups.
Remaining matched groups retain the upstream mixed example.
Every held-out validation group receives both an edit and an insertion.
For smaller corpora, set `--train-mixed` and `--eval-mixed` explicitly.

Review the plan and agree on a generation budget before you run it:

```bash
python finetune/generate.py run --plan-dir data/generation \
  --max-calls 800 --minutes 60 --workers 4
```

The tool permits at most two attempts per task.
It saves prompts, model identities, hashes, checks, and results without API keys.
A repeat command reuses accepted outputs and does not reset failed-attempt limits.
Authentication and rate-limit failures stop new calls.
Review each run summary before assembly.

Length, number, lexical-retention, and copy checks do not prove semantic fidelity.
Review a sample for changes in facts, meaning, or viewpoint.
Do not use detector outputs to select generated examples.

## 4. Freeze the four-class dataset

```bash
python finetune/assemble.py --parents data/parents \
  --generation data/generation --model "$CLEFSIFT_MODEL" \
  --out-dir data/frozen
clef-finetune validate data/frozen/train.jsonl data/frozen/eval.jsonl \
  data/frozen/holdout.jsonl data/frozen/regression.jsonl
```

Assembly verifies parent and generation hashes and identities.
It decodes HTML entities and escaped whitespace, then collapses whitespace for every new class.
The tool excludes whole groups with failed generation, duplicate text, or requests above 1,536 tokens.
Each retained group has one example per class.

| File | Purpose |
|---|---|
| `train.jsonl` | Development training examples |
| `eval.jsonl` | Matched validation for checkpoint selection |
| `holdout.jsonl` | Held-out generator evaluation after selection |
| `regression.jsonl` | Byte-identical copy of the original parent validation requests |
| `manifest.json` | Provenance, exclusions, counts, hashes, and input limits |

Matched and held-out panels share parent groups and human/fully-AI controls.
Do not pool their scores. Distinct model IDs do not establish unrelated model-family coverage.

## 5. Train and select

Check GPU ownership and free disk first. Do not stop another process to make room.
The runner uses the [128-step recipe](../finetune/recipe.yaml), starting from the fresh pinned reference.

```bash
python finetune/run.py --data-dir data/frozen --model "$CLEFSIFT_MODEL" \
  --out-dir runs/v2 --minutes 90 --gpu 0
```

The runner evaluates the base and checkpoints 64 and 128 on matched validation.
It selects the highest fixed-four macro F1, then lower Brier, then the earlier step.
It writes `selection.json` before held-out evaluation.
It also scores the selected checkpoint on the original regression requests.

The time limit includes model loading, training, saves, and evaluation.
The runner stops its own process group on failure, non-finite training values, interruption, or resource limits.
It does not delete artifacts or change a running server.
Logs, package versions, configuration, dataset hashes, and selection records remain in the run directory.

To repeat v1 first, use the same recipe with three-target parent validation:

```bash
python finetune/run.py --data-dir data/parents --model "$CLEFSIFT_MODEL" \
  --out-dir runs/v1 --minutes 90 --gpu 0 --three-targets
```

For the v2 baseline comparison, add `--prior-checkpoint` with the checkpoint from `runs/v1/selection.json`.
That checkpoint supplies a comparison only. v2 still starts from the reference.

## Metrics and export

Each evaluation saves toolkit predictions and `fixed_metrics.json`.
Accuracy, F1, and human false positives use the winning class before the display uncertainty rule.
Fixed-four F1 uses all four labels. Original v1 validation uses fixed-three target F1 but retains all four prediction options.
Multiclass Brier sums squared error across all four probabilities. Lower is better.
A human false positive is any non-human prediction on a provisional human target.

Recompute metrics from saved predictions without inference:

```bash
python finetune/evaluate.py summarize --data data/frozen/eval.jsonl \
  --predictions runs/v2/step-128-matched/tuned_predictions.jsonl \
  --out reports/matched.json
```

Use the selected step from `selection.json` if it differs from 128.
Continue with [export and serving](serving.md).
See [recorded results](results.md) for the original run and its limits.
