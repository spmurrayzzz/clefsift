# Model export and serving

Model card and files: [spmurrayzzz/clefsift](https://huggingface.co/spmurrayzzz/clefsift).
The weight upload is in progress. Download commands apply after the files arrive.
The model repo will contain the merged BF16 reference and `ClefSift-Q8_0.gguf`.

Run commands from the Git repo root.
Use new output paths. Preserve checkpoints, datasets, logs, and intermediates.
Do not replace or restart an existing server without approval.

## Use the published Q8 model

Install the Hugging Face CLI before downloading.
Set `CLEFSIFT_REVISION` to a Hub commit that contains the uploaded model files.
Use that fixed commit for repeatable deployment.

```bash
hf download spmurrayzzz/clefsift ClefSift-Q8_0.gguf \
  --revision "$CLEFSIFT_REVISION" --local-dir models/clefsift
```

The Q8 file occupies 26.76 GiB.
Its SHA-256 is `1fbdedec7989395973e5364a47dcd4a01a5f419a425b26ba476faed5556cb7fa`.
Verify the file hash before serving it.
The [model card](https://huggingface.co/spmurrayzzz/clefsift) explains direct reference inference and its head-dtype differences.

## Build the pinned llama.cpp tools

Conversion and Q8 evaluation used llama.cpp commit `a46709b683aba9274d8ab29f5b42f7e551d1dff6`, build 11450.
Older builds can lack Clef decision support.

```bash
git clone https://github.com/ggml-org/llama.cpp tools/llama.cpp
git -C tools/llama.cpp checkout a46709b683aba9274d8ab29f5b42f7e551d1dff6
cmake -S tools/llama.cpp -B tools/llama.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build tools/llama.cpp/build --config Release \
  --target llama-server llama-quantize -j 4
```

For NVIDIA GPU serving, add `-DGGML_CUDA=ON` to the CMake configuration command.
The default macOS build supports Metal.
Follow [upstream build instructions](https://github.com/ggml-org/llama.cpp/blob/a46709b683aba9274d8ab29f5b42f7e551d1dff6/docs/build.md) for required system tools.

## Serve text decisions locally

Check that port 8000 is free before starting the server.
This loopback listener does not expose the model to other network hosts.

```bash
tools/llama.cpp/build/bin/llama-server \
  --host 127.0.0.1 --port 8000 \
  --model models/clefsift/ClefSift-Q8_0.gguf \
  -b 32768 -ub 32768
```

Keep both batch values at `32768`.
The default physical batch of 512 fails on longer decision requests.
Text-only requests do not need a vision projector.
The original evaluations reused the upstream projector, but they did not test vision behavior.

From another terminal, inspect the service:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
```

Use `http://127.0.0.1:8000/v1` in extension settings.
Leave the key empty for this local listener.
Select the exact ID from the model list, save, and test the connection.
Reload LinkedIn after a model change.

## Export a new checkpoint

[Training](training.md) produces `selection.json` with the selected checkpoint path.
Keep the original pinned reference snapshot in `CLEFSIFT_MODEL`.

CPU merging needs enough RAM for the backbone and adapter merge.
Plan disk space for the base weights, a roughly 55 GB merged release, a roughly 54 GB F16 file, and Q8.
These sizes exclude checkpoints and temporary files.

```bash
export CLEFSIFT_CHECKPOINT=$(python -c \
  'import json; print(json.load(open("runs/v2/selection.json"))["checkpoint"])')
clef-finetune merge --model "$CLEFSIFT_MODEL" \
  --checkpoint "$CLEFSIFT_CHECKPOINT" --output releases/clefsift/reference \
  --dtype bfloat16 --head-dtype float32
```

The trained decision head must remain FP32.
The toolkit merge default is BF16. Do not omit `--head-dtype float32`.
The toolkit merges on CPU, even on a GPU host.

Conversion used a separate environment with Python 3.11, torch 2.11.0, and transformers 4.57.6.
Do not reuse an old llama.cpp tokenizer environment.

```bash
uv venv --python 3.11 .venv-convert
uv pip install --python .venv-convert/bin/python \
  torch==2.11.0 transformers==4.57.6 numpy sentencepiece safetensors
PYTHONPATH=tools/llama.cpp/gguf-py .venv-convert/bin/python \
  tools/llama.cpp/convert_hf_to_gguf.py releases/clefsift/reference \
  --outfile releases/clefsift/ClefSift-F16.gguf --outtype f16
tools/llama.cpp/build/bin/llama-quantize \
  releases/clefsift/ClefSift-F16.gguf \
  releases/clefsift/ClefSift-Q8_0.gguf Q8_0
```

The converter can warn that transformers does not recognize `qwen3_5`.
The original conversion succeeded through its `config.json` fallback.
Keep the joint decision head with the merged reference. A backbone-only export is incomplete.

## Compare Q8 with the selected reference

Use identical frozen requests and the selected pre-merge reference predictions.
Start a separate Q8 listener on a free port, such as 8002.
Do not replace the production listener for a comparison.
Use the model ID that the comparison server lists.

```bash
python finetune/evaluate.py http --data data/frozen/eval.jsonl \
  --base-url http://127.0.0.1:8002/v1 --model "$CLEFSIFT_Q8_MODEL_ID" \
  --out-dir reports/q8-matched --max-requests 188
python finetune/evaluate.py compare --data data/frozen/eval.jsonl \
  --reference runs/v2/step-128-matched/tuned_predictions.jsonl \
  --candidate reports/q8-matched/tuned_predictions.jsonl \
  --out reports/q8-comparison.json
```

Use your actual selected step and request count.
The HTTP tool rejects a dataset larger than `--max-requests` before making requests.
It saves each successful prediction and all four probabilities.
An optional `CLEFSIFT_API_KEY` environment variable supplies a Bearer key.

Repeat separately for held-out and regression requests.
Do not pool shared validation panels.
The report includes class agreement, human false positives, recall, Brier, probability drift, and display changes.
Probability differences are diagnostics, not automatic export blockers.
Keep the extension thresholds unchanged unless independent calibration supports a separate change.

Merging, conversion, quantization, and backend differences all affect this comparison.
It does not isolate quantization alone.
