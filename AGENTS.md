# Agent instructions

Read [README.md](README.md) and the affected guide in [docs/](docs/) before changes.

- Keep extension code plain JavaScript and CSS. Do not add a build step or dependencies.
- Keep Python wrappers stdlib-only, except for the existing training toolkit and tokenizer APIs.
- Do not add code comments, tests, or unrelated features without a request.
- Keep technical instructions short and direct. Update affected docs with code changes.
- Preserve the extension permissions, `systemone_settings`, `systemone_history`, and `/v1/systemone` contract.
- Keep the product question in `extension/background.js` identical to `finetune/common.py`.
- Keep display thresholds at `0.35` and `0.10` in the worker, popup, and feed scanner.
- Preserve provenance and source groups. Keep explicit source splits. Do not use final-test content for generation, development, selection, or scoring.
- Treat supplied authorship labels as provisional unless evidence establishes otherwise. Never use commercial detector outputs as training labels.
- Use Luna models only for example generation, never coding work.
- Select checkpoints on matched validation before held-out scoring. Do not pool panels that share source groups or controls.
- Keep the trained decision head in FP32 during export. Treat probability drift as diagnostic, not an automatic export blocker.
- Ask before generation, GPU jobs, large downloads, or other long jobs. Agree on numeric resource limits first.
- Do not delete datasets, checkpoints, logs, reports, or model intermediates. Preserve at least 30 GiB of free disk for training.
- Keep corpus text, generated examples, credentials, environments, and weights out of Git. Never add proprietary extension code or assets.

## Verification

Run `node --check` on changed JavaScript files.
Run Python syntax checks and CLI help checks without model loading.
Use `tools/mock_server.py` for interface checks.
Do not run training or inference as a routine syntax check.

## Release state

The initial source-only release includes the extension, portable training tools, pinned configuration, and recorded results.
No public model or dataset release is included. See [results](docs/results.md) for open evaluation limits.
