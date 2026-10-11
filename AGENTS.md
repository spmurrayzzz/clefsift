# Agent instructions

Read [README.md](README.md) and the affected guide in [docs/](docs/) before changes.

- Keep extension code plain JavaScript and CSS. Do not add a build step or dependencies.
- Keep Python wrappers stdlib-only, except for the existing training toolkit/tokenizer APIs and the approved Parquet/7z source readers in `finetune/requirements-ingest.txt`. Install readers only in a separate uv environment.
- Do not add code comments, tests, or unrelated features without a request.
- Keep technical instructions short and direct. Update affected docs with code changes.
- Preserve the extension permissions, `systemone_settings`, `systemone_history`, and `/v1/systemone` contract.
- Keep the product question in `extension/background.js` identical to `finetune/common.py`.
- Keep display thresholds at `0.35` and `0.10` in the worker, popup, and feed scanner.
- Preserve provenance and source groups. Keep explicit source splits. Never import upstream test files or reuse the historical final test for generation, development, selection, or scoring.
- The separate `clefsift-general-v1` workflow may construct its new final holdout under a frozen recipe with explicit `--construct-final` approval. Do not inspect that prose or use it to revise prompts or acceptance rules. Default dataset validation and training must never open or hash its final-holdout file. Final scoring requires separate approval after selection.
- Treat supplied authorship labels as provisional unless evidence establishes otherwise. Never use commercial detector outputs as training labels.
- Use Luna models only for example generation, never coding work.
- Select checkpoints on validation before held-out scoring. Legacy matched/held-out panels share groups and controls; do not pool them. The general corpus has disjoint train/validation/final-holdout groups, exact class/domain/parent quotas and historical planned-route strata, and a separate epoch-sized recipe. Never fall back to the historical 128-step recipe for that corpus.
- Keep the trained decision head in FP32 during export. Treat probability drift as diagnostic, not an automatic export blocker.
- Ask before generation, GPU jobs, large downloads, or other long jobs. Agree on numeric resource limits first.
- Do not delete datasets, checkpoints, logs, reports, or model intermediates. Preserve at least 30 GiB of free disk for training.
- Keep corpus text, generated examples, credentials, environments, and weights out of Git. Never add proprietary extension code or assets.

The corpus control path is now one config-driven generation runner in `control_corpus.py`; `build_corpus.py` is removed. `tools/corpus` uses a non-virtual Python 3.11 selected through uv, without downloads or installs. It does not use the legacy training environment. Default config is `./corpus.json`; explicit `--config` and `CLEFSIFT_CORPUS_CONFIG` work, with paths relative to the config. `concurrency` is the sole model/worker authority. Legacy approval, hash, import, model, and interpreter config fields are ignored. No approval file or build-contract check may be reintroduced. Each run executes a source/config snapshot; hashes are audit metadata. Keep PID birth/owner/session checks and duplicate-run locks. `stop` returns immediately; `--wait` waits visibly. Status uses the terminal summary, not stale sampled counters. Existing clocks and cumulative attempt counts survive restarts; final construction requires explicit `--construct-final`. Preparation and assembly remain separate; this CLI never trains.

Normal generation resume reads small request/result/dispatch records and counts attempts. Do not restore repeated transcript hashing or acceptance checks on terminal results. Interrupted responses without a result use deep recovery; assembly keeps full verification. There is no new database or cache.

## Verification

Run `node --check` on changed JavaScript files.
Run Python syntax checks and CLI help checks without model loading.
Use `tools/mock_server.py` for interface checks.
Do not run training or inference as a routine syntax check.

## Release state

The initial source-only release includes the extension, portable training tools, pinned configuration, and recorded results.
No public model or dataset release is included. See [results](docs/results.md) for open evaluation limits.

The [general corpus workflow](docs/general-corpus.md) includes human-only preparation, Pi generation, assembly, and validation-only checkpoint selection. All configured routes draw remaining tasks from a shared queue. Per-attempt `dispatch.json` binds the actual route to the immutable task and plan. Store and validate actual provider/model identities, preserve old outputs, and retain source/domain/class/split/parent quotas. Actual generator balance is throughput-driven; planned-route quotas are historical parent strata. Final scoring needs separate approval.

The explicit `finetune/pi-generation.ts` request hook registers no tools. It records native output-schema options; Qwen uses reasoning and no word-count regex because the local grammar compiler rejected that pattern. Luna uses the existing word-range constraint in its schema. Discovered extensions remain disabled. Do not change acceptance checks or overwrite old attempts to make generation pass.

Isolated timeouts, completion-output limits, and retryable transport failures (including Codex WebSocket closed 1000) retry within the existing two-attempt bound without cancelling other workers. Empty responses also retry within two attempts. Repeated provider failures and authentication/configuration failures disable only the affected route; active calls finish, and remaining retries go to healthy routes. Invalid-content rejection rates never stop the pool. Integrity, malformed protocol, budget, and resource failures still stop the run. Runtime-only patches can use the explicit frozen-recipe compatibility registry only when the data semantics fingerprint matches; preserve all old artifacts and record the actual runtime code hash.

Full Pi routes and positive worker counts come only from config. Preserve historical approvals as local artifacts, but never read them as startup gates. New runs specify their clock duration in config; cumulative attempts default to the plan limit and can be lowered with `max_attempts`. Existing clocks and cumulative attempt counts must survive restarts.
