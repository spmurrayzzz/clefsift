# General-purpose corpus workflow

This workflow starts from human-only source records. It is separate from the historical LLMTrace preparation, API generation, and paired validation workflow.

The workflow covers acquisition, raw-source import, preparation, Pi generation, and assembly.

Set download, preparation, generation, and disk limits for each run. Model routes and worker counts come from the run config; there are no approval-file or per-route-cap checks.

## Dataset contract

| Split | Human parent groups | Rows per class | Total rows |
|---|---:|---:|---:|
| Training | 20,000 | 20,000 | 80,000 |
| Validation | 2,000 | 2,000 | 8,000 |
| Final holdout | 2,000 | 2,000 | 8,000 |

Each accepted group contains its human original, one copy edit, one mixed insertion, and one AI rewrite or seed-conditioned draft. All configured full provider/model routes draw remaining tasks from a shared queue. Variants within a group can use different generators.

The immutable plan retains its original Qwen/Luna assignments as historical parent strata. Planned-route quotas do not specify actual generator balance. The actual model mix depends on throughput, with no guaranteed half-Qwen/half-Luna split. Source, domain, class, split, and parent quotas remain unchanged. The final holdout measures new source groups, not an unseen generator family.

The existing classification question, reference revision, 1,536-token request cap, and display thresholds remain unchanged. This workflow currently accepts human excerpts of 50–500 words. It does not establish performance on shorter inputs or full long documents.

## Source selection

[general-quotas.json](../finetune/general-quotas.json) proposes these human training counts, with 10% as many in each evaluation split:

| Domain | Training groups | Candidate source |
|---|---:|---|
| News | 3,000 | CNN/DailyMail, 2007–2015 |
| Reviews | 4,000 | SNAP Amazon reviews through March 2013 |
| Q&A and advice | 5,000 | December 2018 Stack Exchange archive |
| Personal writing | 2,000 | Selected pre-2019 FineWeb pages |
| Professional writing | 2,000 | Selected pre-2019 FineWeb pages |
| Technical explanation | 2,000 | Selected pre-2019 FineWeb pages |
| Discussion | 2,000 | Selected pre-2019 FineWeb pages |

[general-sources.json](../finetune/general-sources.json) records candidate files and published hashes where available. Download only approved, named files. Never download an entire FineWeb crawl for this selection. Do not use its mixed-year sample configurations.

The historical dates support provisional human labels. They do not establish individual authorship. Dataset licenses do not automatically clear the underlying texts. The operator approved exploratory use with these limitations retained in each source record. That approval does not establish reuse rights. Do not publish raw text or generation transcripts.

`acquire.py` downloads only catalog files, verifies published hashes, records local SHA-256 values, and preserves failed attempts. Its cumulative byte ledger and deadline survive resume. `import_sources.py` reads the approved Parquet, 7z/XML, and compressed review files. It uses the pinned readers in [requirements-ingest.txt](../finetune/requirements-ingest.txt), installed in a separate uv environment. The training environment is unchanged.

The importer checks source dates and uses a recorded domain rule set. It selects one contiguous sentence excerpt per source group and keeps identified authors, threads, URLs, and duplicate texts together. Domain and language checks are heuristic. Invalid individual records are counted and excluded; invalid schemas, hashes, archive paths, or resource limits stop the import. No detector or LLM selects human examples.

Keep document, thread, author, canonical-URL, and known duplicate clusters together. Qualify identities with their source namespace. Omit missing author IDs rather than treating all deleted authors as one person. Excerpts from one document must share a group. The preparer retains only one eligible original per group.

Preserve official source assignments. An upstream `train` record can only enter training. An upstream `valid` record can enter validation or the new final holdout, with that allocation fixed before generation. Previously unsplit archives can supply any split. Upstream test files and the existing LLMTrace final test remain excluded.

## Acquire and import

Keep raw files, failed attempts, extracted XML, the import index, and logs outside Git. Set a shared preparation deadline before starting both commands:

```bash
uv venv --python 3.11 "$WORK/ingest-venv"
uv pip install --python "$WORK/ingest-venv/bin/python" -r finetune/requirements-ingest.txt
uv run --no-project --python "$WORK/ingest-venv/bin/python" python finetune/acquire.py \
  --catalog finetune/general-sources.json --out-dir "$WORK/acquired" \
  --max-bytes 10737418240 --minutes 120 --deadline-utc "$PREP_DEADLINE"
uv run --no-project --python "$WORK/ingest-venv/bin/python" python finetune/import_sources.py \
  --acquired-dir "$WORK/acquired" --catalog finetune/general-sources.json \
  --quotas finetune/general-quotas.json --out-dir "$WORK/import-v1" \
  --minutes 120 --deadline-utc "$PREP_DEADLINE" --candidate-factor 2
```

These commands require a fresh approval for another experiment. `WORK` is a dedicated artifact directory; `PREP_DEADLINE` is an absolute UTC deadline. Use a new import output directory for a corrected importer. Preserve the failed import and its report. The candidate pool is larger than the final parent selection to allow token and duplicate checks. It does not increase the approved generation parent limit.

## Prepare normalized source records

Supply JSONL with these fields:

```json
{
  "id": "archive:document-1:excerpt-1",
  "text": "<original human excerpt of 50–500 words>",
  "label": "human",
  "group": "archive:document-1",
  "document_id": "archive:document-1",
  "author_id": "archive:author-1",
  "split": "train",
  "source": "<source dataset or archive>",
  "source_split": "unsplit",
  "source_revision": "<pinned revision or archive checksum>",
  "archive_date": "2018-12-01",
  "domain": "personal",
  "lang": "eng",
  "label_status": "historical-source-provisional",
  "authorship_status": "not-independently-verified",
  "rights_status": "<documented license or permission>"
}
```

`split` is `train`, `validation`, or `final_holdout`. `source_split` is `train`, `valid`, or `unsplit`. Archive evidence must predate 2019. An old publication date on a current page is insufficient. Preserve additional provenance fields. Optional `brief`, `headline`, and `voice` fields must come from the source preparation process.

Use a new artifact directory on a filesystem with enough space. The following paths are examples; set them for the host:

```bash
cd /path/to/clefsift
export WORK=/path/to/corpus-work
export CLEFSIFT_MODEL=<local-pinned-reference-snapshot>
uv run --no-sync python finetune/corpus.py prepare --source "$WORK/source.jsonl" \
  --model "$CLEFSIFT_MODEL" --quotas finetune/general-quotas.json \
  --reserve-fraction 0.10 --out-dir "$WORK/parents"
python finetune/corpus.py plan --parents "$WORK/parents" \
  --out-dir "$WORK/generation"
```

Preparation checks source groups, supplied identity clusters, exact duplicates, and near-duplicate candidates. The near-duplicate screen uses minimum-hash candidates and five-word-shingle Jaccard similarity. It is a heuristic, not proof of independence. Its report records skipped saturated candidate buckets.

No model weights load during token counting. Complete classifier requests are not truncated. Source import explicitly records contiguous excerpt offsets and hashes of the full normalized source. All new classes decode entities, collapse whitespace, and remove spaces before punctuation. This removes tokenized punctuation present in the news dataset without changing the legacy workflow. Source import also removes initial CNN publisher prefixes.

Reserves default to zero. An explicit 10% reserve permits up to 26,400 parents and 79,200 generated outputs, including all splits. At most two attempts per task gives a maximum of 158,400 Pi generation invocations. Actual counts appear in the plan. Reserves stay within their split and domain. Assembly will report a shortfall instead of changing quotas or shrinking the dataset.

## Bounded Pi generation

Agree on numeric call, wall-time, concurrency, and disk bounds first. Keep the local inference service available. Check both the artifact filesystem and root, including cache and temporary-file locations.

```bash
python finetune/corpus.py run --plan-dir "$WORK/generation" \
  --split development --max-calls <approved-call-limit> \
  --minutes <approved-wall-minutes> --workers 4 --timeout 120
```

Each request runs in a new Pi process and an owned attempt directory. Tools, discovered extensions, skills, context files, prompt templates, themes, and persisted sessions are disabled. An empty owned append-prompt file suppresses the user append prompt. Agent and provider retries and compaction are disabled through owned project settings. Global Pi settings and credentials are not copied or edited by the adapter.

`CLEFSIFT_STRUCTURED_GENERATION=1` loads only the explicit [request-options extension](../finetune/pi-generation.ts). It registers no tools. It requests native JSON-schema output and records the options in each attempt. Luna also receives a word-range schema that restates the existing prompt constraint. Qwen uses a JSON schema without the word-range regex and enables reasoning for constraint adherence. Source text, base prompts, labels, and acceptance checks remain unchanged.

An unauthenticated local endpoint can use `CLEFSIFT_LOCAL_UNAUTHENTICATED=1`. This supplies the nonsecret `local` placeholder through Pi's per-call API-key option so Pi can select the model. Do not set it for an authenticated local endpoint. It does not modify `models.json` or store a credential.

The base client requests thinking off. The scoped request-options extension explicitly enables Qwen reasoning because its no-thinking outputs often failed the existing length checks. Luna keeps the off request. Any returned thinking is excluded from the generated text. Raw JSONL, usage, selected route, optional concrete response model, Pi version, prompts, and configuration remain in the attempt artifacts. Pi-reported identity is not independent attestation of a local backend's weights.

The client requires one complete assistant response and a settled run. It rejects tool calls, automatic retries, compaction, malformed events, incomplete output, and unexpected identities. Per-process logs are capped at 8 MiB each. Timeout and cancellation terminate only owned Pi process groups, with a five-second shutdown grace before forced termination.

An isolated empty response, request timeout, completion-output limit, or retryable transport failure consumes that task's attempt and permits one retry after a two-second delay. This includes a Codex WebSocket closure with code 1000. Other workers continue. A second failure exhausts the task; assembly can use a matching reserve group. The 120-second request bound and two-attempt limit remain unchanged.

Three consecutive retryable failures from one route disable only that route for the current run. Provider authentication, rate-limit, and configuration failures also disable only their route. Active calls finish and retain valid outputs; pending retries return to the shared queue for healthy workers without resetting attempt counts. Status reports disabled routes and error details. If all routes are disabled with work remaining, the run stops. Integrity, malformed protocol, cancellation, disk, deadline, and attempt-budget failures remain global stops. Invalid generated JSON or content gets at most one retry and remains excluded if rejected again; its rejection rate never cancels other requests.

Each attempt's `dispatch.json` binds its actual route to the immutable original task and plan. The actual provider/model is stored and validated. Old outputs retain their original identities and remain available for reuse.

Reservations consume an attempt even after interruption. Resume verifies prior artifacts, reuses accepted results, and recovers a saved complete response without another model call. It never resets attempt limits.

The dataset loader retains the existing [recipe compatibility registry](../finetune/general-recipe-compatibility.json) for source normalization, grouping, prompts, labels, and acceptance rules. The corpus wrapper does not compare code against a build contract. Each launch records and executes a source snapshot, so later checkout edits do not alter an active run. Original manifests, plans, outputs, and clocks are not rewritten.

AI-assisted outputs receive length, number-retention, and lexical checks. Mixed outputs insert a generated paragraph at a recorded sentence boundary while preserving the human substrings. AI outputs alternate complete rewrites and fresh drafts conditioned on a supplied brief, headline, or short extract. These are process labels. Checks do not prove semantic fidelity or authorship.

## Construct the new final holdout

Freeze the recipe before final construction. Do not inspect final prose or use its generation failures to revise prompts or acceptance rules. Construction uses the same rules as development and does not score a detector.

```bash
python finetune/corpus.py run --plan-dir "$WORK/generation" \
  --split final_holdout --construct-final \
  --max-calls <approved-final-call-limit> --minutes <approved-wall-minutes> \
  --workers 4 --timeout 120
uv run --no-sync python finetune/corpus.py assemble --parents "$WORK/parents" \
  --generation "$WORK/generation" --model "$CLEFSIFT_MODEL" \
  --out-dir "$WORK/frozen" --construct-final
```

Assembly requires complete four-class groups. It rechecks response identities, artifact hashes, full-request tokens, duplicates, and exact domain and historical planned-route parent quotas. A failed group can only use a matching reserve. A shortfall creates an audit failure, not a frozen manifest.

Sealing is a protocol and hash boundary, not filesystem access control. The final parents, requests, responses, and `final-holdout.jsonl` remain accessible to the file owner. Keep them outside development inspections. The completed holdout below was later published; treat it as public but excluded from training, selection, and prompt or acceptance revision.

## Assembly from existing development outputs

`assemble --from-development` treats exact size/domain quotas as advisory. It retains all valid complete four-class development groups and assigns about 10% of training provenance components to holdout by an ID hash, before output quality checks. All variants and connected source identities stay together; validation is unchanged. This mode does not generate examples or load the original final-parent panel.

The manifest records `selection_policy: all-valid-development-groups-v1`, actual groups and domain/model contributions, historical targets, and the holdout selection method. Default validation does not open or hash the final corpus. Final scoring requires separate approval; the public holdout stays out of training and selection.

## Public text/label export

`export` validates a frozen dataset then writes pooled `{text, label}` JSONL: train rows, validation rows, then holdout rows, each in frozen file order. Lines use compact UTF-8 JSON with the `text` key first. Groups, splits, and provenance are dropped, so the export is not a training input. Holdout rows are included, which requires explicit `--include-final`:

```bash
clefsift corpus export --frozen "$WORK/frozen" --out "$WORK/clefsift-cleaned.jsonl" --include-final
```

This command reproduces [spmurrayzzz/clefsift-cleaned](https://huggingface.co/datasets/spmurrayzzz/clefsift-cleaned) byte for byte from its frozen source dataset.

## Start, stop, and status

`tools/corpus` now controls generation only. It calls `corpus.run` directly instead of chaining through `build_corpus.py`. Preparation and assembly remain separate commands above; neither training nor final-holdout generation starts automatically.

The wrapper requires Linux, uv, an installed Python 3.11, and Pi. It selects a non-virtual Python through uv without downloads or package installs. Generation uses the standard library and Pi, not the old training environment. Preparation and assembly use Transformers and the local reference encoder for token counting. They do not import Unsloth or load model weights.

Use one JSON config. The default is `./corpus.json`, or set `CLEFSIFT_CORPUS_CONFIG`. `--config` accepts an explicit file before or after the command. Relative paths resolve from that file's directory:

```json
{
  "work_dir": "./corpus-work",
  "concurrency": {
    "openai-codex/gpt-6-luna": 16,
    "meta/muse-spark-1.3-contributor": 8
  },
  "generation_hours": 72,
  "request_timeout": 120
}
```

`work_dir` must contain the prepared `generation/plan.json` and its parent artifacts. `pi_bin_dir` is optional when node and Pi are already on PATH. Set it to their directory otherwise. Only `work_dir` and `concurrency` are required for a run with an existing clock. `generation_hours` sets the first clock; restarting never resets it. An optional `deadline_utc` can shorten, but cannot extend, that clock.

`max_attempts` is an optional cumulative limit across all launches. Its default is the saved plan's limit. The lower of these two limits applies. Each task still has at most two attempts. `request_timeout` defaults to 120 seconds and accepts values up to 600.

```bash
./tools/corpus start --config /path/to/corpus.json
./tools/corpus status --config /path/to/corpus.json
./tools/corpus stop --config /path/to/corpus.json
```

`start` detaches and resumes unfinished development tasks. All configured full provider/model routes share the queue. Add or remove routes, or change positive integer worker counts, in this config only. There are no per-route approval caps. Models must be available and authenticated in Pi.

`stop` returns immediately after requesting a drain. No new requests start; active requests finish. Use `stop --wait` for visible waiting, or `--wait-seconds N` for a bounded wait. Ctrl-C stops waiting without cancelling the job. `--force` sends SIGTERM only to verified controller processes; it does not signal an inference or training server. Wait for process exit before editing the config and starting again.

`status` and `stop` do not require working model settings or Pi executables. Status shows the launched and configured workers, task-loading progress, disabled routes, stop reasons, and final summary counts. `completed-with-shortfalls` means all tasks were attempted but some outputs failed. It does not claim a complete assembled dataset. Add `--json` for machine-readable output.

Each launch saves its resolved config, source snapshot, and source hashes for provenance. It executes that snapshot and maintains one active-process record. Code edits no longer require a build-contract update. Duplicate-run locks, process ownership checks, artifact checks, and the original clock remain in effect.

Existing `generation-control.json` files remain readable with an explicit `--config`. Legacy interpreter, import, model, and concurrency-approval fields are ignored by generation. No old approval file or environment is required.

`start --construct-final` explicitly selects final-holdout construction instead of development. Use it only under an agreed final-construction protocol. Default generation never reads final-parent prose.

The old build controller is removed. This is the first control-path cleanup, not a migration of the preparation, assembly, or training interfaces.

## Training handoff

Do not stop existing inference services without approval. Verify that the target GPU is idle before baseline evaluation or training.

Use the approved checkout and the [uv training setup](training.md#setup). Keep datasets, checkpoints, and logs on the selected artifact filesystem. Preserve previous runs and at least 30 GiB free space.

Set `model.path`, `data.directory`, `output_dir`, `minutes`, and `train.num_train_epochs` in a local copy of the [recipe](../finetune/recipe-general.yaml). Paths resolve from that file. Then run:

```bash
uv run --no-sync tools/clefsift train --config /path/to/recipe.yaml
```

The general workflow requires explicit epochs. Its recipe saves every 5,000 steps. With 80,000 rows and a global batch of four, one epoch uses 20,000 optimizer steps. The final step is saved and evaluated even between save intervals. The historical 128-step recipe is never a fallback. See [multi-GPU training](training.md#multiple-gpus) for standard DDP launching and batch-size accounting.

Unsloth trains BF16 LoRA adapters and the FP32 decision head, then selects on validation. Selection uses fixed-four macro F1, then Brier score, then the earlier step. `selection.json` binds the selected checkpoint files, dataset manifest, configuration, and predictions. The runner exports the selected model and ends with `selected-final-sealed`; it does not claim final evaluation is complete.

Default validation never opens or hashes the final-holdout file. It checks its text-free manifest group/hash inventory for overlap. Actual final-file integrity and scoring require a separately authorized step after selection. No automatic final-scoring command is provided yet.

No new accuracy or false-positive claim follows from successful pipeline checks.
