import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from common import LABELS, MODEL_REVISION, QUESTION, TokenCounter, file_hash, normalize, sha256, validate_records, write_json, write_jsonl
from generate import SYSTEM, load_json, load_parents, load_plan, object_hash, prompt_for, rank, utc_now, validate, validate_result, verify_plan_parents


MAX_INPUT_TOKENS = 1536
PREPROCESSING = "Decode HTML entities and literal escaped newlines/tabs; collapse whitespace for every class. Original text and metadata remain in the parent dataset."


def inherited(record, source_file, source_hash):
    metadata = {**record["metadata"], "upstream_metadata": record["metadata"], "parent_id": record["id"],
                "parent_text_sha256": sha256(record["state"]), "parent_record_sha256": object_hash(record),
                "parent_source_file": source_file, "parent_source_file_sha256": source_hash, "preprocessing": PREPROCESSING}
    metadata.pop("input_tokens", None)
    metadata.pop("ai_char_intervals", None)
    return {**record, "id": record["id"] + ":fourway", "state": normalize(record["state"]), "metadata": metadata}


def generated(task, result, result_hash, plan_hash, label):
    value, checks = validate(task, result["output"])
    original = task["original_text"]
    metadata = {**task["source_metadata"], "source": "controlled-api-derivative",
                "label_status": "controlled-construction-with-provisional-human-parent",
                "upstream_metadata": task["source_metadata"], "parent_id": task["parent_id"],
                "parent_text_sha256": task["original_sha256"], "parent_record_sha256": task["parent_record_sha256"],
                "parent_source_file": task["source_file"], "parent_source_file_sha256": task["source_file_sha256"],
                "provider": task["provider"], "base_url": task["base_url"], "model": task["model"],
                "requested_model": task["model"], "response_model": result["response_model"], "generator_role": task["role"],
                "task_id": task["task_id"], "generation_result": f'tasks/{task["task_id"]}/result.json',
                "generation_result_sha256": result_hash, "generation_plan_sha256": plan_hash,
                "system_sha256": sha256(SYSTEM), "prompt_sha256": sha256(prompt_for(task)), "checks": checks,
                "preprocessing": PREPROCESSING, "sampling": "OpenAI-compatible API defaults; no sampling seed set"}
    metadata.pop("input_tokens", None)
    metadata.pop("ai_char_intervals", None)
    if label == "ai-assisted":
        text = normalize(value["polished"])
        metadata["construction"] = "AI copy edit of the full human source; semantic fidelity not independently verified"
    elif label == "mixed" and task["generate_mixed"]:
        offset = task["insertion_offset"]
        left, right = normalize(original[:offset]), normalize(original[offset:])
        addition = normalize(value["addition"])
        if left + " " + right != normalize(original):
            raise ValueError("Insertion boundary changes the normalized human source")
        text = left + " " + addition + " " + right
        start = len(left) + 1
        metadata.update(construction="One generated paragraph inserted between two unchanged, normalized human substrings",
                        ai_char_intervals=[[start, start + len(addition)]], insertion_offset_in_raw_parent=offset,
                        ai_character_fraction=len(addition) / len(text))
    else:
        raise ValueError("Unsupported generated label")
    return {"id": f'controlled-api:{task["task_id"]}:{label}', "state": text, "questions": {"classification": QUESTION},
            "labels": {"classification": label}, "metadata": metadata}


def count_tokens(counter, record):
    count = counter(record)
    if type(count) is not int or count < 1:
        raise ValueError("Invalid complete request token count")
    return count


def panel_groups(records):
    groups = defaultdict(dict)
    for record in records:
        group, label = record["metadata"]["group"], record["labels"]["classification"]
        if label in groups[group]:
            raise ValueError("Duplicate panel label within a source group")
        groups[group][label] = record
    if not groups or any(set(members) != set(LABELS) for members in groups.values()):
        raise ValueError("Every panel needs nonempty, balanced four-label groups")
    return groups


def assemble(parents_dir, generation_dir, model_path, out_dir):
    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError(f"Output already exists: {out_dir}")
    previous, parent_files = load_parents(parents_dir)
    plan = load_plan(generation_dir)
    verify_plan_parents(plan, parents_dir, parent_files)
    plan_hash = file_hash(generation_dir / "plan.json")
    source_hashes = plan["source_files"]
    counter = TokenCounter(model_path)
    regression = parent_files["eval.jsonl"]
    regression_lengths = {}
    for record in regression:
        count = count_tokens(counter, record)
        if count > MAX_INPUT_TOKENS:
            raise ValueError("Unchanged parent validation exceeds 1536 complete input tokens")
        if record["metadata"].get("input_tokens") != count:
            raise ValueError("Parent validation token count differs from the local model measurement")
        regression_lengths[record["id"]] = count
    regression_text_groups = defaultdict(set)
    for record in regression:
        regression_text_groups[normalize(record["state"]).casefold()].add(record["metadata"]["group"])
    if any(len(groups) > 1 for groups in regression_text_groups.values()):
        raise ValueError("Unchanged parent validation contains normalized cross-group duplicates")
    parents = defaultdict(dict)
    for records in parent_files.values():
        for record in records:
            parents[record["metadata"]["group"]][record["labels"]["classification"]] = record
    tasks = {(task["group"], task["role"]): task for task in plan["tasks"]}
    results, result_hashes, result_errors = {}, {}, {}
    for task in plan["tasks"]:
        path = generation_dir / "tasks" / task["task_id"] / "result.json"
        try:
            if not path.is_file():
                raise ValueError("No saved accepted generation")
            result_hashes[task["task_id"]] = file_hash(path)
            result = load_json(path)
            validate_result(task, result, plan_hash)
            results[task["task_id"]] = result
        except (OSError, ValueError, KeyError, TypeError) as error:
            result_errors[task["task_id"]] = str(error) if isinstance(error, (OSError, ValueError)) else "Invalid saved generation schema"
    holdout_models = {plan["holdout_model"]} | {result["response_model"] for result in results.values() if result["role"] == "holdout"}
    candidates = {"train.jsonl": [], "eval.jsonl": [], "holdout.jsonl": []}
    excluded = {}

    def exclude(group, reason):
        members = parents[group]
        human = members["human"]
        if group not in excluded:
            group_tasks = [task for task in plan["tasks"] if task["group"] == group]
            excluded[group] = {"group": group, "split": human["metadata"]["upstream_split"], "reasons": [],
                               "parents": {label: {"id": record["id"], "text_sha256": sha256(record["state"]),
                                                   "record_sha256": object_hash(record), "metadata": record["metadata"]}
                                           for label, record in members.items()},
                               "generation": [{"task_id": task["task_id"], "role": task["role"], "requested_model": task["model"],
                                               "provider": task["provider"], "base_url": task["base_url"],
                                               "prompt_sha256": task["prompt_sha256"],
                                               "result_sha256": result_hashes.get(task["task_id"]),
                                               "response_model": results.get(task["task_id"], {}).get("response_model"),
                                               "failure": result_errors.get(task["task_id"])} for task in group_tasks]}
        if reason not in excluded[group]["reasons"]:
            excluded[group]["reasons"].append(reason)
        excluded[group]["reason"] = "; ".join(excluded[group]["reasons"])

    for group, members in sorted(parents.items()):
        split = members["human"]["metadata"]["upstream_split"]
        roles = ("development",) if split == "train" else ("development", "holdout")
        prepared, reasons = {}, []
        for role in roles:
            task = tasks[group, role]
            if task["task_id"] in result_errors:
                reasons.append(f'{role}: {result_errors[task["task_id"]]}')
                continue
            try:
                result = results[task["task_id"]]
                records = [inherited(members[label], task["source_file"], source_hashes[task["source_file"]]) for label in ("human", "ai")]
                records.append(generated(task, result, result_hashes[task["task_id"]], plan_hash, "ai-assisted"))
                records.append(generated(task, result, result_hashes[task["task_id"]], plan_hash, "mixed") if task["generate_mixed"]
                               else inherited(members["mixed"], task["source_file"], source_hashes[task["source_file"]]))
                if len({normalize(record["state"]).casefold() for record in records}) != len(LABELS):
                    raise ValueError("Duplicate normalized text within a four-label group")
                if split == "train" and any(record["metadata"].get("generator_role") == "holdout" or
                                           any(record["metadata"].get(key) in holdout_models for key in ("model", "requested_model", "response_model"))
                                           for record in records):
                    raise ValueError("Holdout generator identity appears in training")
                for record in records:
                    count = count_tokens(counter, record)
                    record["metadata"]["input_tokens"] = count
                    if count > MAX_INPUT_TOKENS:
                        raise ValueError(f'{record["labels"]["classification"]} request exceeds 1536 complete input tokens')
                validate_records(records)
                prepared[role] = records
            except (ValueError, KeyError, TypeError) as error:
                reasons.append(f"{role}: {error}" if isinstance(error, ValueError) else f"{role}: Invalid generation or parent schema")
        if reasons:
            for reason in reasons:
                exclude(group, reason)
            continue
        candidates["train.jsonl" if split == "train" else "eval.jsonl"].extend(prepared["development"])
        if split == "valid":
            candidates["holdout.jsonl"].extend(prepared["holdout"])
    text_groups = defaultdict(set)
    for records in [*candidates.values(), regression]:
        for record in records:
            text_groups[normalize(record["state"]).casefold()].add(record["metadata"]["group"])
    duplicate_groups = {group for groups in text_groups.values() if len(groups) > 1 for group in groups}
    for group in sorted(duplicate_groups):
        exclude(group, "Duplicate normalized text across source groups")
    candidates = {filename: [record for record in records if record["metadata"]["group"] not in excluded]
                  for filename, records in candidates.items()}
    indexed = {filename: panel_groups(records) for filename, records in candidates.items()}
    train_groups, eval_groups, holdout_groups = (set(indexed[filename]) for filename in candidates)
    regression_groups = {record["metadata"]["group"] for record in regression}
    if train_groups & (eval_groups | holdout_groups | regression_groups) or eval_groups != holdout_groups:
        raise ValueError("Training separation or validation pairing mismatch")
    for group in eval_groups:
        for label in ("human", "ai"):
            if indexed["eval.jsonl"][group][label] != indexed["holdout.jsonl"][group][label]:
                raise ValueError("Paired validation controls differ")
    if any(record["metadata"].get("generator_role") == "holdout" or
           any(record["metadata"].get(key) in holdout_models for key in ("model", "requested_model", "response_model"))
           for record in candidates["train.jsonl"]):
        raise ValueError("Holdout generator leaked into training")
    for records in candidates.values():
        validate_records(records)
        records.sort(key=lambda record: rank(record["id"]))
    candidates["regression.jsonl"] = regression
    verify_plan_parents(plan, parents_dir, parent_files)
    if file_hash(generation_dir / "plan.json") != plan_hash:
        raise ValueError("Generation plan changed during assembly")
    for task_id, expected_hash in result_hashes.items():
        if file_hash(generation_dir / "tasks" / task_id / "result.json") != expected_hash:
            raise ValueError("Generation result changed during assembly")
    out_dir.mkdir(parents=True, exist_ok=False)
    files = []
    roles = {"train.jsonl": "train", "eval.jsonl": "matched_validation", "holdout.jsonl": "heldout_model", "regression.jsonl": "prior_validation"}
    for filename, records in candidates.items():
        path = out_dir / filename
        if filename == "regression.jsonl":
            with path.open("xb") as output:
                output.write((parents_dir / "eval.jsonl").read_bytes())
            if file_hash(path) != source_hashes["eval.jsonl"]:
                raise ValueError("Regression copy differs from parent validation")
            lengths = list(regression_lengths.values())
        else:
            write_jsonl(path, records)
            lengths = [record["metadata"]["input_tokens"] for record in records]
        files.append({"path": filename, "sha256": file_hash(path), "rows": len(records),
                      "groups": len({record["metadata"]["group"] for record in records}), "role": roles[filename],
                      "labels": dict(Counter(record["labels"]["classification"] for record in records)), "max_tokens": max(lengths)})
    manifest = {"stage": "frozen", "created_at_utc": utc_now(), "question": QUESTION, "model_revision": MODEL_REVISION,
                "model_path": str(model_path.resolve()), "max_input_tokens": MAX_INPUT_TOKENS, "truncation": False,
                "parent_directory": str(parents_dir.resolve()), "parent_manifest_sha256": file_hash(parents_dir / "manifest.json"),
                "parent_files": source_hashes, "parent_stage": previous["stage"], "generation_directory": str(generation_dir.resolve()),
                "generation_plan_sha256": plan_hash, "generation_results_sha256": result_hashes,
                "preparer_sha256": file_hash(Path(__file__)), "generation_validator_sha256": file_hash(Path(__file__).with_name("generate.py")),
                "development_model": plan["development_model"], "holdout_model": plan["holdout_model"],
                "files": files, "excluded_groups": [excluded[group] for group in sorted(excluded)], "preprocessing": PREPROCESSING,
                "label_status": "Upstream labels provisional; new labels describe controlled edits or insertions, not verified human authorship or rights",
                "heldout_scope": "Holdout generators are absent from training. Validation panels share source groups and identical human/AI controls. Distinct model IDs do not establish model-family generalization. Do not pool panels.",
                "selection": "Use matched validation for checkpoint selection; evaluate the heldout panel after selection"}
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps({"stage": manifest["stage"], "files": files, "excluded_groups": len(excluded)}, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Freeze balanced four-label panels from parents and accepted API derivatives.")
    parser.add_argument("--parents", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        assemble(args.parents, args.generation, args.model, args.out_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        if isinstance(error, (KeyError, TypeError)):
            parser.error("Invalid local dataset or generation artifact schema")
        parser.error(str(error))


if __name__ == "__main__":
    main()
