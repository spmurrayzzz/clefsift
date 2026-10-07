import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from common import MODEL_REVISION, QUESTION, TokenCounter, file_hash, normalize, read_jsonl, sha256, validate_records, write_json, write_jsonl


PARENT_LABELS = ("human", "ai", "mixed")


def import_llmtrace(directory):
    selection = json.loads(Path(__file__).with_name("llmtrace-selection.json").read_text())
    revision = selection["revision"]
    result = []
    for split, groups in selection["groups"].items():
        path = directory / f"{split}.jsonl"
        if file_hash(path) != selection["files"][split]["sha256"]:
            raise ValueError(f"Pinned corpus hash mismatch: {split}")
        needed = {line: (topic, label) for topic, *lines in groups for label, line in zip(PARENT_LABELS, lines)}
        with path.open() as source:
            for line, text in enumerate(source, 1):
                if line not in needed:
                    continue
                row = json.loads(text)
                topic, label = needed.pop(line)
                if row["topic_id"] != topic or row["label"] != label or row["lang"] != "eng":
                    raise ValueError(f"Source locator mismatch: {split}:{line}")
                result.append({"id": f"llmtrace:{revision[:12]}:{split}:{line}", "state": row["text"],
                               "questions": {"classification": QUESTION}, "labels": {"classification": label},
                               "metadata": {**{key: value for key, value in row.items() if key not in ("text", "label")},
                                            "group": f"llmtrace:{topic}", "source": selection["dataset"], "source_revision": revision,
                                            "upstream_split": split, "upstream_line": line,
                                            "label_status": "upstream-provisional", "authorship_status": "not-independently-verified",
                                            "rights_status": "not-independently-verified"}})
        if needed:
            raise ValueError(f"Missing selected corpus rows: {split}")
    return result


def import_source(path):
    result = []
    for example in read_jsonl(path):
        if not isinstance(example, dict) or example.get("split") not in ("train", "eval"):
            raise ValueError("Each source row requires split=train or split=eval")
        metadata = {key: value for key, value in example.items() if key not in ("id", "text", "label", "split")}
        metadata["upstream_split"] = "train" if example["split"] == "train" else "valid"
        result.append({"id": example.get("id"), "state": example.get("text"),
                       "questions": {"classification": QUESTION}, "labels": {"classification": example.get("label")},
                       "metadata": metadata})
    return result


def main():
    parser = argparse.ArgumentParser(description="Prepare English parent groups without changing source splits.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", type=Path, help="Source JSONL with explicit train/eval splits and provenance.")
    source.add_argument("--llmtrace", type=Path, help="Directory with pinned train.jsonl and valid.jsonl, acquired separately.")
    parser.add_argument("--model", type=Path, required=True, help="Local pinned Clef snapshot for complete-request token counts.")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists() or args.out_dir.is_symlink():
        parser.error("Output directory already exists")
    records = import_llmtrace(args.llmtrace) if args.llmtrace else import_source(args.source)
    validate_records(records)
    groups = defaultdict(list)
    text_groups = {}
    for row in records:
        group = row["metadata"]["group"]
        text = normalize(row["state"]).casefold()
        if text in text_groups and text_groups[text] != group:
            raise ValueError("Duplicate text across source groups")
        text_groups[text] = group
        groups[group].append(row)
    counter = TokenCounter(args.model)
    excluded = []
    panels = {"train.jsonl": [], "eval.jsonl": []}
    for group, members in groups.items():
        if Counter(row["labels"]["classification"] for row in members) != Counter(PARENT_LABELS):
            raise ValueError(f"One human, AI, and mixed parent per group is required: {group}")
        splits = {row["metadata"]["upstream_split"] for row in members}
        if len(splits) != 1:
            raise ValueError(f"Source group crosses splits: {group}")
        if len({normalize(row["state"]).casefold() for row in members}) != 3:
            raise ValueError(f"Duplicate parent texts in group: {group}")
        for row in members:
            row["metadata"]["input_tokens"] = counter(row)
        if any(row["metadata"]["input_tokens"] > 1536 for row in members):
            excluded.append({"group": group, "reason": "Complete parent request exceeds 1536 tokens"})
            continue
        name = "train.jsonl" if splits == {"train"} else "eval.jsonl"
        panels[name].extend(members)
    if any(not rows for rows in panels.values()):
        raise ValueError("Both source splits require complete parent groups")
    args.out_dir.mkdir(parents=True)
    files = []
    for name, rows in panels.items():
        rows.sort(key=lambda row: sha256(f"llmtrace-en-pilot-v1:{row['id']}"))
        path = args.out_dir / name
        write_jsonl(path, rows)
        files.append({"path": name, "sha256": file_hash(path), "rows": len(rows),
                      "groups": len(rows) // 3, "labels": dict(Counter(row["labels"]["classification"] for row in rows))})
    write_json(args.out_dir / "manifest.json", {
        "stage": "parents", "created_at_utc": datetime.now(timezone.utc).isoformat(), "question": QUESTION,
        "model_revision": MODEL_REVISION, "max_input_tokens": 1536, "truncation": False, "files": files,
        "source_sha256": file_hash(Path(__file__).with_name("llmtrace-selection.json") if args.llmtrace else args.source),
        "excluded_groups": excluded, "final_test": "Not imported or used",
        "label_status": "Supplied labels retain their source provenance; authorship is not independently verified by this tool",
    })
    print(json.dumps({"files": files, "excluded_groups": excluded}, indent=2))


if __name__ == "__main__":
    main()
