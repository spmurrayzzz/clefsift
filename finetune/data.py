from collections import Counter, namedtuple
from contextlib import closing, ExitStack
import hashlib
import json
import math
from pathlib import Path
import random
import shutil

from common import LABELS, QUESTION, normalize, provenance_hashes, sha256, write_json


_Row = namedtuple("_Row", "row row_index line_index text_key groups split encoded")
SPLITS = ("train", "validation", "test")
FILES = {split: f"{split}.jsonl" for split in SPLITS}
ALIASES = {alias: split for split, aliases in (
    ("train", ("train", "training")),
    ("validation", ("eval", "valid", "validation")),
    ("test", ("test", "holdout", "final_holdout", "final-holdout")),
) for alias in aliases}


def validate_splits(ratios, split_rows, seed):
    if ratios is not None and split_rows is not None:
        raise ValueError("ratios and split_rows are mutually exclusive")
    if ratios is not None:
        if (not isinstance(ratios, (tuple, list)) or len(ratios) != 3
                or any(type(value) not in (int, float) for value in ratios)):
            raise ValueError("ratios must contain three finite numbers")
        try:
            valid = (all(math.isfinite(value) for value in ratios)
                     and ratios[0] > 0 and ratios[1] > 0 and ratios[2] >= 0
                     and math.isclose(math.fsum(ratios), 1.0, rel_tol=0, abs_tol=1e-12))
        except (OverflowError, ValueError):
            valid = False
        if not valid:
            raise ValueError("ratios must sum to one with positive train/validation and nonnegative test")
        ratios = tuple(float(value) for value in ratios)
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2 ** 32):
        raise ValueError("seed must be a nonnegative 32-bit integer")
    if ratios is not None and seed is None:
        raise ValueError("Random splitting requires an explicit nonnegative 32-bit integer seed")
    if split_rows is not None:
        if (not isinstance(split_rows, (tuple, list)) or len(split_rows) != 3
                or any(type(value) is not int for value in split_rows)
                or split_rows[0] <= 0 or split_rows[1] <= 0 or split_rows[2] < 0):
            raise ValueError("split_rows requires positive train/validation counts and a nonnegative test count")
        split_rows = tuple(split_rows)
    return ratios, split_rows


def _sources(source):
    source = Path(source).resolve()
    if source.is_dir():
        if (source / "manifest.json").exists():
            raise ValueError("Native manifest directories must be handled by the caller")
        candidates = sorted(source.glob("*.jsonl"))
        names = {path.name: path for path in candidates}
        validation = [name for name in ("validation.jsonl", "eval.jsonl") if name in names]
        tests = [name for name in ("test.jsonl", "holdout.jsonl", "final-holdout.jsonl", "final_holdout.jsonl") if name in names]
        if "train.jsonl" in names and len(validation) == 1 and len(tests) <= 1:
            expected = {"train.jsonl", validation[0], *tests}
            if set(names) == expected and all(path.is_file() for path in candidates):
                paths = {"train": names["train.jsonl"], "validation": names[validation[0]]}
                if tests:
                    paths["test"] = names[tests[0]]
                return paths, True
        if len(candidates) != 1:
            raise ValueError("Ambiguous source directory: provide one raw JSONL or explicit train/validation files")
        source = candidates[0]
    if not source.is_file() or source.suffix != ".jsonl":
        raise ValueError("Source must be a JSONL file or an unambiguous JSONL directory")
    return {"raw": source}, False


def _row_keys(row, location):
    if not isinstance(row, dict):
        raise ValueError(f"Row must be a JSON object: {location}")
    text = row.get("text")
    if not isinstance(text, str) or not text.strip() or len(text) > 50000:
        raise ValueError(f"Text must be nonempty and at most 50000 characters: {location}")
    if not isinstance(row.get("label"), str) or row["label"] not in LABELS:
        raise ValueError(f"Invalid classification label: {location}")
    metadata = row.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"Metadata must be a JSON object: {location}")
    groups = provenance_hashes(row)
    return sha256(normalize(text).casefold()), set(groups)


def _assignment(row, location):
    containers = (row, row.get("metadata", {}))
    values = [container["split"] for container in containers if "split" in container]
    if any(not isinstance(value, str) or value not in ALIASES for value in values):
        raise ValueError(f"Invalid split assignment: {location}")
    splits = {ALIASES[value] for value in values}
    if len(splits) > 1:
        raise ValueError(f"Conflicting split assignments: {location}")
    allowed = set(SPLITS)
    restricted = False
    for container in containers:
        for key in ("source_split", "upstream_split"):
            if key not in container or container[key] == "unsplit":
                continue
            value = container[key]
            if not isinstance(value, str) or value not in ALIASES:
                raise ValueError(f"Invalid source split: {location}")
            source = ALIASES[value]
            allowed &= {"validation", "test"} if source == "validation" else {source}
            restricted = True
    assigned = next(iter(splits), None)
    if not allowed or assigned is not None and assigned not in allowed:
        raise ValueError(f"Assignment conflicts with the source split: {location}")
    if assigned is None and restricted:
        assigned = next(split for split in SPLITS if split in allowed)
    return assigned


def _rows(path, stats, check_resources=None, limit=None):
    digest = hashlib.sha256()
    row_index, line_index, byte_count = 0, 0, 0
    with ExitStack() as stack:
        stream = stack.enter_context(path.open("rb"))
        while limit is None or row_index < limit:
            if check_resources is not None and line_index % 256 == 0:
                check_resources()
            raw = stream.readline()
            if not raw:
                if limit is not None:
                    raise ValueError(f"Source ended after {row_index} records; {limit} required: {path}")
                break
            line_index += 1
            byte_count += len(raw)
            digest.update(raw)
            if not raw.strip():
                continue
            location = f"{path}:{line_index}"
            if raw.startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError("The source is a Git LFS pointer; fetch the JSONL file first")
            try:
                row = json.loads(raw.decode("utf-8"))
                encoded = (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
            except (UnicodeError, ValueError):
                raise ValueError(f"Invalid UTF-8 JSON record: {location}") from None
            text, groups = _row_keys(row, location)
            assigned = _assignment(row, location)
            row_index += 1
            yield _Row(row, row_index, line_index, text, groups, assigned, encoded)
    stats.update(path=str(path), read_rows=row_index, read_bytes=byte_count,
                 read_sha256=digest.hexdigest(), complete=limit is None)


def _check_overlap(owners, split, text, groups, location):
    for identity in [("text", text), *(("group", group) for group in groups)]:
        if identity in owners and owners[identity] != split:
            raise ValueError(f"Normalized text or supplied group overlaps splits: {location}")
        owners[identity] = split


def _random_partitions(rows, ratios, seed, check_resources):
    parents = list(range(len(rows)))
    identities = {}

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for index, item in enumerate(rows):
        if check_resources is not None and index % 256 == 0:
            check_resources()
        for identity in [("text", item.text_key), *(("group", group) for group in sorted(item.groups))]:
            if identity in identities:
                left, right = root(index), root(identities[identity])
                parents[max(left, right)] = min(left, right)
            else:
                identities[identity] = index
    components = {}
    for index in range(len(rows)):
        components.setdefault(root(index), []).append(index)
    active = [index for index, ratio in enumerate(ratios) if ratio > 0]
    rng = random.Random(seed)
    partitions = [[] for _ in SPLITS]
    grouped = any(len(group) > 1 for group in components.values())
    if not grouped:
        for label in LABELS:
            indices = [index for index, item in enumerate(rows) if item.row["label"] == label]
            if not indices:
                continue
            if len(indices) < len(active):
                raise ValueError(f"Too few unique rows for label {label!r} to populate every requested split")
            rng.shuffle(indices)
            targets = [len(indices) * ratio for ratio in ratios]
            counts = [max(1, math.floor(targets[index])) if index in active else 0 for index in range(3)]
            while sum(counts) > len(indices):
                index = max((index for index in active if counts[index] > 1), key=lambda index: counts[index] - targets[index])
                counts[index] -= 1
            while sum(counts) < len(indices):
                index = max(active, key=lambda index: targets[index] - counts[index])
                counts[index] += 1
            offset = 0
            for partition, count in zip(partitions, counts):
                partition.extend(indices[offset:offset + count])
                offset += count
    else:
        groups = list(components.values())
        if len(groups) < len(active):
            raise ValueError("Too few independent groups to populate every requested split")
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)
        targets = [len(rows) * ratio for ratio in ratios]
        for position, group in enumerate(groups):
            if check_resources is not None and position % 256 == 0:
                check_resources()
            empty = [index for index in active if not partitions[index]]
            candidates = empty if len(groups) - position == len(empty) else active
            index = max(candidates, key=lambda index: targets[index] - len(partitions[index]))
            partitions[index].extend(group)
    return {split: sorted(indices) for split, indices in zip(SPLITS, partitions)}, grouped


def prepare_data(source: Path, output: Path, *, ratios=None, split_rows=None, seed=None, check_resources=None) -> dict:
    ratios, split_rows = validate_splits(ratios, split_rows, seed)
    paths, explicit = _sources(source)
    if explicit and (ratios is not None or split_rows is not None):
        raise ValueError("Explicit split files cannot be repartitioned")
    output = Path(output).resolve()
    for name in (*FILES.values(), "eval.jsonl", "preparation.json"):
        path = output / name
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Output already exists: {path}")
    reserve = 3 * sum(path.stat().st_size for split, path in paths.items() if split != "test") + 65536
    if check_resources is not None:
        check_resources(reserve)
    else:
        ancestor = output
        while not ancestor.exists():
            ancestor = ancestor.parent
        if shutil.disk_usage(ancestor).free < reserve:
            raise RuntimeError("Insufficient disk space for prepared data")
    reads, owners, files = [], {}, {}
    counts = {split: Counter({label: 0 for label in LABELS}) for split in SPLITS}
    test = {"status": "absent"}
    grouped = None
    with ExitStack() as stack:
        streams, digests = {}, {}

        def write(split, path, item):
            if split not in streams:
                output.mkdir(parents=True, exist_ok=True)
                streams[split] = stack.enter_context((output / FILES[split]).open("xb"))
                digests[split] = hashlib.sha256()
                files[split] = {"path": FILES[split], "source": str(path), "source_rows": [], "source_lines": []}
            streams[split].write(item.encoded)
            digests[split].update(item.encoded)
            files[split]["source_rows"].append(item.row_index)
            files[split]["source_lines"].append(item.line_index)

        if explicit:
            mode = "explicit"
            for split in SPLITS[:2]:
                path, stats = paths[split], {}
                reads.append(stats)
                for item in _rows(path, stats, check_resources):
                    if item.split is not None and item.split != split:
                        raise ValueError(f"Row assignment conflicts with explicit file: {path}:{item.line_index}")
                    _check_overlap(owners, split, item.text_key, item.groups, f"{path}:{item.line_index}")
                    counts[split][item.row["label"]] += 1
                    write(split, path, item)
            if "test" in paths:
                test = {"status": "existing-unread", "path": str(paths["test"]), "verified": False}
        else:
            path, stats = paths["raw"], {}
            reads.append(stats)
            limit = sum(split_rows[:2]) if split_rows is not None else None
            reader = stack.enter_context(closing(_rows(path, stats, check_resources, limit)))
            first = next(reader, None)
            if first is None:
                raise ValueError("Source contains no records")
            assigned = first.split is not None
            if assigned and (ratios is not None or split_rows is not None):
                raise ValueError("Preassigned rows cannot be repartitioned")
            if not assigned and ratios is None and split_rows is None:
                raise ValueError("Unassigned rows require ratios with seed or explicit split_rows")
            mode = "preassigned" if assigned else "random" if ratios is not None else "ordered"
            rows = []
            item = first
            while item is not None:
                if (item.split is not None) != assigned:
                    raise ValueError(f"Mixed assigned/unassigned records: {path}:{item.line_index}")
                if mode == "random":
                    rows.append(item)
                else:
                    split = item.split if assigned else "train" if item.row_index <= split_rows[0] else "validation"
                    _check_overlap(owners, split, item.text_key, item.groups, f"{path}:{item.line_index}")
                    counts[split][item.row["label"]] += 1
                    if split != "test":
                        write(split, path, item)
                item = next(reader, None)
            if mode == "random":
                partitions, grouped = _random_partitions(rows, ratios, seed, check_resources)
                for split, indices in partitions.items():
                    for position, index in enumerate(indices):
                        if check_resources is not None and position % 256 == 0:
                            check_resources()
                        item = rows[index]
                        counts[split][item.row["label"]] += 1
                        write(split, path, item)
                if ratios[2] > 0:
                    test = {"status": "generated", "path": FILES["test"], "verified": True}
            elif mode == "ordered":
                test = {"status": "declared-unverified", "declared_rows": split_rows[2],
                        "verified": False, "remainder_not_read": True, "path": str(path)}
            elif counts["test"].total():
                test = {"status": "existing-tagged", "path": str(path), "copied": False, "verified": True}
        if any(not counts[split].total() for split in SPLITS[:2]):
            raise ValueError("Nonempty training and validation splits are required")
        for split, entry in files.items():
            entry.update(rows=counts[split].total(), labels=dict(counts[split]), sha256=digests[split].hexdigest())
    actual = {split: {"rows": counts[split].total(), "labels": dict(counts[split])} for split in SPLITS}
    if mode in ("ordered", "explicit"):
        actual["test"] = None
    report = {"mode": mode, "ratios": list(ratios) if ratios is not None else None,
              "split_rows": list(split_rows) if split_rows is not None else None, "seed": seed,
              "sources": {split: str(path) for split, path in paths.items()}, "reads": reads,
              "grouped": grouped, "actual": actual, "test": test,
              "files": [files[split] for split in SPLITS if split in files]}
    if check_resources is not None:
        check_resources()
    write_json(output / "preparation.json", report)
    return report


def _metadata_value(metadata, key, value):
    while key in metadata:
        key += "_"
    metadata[key] = value


def load_data(output: Path) -> dict:
    output = Path(output)
    report = json.loads((output / "preparation.json").read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("mode") not in ("ordered", "random", "preassigned", "explicit"):
        raise ValueError("Invalid preparation report")
    files = report.get("files")
    if (not isinstance(files, list) or any(not isinstance(item, dict) or item.get("path") not in FILES.values() for item in files)
            or len({item["path"] for item in files}) != len(files)):
        raise ValueError("Invalid prepared file entries")
    entries = {item["path"]: item for item in files}
    if not {FILES["train"], FILES["validation"]} <= entries.keys():
        raise ValueError("Prepared training and validation files are required")
    datasets, owners = {}, {}
    for split, key in (("train", "train.jsonl"), ("validation", "eval.jsonl")):
        name = FILES[split]
        entry = entries[name]
        count = entry.get("rows")
        source_rows, source_lines = entry.get("source_rows"), entry.get("source_lines")
        if (type(count) is not int or count <= 0 or not isinstance(entry.get("source"), str)
                or not isinstance(entry.get("labels"), dict) or set(entry["labels"]) != set(LABELS)
                or any(type(value) is not int or value < 0 for value in entry["labels"].values())
                or sum(entry["labels"].values()) != count
                or any(not isinstance(values, list) or len(values) != count
                       or any(type(value) is not int or value <= 0 for value in values)
                       for values in (source_rows, source_lines))):
            raise ValueError(f"Invalid preparation counts or source locators: {name}")
        records, stats = [], {}
        labels = Counter({label: 0 for label in LABELS})
        for item in _rows(output / name, stats):
            row, index, line, text, groups, assigned, _ = item
            if index > count:
                raise ValueError(f"Prepared file length mismatch: {name}")
            if assigned is not None and assigned != split:
                raise ValueError(f"Prepared row assignment mismatch: {name}:{line}")
            _check_overlap(owners, split, text, groups, f"{name}:{line}")
            metadata = dict(row.get("metadata", {}))
            if "group" in row:
                if "group" not in metadata:
                    metadata["group"] = row["group"]
                elif metadata["group"] != row["group"]:
                    _metadata_value(metadata, "source_group", row["group"])
            _metadata_value(metadata, "source_row_locator", {
                "path": entry["source"], "row": source_rows[index - 1],
                "line": source_lines[index - 1], "indexing": "nonblank-one-based",
            })
            records.append({"id": f"{Path(key).stem}:{index}", "state": row["text"],
                            "questions": {"classification": QUESTION}, "labels": {"classification": row["label"]},
                            "metadata": metadata})
            labels[row["label"]] += 1
        if stats["read_sha256"] != entry.get("sha256"):
            raise ValueError(f"Prepared file hash mismatch: {name}")
        if len(records) != count or dict(labels) != entry["labels"]:
            raise ValueError(f"Prepared file count mismatch: {name}")
        datasets[key] = records
    return datasets
