from collections import Counter
import hashlib
import html
import ipaddress
import json
from pathlib import Path
import urllib.parse
import urllib.request


LABELS = ("human", "ai-assisted", "mixed", "ai")
MODEL_REVISION = "2f3de3dd85f379784083b0814d997ab627200f0c"
QUESTION = {
    "type": "choice",
    "instructions": "Judge only the writing style, not the topic or quality. Human writing tends to show irregular grammar, typos, uneven rhythm, slang, humor, and lived personal detail. AI-generated writing tends to show flawless grammar, uniformly balanced sentences, generic abstraction, hedging, and formulaic transitions.",
    "criteria": {
        "human": "Irregular grammar or typos, colloquial voice, slang or humor, uneven sentence rhythm, concrete personal detail from lived experience",
        "ai-assisted": "Mostly uniform, polished, machine-like prose, but with some personal specifics or human irregularities breaking through",
        "mixed": "Contains some clearly human-written passages mixed with some clearly machine-generated passages",
        "ai": "Flawless uniform prose, balanced clause structures, generic abstract content, hedging language, formulaic transitions, no personal irregularities",
    },
}


def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def normalize(text):
    return " ".join(html.unescape(text).replace("\\n", " ").replace("\\r", " ").replace("\\t", " ").split())


def read_jsonl(path):
    with Path(path).open() as source:
        return [json.loads(line) for line in source if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def validate_records(records):
    if not records:
        raise ValueError("Empty dataset")
    seen = set()
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("Dataset rows must be objects")
        if not isinstance(row.get("id"), str) or not row["id"].strip() or row["id"] in seen:
            raise ValueError("Invalid or duplicate record ID")
        seen.add(row["id"])
        if not isinstance(row.get("state"), str) or not row["state"].strip() or len(row["state"]) > 50000:
            raise ValueError(f"Invalid text: {row['id']}")
        if row.get("questions") != {"classification": QUESTION}:
            raise ValueError(f"Question mismatch: {row['id']}")
        if not isinstance(row.get("labels"), dict) or set(row["labels"]) != {"classification"} or row["labels"]["classification"] not in LABELS:
            raise ValueError(f"Invalid label: {row['id']}")
        metadata = row.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("lang") != "eng":
            raise ValueError(f"English provenance is required: {row['id']}")
        for key in ("group", "source", "label_status", "authorship_status", "rights_status"):
            if not isinstance(metadata.get(key), str) or not metadata[key].strip():
                raise ValueError(f"Missing {key}: {row['id']}")
        if metadata.get("upstream_split") not in ("train", "valid"):
            raise ValueError(f"Unexpected source split: {row['id']}")


def validate_dataset(directory, balanced=False):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("question") != QUESTION or manifest.get("model_revision") != MODEL_REVISION:
        raise ValueError("Dataset question or model revision mismatch")
    if manifest.get("max_input_tokens") != 1536 or manifest.get("truncation") is not False:
        raise ValueError("Dataset requires full-request token checks without truncation")
    if manifest.get("stage") not in ("parents", "frozen"):
        raise ValueError("Dataset is not ready")
    datasets = {}
    for item in manifest["files"]:
        name = item["path"]
        if name not in ("train.jsonl", "eval.jsonl", "holdout.jsonl", "regression.jsonl") or name in datasets:
            raise ValueError("Unexpected dataset file")
        if file_hash(directory / name) != item["sha256"]:
            raise ValueError(f"Dataset hash mismatch: {name}")
        records = read_jsonl(directory / name)
        validate_records(records)
        if len(records) != item["rows"]:
            raise ValueError(f"Dataset count mismatch: {name}")
        for row in records:
            tokens = row["metadata"].get("input_tokens")
            if type(tokens) is not int or not 0 < tokens <= 1536:
                raise ValueError(f"Invalid token count: {row['id']}")
            expected_split = "train" if name == "train.jsonl" else "valid"
            if row["metadata"]["upstream_split"] != expected_split:
                raise ValueError(f"Source split mismatch: {row['id']}")
        datasets[name] = records
    if not {"train.jsonl", "eval.jsonl"} <= set(datasets):
        raise ValueError("Training and validation files are required")
    groups = {name: {row["metadata"]["group"] for row in records} for name, records in datasets.items()}
    if groups["train.jsonl"] & set().union(*(group for name, group in groups.items() if name != "train.jsonl")):
        raise ValueError("Training groups overlap validation groups")
    if "holdout.jsonl" in groups and groups["eval.jsonl"] != groups["holdout.jsonl"]:
        raise ValueError("Validation panels do not share source groups")
    text_groups = {}
    for records in datasets.values():
        for row in records:
            text = normalize(row["state"]).casefold()
            group = row["metadata"]["group"]
            if text in text_groups and text_groups[text] != group:
                raise ValueError("Duplicate normalized text across source groups")
            text_groups[text] = group
    if balanced:
        for name, records in datasets.items():
            if name == "regression.jsonl":
                continue
            counts = Counter((row["metadata"]["group"], row["labels"]["classification"]) for row in records)
            if any(counts[group, label] != 1 for group in groups[name] for label in LABELS):
                raise ValueError(f"Incomplete four-class group: {name}")
    return manifest, datasets


class TokenCounter:
    def __init__(self, model):
        from clef_finetune.jsm import load_jsm
        from transformers import AutoTokenizer

        model = Path(model)
        if not model.is_dir():
            raise ValueError("A local pinned Clef snapshot is required")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
        self.encoder = load_jsm(release_dir=model)

    def __call__(self, row):
        return len(self.encoder.encode_record(self.tokenizer, row, max_length=1000000).input_ids)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, url):
        return None


def api_endpoint(base_url, route):
    if any(character.isspace() or ord(character) < 32 for character in base_url):
        raise ValueError("Invalid API base URL")
    parts = urllib.parse.urlsplit(base_url)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username is not None or parts.password is not None or parts.query or parts.fragment:
        raise ValueError("Use an HTTP(S) base URL without credentials, query, or fragment")
    if parts.port is not None and not 1 <= parts.port <= 65535:
        raise ValueError("Invalid API port")
    if parts.scheme == "http" and parts.hostname != "localhost":
        try:
            local = ipaddress.ip_address(parts.hostname).is_loopback
        except ValueError:
            local = False
        if not local:
            raise ValueError("Remote API URLs require HTTPS")
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/") + "/" + route.lstrip("/"), "", ""))
