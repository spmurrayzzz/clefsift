from collections import Counter
import hashlib
import html
import importlib.util
import ipaddress
import json
import re
from pathlib import Path
import sys
import urllib.parse
import urllib.request


GENERAL_WORKFLOW = "clefsift-general-v1"
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


def provenance_hashes(source):
    identities = set()

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                match = re.fullmatch(r"(author|document|thread|cluster)(?:s|_id|_ids|_group|_cluster|_cluster_id|_cluster_ids|_name)?", key.lower())
                kind = match[1] if match else "url" if key in ("canonical_url", "source_url") else "group" if key == "group" else None
                if kind and item is not None:
                    for identity in item if isinstance(item, list) else [item]:
                        if not isinstance(identity, (str, int)) or isinstance(identity, bool) or not str(identity).strip():
                            raise ValueError("Invalid provenance identity")
                        identities.add(sha256(json.dumps([kind, str(identity)], ensure_ascii=False)))
                elif isinstance(item, (dict, list)):
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(source)
    return sorted(identities)


def validate_records(records, *, workflow=None, split=None):
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
        if workflow is not None and metadata.get("workflow") != workflow:
            raise ValueError(f"Workflow mismatch: {row['id']}")
        if metadata.get("workflow") == GENERAL_WORKFLOW:
            if metadata.get("split") not in ("train", "validation", "final_holdout"):
                raise ValueError(f"Unexpected dataset split: {row['id']}")
            if split is not None and metadata["split"] != split:
                raise ValueError(f"Dataset split mismatch: {row['id']}")
            if metadata.get("source_split") not in ("train", "valid", "unsplit"):
                raise ValueError(f"Unexpected source split: {row['id']}")
            original_split = metadata["split"]
            if metadata.get("selection_policy") == "all-valid-development-groups-v1":
                original_split = metadata.get("original_split")
                if original_split not in ("train", "validation") or metadata["split"] not in (("train", "final_holdout") if original_split == "train" else ("validation",)):
                    raise ValueError(f"Invalid development split assignment: {row['id']}")
            if metadata["source_split"] != "unsplit" and (original_split == "train") != (metadata["source_split"] == "train"):
                raise ValueError(f"Source split reassigned: {row['id']}")
            tokens = metadata.get("input_tokens")
            if type(tokens) is not int or not 0 < tokens <= 1536:
                raise ValueError(f"Invalid token count: {row['id']}")
        elif metadata.get("upstream_split") not in ("train", "valid"):
            raise ValueError(f"Unexpected source split: {row['id']}")


def dataset_stage(manifest):
    stage = manifest.get("stage")
    if stage in ("parents", "assembled"):
        return stage
    files = manifest.get("files")
    if not isinstance(files, list) or any(not isinstance(item, dict) or not isinstance(item.get("path"), str) for item in files):
        raise ValueError("Dataset is not ready")
    names = {item.get("path") for item in files}
    if (manifest.get("workflow") == GENERAL_WORKFLOW and manifest.get("version") == 1
            and names == {"train.jsonl", "eval.jsonl", "final-holdout.jsonl"}
            and {"recipe_sha256", "parent_manifest_sha256", "target_groups"} <= manifest.keys()):
        return "assembled"
    if (manifest.get("parent_stage") == "parents" and {"train.jsonl", "eval.jsonl"} <= names
            and {"parent_manifest_sha256", "generation_plan_sha256", "generation_results_sha256"} <= manifest.keys()):
        return "assembled"
    raise ValueError("Dataset is not ready")


def _validate_general_dataset(directory, manifest, include_final):
    if type(include_final) is not bool:
        raise ValueError("include_final must be an explicit boolean")
    roles = {"train.jsonl": "train", "eval.jsonl": "validation", "final-holdout.jsonl": "final_holdout"}
    development = manifest.get("selection_policy") == "all-valid-development-groups-v1"
    if "selection_policy" in manifest and not development:
        raise ValueError("Unknown selection policy")
    targets = manifest.get("actual_groups" if development else "target_groups")
    if dataset_stage(manifest) != "assembled" or not isinstance(targets, dict) or set(targets) != set(roles.values()):
        raise ValueError("General workflow requires an assembled dataset and explicit target groups")
    if any(type(count) is not int or count <= 0 or (not development and count % 2) for count in targets.values()):
        raise ValueError("Target group counts must be positive even integers")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != len(roles):
        raise ValueError("General workflow requires exactly train, eval, and final-holdout entries")
    expected_routes = manifest.get("routes")
    if not isinstance(expected_routes, list) or len(expected_routes) != 2 or any(not isinstance(route, str) or "/" not in route for route in expected_routes) or len(set(expected_routes)) != 2:
        raise ValueError("General workflow requires two explicit generator routes")
    expected_routes = {tuple(route.split("/", 1)) for route in expected_routes}
    dynamic = manifest.get("generation_assignment") == "per-attempt-v1"
    if "generation_assignment" in manifest and not dynamic:
        raise ValueError("Unknown generation assignment")
    if dynamic and manifest.get("route_quota_basis") != "planned-parent":
        raise ValueError("Dynamic generation requires planned-parent quotas")
    actual_routes = manifest.get("actual_routes") if development else manifest["routes"]
    if not isinstance(actual_routes, list) or not actual_routes or any(not isinstance(route, str) or "/" not in route for route in actual_routes):
        raise ValueError("Invalid actual routes")
    if development:
        expected_routes = {tuple(route.split("/", 1)) for route in actual_routes}
    domain_targets = manifest.get("actual_domain_route_counts" if development else "domain_route_targets")
    if not isinstance(domain_targets, dict) or set(domain_targets) != set(roles.values()):
        raise ValueError("General workflow requires domain and route quotas for every split")
    for role, domains in domain_targets.items():
        if not isinstance(domains, dict) or not domains:
            raise ValueError("Invalid domain quotas")
        totals = Counter()
        for domain, counts in domains.items():
            if not isinstance(domain, str) or not domain.strip() or not isinstance(counts, dict) or set(counts) != set(manifest["routes"]):
                raise ValueError("Invalid domain route quota")
            if any(type(count) is not int or count < 0 for count in counts.values()):
                raise ValueError("Invalid domain group count")
            totals.update(counts)
        if sum(totals.values()) != targets[role] or (not development and any(totals[route] != targets[role] // 2 for route in manifest["routes"])):
            raise ValueError("Domain quotas differ from split targets")
    entries, all_groups, all_text = {}, set(), set()
    all_provenance, all_clusters = set(), set()

    def is_hash(value):
        return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)

    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("Invalid dataset file entry")
        name = item["path"]
        if name not in roles or name in entries or item.get("role") != roles[name]:
            raise ValueError("Unexpected dataset file or role")
        count = targets[roles[name]]
        if type(item.get("groups")) is not int or item["groups"] != count or type(item.get("rows")) is not int or item["rows"] != count * len(LABELS):
            raise ValueError(f"Dataset target count mismatch: {name}")
        labels = item.get("labels")
        if not isinstance(labels, dict) or set(labels) != set(LABELS) or any(type(value) is not int or value != count for value in labels.values()):
            raise ValueError(f"Dataset label count mismatch: {name}")
        if dynamic:
            planned = item.get("route_counts")
            actual = item.get("actual_route_counts")
            if not isinstance(planned, dict) or set(planned) != set(LABELS):
                raise ValueError(f"Invalid planned route counts: {name}")
            for counts in planned.values():
                if not isinstance(counts, dict) or not set(counts) <= set(manifest["routes"]) or any(type(value) is not int or value <= 0 or (not development and value != count // 2) for value in counts.values()) or sum(counts.values()) != count:
                    raise ValueError(f"Unbalanced planned route counts: {name}")
            if not isinstance(actual, dict) or set(actual) != set(LABELS[1:]):
                raise ValueError(f"Invalid actual route counts: {name}")
            for counts in actual.values():
                if not isinstance(counts, dict) or not set(counts) <= set(actual_routes) or any(type(value) is not int or value <= 0 for value in counts.values()) or sum(counts.values()) != count:
                    raise ValueError(f"Invalid actual route contribution: {name}")
        group_ids, text_hashes = item.get("group_ids"), item.get("text_sha256")
        if not isinstance(group_ids, list) or any(not isinstance(group, str) or not group.strip() for group in group_ids):
            raise ValueError(f"Invalid manifest group IDs: {name}")
        if len(group_ids) != count or group_ids != sorted(set(group_ids)):
            raise ValueError(f"Manifest group count, order, or uniqueness mismatch: {name}")
        if not is_hash(item.get("sha256")) or not isinstance(text_hashes, list) or any(not is_hash(value) for value in text_hashes):
            raise ValueError(f"Invalid manifest hashes: {name}")
        if len(text_hashes) != item["rows"] or text_hashes != sorted(set(text_hashes)):
            raise ValueError(f"Manifest text count, order, or uniqueness mismatch: {name}")
        if all_groups.intersection(group_ids) or all_text.intersection(text_hashes):
            raise ValueError("General workflow splits overlap in parent groups or normalized text")
        if "max_tokens" in item and (type(item["max_tokens"]) is not int or not 0 < item["max_tokens"] <= 1536):
            raise ValueError(f"Invalid manifest token count: {name}")
        if development:
            for key, seen in (("provenance_sha256", all_provenance), ("cluster_ids", all_clusters)):
                inventory = item.get(key)
                if not isinstance(inventory, list) or not inventory or any(not is_hash(value) for value in inventory) or inventory != sorted(set(inventory)) or seen.intersection(inventory):
                    raise ValueError(f"Invalid or overlapping provenance inventory: {name}")
                seen.update(inventory)
        all_groups.update(group_ids)
        all_text.update(text_hashes)
        entries[name] = item
    datasets, seen_ids, parent_ids, parent_texts, routes = {}, set(), {}, {}, {}
    for name, role in roles.items():
        if role == "final_holdout" and not include_final:
            continue
        item = entries[name]
        if file_hash(directory / name) != item["sha256"]:
            raise ValueError(f"Dataset hash mismatch: {name}")
        records = read_jsonl(directory / name)
        validate_records(records, workflow=GENERAL_WORKFLOW, split=role)
        if not development and any("selection_policy" in row["metadata"] for row in records):
            raise ValueError("Development row in strict dataset")
        if development:
            identities, clusters, group_clusters = set(), set(), {}
            for row in records:
                metadata = row["metadata"]
                cluster = metadata.get("provenance_cluster")
                if metadata.get("selection_policy") != manifest["selection_policy"] or not is_hash(cluster):
                    raise ValueError("Invalid development selection metadata")
                group = metadata["group"]
                if group in group_clusters and group_clusters[group] != cluster:
                    raise ValueError("Group has inconsistent provenance cluster")
                group_clusters[group] = cluster
                identities.update(provenance_hashes(metadata["provenance"]))
                clusters.add(cluster)
            if sorted(identities) != item["provenance_sha256"] or sorted(clusters) != item["cluster_ids"]:
                raise ValueError("Loaded provenance differs from manifest")
        groups = sorted({row["metadata"]["group"] for row in records})
        texts = sorted(sha256(normalize(row["state"]).casefold()) for row in records)
        labels = Counter(row["labels"]["classification"] for row in records)
        if len(records) != item["rows"] or groups != item["group_ids"] or texts != item["text_sha256"] or labels != item["labels"]:
            raise ValueError(f"Loaded rows differ from manifest metadata: {name}")
        if "max_tokens" in item and max(row["metadata"]["input_tokens"] for row in records) != item["max_tokens"]:
            raise ValueError(f"Loaded token counts differ from manifest metadata: {name}")
        counts, generators, domain_counts = Counter(), {}, Counter()
        parent_strata = {}
        planned_counts = {label: Counter() for label in LABELS}
        routes[name] = {label: Counter() for label in LABELS if label != "human"}
        for row in records:
            metadata, label = row["metadata"], row["labels"]["classification"]
            group = metadata["group"]
            if row["id"] in seen_ids:
                raise ValueError("Record IDs overlap across splits")
            seen_ids.add(row["id"])
            for identity in (row["id"], metadata.get("parent_id", row["id"])):
                if not isinstance(identity, str) or not identity.strip():
                    raise ValueError("Invalid parent ID")
                if identity in parent_ids and parent_ids[identity] != group:
                    raise ValueError("Parent IDs overlap across source groups")
                parent_ids[identity] = group
            if "parent_text_sha256" in metadata:
                digest = metadata["parent_text_sha256"]
                if not is_hash(digest):
                    raise ValueError("Invalid parent text hash")
                if digest in parent_texts and parent_texts[digest] != group:
                    raise ValueError("Parent text overlaps across source groups")
                parent_texts[digest] = group
            counts[group, label] += 1
            if dynamic:
                assigned = metadata.get("assigned_route")
                domain = metadata.get("domain")
                if not isinstance(assigned, str) or assigned not in manifest["routes"] or not isinstance(domain, str) or domain not in domain_targets[role]:
                    raise ValueError("Invalid planned parent stratum")
                stratum = (domain, assigned)
                if group in parent_strata and parent_strata[group] != stratum:
                    raise ValueError("Rows for a parent must share the planned stratum")
                parent_strata[group] = stratum
                planned_counts[label][assigned] += 1
                if label == "human":
                    domain_counts[stratum] += 1
            if label != "human":
                route = (metadata.get("provider"), metadata.get("model"))
                if any(not isinstance(value, str) or not value.strip() for value in route):
                    raise ValueError("Generated rows require provider and model")
                if route not in expected_routes:
                    raise ValueError("Unexpected generator route")
                if dynamic:
                    if metadata.get("actual_route") != "/".join(route):
                        raise ValueError("Actual route differs from provider and model")
                else:
                    if metadata.get("assigned_route") != "/".join(route):
                        raise ValueError("Generated row differs from the planned route")
                    if label == "ai":
                        domain_counts[metadata.get("domain"), "/".join(route)] += 1
                    if group in generators and generators[group] != route:
                        raise ValueError("Generated variants for a parent must use the same provider and model")
                generators[group] = route
                routes[name][label][route] += 1
        if any(counts[group, label] != 1 for group in groups for label in LABELS):
            raise ValueError(f"Incomplete four-class group: {name}")
        expected_domains = Counter({(domain, route): count for domain, counts in domain_targets[role].items() for route, count in counts.items()})
        if domain_counts != expected_domains:
            raise ValueError(f"Loaded rows differ from domain route quotas: {name}")
        if dynamic:
            actual_counts = {label: {"/".join(route): count for route, count in counts.items()} for label, counts in routes[name].items()}
            if planned_counts != item["route_counts"] or actual_counts != item["actual_route_counts"]:
                raise ValueError(f"Loaded route counts differ from manifest metadata: {name}")
        datasets[name] = records
    for name, panel in routes.items():
        if not dynamic and any(set(counts) != expected_routes or len(set(counts.values())) != 1 for counts in panel.values()):
            raise ValueError(f"Unbalanced generator route contribution: {name}")
    return manifest, datasets


def validate_dataset(directory, balanced=False, *, include_final=False):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("question") != QUESTION or manifest.get("model_revision") != MODEL_REVISION:
        raise ValueError("Dataset question or model revision mismatch")
    if manifest.get("max_input_tokens") != 1536 or manifest.get("truncation") is not False:
        raise ValueError("Dataset requires full-request token checks without truncation")
    dataset_stage(manifest)
    if manifest.get("workflow") == GENERAL_WORKFLOW:
        return _validate_general_dataset(directory, manifest, include_final)
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


def load_schema(model):
    path = Path(model).resolve() / "joint_schema_model.py"
    if not path.is_file():
        raise ValueError("A local Clef snapshot with joint_schema_model.py is required")
    name = "_clefsift_schema_" + sha256(str(path))[:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    return sys.modules[name]


class TokenCounter:
    def __init__(self, model):
        from transformers import AutoTokenizer

        self.encoder = load_schema(model)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)

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
