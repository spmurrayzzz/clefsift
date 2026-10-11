import argparse
from bisect import bisect_left
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import heapq
from html.parser import HTMLParser
import importlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import sqlite3
import sys
import time
import traceback
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import xml.etree.ElementTree as ET

from common import sha256, write_json
from corpus import SPLITS, normalize, provenance_identities, source_check, strict_json


SEED = "clefsift-general-v1:source-import:1"
DOMAINS = ("news", "reviews", "qa", "personal", "professional", "technical", "discussion")
READERS = {"pyarrow": "26.0.0", "py7zr": "1.1.3"}
GIB = 1024 ** 3
MAX_RECORD = 1024 ** 2
MAX_TARGET_BYTES = 4 * GIB
MAX_ARCHIVE_BYTES = 16 * GIB
MAX_EXTRACTED_BYTES = 16 * GIB
TRACKING = ("fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid", "ref_src", "ref_url")
DOMAIN_RULES = {
    "discussion_url": r"\b(?:forums?|threads?|viewtopic|boards?|community|reddit)\b",
    "technical_url": r"\b(?:docs|documentation|tutorials?|wiki|readthedocs)\b",
    "technical_text": r"\b(?:algorithm|compiler|database|programming|function|software|hardware|protocol|equation|theorem|javascript|python|linux|circuit)\b",
    "professional_text": r"\b(?:business|work|career|customer|marketing|workplace|employee|management|client|hiring|recruitment)\b",
    "personal_text": r"\b(?:family|mother|father|husband|wife|daughter|son|childhood|birthday|diary|vacation|wedding|my life|our home)\b",
    "first_person": r"\b(?:i|my|me|we|our)\b",
    "blog_url": r"\b(?:blog|blogs|wordpress|blogspot)\b",
}
BOILERPLATE = r"(?:all rights reserved\.?|privacy policy|terms (?:of service|and conditions)|accept (?:all )?cookies|cookie policy|share this article|sign up for our newsletter)"
POLICY = {
    "seed": SEED,
    "readers": READERS,
    "archive_before": "2019-01-01",
    "domain_priority": ["discussion", "technical", "professional", "personal"],
    "domain_rules": DOMAIN_RULES,
    "domain_text_window_words": 512,
    "personal_rule": "first_person AND (blog_url OR personal_text)",
    "english_screen": "source evidence AND ASCII letters >=70% of letters AND >=2 distinct listed function words; not verified language",
    "english_function_words": "the a an and or but if to of in on for with is are was were be it this that i my we you",
    "normalization": "HTMLParser for reviews/Stack Exchange; drop code/pre/script/style/nav/header/footer; remove exact standalone boilerplate lines; corpus.normalize to fixed point (at most 16 passes; reject nonconvergence), including spaces before punctuation; remove initial CNN publisher prefix",
    "boilerplate_pattern": BOILERPLATE,
    "excerpt": "one contiguous normalized sentence span, 50-300 whitespace words; stable target 60/120/220/300; nearest length then stable offset rank; >=2 sentences with >=10 words on each side of a boundary",
    "offset_unit": "Unicode code points in full normalized text; end exclusive",
    "url_policy": "collapse HTTP/HTTPS and www; drop fragments and tracking parameters; retain all other query keys/values in sorted order; retain path case",
    "tracking_parameters": list(TRACKING),
    "tracking_prefix": "utm_",
    "grouping": "disk-backed union of qualified corpus identities, full normalized casefold hash, excerpt casefold hash, canonical URL and identified thread/blog-host clusters; one lowest-ranked eligible original per component",
    "stackexchange_selection": "union duplicate PostLinks and answer ParentId edges in a first Posts pass; select lowest-ranked eligible post per canonical thread in a second pass before author grouping; no votes or scores",
    "unsplit_assignment": "stable canonical-component hash modulo 12: 0..9 train, 10 validation, 11 final_holdout",
    "official_assignment": "train fixed; original validation stable canonical-component hash modulo 2; conflicting official components excluded",
    "candidate_selection": "lowest stable group hashes per split/domain; bounded heaps; disk-backed identity and component state",
    "limits": {"record_bytes_or_characters": MAX_RECORD, "archive_uncompressed_bytes": MAX_ARCHIVE_BYTES,
               "archive_target_bytes": MAX_TARGET_BYTES, "total_extracted_bytes": MAX_EXTRACTED_BYTES,
               "minimum_free_bytes": 30 * GIB, "preparation_minutes": 120},
    "downstream": "CPU token counting and near-duplicate QC remain in corpus.py prepare; use reserve_fraction 0.10; at most 26400 generation parents under the approved quotas",
    "authorship": "historical-source-provisional; not independently verified; no detector or model used",
}


class ImportFailure(ValueError):
    pass


class RecordRejected(ValueError):
    pass


def record_require(condition, reason):
    if not condition:
        raise RecordRejected(reason)


def text_field(value):
    record_require(isinstance(value, str), "field_type")
    record_require(len(value) <= MAX_RECORD, "record_size")
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise RecordRejected("unicode") from None
    record_require("\x00" not in value, "unicode")
    return value


def require(condition, message):
    if not condition:
        raise ImportFailure(message)


def pack(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def rank(value):
    return sha256(SEED + ":" + value)


def load(path):
    return strict_json(Path(path).read_text(encoding="utf-8"))


def reader(name):
    try:
        module = importlib.import_module(name)
    except ImportError:
        raise ImportFailure(f"Required reader missing: {name}=={READERS[name]}; use the separate approved reader environment") from None
    require(module.__version__ == READERS[name], f"Required reader version: {name}=={READERS[name]}")
    return module


class Budget:
    def __init__(self, args):
        require(math.isfinite(args.minutes) and 0 < args.minutes <= 120, "--minutes must be positive and at most 120")
        self.started = time.monotonic()
        seconds = args.minutes * 60
        if args.deadline_utc:
            try:
                deadline = datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00"))
                require(deadline.utcoffset() is not None and deadline.utcoffset().total_seconds() == 0, "--deadline-utc must specify UTC")
                seconds = min(seconds, (deadline - datetime.now(timezone.utc)).total_seconds())
            except ValueError:
                raise ImportFailure("Invalid --deadline-utc") from None
        require(seconds > 0, "Preparation deadline has expired")
        self.deadline = self.started + seconds
        self.deadline_utc = datetime.fromtimestamp(time.time() + seconds, timezone.utc).isoformat()
        self.next_disk = 0
        self.directory = args.out_dir

    def check(self):
        now = time.monotonic()
        require(now < self.deadline, "Preparation deadline reached")
        if now >= self.next_disk:
            require(all(shutil.disk_usage(path).free >= 30 * GIB for path in (self.directory, Path.home(), Path('/'))), "Less than 30 GiB of free artifact or root disk")
            self.next_disk = now + 1


def hash_file(path, budget=None, algorithms=("sha256",)):
    hashes = {name: hashlib.new(name) for name in algorithms}
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 ** 2):
            if budget:
                budget.check()
            for value in hashes.values():
                value.update(block)
    return {name: value.hexdigest() for name, value in hashes.items()}


def file_state(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def raw_path(directory, name):
    require(isinstance(name, str) and name and "\\" not in name and "\x00" not in name, "Invalid relative raw path")
    parts = name.split("/")
    require(all(part not in ("", ".", "..") for part in parts) and not PurePosixPath(name).is_absolute() and ":" not in parts[0], "Unsafe relative raw path")
    path = directory
    for part in parts:
        path = path / part
        require(not path.is_symlink(), "Raw paths must not contain symbolic links")
    require(path.is_file() and path.resolve().is_relative_to(directory.resolve()), "Raw file is missing or outside acquired directory")
    return path


def inventory(args, budget, audit):
    catalog, quotas = load(args.catalog), load(args.quotas)
    manifest = load(args.acquired_dir / "manifest.json")
    catalog_hash = hash_file(args.catalog, budget)["sha256"]
    require(catalog.get("workflow") == "clefsift-general-v1" and catalog.get("status") == "approved-exploratory-acquisition", "Catalog lacks approved exploratory status")
    require(catalog.get("archive_before") == "2019-01-01" and isinstance(catalog.get("rights_policy"), str), "Invalid catalog policy")
    require(manifest.get("stage") == "acquired" and manifest.get("catalog_sha256") == catalog_hash, "Acquisition stage or catalog hash mismatch")
    require(isinstance(quotas, dict) and set(quotas) == set(SPLITS), "Quotas must contain all three splits")
    for domains in quotas.values():
        require(isinstance(domains, dict) and set(domains) == set(DOMAINS), "Quotas must contain the seven source domains")
        require(all(type(n) is int and n > 0 and n % 2 == 0 for n in domains.values()), "Quotas must be positive even integers")
    require(sum(n + 2 * math.ceil(n * 0.10 / 2) for domains in quotas.values() for n in domains.values()) <= 26400, "Quotas with 10% reserves exceed 26400 generation parents")
    expected = {}
    sources = catalog.get("sources")
    require(isinstance(sources, list) and len(sources) == 4 and {s.get("id") for s in sources} == {"cnn-dailymail", "snap-amazon-2013", "stackexchange-2018-12", "fineweb-2018-30"}, "Unexpected source catalog")
    for source in sources:
        require(isinstance(source.get("rights_status"), str) and source["rights_status"].strip() and isinstance(source.get("metadata_url"), str), "Missing source rights or evidence URL")
        if source["id"] in ("cnn-dailymail", "fineweb-2018-30"):
            require(isinstance(source.get("revision"), str) and re.fullmatch(r"[0-9a-f]{40}", source["revision"]), "Missing pinned source revision")
        for spec in source["files"]:
            name = spec.get("path", spec.get("url"))
            require(all(record.get(key, "unsplit") in ("train", "valid", "validation", "unsplit") for record in (source, spec) for key in ("split", "source_split")), "Upstream test or holdout files prohibited")
            require(isinstance(name, str) and name and (source["id"], name) not in expected, "Duplicate or missing catalog file locator")
            if source["id"] == "cnn-dailymail":
                require(re.fullmatch(r"3\.0\.0/(?:train|validation)-\d+-of-\d+\.parquet", name) is not None, "Only CNN training and validation files are allowed")
            elif source["id"] == "fineweb-2018-30":
                require(name == "data/CC-MAIN-2018-30/000_00000.parquet", "Only the named FineWeb 2018 shard is allowed")
            elif source["id"] == "stackexchange-2018-12":
                require(re.fullmatch(r"[a-z]+\.stackexchange\.com\.7z", name) is not None and source.get("archive") == "StackExchange2018-12", "Unexpected Stack Exchange archive")
            else:
                require(name.startswith("https://snap.stanford.edu/data/amazon/") and name.endswith(".txt.gz"), "Unexpected SNAP file")
            expected[source["id"], name] = (source, spec)
    entries = manifest.get("files")
    require(isinstance(entries, list) and len(entries) == len(expected), "Acquired file inventory does not match catalog")
    found, paths, result = set(), set(), []
    for entry in entries:
        require(isinstance(entry, dict) and all(k in entry for k in ("source_id", "catalog_path", "local_path", "bytes", "sha256", "source_url", "verified_hash")), "Missing acquired file fields")
        matches = []
        for key, (source, spec) in expected.items():
            aliases = {key[1]}
            if "url" in spec:
                aliases.update((urlsplit(spec["url"]).path.lstrip("/"), PurePosixPath(urlsplit(spec["url"]).path).name))
            if entry["source_id"] == key[0] and entry["catalog_path"] in aliases:
                matches.append(key)
        require(len(matches) == 1 and matches[0] not in found, "Unlisted or repeated acquired file")
        key = matches[0]
        source, spec = expected[key]
        require(isinstance(entry["source_url"], str) and urlsplit(entry["source_url"]).scheme in ("http", "https") and urlsplit(entry["source_url"]).hostname, "Missing acquired source URL")
        require("url" not in spec or entry["source_url"] == spec["url"], "SNAP source URL differs from catalog")
        require(type(entry["bytes"]) is int and entry["bytes"] > 0 and isinstance(entry["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), "Invalid acquired size or SHA-256")
        audit["current_file"] = {"source_id": source["id"], "catalog_path": key[1], "operation": "verify_hash"}
        path = raw_path(args.acquired_dir, entry["local_path"])
        require(path.resolve() not in paths, "Repeated local raw path")
        before = file_state(path)
        require(before[2] == entry["bytes"] and ("bytes" not in spec or before[2] == spec["bytes"]), "Raw file size mismatch")
        hashes = hash_file(path, budget, ("sha256", "sha1") if "sha1" in spec else ("sha256",))
        require(file_state(path) == before and hashes["sha256"] == entry["sha256"], "Raw file changed or acquisition SHA-256 mismatch")
        require(all(hashes[k] == spec[k] for k in ("sha256", "sha1") if k in spec), "Catalog checksum mismatch")
        verified = {**entry, "catalog_path": key[1], "sha256": hashes["sha256"], "checked_hashes": hashes,
                    "catalog_hash_verified": any(k in spec for k in ("sha256", "sha1"))}
        audit["files"].append(verified)
        result.append((source, verified, path, before))
        found.add(key)
        paths.add(path.resolve())
    require(found == set(expected), "Missing catalog files")
    audit.update(catalog_sha256=catalog_hash, quotas_sha256=hash_file(args.quotas, budget)["sha256"],
                 acquired_manifest_sha256=hash_file(args.acquired_dir / "manifest.json", budget)["sha256"])
    return catalog, quotas, sorted(result, key=lambda item: (item[0]["id"], item[1]["catalog_path"]))


class Markup(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.hidden = [], []

    def handle_starttag(self, tag, attrs):
        if tag in ("code", "pre", "script", "style", "nav", "header", "footer"):
            self.hidden.append(tag)
        elif not self.hidden and tag in ("br", "p", "div", "li", "blockquote", "h1", "h2", "h3", "tr"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.hidden:
            self.hidden = self.hidden[:self.hidden.index(tag)]
            self.parts.append("\n")
        elif not self.hidden and tag in ("p", "div", "li", "blockquote", "h1", "h2", "h3", "tr"):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def clean(text, markup=False):
    text_field(text)
    if markup:
        parser = Markup()
        try:
            parser.feed(text)
            parser.close()
        except ImportFailure:
            raise
        except (ValueError, AssertionError):
            raise RecordRejected("markup") from None
        text = "".join(parser.parts)
    text = "\n".join(line for line in text.splitlines() if not re.fullmatch(BOILERPLATE, line.strip(), re.I))
    for _ in range(16):
        normalized = normalize(text)
        if normalized == text:
            return text
        text = normalized
    record_require(normalize(text) == text, "normalization")
    return text


def excerpt(text, identity):
    spans = [(m.start(), m.end()) for m in re.finditer(r'.+?(?:[.!?][\"\u201d\u2019\x27)]*(?=\s|$)|$)', text)]
    spans = [(start + len(text[start:end]) - len(text[start:end].lstrip()), end) for start, end in spans if text[start:end].strip()]
    lengths = [len(text[start:end].split()) for start, end in spans]
    prefix = [0]
    for length in lengths:
        prefix.append(prefix[-1] + length)
    target = (60, 120, 220, 300)[int(rank(identity + ":length"), 16) % 4]
    best = None
    for first, (start, _) in enumerate(spans):
        boundary = bisect_left(prefix, prefix[first] + 10, first + 1)
        for last in range(first, len(spans)):
            count = prefix[last + 1] - prefix[first]
            if count > 300:
                break
            end = spans[last][1]
            if count >= 50 and boundary <= last and prefix[last + 1] - prefix[boundary] >= 10 and (best is None or abs(count - target) <= best[0]):
                candidate = (abs(count - target), rank(f"{identity}:offset:{start}:{end}"), start, end, count)
                if best is None or candidate < best:
                    best = candidate
    if best is None:
        return None
    _, _, start, end, count = best
    return text[start:end], {"start": start, "end": end, "words": count, "target_words": target, "unit": POLICY["offset_unit"]}


def canonical_url(value):
    text_field(value)
    record_require(not any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value) and "\\" not in value and not re.search(r"%(?![0-9a-fA-F]{2})", value), "url")
    try:
        parsed = urlsplit(value)
        host, port = parsed.hostname, parsed.port
        query = sorted((k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True, errors="strict") if not k.lower().startswith("utm_") and k.lower() not in TRACKING)
    except ImportFailure:
        raise
    except ValueError:
        raise RecordRejected("url") from None
    record_require(parsed.scheme.lower() in ("http", "https") and host and not parsed.username and not parsed.password, "url")
    host = host.lower().removeprefix("www.")
    record_require(host and host.strip("."), "url")
    authority = host if port in (None, 80, 443) else f"{host}:{port}"
    return urlunsplit(("https", authority, parsed.path or "/", urlencode(query), "")), host


def fineweb_domain(url, text, counts):
    preview = " ".join(text.split(maxsplit=512)[:512])
    matches = {key: bool(re.search(pattern, url if key.endswith("url") else preview, re.I)) for key, pattern in DOMAIN_RULES.items()}
    possible = {"discussion": matches["discussion_url"], "technical": matches["technical_url"] or matches["technical_text"],
                "professional": matches["professional_text"], "personal": matches["first_person"] and (matches["blog_url"] or matches["personal_text"])}
    for domain, matched in possible.items():
        if matched:
            counts[f"domain_match:{domain}"] += 1
    return next((domain for domain in POLICY["domain_priority"] if possible[domain]), None)


def overlap_host(host):
    fixed = ("cnn.com", "cnn.net", "dailymail.co.uk", "mailonsunday.co.uk", "stackexchange.com", "stackoverflow.com", "serverfault.com", "superuser.com", "askubuntu.com", "mathoverflow.net", "stackapps.com")
    return any(host == name or host.endswith("." + name) for name in fixed) or re.search(r"(?:^|\.)(?:amazon|amzn)\.[a-z.]+$", host) is not None


def web_clusters(url, host):
    clusters = []
    if re.fullmatch(r"[^.]+\.(?:wordpress\.com|blogspot\.[a-z.]+)", host):
        clusters.append("web:blog-host:" + host)
    parsed = urlsplit(url)
    for key, value in parse_qsl(parsed.query):
        if key.lower() in ("t", "threadid", "thread_id", "topic", "topicid", "topic_id") and value:
            clusters.append("web:thread:" + host + parsed.path + "?" + key.lower() + "=" + value)
    match = re.search(r"/(?:comments|threads?|topics?)/([^/]+)", parsed.path, re.I)
    if match:
        clusters.append("web:thread:" + host + match.group(0))
    return clusters


class Groups:
    def __init__(self, db, table):
        self.db, self.table = db, table
        db.execute(f"CREATE TABLE {table} (key TEXT PRIMARY KEY, parent TEXT NOT NULL)")

    def root(self, key):
        self.db.execute(f"INSERT OR IGNORE INTO {self.table} VALUES (?, ?)", (key, key))
        path, current = [], key
        while True:
            parent = self.db.execute(f"SELECT parent FROM {self.table} WHERE key=?", (current,)).fetchone()[0]
            if parent == current:
                break
            path.append(current)
            current = parent
        self.db.executemany(f"UPDATE {self.table} SET parent=? WHERE key=?", ((current, item) for item in path))
        return current

    def join(self, keys):
        roots = sorted({self.root(key) for key in keys})
        root = roots[0]
        self.db.executemany(f"UPDATE {self.table} SET parent=? WHERE key=?", ((root, old) for old in roots[1:]))
        return root, roots


class Importer:
    def __init__(self, args, budget, catalog, audit):
        self.args, self.budget, self.catalog, self.audit = args, budget, catalog, audit
        self.db = sqlite3.connect(args.out_dir / "import-state.sqlite")
        self.db.execute("PRAGMA cache_size=-16384")
        self.db.execute("PRAGMA temp_store=FILE")
        self.groups = Groups(self.db, "identities")
        self.threads = Groups(self.db, "threads")
        self.db.execute("CREATE TABLE components (root TEXT PRIMARY KEY, mask INTEGER NOT NULL, rank TEXT, payload TEXT)")
        self.db.execute("CREATE TABLE users (site TEXT, id TEXT, account TEXT, name TEXT, PRIMARY KEY (site, id))")
        self.db.execute("CREATE TABLE posts (thread TEXT PRIMARY KEY, rank TEXT NOT NULL, payload TEXT NOT NULL)")
        self.counts = Counter()
        self.extracted = 0

    @contextmanager
    def record(self, stage=""):
        try:
            yield
        except RecordRejected as exc:
            self.counts["excluded:" + stage + str(exc)] += 1

    def scanned(self, source):
        self.budget.check()
        key = source["id"] + ":scanned"
        self.counts[key] += 1
        if self.counts[key] % 10000 == 0:
            print(pack({"stage": "import-progress", "source_id": source["id"], "scanned": self.counts[key],
                        "eligible_records": self.counts[source["id"] + ":eligible_records"],
                        "seconds": time.monotonic() - self.budget.started}), flush=True)

    def base(self, source, entry, locator, document, archive_date, evidence):
        return {"id": document + ":excerpt", "document_id": document, "source": source["id"], "source_split": "unsplit",
                "label": "human", "lang": "eng", "label_status": "historical-source-provisional",
                "authorship_status": "not-independently-verified", "author_identity_status": "unknown",
                "language_source_evidence": {"cnn-dailymail": "English-language CNN/DailyMail corpus",
                    "snap-amazon-2013": "US Amazon review collection; English expected, not guaranteed",
                    "stackexchange-2018-12": "Selected English-language Stack Exchange sites",
                    "fineweb-2018-30": "Source language=en and language_score>=0.85"}.get(source["id"], "Unverified source language"),
                "rights_status": source["rights_status"] + "; " + self.catalog["rights_policy"],
                "archive_date": archive_date, "evidence": evidence, "evidence_url": source["metadata_url"],
                "source_revision": source.get("revision", source.get("archive", "sha256:" + entry["sha256"])),
                "raw_source_file": entry["local_path"], "raw_source_sha256": entry["sha256"], "record_locator": locator,
                "raw_source_url": entry["source_url"], "catalog_path": entry["catalog_path"],
                "source_metadata": {k: source[k] for k in ("dataset", "archive", "source_dates", "provenance", "license_url", "rights_status", "split_policy") if k in source}}

    def original(self, row, raw, markup=False):
        self.budget.check()
        self.counts[row["source"] + ":records"] += 1
        text_field(raw)
        text = clean(raw, markup)
        if row["source"] == "cnn-dailymail":
            text = re.sub(r"^.{0,100}?\(CNN\)\s*--\s*", "", text).strip()
        letters = [c for c in text if c.isalpha()]
        function_words = set(re.findall(r"\b[a-z]+\b", text.casefold())) & set(POLICY["english_function_words"].split())
        if not letters or sum(c.isascii() for c in letters) / len(letters) < 0.70 or len(function_words) < 2:
            self.counts["excluded:english_screen"] += 1
            return None
        selected = excerpt(text, row["document_id"])
        if selected is None:
            self.counts["excluded:sentence_excerpt"] += 1
            return None
        row.update(text=selected[0], normalization={"policy": "source-import-v1", "raw_text_sha256": sha256(raw),
                   "full_normalized_sha256": sha256(text), "full_normalized_casefold_sha256": sha256(text.casefold()),
                   "full_normalized_characters": len(text), "markup_removed": markup}, excerpt=selected[1],
                   language_evidence="Source-specific English evidence plus light lexical screen; not independently verified")
        self.counts[row["source"] + ":eligible_records"] += 1
        return row

    def add(self, row):
        if row is None:
            return
        self.budget.check()
        identities = [kind + ":" + str(value) for kind, value in provenance_identities(row)]
        texts = ["full-text:" + row["normalization"]["full_normalized_casefold_sha256"], "excerpt-text:" + sha256(row["text"].casefold())]
        if self.db.execute("SELECT 1 FROM identities WHERE key IN (?, ?) LIMIT 1", texts).fetchone():
            self.counts["exact_duplicate_records"] += 1
        identities += texts
        root, roots = self.groups.join(identities)
        mask = {"train": 1, "valid": 2, "unsplit": 0}[row["source_split"]]
        best = (rank(row["id"] + ":" + row["raw_source_sha256"] + ":" + pack(row["record_locator"])), pack(row))
        for old in roots:
            existing = self.db.execute("SELECT mask, rank, payload FROM components WHERE root=?", (old,)).fetchone()
            if existing:
                mask |= existing[0]
                best = min(best, existing[1:])
                self.db.execute("DELETE FROM components WHERE root=?", (old,))
                self.counts["merged_component_candidates"] += 1
        self.db.execute("INSERT INTO components VALUES (?, ?, ?, ?)", (root, mask, *best))
        if self.counts["added_records"] % 1000 == 0:
            self.db.commit()
        self.counts["added_records"] += 1

    def parquet(self, source, entry, path):
        reader("pyarrow")
        parquet = importlib.import_module("pyarrow.parquet")
        cnn = source["id"] == "cnn-dailymail"
        fields = ("article", "highlights", "id") if cnn else ("text", "id", "url", "date", "dump", "language", "language_score", "file_path")
        with parquet.ParquetFile(path) as file:
            require(set(fields) <= set(file.schema_arrow.names), "Parquet schema missing required columns: " + ",".join(fields))
            for group in range(file.num_row_groups):
                position = 0
                for batch in file.iter_batches(batch_size=64, row_groups=[group], columns=list(fields), use_threads=False):
                    for raw in batch.to_pylist():
                        self.scanned(source)
                        locator = {"format": "parquet", "row_group": group, "row_in_group": position}
                        position += 1
                        if not cnn:
                            require(raw.get("dump") == "CC-MAIN-2018-30", "FineWeb row belongs to a different crawl")
                        with self.record():
                            record_require(all(raw.get(k) is not None for k in fields), "null_field")
                            record_require(sum(len(value) for value in raw.values() if isinstance(value, str)) <= MAX_RECORD, "record_size")
                            for key in fields:
                                if key not in ("date", "language_score"):
                                    text_field(raw[key])
                            if cnn:
                                record_require(re.fullmatch(r"[0-9a-fA-F]{40}", raw["id"]), "document_id")
                                row = self.base(source, entry, locator, "cnn-dailymail:url-sha1:" + raw["id"].lower(), "2015-04-30",
                                                "CNN/DailyMail catalog release cutoff April 2015; month-end upper bound, not an individual article timestamp")
                                row.update(domain="news", source_split="train" if PurePosixPath(entry["catalog_path"]).name.startswith("train-") else "valid", url_sha1=raw["id"], source_document_id=raw["id"])
                                brief = clean(raw["highlights"])
                                if brief:
                                    row.update(brief=brief, brief_origin="source highlights; provisional human summary")
                                self.add(self.original(row, raw["article"]))
                            else:
                                record_require(raw["id"].strip() and raw["file_path"].strip(), "document_id")
                                capture = timestamp(raw["date"])
                                record_require(type(raw["language_score"]) in (int, float) and 0 <= raw["language_score"] <= 1 and math.isfinite(raw["language_score"]), "language_score")
                                if capture >= datetime(2019, 1, 1, tzinfo=timezone.utc):
                                    self.counts["excluded:fineweb_capture"] += 1
                                    continue
                                if raw["language"] != "en" or raw["language_score"] < 0.85:
                                    self.counts["excluded:fineweb_language"] += 1
                                    continue
                                url, host = canonical_url(raw["url"])
                                if overlap_host(host):
                                    self.counts["excluded:fineweb_source_overlap"] += 1
                                    continue
                                domain = fineweb_domain(url, raw["text"], self.counts)
                                if domain is None:
                                    self.counts["excluded:fineweb_domain"] += 1
                                    continue
                                self.counts["domain_selected:" + domain] += 1
                                row = self.base(source, entry, locator, "fineweb:cc:" + raw["id"], capture.date().isoformat(), "Common Crawl capture timestamp from the named CC-MAIN-2018-30 shard; not a publication timestamp")
                                row.update(domain=domain, source_url=raw["url"], canonical_url=url, capture_timestamp=capture.isoformat(),
                                           crawl_dump=raw["dump"], warc_path=raw["file_path"], source_document_id=raw["id"],
                                           source_language=raw["language"], source_language_score=raw["language_score"],
                                           cluster_ids=["web:url:" + url, *web_clusters(url, host)])
                                self.add(self.original(row, raw["text"]))

    def reviews(self, source, entry, path):
        required = ("product/productId", "review/userId", "review/profileName", "review/time", "review/summary", "review/text")
        for raw, locator in review_records(path, self.budget):
            self.scanned(source)
            with self.record():
                if isinstance(raw, RecordRejected):
                    raise raw
                record_require(all(key in raw for key in required), "missing_field")
                for value in raw.values():
                    text_field(value)
                record_require(raw["product/productId"].strip(), "document_id")
                record_require(re.fullmatch(r"\d+", raw["review/time"]) is not None, "timestamp")
                try:
                    stamp = datetime.fromtimestamp(int(raw["review/time"]), timezone.utc)
                except ImportFailure:
                    raise
                except (ValueError, OverflowError, OSError):
                    raise RecordRejected("timestamp") from None
                if stamp > datetime(2013, 3, 31, 23, 59, 59, tzinfo=timezone.utc):
                    self.counts["excluded:review_date"] += 1
                    continue
                document = f"snap-amazon-2013:{entry['sha256']}:{locator['record']}"
                row = self.base(source, entry, locator, document, "2013-03-31", "SNAP 2013 dataset capture through March 2013; month-end cutoff, not the individual capture timestamp; source review timestamp checked separately")
                row.update(domain="reviews", source_timestamp=stamp.isoformat(), product_id=raw["product/productId"],
                           source_user_id=raw["review/userId"], source_review_metadata={k: v for k, v in raw.items() if k != "review/text"})
                author = raw["review/userId"].strip()
                if author.casefold() not in ("", "unknown", "unknownuser", "anonymous", "deleted", "none", "null", "0", "-1"):
                    row.update(author_id="snap-amazon-2013:reviewer:" + author, author_identity_status="source reviewer ID; not verified identity",
                               cluster_id="snap-amazon-2013:review:" + sha256(pack([author, raw["product/productId"], raw["review/time"]])))
                if raw["review/profileName"].strip():
                    row["attribution_name"] = raw["review/profileName"]
                summary = clean(raw["review/summary"], True)
                if summary:
                    row["headline"] = summary
                self.add(self.original(row, raw["review/text"], True))

    def stackexchange(self, source, entry, path):
        site = PurePosixPath(entry["catalog_path"]).name.removesuffix(".7z")
        directory = self.args.out_dir / "xml" / site
        directory.mkdir(parents=True, exist_ok=False)
        sizes = extract_xml(path, directory, self.budget, MAX_EXTRACTED_BYTES - self.extracted)
        self.extracted += sum(sizes.values())
        self.audit.setdefault("extracted", []).append({"source_file": entry["local_path"], "members": sizes})
        for raw, position in xml_rows(directory / "PostLinks.xml", "postlinks", self.budget):
            with self.record("stackexchange_links:"):
                record_require(positive_id(raw.get("LinkTypeId")), "link_type")
                if raw["LinkTypeId"] == "3":
                    record_require(all(positive_id(raw.get(k)) for k in ("PostId", "RelatedPostId")), "document_id")
                    self.threads.join([f"stackexchange:{site}:thread:{raw[k]}" for k in ("PostId", "RelatedPostId")])
                    self.counts["stackexchange:duplicate_links"] += 1
        for raw, position in xml_rows(directory / "Posts.xml", "posts", self.budget):
            with self.record("stackexchange_edges:"):
                record_require(positive_id(raw.get("PostTypeId")), "post_type")
                if raw["PostTypeId"] == "2":
                    record_require(positive_id(raw.get("Id")) and positive_id(raw.get("ParentId")), "document_id")
                    self.threads.join([f"stackexchange:{site}:thread:{raw[k]}" for k in ("Id", "ParentId")])
                    self.counts["stackexchange:answer_parent_links"] += 1
        self.db.commit()
        for raw, position in xml_rows(directory / "Users.xml", "users", self.budget):
            with self.record("stackexchange_users:"):
                record_require(isinstance(raw.get("Id"), str) and re.fullmatch(r"-?\d+", raw["Id"]), "author_id")
                if positive_id(raw["Id"]):
                    for key in ("AccountId", "DisplayName"):
                        if key in raw:
                            text_field(raw[key])
                    record_require("AccountId" not in raw or re.fullmatch(r"-?\d+", raw["AccountId"]), "author_id")
                    self.db.execute("INSERT INTO users VALUES (?, ?, ?, ?)", (site, raw["Id"], raw.get("AccountId") if positive_id(raw.get("AccountId")) else None, raw.get("DisplayName")))
        for raw, position in xml_rows(directory / "Posts.xml", "posts", self.budget):
            self.scanned(source)
            with self.record():
                record_require(positive_id(raw.get("PostTypeId")), "post_type")
                if raw["PostTypeId"] not in ("1", "2"):
                    continue
                record_require(positive_id(raw.get("Id")), "document_id")
                record_require(all(k in raw for k in ("Body", "CreationDate")), "missing_field")
                record_require(sum(len(value) for value in raw.values() if isinstance(value, str)) <= MAX_RECORD, "record_size")
                for value in raw.values():
                    text_field(value)
                for key in ("OwnerUserId", "LastEditorUserId"):
                    record_require(key not in raw or re.fullmatch(r"-?\d+", raw[key]), "author_id")
                if any(timestamp(raw[k]) >= datetime(2019, 1, 1, tzinfo=timezone.utc) for k in ("CreationDate", "LastEditDate", "LastActivityDate") if k in raw):
                    self.counts["excluded:post_date"] += 1
                    continue
                parent = raw["Id"] if raw["PostTypeId"] == "1" else raw.get("ParentId")
                record_require(positive_id(parent), "document_id")
                thread = self.threads.root(f"stackexchange:{site}:thread:{parent}")
                locator = {"format": "7z/xml", "member": "Posts.xml", "row": position, "post_id": raw["Id"]}
                row = self.base(source, entry, locator, f"stackexchange:{site}:post:{raw['Id']}", "2018-12-31", "Catalog December 2018 Stack Exchange snapshot; month-end upper bound, not an individual capture timestamp; content dates checked")
                row.update(domain="qa", thread_id=thread, source_thread_id=parent, source_post_id=raw["Id"],
                           source_url=f"https://{site}/" + ("questions/" if raw["PostTypeId"] == "1" else "a/") + raw["Id"],
                           source_creation_date=raw["CreationDate"], source_post_type=raw["PostTypeId"])
                for key in ("LastEditDate", "LastActivityDate", "OwnerUserId", "OwnerDisplayName", "LastEditorUserId", "LastEditorDisplayName", "Title", "Tags", "ContentLicense"):
                    if key in raw:
                        row.setdefault("post_metadata", {})[key] = raw[key]
                if "Title" in raw:
                    headline = clean(raw["Title"])
                    if headline:
                        row["headline"] = headline
                row = self.original(row, raw["Body"], True)
                if row:
                    self.db.execute("INSERT INTO posts VALUES (?, ?, ?) ON CONFLICT(thread) DO UPDATE SET rank=excluded.rank, payload=excluded.payload WHERE excluded.rank < posts.rank",
                                    (thread, rank(row["document_id"]), pack(row)))
        self.db.commit()
        for (payload,) in self.db.execute("SELECT payload FROM posts ORDER BY thread"):
            self.counts["stackexchange:selected_threads_before_authors"] += 1
            row = json.loads(payload)
            metadata = row.get("post_metadata", {})
            owner = metadata.get("OwnerUserId")
            name = metadata.get("OwnerDisplayName")
            if positive_id(owner):
                user = self.db.execute("SELECT account, name FROM users WHERE site=? AND id=?", (site, owner)).fetchone()
                account = user[0] if user else None
                row.update(author_id="stackexchange:account:" + account if account else f"stackexchange:{site}:user:{owner}",
                           author_identity_status="source account ID" if account else "source site user ID")
                if user:
                    row["attribution_locator"] = {"member": "Users.xml", "user_id": owner}
                if account:
                    row["source_account_id"] = account
                name = (user[1] if user else None) or name
            if name:
                row["attribution_name"] = name
            self.add(row)
        self.db.execute("DELETE FROM posts")
        self.db.commit()

    def select(self, quotas, factor):
        heaps = {(split, domain): [] for split in SPLITS for domain in DOMAINS}
        available = Counter()
        for root, mask, payload in self.db.execute("SELECT root, mask, payload FROM components"):
            self.budget.check()
            if mask == 3:
                self.counts["excluded:official_split_conflict_components"] += 1
                continue
            number = int(rank("assignment:" + root), 16)
            if mask == 1:
                split = "train"
            elif mask == 2:
                split = SPLITS[1 + number % 2]
            else:
                split = "train" if number % 12 < 10 else SPLITS[number % 12 - 9]
            row = json.loads(payload)
            row.update(group="source-import:" + sha256(root), split=split)
            source_check(row)
            key = split, row["domain"]
            available[key] += 1
            limit = math.ceil(quotas[split][row["domain"]] * factor)
            heap = heaps[key]
            item = (-int(rank("candidate:" + root), 16), row["id"], pack(row))
            if len(heap) < limit:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)
        selected = [json.loads(item[2]) for key in sorted(heaps) for item in sorted(heaps[key], reverse=True)]
        check_isolation(selected)
        self.audit["yields"] = {split: {domain: {"eligible_groups": available[split, domain], "candidate_target": math.ceil(quotas[split][domain] * factor),
                                          "selected_candidates": len(heaps[split, domain]), "parent_quota": quotas[split][domain]} for domain in DOMAINS} for split in SPLITS}
        self.audit["shortfalls"] = {f"{split}:{domain}": math.ceil(quotas[split][domain] * factor) - len(heaps[split, domain]) for split in SPLITS for domain in DOMAINS if len(heaps[split, domain]) < math.ceil(quotas[split][domain] * factor)}
        return selected


def timestamp(value):
    try:
        if isinstance(value, datetime):
            result = value
        elif isinstance(value, str):
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            raise ValueError
        return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)
    except ImportFailure:
        raise
    except (ValueError, OverflowError, OSError):
        raise RecordRejected("timestamp") from None


def positive_id(value):
    return isinstance(value, str) and re.fullmatch(r"[1-9]\d*", value) is not None


def review_records(path, budget):
    with gzip.open(path, "rb") as stream:
        raw, key, index, start, line_number, first_line, size = {}, None, 0, 0, 0, 1, 0
        rejected = None
        while True:
            budget.check()
            offset = stream.tell()
            line = stream.readline(MAX_RECORD + 1)
            line_number += 1
            oversized = len(line) > MAX_RECORD
            if not line or (not oversized and not line.strip()):
                if size:
                    yield rejected if rejected is not None else raw, {"format": "gzip/plaintext", "record": index, "uncompressed_byte_offset": start, "line": first_line}
                    index += 1
                raw, key, size, rejected = {}, None, 0, None
                if not line:
                    break
                continue
            if not size:
                require(re.match(rb"^(?:product|review)/[A-Za-z]+:", line) is not None, "SNAP continuation without a field")
                start, first_line = offset, line_number
            size += len(line)
            if oversized:
                tail = line
                while tail and not tail.endswith(b"\n"):
                    budget.check()
                    tail = stream.readline(MAX_RECORD + 1)
                rejected = rejected or RecordRejected("record_size")
            if size > MAX_RECORD:
                rejected = rejected or RecordRejected("record_size")
            if rejected is not None:
                raw.clear()
                continue
            try:
                text = line.decode("utf-8").rstrip("\r\n")
            except UnicodeDecodeError:
                rejected = RecordRejected("unicode")
                raw.clear()
                continue
            match = re.match(r"^((?:product|review)/[A-Za-z]+):\s?(.*)$", text)
            if match:
                key, value = match.groups()
                require(key not in raw, "Duplicate SNAP field without record delimiter")
                raw[key] = value
            else:
                require(key is not None, "SNAP continuation without a field")
                raw[key] += "\n" + text


class XMLInput:
    def __init__(self, stream, budget):
        self.stream, self.budget, self.tail = stream, budget, b""

    def read(self, size=-1):
        self.budget.check()
        block = self.stream.read(min(size, 65536) if size >= 0 else 65536)
        checked = (self.tail + block).upper()
        require(b"\x00" not in block and b"<!DOCTYPE" not in checked and b"<!ENTITY" not in checked, "XML DTD, entity declaration, or NUL encoding prohibited")
        self.tail = checked[-16:]
        return block


def xml_rows(path, expected_root, budget):
    with path.open("rb") as stream:
        events = ET.iterparse(XMLInput(stream, budget), events=("start", "end"))
        event, root = next(events)
        require(event == "start" and root.tag.lower() == expected_root, "Unexpected XML root")
        position, depth = 0, 0
        for event, element in events:
            if event == "start":
                depth += 1
                require(depth == 1 and element.tag == "row", "Unexpected XML element")
            elif element is not root:
                require(not (root.text or "").strip() and not (element.text or "").strip() and not (element.tail or "").strip(), "Unexpected XML row content")
                values = dict(element.attrib)
                element.clear()
                root.clear()
                depth -= 1
                yield values, position
                position += 1
            else:
                require(depth == 0 and not (root.text or "").strip(), "Unexpected XML root content")


def extract_xml(path, directory, budget, remaining):
    py7zr = reader("py7zr")
    with py7zr.SevenZipFile(path, "r") as archive:
        require(not archive.needs_password(), "Encrypted Stack Exchange archive prohibited")
        members, total, targets, names = archive.list(), 0, {}, set()
        for member in members:
            name = member.filename
            require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", name) and name not in (".", "..") and name not in names, "Unsafe or duplicate archive member path")
            require(member.is_file and not member.is_directory and not member.is_symlink, "Archive links and nonregular members prohibited")
            require(type(member.uncompressed) is int and member.uncompressed >= 0, "Unknown archive member size")
            names.add(name)
            total += member.uncompressed
            if name in ("Posts.xml", "PostLinks.xml", "Users.xml"):
                targets[name] = member.uncompressed
        require(total <= MAX_ARCHIVE_BYTES, "Archive uncompressed size exceeds limit")
        require({"Posts.xml", "PostLinks.xml", "Users.xml"} == set(targets), "Archive missing Posts.xml, PostLinks.xml, or Users.xml")
        require(sum(targets.values()) <= min(MAX_TARGET_BYTES, remaining), "XML extraction exceeds target size budget")
        require(shutil.disk_usage(directory).free >= 30 * GIB + sum(targets.values()), "Insufficient disk for bounded extraction")

        class Output(py7zr.Py7zIO):
            def __init__(self, name):
                self.limit = targets[name]
                self.file = (directory / name).open("x+b")
                self.length = 0

            def write(self, data):
                budget.check()
                require(self.file.tell() + len(data) <= self.limit, "XML exceeds declared member size")
                count = self.file.write(data)
                self.length = max(self.length, self.file.tell())
                return count

            def read(self, size=None):
                return self.file.read(-1 if size is None else size)

            def seek(self, offset, whence=0):
                position = self.file.seek(offset, whence)
                require(0 <= position <= self.limit, "XML seek exceeds declared member size")
                return position

            def flush(self):
                self.file.flush()

            def size(self):
                return self.length

        class Factory(py7zr.WriterFactory):
            def __init__(self):
                self.outputs = {}

            def create(self, filename):
                require(filename in targets and filename not in self.outputs, "Unexpected extraction target")
                self.outputs[filename] = Output(filename)
                return self.outputs[filename]

        factory = Factory()
        try:
            archive.extract(targets=list(targets), factory=factory)
            require(set(factory.outputs) == set(targets), "Missing extracted member")
            require(all(output.size() == targets[name] for name, output in factory.outputs.items()), "Extracted member size mismatch")
        finally:
            for output in factory.outputs.values():
                output.file.close()
        budget.check()
        return targets


def check_isolation(rows):
    ids, groups, texts, identities = set(), set(), set(), {}
    for row in rows:
        source_check(row)
        require(row["id"] not in ids and row["group"] not in groups, "Repeated selected original or canonical group")
        ids.add(row["id"])
        groups.add(row["group"])
        key = sha256(normalize(row["text"]).casefold())
        require(key not in texts, "Duplicate selected excerpt")
        texts.add(key)
        for identity in provenance_identities(row):
            require(identity not in identities or identities[identity] == row["split"], "Selected source identities cross splits")
            identities[identity] = row["split"]


def main():
    parser = argparse.ArgumentParser(description="Import approved historical raw sources into provisional human candidates. No models or network access.")
    parser.add_argument("--acquired-dir", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--quotas", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--minutes", type=float, required=True)
    parser.add_argument("--deadline-utc", help="Shared preparation deadline, ISO 8601 UTC")
    parser.add_argument("--candidate-factor", type=float, default=2)
    args = parser.parse_args()
    audit = {"stage": "import-failed", "files": [], "created_at": datetime.now(timezone.utc).isoformat(), "policy": POLICY,
             "minutes": args.minutes, "shared_deadline_utc": args.deadline_utc, "candidate_factor": args.candidate_factor}
    importer, budget, owned, previous = None, None, False, {}
    try:
        require(math.isfinite(args.candidate_factor) and 1 <= args.candidate_factor <= 4, "--candidate-factor must be between 1 and 4")
        require(not args.out_dir.exists(), "Output directory must be new")
        budget = Budget(args)
        args.out_dir.mkdir(parents=True, exist_ok=False)
        owned = True
        budget.check()
        audit["effective_deadline_utc"] = budget.deadline_utc
        audit["implementation_sha256"] = hash_file(__file__, budget)["sha256"]
        audit["corpus_implementation_sha256"] = hash_file(Path(__file__).with_name("corpus.py"), budget)["sha256"]

        def stop(signum, frame):
            raise ImportFailure("Preparation deadline reached" if signum == signal.SIGALRM else "Import interrupted")

        for sig in (signal.SIGALRM, signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, budget.deadline - time.monotonic()))
        catalog, quotas, files = inventory(args, budget, audit)
        importer = Importer(args, budget, catalog, audit)
        for source, entry, path, before in files:
            audit["current_file"] = {"source_id": source["id"], "catalog_path": entry["catalog_path"], "operation": "parse"}
            before_counts = importer.counts.copy()
            if source["id"] in ("cnn-dailymail", "fineweb-2018-30"):
                importer.parquet(source, entry, path)
            elif source["id"] == "snap-amazon-2013":
                importer.reviews(source, entry, path)
            else:
                importer.stackexchange(source, entry, path)
            require(file_state(path) == before, "Raw source changed during import")
            entry["scan_counts"] = dict(sorted((importer.counts - before_counts).items()))
            importer.db.commit()
        selected = importer.select(quotas, args.candidate_factor)
        require(all(file_state(path) == before for _, _, path, before in files), "Raw sources changed before output")
        audit["stage"] = "import-shortfall" if audit["shortfalls"] else "imported"
        output_files = []
        if not audit["shortfalls"]:
            output = args.out_dir / "source.jsonl.pending"
            with output.open("x", encoding="utf-8") as stream:
                for row in selected:
                    budget.check()
                    stream.write(pack(row) + "\n")
            output_files.append({"path": "source.jsonl", "sha256": hash_file(output, budget)["sha256"], "records": len(selected), "bytes": output.stat().st_size})
            output.rename(args.out_dir / "source.jsonl")
        audit["output_files"] = output_files
        audit["source_group_isolation_checked"] = True
    except Exception as exc:
        audit.update(stage="import-failed", error=str(exc) if isinstance(exc, ImportFailure) else "Reader or schema failure: " + type(exc).__name__,
                     error_locations=[{"file": frame.filename, "line": frame.lineno, "function": frame.name} for frame in traceback.extract_tb(exc.__traceback__)])
    finally:
        if previous:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        if importer:
            audit["counts"] = dict(sorted(importer.counts.items()))
            importer.db.commit()
            importer.db.close()
        if budget:
            audit["seconds"] = time.monotonic() - budget.started
        if owned:
            write_json(args.out_dir / "audit.json", audit)
            write_json(args.out_dir / "manifest.json", {"workflow": "clefsift-general-v1", "version": 1, "stage": audit["stage"],
                       "catalog_sha256": audit.get("catalog_sha256"), "quotas_sha256": audit.get("quotas_sha256"),
                       "acquired_manifest_sha256": audit.get("acquired_manifest_sha256"), "policy_sha256": sha256(pack(POLICY)),
                       "audit_sha256": hash_file(args.out_dir / "audit.json")["sha256"], "files": audit.get("output_files", []),
                       "candidate_factor": args.candidate_factor, "generation_parent_limit": 26400,
                       "provisional_rights_and_authorship": True, "generation_performed": False, "final_scoring_performed": False})
    summary = {k: audit[k] for k in ("stage", "error", "yields", "shortfalls", "counts", "seconds") if k in audit}
    print(pack(summary))
    return 0 if audit["stage"] == "imported" else 1


if __name__ == "__main__":
    sys.exit(main())
