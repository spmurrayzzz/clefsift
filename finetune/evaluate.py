import argparse
from collections import Counter
from http.client import HTTPException
import json
import math
import os
from pathlib import Path
import urllib.request
import uuid

from common import LABELS, QUESTION, NoRedirect, api_endpoint, read_jsonl, file_hash, write_json


THREE_TARGETS = ("human", "mixed", "ai")


def index_records(records):
    indexed = {}
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"Invalid dataset record at row {index}")
        record_id, labels = record.get("id"), record.get("labels")
        if not isinstance(record_id, str) or not record_id.strip() or record_id in indexed:
            raise ValueError(f"Invalid or duplicate dataset ID at row {index}")
        if not isinstance(labels, dict) or set(labels) != {"classification"} or labels["classification"] not in LABELS:
            raise ValueError(f"Invalid dataset label at row {index}")
        indexed[record_id] = record
    if not indexed:
        raise ValueError("Dataset is empty")
    return indexed


def canonical_row(record, probabilities, choice=None):
    if not isinstance(probabilities, dict) or set(probabilities) != set(LABELS):
        raise ValueError("Probabilities must contain exactly the four labels")
    if any(type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value) for value in probabilities.values()):
        raise ValueError("Probabilities must be finite numbers between zero and one")
    if not math.isclose(sum(probabilities.values()), 1, rel_tol=0, abs_tol=0.0002):
        raise ValueError("Probabilities do not sum to one within four-decimal rounding")
    if choice is None:
        choice = max(probabilities, key=probabilities.get)
    if choice not in LABELS or probabilities[choice] != max(probabilities.values()):
        raise ValueError("Choice differs from probability argmax")
    target = record["labels"]["classification"]
    return {"id": record["id"], "target": target,
            "prediction": choice, "probabilities": probabilities,
            "brier": sum((probabilities[label] - (label == target)) ** 2 for label in LABELS),
            "nll": -math.log(max(probabilities[target], 1e-12))}


def load_predictions(records, path):
    indexed = index_records(records)
    raw = read_jsonl(Path(path))
    if len(raw) != len(indexed):
        raise ValueError("Prediction coverage mismatch")
    result, seen = [], set()
    for index, row in enumerate(raw, 1):
        if not isinstance(row, dict):
            raise ValueError(f"Invalid prediction at row {index}")
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or record_id not in indexed or record_id in seen:
            raise ValueError(f"Unknown or duplicate prediction ID at row {index}")
        options, probabilities, target = row.get("option_ids"), row.get("probs"), row.get("target")
        if row.get("question_id") != "classification" or row.get("type") != "choice":
            raise ValueError(f"Invalid prediction question or type at row {index}")
        if not isinstance(options, list) or len(options) != 4 or any(not isinstance(label, str) for label in options) or set(options) != set(LABELS):
            raise ValueError(f"Invalid prediction options at row {index}")
        if not isinstance(probabilities, list) or len(probabilities) != 4:
            raise ValueError(f"Invalid probability vector at row {index}")
        if type(target) is not int or not 0 <= target < 4:
            raise ValueError(f"Invalid target index at row {index}")
        if options[target] != indexed[record_id]["labels"]["classification"]:
            raise ValueError(f"Prediction target mismatch at row {index}")
        result.append(canonical_row(indexed[record_id], dict(zip(options, probabilities)), row.get("choice")))
        seen.add(record_id)
    return sorted(result, key=lambda row: row["id"])


def metrics(predictions):
    count = len(predictions)
    if not count:
        raise ValueError("Predictions are empty")
    confusion = Counter((row["target"], row["prediction"]) for row in predictions)
    classes = {}
    for label in LABELS:
        support = sum(value for (target, prediction), value in confusion.items() if target == label)
        predicted = sum(value for (target, prediction), value in confusion.items() if prediction == label)
        correct = confusion[label, label]
        classes[label] = {"support": support, "predicted": predicted, "correct": correct,
                          "precision": correct / predicted if predicted else None,
                          "recall": correct / support if support else None,
                          "f1": 2 * correct / (support + predicted) if support + predicted else 0}
    human_errors = classes["human"]["support"] - classes["human"]["correct"]
    return {"n": count, "accuracy": sum(row["target"] == row["prediction"] for row in predictions) / count,
            "macro_f1_fixed_four": sum(classes[label]["f1"] for label in LABELS) / 4,
            "macro_f1_fixed_three": sum(classes[label]["f1"] for label in THREE_TARGETS) / 3,
            "human_false_positives": human_errors,
            "human_false_positive_rate": human_errors / classes["human"]["support"] if classes["human"]["support"] else None,
            "multiclass_brier": sum(row["brier"] for row in predictions) / count,
            "nll": sum(row["nll"] for row in predictions) / count, "per_class": classes,
            "confusion": {target: {prediction: confusion[target, prediction] for prediction in LABELS} for target in LABELS}}


def verdict(row):
    probabilities = sorted(row["probabilities"].values(), reverse=True)
    return row["prediction"] if probabilities[0] >= 0.35 and probabilities[0] - probabilities[1] >= 0.1 else "uncertain"


def compare(reference, candidate):
    reference, candidate = (sorted(rows, key=lambda row: row["id"]) for rows in (reference, candidate))
    if not reference or [(row["id"], row["target"]) for row in reference] != [(row["id"], row["target"]) for row in candidate]:
        raise ValueError("Comparison coverage or targets differ")
    pairs = list(zip(reference, candidate))
    differences = [abs(before["probabilities"][label] - after["probabilities"][label]) for before, after in pairs for label in LABELS]
    changed = [{"id": before["id"], "target": before["target"], "reference": before["prediction"], "candidate": after["prediction"],
                "reference_probabilities": before["probabilities"], "candidate_probabilities": after["probabilities"]}
               for before, after in pairs if before["prediction"] != after["prediction"]]
    display = [(before["id"], verdict(before), verdict(after)) for before, after in pairs]
    before_metrics, after_metrics = metrics(reference), metrics(candidate)
    keys = ("accuracy", "macro_f1_fixed_four", "macro_f1_fixed_three", "human_false_positives", "human_false_positive_rate", "multiclass_brier", "nll")
    return {"n": len(reference), "probability_changes_are_diagnostic": True,
            "argmax_agreement": (len(reference) - len(changed)) / len(reference),
            "argmax_agreement_count": len(reference) - len(changed),
            "drift": {"changed_choice_rate": len(changed) / len(reference),
                      "mean_absolute_probability_difference": sum(differences) / len(differences),
                      "maximum_absolute_probability_difference": max(differences)},
            "reference_metrics": before_metrics, "candidate_metrics": after_metrics,
            "metric_deltas": {key: after_metrics[key] - before_metrics[key] if before_metrics[key] is not None and after_metrics[key] is not None else None for key in keys},
            "changed_choices": changed,
            "changed_display_verdicts": [record_id for record_id, before, after in display if before != after],
            "display_uncertainty": {"reference_uncertain": sum(before == "uncertain" for _, before, after in display),
                                    "candidate_uncertain": sum(after == "uncertain" for _, before, after in display),
                                    "became_uncertain": [record_id for record_id, before, after in display if before != "uncertain" and after == "uncertain"],
                                    "became_certain": [record_id for record_id, before, after in display if before == "uncertain" and after != "uncertain"]},
            "largest_probability_changes": sorted(
                [{"id": before["id"], "maximum_difference": max(abs(before["probabilities"][label] - after["probabilities"][label]) for label in LABELS)} for before, after in pairs],
                key=lambda row: row["maximum_difference"], reverse=True)[:10]}


def unique_object(pairs):
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError("Duplicate JSON keys")
    return value


def run_http(args, records):
    indexed = index_records(records)
    if args.max_requests < 1 or len(indexed) > args.max_requests:
        raise ValueError("Dataset exceeds --max-requests, or the bound is not positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("Timeout must be finite and positive")
    if not args.model.strip() or not args.api_key_env.strip():
        raise ValueError("Model and API key environment name must be nonempty")
    url = api_endpoint(args.base_url, "systemone")
    for index, record in enumerate(indexed.values(), 1):
        if record.get("questions") != {"classification": QUESTION} or not isinstance(record.get("state"), str) or not record["state"].strip():
            raise ValueError(f"Invalid HTTP request data at row {index}")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    api_key = os.environ.get(args.api_key_env, "")
    if "\r" in api_key or "\n" in api_key:
        raise ValueError("Invalid API key header")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data_hash = file_hash(args.data)
    args.out_dir.mkdir(parents=True)
    opener = urllib.request.build_opener(NoRedirect())
    predictions = []
    observed_models = set()
    options = sorted(LABELS)
    output = args.out_dir / "tuned_predictions.jsonl"
    output.open("x").close()
    for index, record in enumerate(indexed.values(), 1):
        body = {"model": args.model, "state": record["state"], "questions": record["questions"]}
        request = urllib.request.Request(url, data=json.dumps(body, allow_nan=False).encode(),
                                         headers={**headers, "Idempotency-Key": str(uuid.uuid4())}, method="POST")
        try:
            with opener.open(request, timeout=args.timeout) as response:
                if response.status != 200:
                    raise ValueError("Unexpected HTTP status")
                raw = json.loads(response.read(), object_pairs_hook=unique_object)
        except (OSError, ValueError, HTTPException):
            raise ValueError(f"Request {index}: HTTP or JSON failure") from None
        observed_model = raw.get("model") if isinstance(raw, dict) else None
        if observed_model is not None:
            if observed_model != args.model:
                raise ValueError(f"Request {index}: server returned a different model")
            observed_models.add(observed_model)
        answers = raw.get("answers") if isinstance(raw, dict) else None
        if not isinstance(answers, dict) or set(answers) != {"classification"}:
            raise ValueError(f"Request {index}: invalid answers")
        answer = answers["classification"]
        if not isinstance(answer, dict) or answer.get("type") != "choice" or answer.get("choice") not in LABELS:
            raise ValueError(f"Request {index}: invalid answer type or choice")
        row = canonical_row(record, answer.get("probabilities"), answer["choice"])
        predictions.append(row)
        saved = {"record_id": record["id"], "question_id": "classification", "type": "choice",
                 "option_ids": options, "probs": [row["probabilities"][label] for label in options],
                 "target": options.index(row["target"]), "choice": row["prediction"], "model": observed_model}
        with output.open("a") as destination:
            destination.write(json.dumps(saved, allow_nan=False) + "\n")
    if file_hash(args.data) != data_hash:
        raise ValueError("Dataset changed during evaluation")
    write_json(args.out_dir / "metrics.json", {"data_sha256": data_hash, "predictions_sha256": file_hash(output),
                                               "model": args.model, "observed_models": sorted(observed_models),
                                               "max_requests": args.max_requests, **metrics(predictions)})


def main():
    parser = argparse.ArgumentParser(description="Evaluate four-label classification predictions.")
    commands = parser.add_subparsers(dest="command", required=True)
    summarize = commands.add_parser("summarize", help="Summarize toolkit tuned_predictions.jsonl.")
    http = commands.add_parser("http", help="Evaluate bounded, sequential SystemOne requests.")
    comparison = commands.add_parser("compare", help="Compare predictions on identical requests; drift is diagnostic.")
    for command in (summarize, http, comparison):
        command.add_argument("--data", type=Path, required=True, help="Labeled toolkit request JSONL.")
    summarize.add_argument("--predictions", type=Path, required=True)
    summarize.add_argument("--out", type=Path, required=True)
    http.add_argument("--base-url", required=True, help="API base URL including /v1; HTTP only on localhost or loopback.")
    http.add_argument("--model", required=True)
    http.add_argument("--out-dir", type=Path, required=True, help="New output directory; must not exist.")
    http.add_argument("--max-requests", type=int, required=True, help="Positive bound; oversized datasets fail before requests.")
    http.add_argument("--timeout", type=float, default=120, help="Timeout per request in seconds (default: 120).")
    http.add_argument("--api-key-env", default="CLEFSIFT_API_KEY", help="Optional Bearer key environment variable (default: CLEFSIFT_API_KEY).")
    comparison.add_argument("--reference", type=Path, required=True)
    comparison.add_argument("--candidate", type=Path, required=True)
    comparison.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        records = read_jsonl(args.data)
        if args.command == "http":
            run_http(args, records)
            return
        if args.out.exists() or args.out.is_symlink():
            raise FileExistsError(f"Output already exists: {args.out}")
        result = {"data_sha256": file_hash(args.data)}
        if args.command == "summarize":
            result.update(metrics(load_predictions(records, args.predictions)))
            result["predictions_sha256"] = file_hash(args.predictions)
        else:
            result.update(compare(load_predictions(records, args.reference), load_predictions(records, args.candidate)))
            result.update(reference_sha256=file_hash(args.reference), candidate_sha256=file_hash(args.candidate))
        write_json(args.out, result)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
