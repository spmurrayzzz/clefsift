import math
from pathlib import Path

from common import LABELS, file_hash, load_schema
from evaluate import canonical_row, metrics


CHECKPOINT_FILES = ("adapter_config.json", "adapter_model.safetensors", "joint_head.safetensors",
                    "joint_head_config.json", "unsloth_decision_config.json", "trainer_state.json")


def checkpoint_hashes(directory):
    directory = Path(directory)
    if any(not (directory / name).is_file() for name in CHECKPOINT_FILES):
        raise ValueError(f"Incomplete Unsloth checkpoint: {directory}")
    return {name: file_hash(directory / name) for name in CHECKPOINT_FILES}


def adapter_directory(checkpoint):
    checkpoint = Path(checkpoint)
    return checkpoint / "adapter" if (checkpoint / "adapter").is_dir() else checkpoint


def build_items(processor, records, model_path):
    tokenizer = getattr(processor, "tokenizer", processor)
    encoder = load_schema(model_path)
    items = []
    for index, row in enumerate(records):
        encoded = encoder.encode_record(tokenizer, row, max_length=1000000)
        if len(encoded.input_ids) > 1536 or len(encoded.input_ids) != row["metadata"]["input_tokens"]:
            raise ValueError(f"Full-request token count mismatch: {row['id']}")
        if (len(encoded.questions) != 1 or encoded.questions[0].question_id != "classification"
                or encoded.questions[0].option_ids != tuple(sorted(LABELS))):
            raise ValueError(f"Encoded classification question mismatch: {row['id']}")
        question = encoded.questions[0]
        label = question.option_ids.index(row["labels"]["classification"])
        items.append({"input_ids": list(encoded.input_ids), "record": encoded,
                      "qtypes": [question.question_type], "labels": [label], "row": index,
                      "targets": [[float(option == label) for option in range(len(question.option_ids))]],
                      "source": {"state": row["state"], "questions": row["questions"]}})
    return items


def summarize_logits(records, logits):
    options = sorted(LABELS)
    if len(logits) != len(records):
        raise ValueError("Prediction coverage mismatch")
    raw, canonical = [], []
    for row, values in zip(records, logits):
        values = [float(value) for value in values]
        if len(values) != len(options) or any(not math.isfinite(value) for value in values):
            raise ValueError(f"Invalid logits: {row['id']}")
        weights = [math.exp(value - max(values)) for value in values]
        total = sum(weights)
        probabilities = dict(zip(options, (weight / total for weight in weights)))
        result = canonical_row(row, probabilities)
        canonical.append(result)
        raw.append({"record_id": row["id"], "question_id": "classification", "type": "choice",
                    "option_ids": options, "probs": [probabilities[label] for label in options],
                    "target": options.index(row["labels"]["classification"]), "choice": result["prediction"]})
    return raw, metrics(canonical)


def load_checkpoint(model, checkpoint):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    checkpoint = Path(checkpoint)
    result = set_peft_model_state_dict(model.encoder, load_file(str(checkpoint / "adapter_model.safetensors")))
    if result.unexpected_keys or any("lora_" in name for name in result.missing_keys):
        raise ValueError("Checkpoint adapter does not match the training model")
    model.head.load_state_dict(load_file(str(checkpoint / "joint_head.safetensors")), strict=True)
