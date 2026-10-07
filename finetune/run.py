import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from common import MODEL_REVISION, file_hash, read_jsonl, validate_dataset, write_json
from evaluate import load_predictions, metrics


def stop(process):
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except ProcessLookupError:
            process.wait()
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main():
    parser = argparse.ArgumentParser(description="Run bounded reference training and select on matched validation only.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="Local pinned Clef snapshot; no downloads occur.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--minutes", type=float, required=True, help="Total wall-time limit, including all evaluations.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--prior-checkpoint", type=Path, help="Optional v1 baseline; never the training starting point.")
    parser.add_argument("--three-targets", action="store_true", help="Use the original human/mixed/AI parent panel for a v1 run.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("recipe.yaml"))
    args = parser.parse_args()
    if not math.isfinite(args.minutes) or args.minutes <= 0 or args.gpu < 0:
        parser.error("Use a positive time limit and a nonnegative GPU index")
    if args.out_dir.exists() or args.out_dir.is_symlink():
        parser.error("Output directory already exists")
    args.model = args.model.resolve()
    args.data_dir = args.data_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    if not args.model.is_dir() or not (args.model / "joint_head.safetensors").is_file():
        parser.error("A complete local pinned reference snapshot is required")
    if args.prior_checkpoint:
        args.prior_checkpoint = args.prior_checkpoint.resolve()
        if not (args.prior_checkpoint / "training_state.pt").is_file():
            parser.error("Prior checkpoint is incomplete")
    manifest, datasets = validate_dataset(args.data_dir, balanced=not args.three_targets)
    if args.three_targets and manifest["stage"] != "parents":
        parser.error("The v1 run requires a parent dataset")
    if not args.three_targets and manifest["stage"] != "frozen":
        parser.error("The v2 run requires an assembled frozen dataset")
    import yaml
    from clef_finetune.train import load_config

    config = load_config(args.config)
    if config["model"].get("revision") != MODEL_REVISION or config["data"]["max_length"] != 1536:
        raise ValueError("Config reference revision or input limit mismatch")
    if config["data"].get("max_state_tokens") is not None:
        raise ValueError("State truncation is not permitted")
    config["model"]["path"] = str(args.model)
    config["data"]["train"] = str(args.data_dir / "train.jsonl")
    config["data"]["eval"] = str(args.data_dir / "eval.jsonl")
    config["output_dir"] = str(args.out_dir / "train")
    steps, interval = config["train"]["max_steps"], config["train"]["save_every"]
    if type(steps) is not int or type(interval) is not int or not 0 < interval <= steps or steps % interval:
        raise ValueError("Steps must be a positive multiple of the save interval")
    args.out_dir.mkdir(parents=True)
    config_path = args.out_dir / "config.yaml"
    with config_path.open("x") as output:
        yaml.safe_dump(config, output, sort_keys=False)
    shutil.copyfile(args.data_dir / "manifest.json", args.out_dir / "dataset-manifest.json")
    data_hash = file_hash(args.data_dir / "manifest.json")
    config_hash = file_hash(config_path)
    write_json(args.out_dir / "environment.json", {
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "model_revision": MODEL_REVISION, "starting_point": "Fresh pinned reference", "gpu": args.gpu,
        "minutes": args.minutes, "data_manifest_sha256": data_hash, "config_sha256": config_hash,
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "clef-finetune", "huggingface-hub")},
        "source_sha256": {path.name: file_hash(path) for path in Path(__file__).parent.glob("*.py")},
    })
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    prefix = [sys.executable, "-m", "clef_finetune.cli"]
    started = time.monotonic()
    deadline = started + args.minutes * 60
    active = None
    phases = []
    interrupted = False

    def interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True
        stop(active)

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)

    def execute(name, command):
        nonlocal active
        if interrupted or time.monotonic() >= deadline:
            raise RuntimeError("Run interrupted or time limit reached")
        memory = subprocess.check_output(["nvidia-smi", f"--id={args.gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True, timeout=15)
        if int(memory.strip()) > 1024:
            raise RuntimeError("GPU is not idle; no existing process was stopped")
        if file_hash(config_path) != config_hash or file_hash(args.data_dir / "manifest.json") != data_hash:
            raise RuntimeError("Config or dataset manifest changed")
        validate_dataset(args.data_dir, balanced=not args.three_targets)
        if shutil.disk_usage(args.out_dir).free < 30 * 1024 ** 3:
            raise RuntimeError("Free disk below 30 GiB")
        phase_start = time.monotonic()
        print(name, flush=True)
        with (args.out_dir / f"{name}.log").open("x") as output:
            active = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=output,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while active.poll() is None:
                    if interrupted or time.monotonic() >= deadline:
                        raise RuntimeError("Run interrupted or time limit reached")
                    if shutil.disk_usage(args.out_dir).free < 30 * 1024 ** 3:
                        raise RuntimeError("Free disk below 30 GiB")
                    log = args.out_dir / "train/train_log.jsonl"
                    if name == "train" and log.exists():
                        for line in log.read_text().splitlines():
                            try:
                                entry = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if any(not math.isfinite(entry[key]) for key in ("loss", "grad_norm")):
                                raise RuntimeError("Non-finite training value")
                    time.sleep(1)
                if active.returncode:
                    raise RuntimeError(f"Phase failed: {name}; see {name}.log")
            finally:
                stop(active)
                phases.append({"phase": name, "seconds": time.monotonic() - phase_start, "returncode": active.returncode})
                active = None

    def evaluate(name, filename, checkpoint=None):
        directory = args.out_dir / name
        command = prefix + ["eval", "--model", str(args.model), "--data", str(args.data_dir / filename),
                            "--device", "cuda", "--dtype", "bfloat16", "--max-length", "1536", "--output-dir", str(directory)]
        if checkpoint:
            command += ["--checkpoint", str(checkpoint)]
        execute(name, command)
        result = metrics(load_predictions(datasets[filename], directory / "tuned_predictions.jsonl"))
        write_json(directory / "fixed_metrics.json", result)
        return result

    status, error = "failed", None
    try:
        evaluate("base-matched", "eval.jsonl")
        if args.prior_checkpoint:
            evaluate("prior-matched", "eval.jsonl", args.prior_checkpoint)
        execute("train", prefix + ["train", str(config_path), "--device", "cuda"])
        log = read_jsonl(args.out_dir / "train/train_log.jsonl")
        if [row["step"] for row in log] != list(range(1, steps + 1)):
            raise RuntimeError("Training log is incomplete")
        matched = {}
        for step in range(interval, steps + 1, interval):
            checkpoint = args.out_dir / "train" / f"checkpoint-{step}"
            if not (checkpoint / "training_state.pt").is_file():
                raise RuntimeError(f"Incomplete checkpoint: {step}")
            matched[step] = evaluate(f"step-{step}-matched", "eval.jsonl", checkpoint)
        metric = "macro_f1_fixed_three" if args.three_targets else "macro_f1_fixed_four"
        selected = min(matched, key=lambda step: (-matched[step][metric], matched[step]["multiclass_brier"], step))
        checkpoint = args.out_dir / "train" / f"checkpoint-{selected}"
        write_json(args.out_dir / "selection.json", {
            "selected_at_utc": datetime.now(timezone.utc).isoformat(), "step": selected, "checkpoint": str(checkpoint),
            "rule": f"Highest matched {metric}, then lower Brier, then earlier step", "matched": matched,
            "prediction_sha256": {str(step): file_hash(args.out_dir / f"step-{step}-matched/tuned_predictions.jsonl") for step in matched},
        })
        if "holdout.jsonl" in datasets:
            evaluate("base-holdout", "holdout.jsonl")
            if args.prior_checkpoint:
                evaluate("prior-holdout", "holdout.jsonl", args.prior_checkpoint)
            evaluate(f"step-{selected}-holdout", "holdout.jsonl", checkpoint)
        if "regression.jsonl" in datasets:
            evaluate(f"step-{selected}-regression", "regression.jsonl", checkpoint)
        status = "complete"
    except Exception as failure:
        error = str(failure)
    finally:
        stop(active)
        write_json(args.out_dir / "summary.json", {"status": status, "error": error, "phases": phases,
                   "wall_seconds": time.monotonic() - started, "finished_at_utc": datetime.now(timezone.utc).isoformat()})
    if status != "complete":
        raise SystemExit(error)


if __name__ == "__main__":
    main()
