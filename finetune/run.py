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

from common import GENERAL_WORKFLOW, MODEL_REVISION, file_hash, validate_dataset, write_json
from evaluate import load_predictions, metrics


def stop(process):
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except ProcessLookupError:
            process.wait()
        except subprocess.TimeoutExpired:
            pass
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def update_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def checkpoint_hashes(checkpoint):
    names = ("adapter/adapter_config.json", "adapter/adapter_model.safetensors", "joint_head.safetensors",
             "training_state.pt", "train_config.json")
    if any(not (checkpoint / name).is_file() for name in names):
        raise RuntimeError(f"Incomplete checkpoint: {checkpoint}")
    return {name: file_hash(checkpoint / name) for name in names}


def main():
    parser = argparse.ArgumentParser(description="Run reference training and select on matched validation only.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="Local pinned Clef snapshot; no downloads occur.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--minutes", type=float, help="Optional total wall-time limit, including all evaluations.")
    parser.add_argument("--eval-gpu", type=int, help="General workflow: evaluate concurrently on a separate GPU.")
    parser.add_argument("--early-stop-patience", type=int, help="Stop after this many checkpoint evaluations without improvement; requires --eval-gpu.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--prior-checkpoint", type=Path, help="Optional v1 baseline; never the training starting point.")
    parser.add_argument("--three-targets", action="store_true", help="Use the original human/mixed/AI parent panel for a v1 run.")
    parser.add_argument("--config", type=Path, help="Defaults to recipe-general.yaml for the general corpus, otherwise recipe.yaml.")
    parser.add_argument("--epochs", type=float, help="General workflow only: derive optimizer steps from row exposures; save_every stays in --config.")
    args = parser.parse_args()
    if (args.minutes is not None and (not math.isfinite(args.minutes) or args.minutes <= 0)) or args.gpu < 0:
        parser.error("Use a positive time limit and a nonnegative GPU index")
    if args.eval_gpu is not None and (args.eval_gpu < 0 or args.eval_gpu == args.gpu):
        parser.error("--eval-gpu must be nonnegative and different from --gpu")
    if args.early_stop_patience is not None and (args.early_stop_patience <= 0 or args.eval_gpu is None):
        parser.error("Positive --early-stop-patience requires --eval-gpu")
    if args.epochs is not None and (not math.isfinite(args.epochs) or args.epochs <= 0):
        parser.error("Epochs must be finite and positive")
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
    general = manifest.get("workflow") == GENERAL_WORKFLOW
    if args.eval_gpu is not None and not general:
        parser.error("--eval-gpu is only supported for clefsift-general-v1")
    if args.epochs is not None and not general:
        parser.error("--epochs is only supported for clefsift-general-v1")
    if general and args.epochs is None:
        parser.error("The general corpus requires an explicitly approved --epochs value")
    args.config = args.config or Path(__file__).with_name("recipe-general.yaml" if general else "recipe.yaml")
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
    schedule = None
    if general:
        row_count, grad_accum = len(datasets["train.jsonl"]), config["train"]["grad_accum"]
        if type(grad_accum) is not int or grad_accum <= 0:
            raise ValueError("Gradient accumulation must be a positive integer")
        if args.epochs is not None:
            requested_steps = row_count * args.epochs / grad_accum
            if not math.isfinite(requested_steps):
                raise ValueError("Epoch request exceeds the supported schedule range")
            config["train"]["max_steps"] = max(1, math.ceil(requested_steps))
    steps, interval = config["train"]["max_steps"], config["train"]["save_every"]
    if general:
        if type(steps) is not int or type(interval) is not int or steps <= 0 or interval <= 0:
            raise ValueError("Steps and save interval must be positive integers")
    elif type(steps) is not int or type(interval) is not int or not 0 < interval <= steps or steps % interval:
        raise ValueError("Steps must be a positive multiple of the save interval")
    checkpoint_steps = sorted(set(range(interval, steps + 1, interval)) | {steps})
    if general:
        exposures = steps * grad_accum
        schedule = {
            "requested_epochs": args.epochs, "train_rows": row_count, "microbatch_rows": 1,
            "grad_accum": grad_accum, "max_steps": steps, "save_every": interval,
            "checkpoint_steps": checkpoint_steps, "row_exposures": exposures,
            "effective_epochs": exposures / row_count, "full_passes": exposures // row_count,
            "rows_in_partial_pass": exposures % row_count, "augment_train": config["train"].get("augment_train", True),
            "sampling": "Shuffled without replacement per pass; accumulation can cross pass boundaries",
        }
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
        "minutes": args.minutes, "eval_gpu": args.eval_gpu, "early_stop_patience": args.early_stop_patience,
        "data_manifest_sha256": data_hash, "config_sha256": config_hash,
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "clef-finetune", "huggingface-hub")},
        "source_sha256": {path.name: file_hash(path) for path in Path(__file__).parent.glob("*.py")},
        **({"workflow": GENERAL_WORKFLOW, "training_schedule": schedule, "final_holdout_status": "sealed"} if general else {}),
    })
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    prefix = [sys.executable, "-m", "clef_finetune.cli"]
    started = time.monotonic()
    deadline = started + args.minutes * 60 if args.minutes is not None else math.inf
    children = {}
    phases = []
    interrupted = False
    early_stopped = False
    actual_steps = 0
    log_offset = 0
    selected = None
    matched, checkpoint_files = {}, {}
    metric = "macro_f1_fixed_three" if args.three_targets else "macro_f1_fixed_four"
    stale = 0

    def interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)

    def monitor():
        nonlocal log_offset, actual_steps
        if interrupted or time.monotonic() >= deadline:
            raise RuntimeError("Run interrupted or time limit reached")
        if shutil.disk_usage(args.out_dir).free < 30 * 1024 ** 3:
            raise RuntimeError("Free disk below 30 GiB")
        log = args.out_dir / "train/train_log.jsonl"
        if log.exists():
            with log.open() as source:
                source.seek(log_offset)
                while line := source.readline():
                    if not line.endswith("\n"):
                        break
                    entry = json.loads(line)
                    if any(not math.isfinite(entry[key]) for key in ("loss", "grad_norm")):
                        raise RuntimeError("Non-finite training value")
                    if entry["step"] != actual_steps + 1:
                        raise RuntimeError("Training log is incomplete")
                    actual_steps = entry["step"]
                    log_offset = source.tell()
        update_json(args.out_dir / "progress.json", {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(), "actual_steps": actual_steps,
            "max_steps": steps, "active": {name: child.pid for name, (child, _) in children.items()},
            "evaluated_steps": list(matched), "best_step": selected, "evaluations_without_improvement": stale,
            "early_stopped": early_stopped,
        })

    def start(name, command, gpu):
        monitor()
        memory = subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True, timeout=15)
        if int(memory.strip()) > 1024:
            raise RuntimeError(f"GPU {gpu} is not idle; no existing process was stopped")
        if file_hash(config_path) != config_hash or file_hash(args.data_dir / "manifest.json") != data_hash:
            raise RuntimeError("Config or dataset manifest changed")
        validate_dataset(args.data_dir, balanced=not args.three_targets)
        monitor()
        print(name, flush=True)
        with (args.out_dir / f"{name}.log").open("x") as output:
            child = subprocess.Popen(command, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)),
                                     stdin=subprocess.DEVNULL, stdout=output,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        children[name] = (child, time.monotonic())

    def finish(name, expected_stop=False):
        child, phase_start = children[name]
        stop(child)
        phases.append({"phase": name, "seconds": time.monotonic() - phase_start,
                       "returncode": child.returncode, "early_stop_requested": expected_stop})
        del children[name]
        if child.returncode and not (expected_stop and child.returncode == -signal.SIGTERM):
            raise RuntimeError(f"Phase failed: {name}; see {name}.log")

    def execute(name, command):
        start(name, command, args.gpu)
        while children[name][0].poll() is None:
            monitor()
            time.sleep(1)
        monitor()
        finish(name)

    def eval_command(name, filename, checkpoint=None):
        command = prefix + ["eval", "--model", str(args.model), "--data", str(args.data_dir / filename),
                            "--device", "cuda", "--dtype", "bfloat16", "--max-length", "1536",
                            "--output-dir", str(args.out_dir / name)]
        if checkpoint:
            command += ["--checkpoint", str(checkpoint)]
        return command

    def eval_result(name, filename):
        directory = args.out_dir / name
        result = metrics(load_predictions(datasets[filename], directory / "tuned_predictions.jsonl"))
        if any(not math.isfinite(result[key]) for key in (metric, "multiclass_brier")):
            raise RuntimeError("Non-finite validation metric")
        write_json(directory / "fixed_metrics.json", result)
        return result

    def evaluate(name, filename, checkpoint=None):
        execute(name, eval_command(name, filename, checkpoint))
        return eval_result(name, filename)

    def rank(step):
        return (-matched[step][metric], matched[step]["multiclass_brier"], step)

    def concurrent_train():
        nonlocal selected, stale, early_stopped
        start("train", prefix + ["train", str(config_path), "--device", "cuda"], args.gpu)
        baselines = [("base-matched", None)]
        if args.prior_checkpoint:
            baselines.append(("prior-matched", args.prior_checkpoint))
        evaluating = None
        pending = iter(checkpoint_steps)
        next_step = next(pending, None)
        while True:
            monitor()
            if "train" in children and children["train"][0].poll() is not None:
                finish("train")
                monitor()
                if actual_steps != steps:
                    raise RuntimeError("Training log is incomplete")
            if evaluating is not None:
                name, step = evaluating
                if children[name][0].poll() is not None:
                    finish(name)
                    result = eval_result(name, "eval.jsonl")
                    evaluating = None
                    if step is not None:
                        checkpoint = args.out_dir / "train" / f"checkpoint-{step}"
                        if checkpoint_hashes(checkpoint) != checkpoint_files[step]:
                            raise RuntimeError(f"Checkpoint changed during validation: {step}")
                        matched[step] = result
                        if selected is None or rank(step) < rank(selected):
                            selected, stale = step, 0
                        else:
                            stale += 1
                        update_json(args.out_dir / "validation-progress.json", {
                            "best_step": selected, "matched": matched, "checkpoint_sha256": checkpoint_files,
                            "evaluations_without_improvement": stale,
                        })
                        if args.early_stop_patience is not None and stale >= args.early_stop_patience:
                            early_stopped = True
                            if "train" in children:
                                finish("train", expected_stop=True)
                            monitor()
                            break
            if evaluating is None:
                if baselines:
                    name, checkpoint = baselines.pop(0)
                    start(name, eval_command(name, "eval.jsonl", checkpoint), args.eval_gpu)
                    evaluating = (name, None)
                elif next_step is not None:
                    checkpoint = args.out_dir / "train" / f"checkpoint-{next_step}"
                    try:
                        saved_config = json.loads((checkpoint / "train_config.json").read_text())
                    except (FileNotFoundError, json.JSONDecodeError):
                        if "train" not in children:
                            raise RuntimeError(f"Incomplete checkpoint: {next_step}")
                    else:
                        if saved_config != config:
                            raise RuntimeError(f"Checkpoint config mismatch: {next_step}")
                        checkpoint_files[next_step] = checkpoint_hashes(checkpoint)
                        name = f"step-{next_step}-matched"
                        start(name, eval_command(name, "eval.jsonl", checkpoint), args.eval_gpu)
                        evaluating = (name, next_step)
                        next_step = next(pending, None)
                elif "train" not in children:
                    break
            time.sleep(1)

    status, error = "failed", None
    try:
        if args.eval_gpu is not None:
            concurrent_train()
        else:
            evaluate("base-matched", "eval.jsonl")
            if args.prior_checkpoint:
                evaluate("prior-matched", "eval.jsonl", args.prior_checkpoint)
            execute("train", prefix + ["train", str(config_path), "--device", "cuda"])
            if actual_steps != steps:
                raise RuntimeError("Training log is incomplete")
            for step in checkpoint_steps:
                checkpoint = args.out_dir / "train" / f"checkpoint-{step}"
                if not (checkpoint / "training_state.pt").is_file():
                    raise RuntimeError(f"Incomplete checkpoint: {step}")
                if general:
                    checkpoint_files[step] = checkpoint_hashes(checkpoint)
                matched[step] = evaluate(f"step-{step}-matched", "eval.jsonl", checkpoint)
                if general and checkpoint_hashes(checkpoint) != checkpoint_files[step]:
                    raise RuntimeError(f"Checkpoint changed during validation: {step}")
        selected = min(matched, key=rank)
        checkpoint = args.out_dir / "train" / f"checkpoint-{selected}"
        selection_binding = {}
        if general:
            if file_hash(config_path) != config_hash or file_hash(args.data_dir / "manifest.json") != data_hash:
                raise RuntimeError("Config or dataset manifest changed")
            validate_dataset(args.data_dir)
            if checkpoint_hashes(checkpoint) != checkpoint_files[selected]:
                raise RuntimeError("Selected checkpoint changed")
            selection_binding = {
                "workflow": GENERAL_WORKFLOW, "data_manifest_sha256": data_hash, "config_sha256": config_hash,
                "checkpoint_sha256": checkpoint_files[selected], "training_schedule": schedule,
                "selected_row_exposures": selected * grad_accum,
                "selected_effective_epochs": selected * grad_accum / row_count,
                "status": "selected-final-sealed", "final_holdout_status": "sealed",
                "final_holdout_evaluated": False,
            }
        write_json(args.out_dir / "selection.json", {
            "selected_at_utc": datetime.now(timezone.utc).isoformat(), "step": selected, "checkpoint": str(checkpoint),
            "rule": f"Highest matched {metric}, then lower Brier, then earlier step", "matched": matched,
            "prediction_sha256": {str(step): file_hash(args.out_dir / f"step-{step}-matched/tuned_predictions.jsonl") for step in matched},
            **selection_binding,
        })
        if not general and "holdout.jsonl" in datasets:
            evaluate("base-holdout", "holdout.jsonl")
            if args.prior_checkpoint:
                evaluate("prior-holdout", "holdout.jsonl", args.prior_checkpoint)
            evaluate(f"step-{selected}-holdout", "holdout.jsonl", checkpoint)
        if not general and "regression.jsonl" in datasets:
            evaluate(f"step-{selected}-regression", "regression.jsonl", checkpoint)
        status = "selected-final-sealed" if general else "complete"
    except Exception as failure:
        error = str(failure)
    finally:
        for name, (child, phase_start) in list(children.items()):
            stop(child)
            phases.append({"phase": name, "seconds": time.monotonic() - phase_start, "returncode": child.returncode})
            del children[name]
        write_json(args.out_dir / "summary.json", {"status": status, "error": error, "phases": phases,
                   "actual_steps": actual_steps, "selected_step": selected,
                   "selected_checkpoint": str(args.out_dir / "train" / f"checkpoint-{selected}") if selected is not None else None,
                   "early_stopped": early_stopped, "early_stop_patience": args.early_stop_patience,
                   "evaluations_without_improvement": stale,
                   "wall_seconds": time.monotonic() - started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                   **({"workflow": GENERAL_WORKFLOW, "training_schedule": schedule,
                       "final_holdout_status": "sealed", "final_holdout_evaluated": False} if general else {})})
    if status not in ("complete", "selected-final-sealed"):
        raise SystemExit(error)


if __name__ == "__main__":
    main()
