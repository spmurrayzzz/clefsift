import argparse
import _thread
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time
import uuid

from common import GENERAL_WORKFLOW, MODEL_REVISION, dataset_stage, file_hash, validate_dataset, write_json, write_jsonl
from data import load_data, prepare_data, validate_splits
from decision import build_items, checkpoint_hashes, load_checkpoint, summarize_logits
from export import export_model


def update_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description="Train, select, and export Clef with Unsloth. Use torchrun for DDP.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("recipe-general.yaml"),
                        help="Training YAML. Paths resolve from this file.")
    parser.add_argument("--data", type=Path, help="Text/label JSONL file or dataset checkout; overrides the recipe input.")
    splits = parser.add_mutually_exclusive_group()
    splits.add_argument("--split", type=float, nargs=3, metavar=("TRAIN", "VALIDATION", "TEST"),
                        help="Runtime fractions for unsplit input; requires --seed.")
    splits.add_argument("--split-rows", type=int, nargs=3, metavar=("TRAIN", "VALIDATION", "TEST"),
                        help="Runtime row counts for ordered input; the declared test tail stays unread.")
    parser.add_argument("--seed", type=int, help="Split and training seed; required with --split.")
    args = parser.parse_args()
    try:
        args.split, args.split_rows = validate_splits(args.split, args.split_rows, args.seed)
    except ValueError as error:
        parser.error(str(error))
    import yaml

    config_path = args.config.resolve()
    try:
        config = yaml.safe_load(config_path.read_text())
        model_path = Path(config["model"]["path"]).expanduser()
        data_source = args.data.expanduser().resolve() if args.data is not None else Path(config["data"].get("path") or config["data"]["directory"]).expanduser()
        output_root = Path(config["output_dir"]).expanduser()
        minutes = config["minutes"]
    except (OSError, KeyError, TypeError, yaml.YAMLError) as error:
        parser.error(f"Invalid training recipe: {error}")
    model_path, data_source, output_root = ((path if path.is_absolute() else config_path.parent / path).resolve()
                                           for path in (model_path, data_source, output_root))
    if type(minutes) not in (int, float) or not math.isfinite(minutes) or minutes <= 0:
        parser.error("minutes must be finite and positive")
    if {"train_rows", "validation_rows"} & config["data"].keys():
        parser.error("Remove data row counts from the recipe; use --split or --split-rows at runtime")
    if (config["model"].get("revision") != MODEL_REVISION or config["model"].get("dtype") != "bfloat16"
            or config["data"]["max_length"] != 1536):
        parser.error("Use the pinned BF16 reference and the 1,536-token limit")
    if not all((model_path / name).is_file() for name in ("config.json", "joint_head.safetensors", "joint_head_config.json", "joint_schema_model.py")):
        parser.error("A complete local pinned reference snapshot is required")
    started = time.monotonic()
    run_dir, trainer, selected, schedule = None, None, None, None
    main_process = int(os.environ.get("RANK", "0")) == 0
    matched, checkpoint_files = {}, {}
    panel_name, panel_records = None, None
    checkpoint_bytes = 0
    status, error, export_status = "failed", None, "not-started"
    actual_steps, general = 0, False

    def check_resources(reserve=0):
        if run_dir is not None and shutil.disk_usage(run_dir).free < 30 * 1024 ** 3 + reserve:
            raise RuntimeError("Insufficient disk to preserve 30 GiB free")
        if time.monotonic() - started >= minutes * 60:
            raise KeyboardInterrupt("Run time limit reached")

    def interrupt(signum, frame):
        raise KeyboardInterrupt("Run time limit reached" if time.monotonic() - started >= minutes * 60 else "Run interrupted")

    previous_handlers = {signum: signal.signal(signum, interrupt) for signum in (signal.SIGINT, signal.SIGTERM)}
    timer = threading.Timer(minutes * 60, _thread.interrupt_main)
    timer.daemon = True
    timer.start()
    try:
        native_data = data_source.is_dir() and (data_source / "manifest.json").is_file()
        three_targets = False
        if native_data:
            if args.split is not None or args.split_rows is not None:
                parser.error("This dataset already has explicit splits; runtime repartitioning is not permitted")
            header = json.loads((data_source / "manifest.json").read_text())
            three_targets = dataset_stage(header) == "parents"
            manifest, datasets = validate_dataset(data_source, balanced=not three_targets)
            general = manifest.get("workflow") == GENERAL_WORKFLOW
        else:
            general = True
        metric = "macro_f1_fixed_three" if three_targets else "macro_f1_fixed_four"
        training = dict(config["train"])
        if args.seed is not None:
            training["seed"] = args.seed
        if general and (training.get("max_steps", -1) != -1 or type(training.get("num_train_epochs")) not in (int, float)
                        or not math.isfinite(training["num_train_epochs"]) or training["num_train_epochs"] <= 0):
            parser.error("The general corpus requires positive num_train_epochs without max_steps")
        interval = training["save_steps"]
        batch_size = training["per_device_train_batch_size"]
        accumulation = training["gradient_accumulation_steps"]
        if any(type(value) is not int or value <= 0 for value in (interval, batch_size, accumulation)):
            parser.error("Batch size, accumulation, and save_steps must be positive integers")
        for key in ("deepspeed", "fsdp", "parallelism_config"):
            if training.get(key):
                parser.error("Use standard single-device training or torchrun DDP")
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        from unsloth import DecisionTrainer, FastDecisionModel
        import torch
        from accelerate import PartialState
        from accelerate.utils import broadcast_object_list
        from transformers import TrainerCallback, TrainingArguments

        distributed = PartialState()
        main_process = distributed.is_main_process
        world_size = distributed.num_processes
        if distributed.device.type != "cuda":
            raise RuntimeError("This recipe requires CUDA; no model was loaded")
        paths = [str(output_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])) if main_process else None]
        broadcast_object_list(paths)
        run_dir = Path(paths[0])
        if main_process:
            run_dir.mkdir(parents=True)
        distributed.wait_for_everyone()
        check_resources()
        if native_data:
            data_record = data_source / "manifest.json"
        else:
            prepared_data = run_dir / "data"
            if main_process:
                preparation = prepare_data(data_source, prepared_data, ratios=args.split, split_rows=args.split_rows,
                                           seed=args.seed, check_resources=check_resources)
                counts = ", ".join(f"{name}={panel['rows']}" if panel is not None else f"{name}=unread"
                                   for name, panel in preparation["actual"].items())
                print(f"Prepared data ({preparation['mode']}): {counts}", flush=True)
            distributed.wait_for_everyone()
            datasets = load_data(prepared_data)
            data_record = prepared_data / "preparation.json"
        panel_records = datasets["eval.jsonl"]
        rows = len(datasets["train.jsonl"])
        local_batches = math.ceil(rows / (batch_size * world_size))
        steps_per_epoch = math.ceil(local_batches / accumulation)
        steps = training.get("max_steps", -1)
        if steps == -1:
            steps = math.ceil(steps_per_epoch * training["num_train_epochs"])
        if type(steps) is not int or steps <= 0:
            raise ValueError("Training must have a positive optimizer-step limit")
        schedule = {"train_rows": rows, "world_size": world_size, "microbatch_rows": batch_size,
                    "gradient_accumulation_steps": accumulation, "global_batch_size": batch_size * accumulation * world_size,
                    "steps_per_epoch": steps_per_epoch, "max_steps": steps, "save_steps": interval,
                    "num_train_epochs": training.get("num_train_epochs"),
                    "sampling": "Unsloth length grouping; Accelerate shards batches and pads uneven distributed tails"}
        fixed = {"eval_strategy": "steps", "eval_steps": interval, "save_strategy": "steps",
                 "save_total_limit": None, "save_only_model": False, "save_on_each_node": False,
                 "eval_on_start": False, "load_best_model_at_end": False, "bf16": True, "fp16": False,
                 "bf16_full_eval": False, "fp16_full_eval": False, "dataloader_drop_last": False,
                 "logging_nan_inf_filter": False, "ddp_find_unused_parameters": False,
                 "train_sampling_strategy": "random", "dataloader_num_workers": 0,
                 "dataloader_multiprocessing_context": None,
                 "accelerator_config": {"split_batches": False, "dispatch_batches": False, "even_batches": True}}
        if "output_dir" in training or any(key in training and training[key] != value for key, value in fixed.items()):
            raise ValueError("Training arguments conflict with split, checkpoint, or precision safeguards")
        train_args = TrainingArguments(**{"per_device_eval_batch_size": batch_size, "logging_steps": 1,
                                          "report_to": "none", "optim": "adamw_torch", "lr_scheduler_type": "linear",
                                          "weight_decay": 0.0, "max_grad_norm": 1.0, **training, **fixed,
                                          "output_dir": str(run_dir / "train")})
        config_hash, data_hash = file_hash(config_path), file_hash(data_record)
        if main_process:
            print(f"Run: {run_dir}\nProcesses: {world_size}; global batch: {schedule['global_batch_size']}; optimizer steps: {steps}", flush=True)
            write_json(run_dir / "config.json", {**config, "model": {**config["model"], "path": str(model_path)},
                       "data": {**config["data"], "path": str(data_source)}, "output_dir": str(output_root),
                       "train": training, "runtime": {"split": args.split, "split_rows": args.split_rows, "seed": args.seed}})
            if native_data:
                shutil.copyfile(data_record, run_dir / "dataset-manifest.json")
            write_json(run_dir / "environment.json", {
                "started_at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                "backend": "unsloth", "model_revision": MODEL_REVISION, "training_schedule": schedule,
                "data_record": str(data_record), "data_record_sha256": data_hash, "config_sha256": config_hash,
                "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "unsloth", "unsloth-zoo", "accelerate")},
                "source_sha256": {path.name: file_hash(path) for path in Path(__file__).parent.glob("*.py")},
            })
        checkpointing = True if world_size > 1 else "unsloth"
        model, processor = FastDecisionModel.from_pretrained(
            model_name=str(model_path), max_seq_length=1536, dtype=torch.bfloat16, load_in_4bit=False,
            local_files_only=True, use_gradient_checkpointing=checkpointing, random_state=train_args.seed,
        )
        if not getattr(model, "is_clef", False) or any(parameter.dtype != torch.float32 for parameter in model.head.parameters()):
            raise RuntimeError("Unsloth must load the existing Clef model with an FP32 head")
        original_head = {name: value.detach().cpu().clone() for name, value in model.head.state_dict().items()}
        model = FastDecisionModel.get_peft_model(model, use_gradient_checkpointing=checkpointing,
                                                 random_state=train_args.seed, **config["lora"])
        train_items = build_items(processor, datasets["train.jsonl"], model_path)
        eval_items = build_items(processor, panel_records, model_path)
        checkpoint_bytes = 4 * sum(parameter.numel() * parameter.element_size() for parameter in model.parameters() if parameter.requires_grad) + 64 * 1024 ** 2

        def rank(step):
            return (-matched[step][metric], matched[step]["multiclass_brier"], step)

        def compute_metrics(prediction):
            check_resources()
            raw, result = summarize_logits(panel_records, prediction.predictions)
            name = panel_name or f"step-{trainer.state.global_step}-matched"
            if main_process:
                write_jsonl(run_dir / name / "tuned_predictions.jsonl", raw)
                write_json(run_dir / name / "fixed_metrics.json", result)
            if panel_name is None:
                matched[trainer.state.global_step] = result
            return {key: result[key] for key in ("accuracy", metric, "multiclass_brier", "nll")}

        class RunCallback(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                check_resources()
                if state.max_steps != steps:
                    raise RuntimeError("Trainer schedule differs from the configured schedule")
                if world_size > 1 and not isinstance(trainer.model_wrapped, torch.nn.parallel.DistributedDataParallel):
                    raise RuntimeError("Distributed training requires synchronized DDP workers")

            def on_step_begin(self, args, state, control, **kwargs):
                check_resources()

            def on_prediction_step(self, args, state, control, **kwargs):
                check_resources()

            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == state.max_steps:
                    control.should_evaluate = control.should_save = True
                if control.should_save and main_process:
                    check_resources(checkpoint_bytes)

            def on_log(self, args, state, control, logs=None, **kwargs):
                check_resources()
                if any(not math.isfinite(value) for value in (logs or {}).values() if type(value) in (int, float)):
                    raise RuntimeError("Non-finite training or evaluation value")
                if main_process:
                    with (run_dir / "train_log.jsonl").open("a") as output:
                        output.write(json.dumps({"step": state.global_step, **(logs or {})}, allow_nan=False) + "\n")

            def on_save(self, args, state, control, **kwargs):
                nonlocal selected
                distributed.wait_for_everyone()
                selected = min(matched, key=rank)
                if main_process:
                    checkpoint_files[state.global_step] = checkpoint_hashes(Path(args.output_dir) / f"checkpoint-{state.global_step}")
                    update_json(run_dir / "progress.json", {"actual_steps": state.global_step, "max_steps": steps,
                                "matched": matched, "best_step": selected, "checkpoint_sha256": checkpoint_files})

        trainer = DecisionTrainer(
            model=model, processing_class=processor, train_dataset=train_items, eval_dataset=eval_items,
            args=train_args, compute_metrics=compute_metrics, callbacks=[RunCallback()],
            permute_fields=False, ordinal_weight=0.0, **config["decision"],
        )
        check_resources(checkpoint_bytes if main_process else 0)
        trainer.train()
        if selected is None or trainer.state.global_step != steps:
            raise RuntimeError("Training ended without completing the configured schedule")
        checkpoint = run_dir / "train" / f"checkpoint-{selected}"
        if main_process:
            if checkpoint_hashes(checkpoint) != checkpoint_files[selected]:
                raise RuntimeError("Selected checkpoint changed")
            if file_hash(config_path) != config_hash or file_hash(data_record) != data_hash:
                raise RuntimeError("Config or dataset metadata changed")
            if native_data:
                validate_dataset(data_source, balanced=not three_targets)
            else:
                load_data(prepared_data)
            write_json(run_dir / "selection.json", {
                "selected_at_utc": datetime.now(timezone.utc).isoformat(), "backend": "unsloth", "step": selected,
                "checkpoint": str(checkpoint), "rule": f"Highest validation {metric}, then lower Brier, then earlier step",
                "matched": matched, "checkpoint_sha256": checkpoint_files[selected], "training_schedule": schedule,
                "data_record": str(data_record), "data_record_sha256": data_hash, "config_sha256": config_hash,
                "prediction_sha256": {str(step): file_hash(run_dir / f"step-{step}-matched/tuned_predictions.jsonl") for step in matched},
                **({"workflow": GENERAL_WORKFLOW, "status": "selected-final-sealed", "final_holdout_evaluated": False} if general else {}),
            })
        distributed.wait_for_everyone()
        panel_name = "base-matched"
        model.head.load_state_dict(original_head, strict=True)
        with model.encoder.disable_adapter():
            trainer.predict(eval_items)
        del original_head
        load_checkpoint(model, checkpoint)
        if not general:
            for filename in ("holdout.jsonl", "regression.jsonl"):
                if filename in datasets:
                    panel_name, panel_records = f"step-{selected}-{filename[:-6]}", datasets[filename]
                    trainer.predict(build_items(processor, panel_records, model_path))
        distributed.wait_for_everyone()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        actual_steps = trainer.state.global_step
        del trainer
        trainer = None
        gc.collect()
        if main_process:
            model.to("cpu")
            torch.cuda.empty_cache()
            release = run_dir / "release"
            release.mkdir()
            export_status = "running"
            export_model(model.encoder, model.head.state_dict(), model_path,
                         checkpoint / "joint_head_config.json", release, check_resources)
            write_json(release / "export.json", {"status": "complete", "checkpoint": str(checkpoint), "head_dtype": "float32"})
            export_status = "complete"
            print(f"Selected checkpoint: {checkpoint}\nRelease: {release}", flush=True)
        status = "selected-final-sealed" if general else "complete"
    except (Exception, KeyboardInterrupt) as failure:
        error = str(failure) or "Run interrupted"
        if export_status == "running":
            export_status = "failed"
    finally:
        timer.cancel()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if main_process and run_dir is not None:
            write_json(run_dir / "summary.json", {
                "status": status, "error": error, "backend": "unsloth", "selected_step": selected,
                "actual_steps": trainer.state.global_step if trainer is not None else actual_steps,
                "export_status": export_status, "training_schedule": schedule, "wall_seconds": time.monotonic() - started,
                **({"final_holdout_status": "sealed", "final_holdout_evaluated": False} if general else {}),
            })
    if error:
        raise SystemExit(error)


if __name__ == "__main__":
    main()
