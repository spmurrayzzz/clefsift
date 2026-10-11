import argparse
import _thread
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import threading
import time

from common import MODEL_REVISION, file_hash, load_schema, write_json
from decision import adapter_directory


RELEASE_FILES = ("joint_schema_model.py", "LICENSE", "tokenizer.json", "tokenizer_config.json",
                 "chat_template.jinja", "processor_config.json", "preprocessor_config.json",
                 "video_preprocessor_config.json", "special_tokens_map.json", "added_tokens.json",
                 "generation_config.json", "tokenizer.model", "vocab.json", "vocab.txt", "merges.txt")


def export_model(backbone, head, reference, head_config_path, output, check_resources):
    import torch
    from safetensors.torch import save_file

    head = {name: value.detach().cpu().contiguous() for name, value in head.items()}
    if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in head.values()):
        raise ValueError("The selected checkpoint must contain a finite FP32 decision head")
    head_config = json.loads(head_config_path.read_text())
    load_schema(reference).JointSchemaHead(**head_config).load_state_dict(head, strict=True)
    release_bytes = sum(path.stat().st_size for path in reference.glob("model*.safetensors")) + sum(value.numel() * value.element_size() for value in head.values()) + 1024 ** 3
    check_resources(release_bytes)
    backbone = backbone.to("cpu").merge_and_unload(safe_merge=True)
    backbone.save_pretrained(str(output), safe_serialization=True, max_shard_size="5GB")
    for name in RELEASE_FILES:
        if (reference / name).is_file():
            shutil.copyfile(reference / name, output / name)
    shutil.copyfile(head_config_path, output / "joint_head_config.json")
    save_file(head, str(output / "joint_head.safetensors"))
    check_resources()


def main():
    parser = argparse.ArgumentParser(description="Merge a Clef LoRA checkpoint on CPU and retain the FP32 decision head.")
    parser.add_argument("--model", type=Path, required=True, help="Local pinned reference snapshot.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Selected Unsloth or historical adapter checkpoint.")
    parser.add_argument("--out-dir", type=Path, required=True, help="New release directory.")
    parser.add_argument("--minutes", type=float, required=True, help="Total CPU export time limit.")
    args = parser.parse_args()
    if not math.isfinite(args.minutes) or args.minutes <= 0:
        parser.error("Use a positive time limit")
    if args.out_dir.exists() or args.out_dir.is_symlink():
        parser.error("Output directory already exists")
    args.model, args.checkpoint, args.out_dir = (path.resolve() for path in (args.model, args.checkpoint, args.out_dir))
    adapter = adapter_directory(args.checkpoint)
    head_path = args.checkpoint / "joint_head.safetensors"
    for path in (args.model / "config.json", args.model / "joint_schema_model.py", head_path,
                 adapter / "adapter_config.json", adapter / "adapter_model.safetensors"):
        if not path.is_file():
            parser.error(f"Missing model or checkpoint file: {path}")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    args.out_dir.mkdir(parents=True)
    started = time.monotonic()

    def check_resources(reserve=0):
        if shutil.disk_usage(args.out_dir).free < 30 * 1024 ** 3 + reserve:
            raise RuntimeError("Insufficient disk to preserve 30 GiB free")
        if time.monotonic() - started >= args.minutes * 60:
            raise KeyboardInterrupt("Export time limit reached")

    def interrupt(signum, frame):
        raise KeyboardInterrupt("Export time limit reached" if time.monotonic() - started >= args.minutes * 60 else "Export interrupted")

    previous_handlers = {signum: signal.signal(signum, interrupt) for signum in (signal.SIGINT, signal.SIGTERM)}
    timer = threading.Timer(args.minutes * 60, _thread.interrupt_main)
    timer.daemon = True
    timer.start()
    status, error = "failed", None
    input_hashes = {}
    try:
        check_resources()
        import torch
        from peft import PeftModel
        from safetensors.torch import load_file
        from transformers import Qwen3_5ForConditionalGeneration

        head = load_file(str(head_path), device="cpu")
        head_config_path = args.checkpoint / "joint_head_config.json"
        if not head_config_path.is_file():
            head_config_path = args.model / "joint_head_config.json"
        input_hashes = {"head": file_hash(head_path), "adapter": file_hash(adapter / "adapter_model.safetensors"),
                        "adapter_config": file_hash(adapter / "adapter_config.json")}
        backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(args.model), dtype=torch.bfloat16, device_map={"": "cpu"}, local_files_only=True,
        )
        backbone = PeftModel.from_pretrained(backbone, str(adapter), local_files_only=True)
        export_model(backbone, head, args.model, head_config_path, args.out_dir, check_resources)
        if input_hashes != {"head": file_hash(head_path), "adapter": file_hash(adapter / "adapter_model.safetensors"),
                            "adapter_config": file_hash(adapter / "adapter_config.json")}:
            raise RuntimeError("Checkpoint changed during export")
        status = "complete"
    except (Exception, KeyboardInterrupt) as failure:
        error = str(failure) or "Export interrupted"
    finally:
        timer.cancel()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        write_json(args.out_dir / "export.json", {
            "status": status, "error": error, "model_revision": MODEL_REVISION,
            "checkpoint": str(args.checkpoint), "head_dtype": "float32", "backbone_dtype": "bfloat16",
            "input_sha256": input_hashes,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(), "wall_seconds": time.monotonic() - started,
        })
    if error:
        raise SystemExit(error)


if __name__ == "__main__":
    main()
