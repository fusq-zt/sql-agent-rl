# Copyright (c) Microsoft. All rights reserved.
# Adapted for SQL Agent RL: portable package imports/entrypoints; see THIRD_PARTY_NOTICES.md.
"""Use VERL's official FSDP merger and preserve a verified source/export manifest.

Usage: python -m sql_agent_rl.export --checkpoint GLOBAL_STEP_DIR --target NEW_HF_DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    source = args.checkpoint.resolve(strict=True)
    target = args.target.resolve()
    if target.exists():
        raise FileExistsError("Choose a new export path; existing models are immutable")
    shards = sorted((source / "actor").glob("model_world_size_*_rank_*.pt"))
    if not shards:
        raise FileNotFoundError("No official FSDP model shards found")
    started = time.time()
    environment = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="2",
        DO_NOT_TRACK="1",
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "verl.model_merger",
            "merge",
            "--backend",
            "fsdp",
            "--local_dir",
            str(source / "actor"),
            "--target_dir",
            str(target),
        ],
        check=True,
        env=environment,
    )
    outputs = sorted(target.glob("*.safetensors"))
    if not outputs or not (target / "config.json").exists():
        raise RuntimeError("Official merger did not produce a complete local HF model")
    manifest = {
        "source_checkpoint": str(source),
        "export_started_unix": started,
        "export_wall_seconds": time.time() - started,
        "merger": "verl.model_merger fsdp, version 0.6.1",
        "export_dtype": "bfloat16 (official merger; original FP32 training checkpoints retained)",
        "source_model_shards": {p.name: {"sha256": file_hash(p), "bytes": p.stat().st_size} for p in shards},
        "output_files": {
            p.name: {"sha256": file_hash(p), "bytes": p.stat().st_size} for p in target.iterdir() if p.is_file()
        },
        "source_optimizer_present": bool(list((source / "actor").glob("optim_world_size_*_rank_*.pt"))),
        "source_driver_rng_present": (source / "driver_rng.pt").exists(),
        "source_dataloader_present": (source / "data.pt").exists(),
    }
    (target / "export_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
