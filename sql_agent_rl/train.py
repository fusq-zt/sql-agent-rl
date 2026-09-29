"""Portable GRPO entrypoint. Default is a CPU-only configuration preview.

python -m sql_agent_rl.train --dry-run
python -m sql_agent_rl.train --execute --model models/qwen2.5-coder-1.5b
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from .config import DEFAULT_RECIPE, load_recipe, load_tasks, make_config, pad_training_rows


def verify_runtime() -> None:
    import importlib.metadata
    import agentlightning

    expected_versions = {"agentlightning": "0.3.1", "verl": "0.6.1", "torch": "2.8.0", "vllm": "0.11.0"}
    for name, expected in expected_versions.items():
        if importlib.metadata.version(name).split("+")[0] != expected:
            raise RuntimeError(f"Expected {name}=={expected}; see docs/usage.md")
    manifest = json.loads((Path(__file__).resolve().parents[1] / "runtime/upstream.json").read_text())
    root = Path(agentlightning.__file__).resolve().parent.parent
    for rel, record in manifest["files"].items():
        actual = hashlib.sha256((root / rel).read_text(encoding="utf-8").encode()).hexdigest()
        if actual != record["sha256"]:
            raise RuntimeError(f"Runtime source differs: {rel}. Run scripts/setup_upstream.py in a fresh destination")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_RECIPE)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--model", type=Path, default=Path("models/qwen2.5-coder-1.5b"))
    p.add_argument("--run-dir", type=Path, default=Path("runs/grpo"))
    p.add_argument("--runtime-dir", type=Path, help="Short absolute Ray temporary path; defaults to a fresh /tmp directory")
    p.add_argument("--port", type=int, default=4747)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true", help="Actually start GPU training")
    args = p.parse_args()
    recipe = load_recipe(args.config)
    data, model, run = (x.resolve() for x in (args.data, args.model, args.run_dir))
    config = make_config(recipe, data, model, run)
    if not args.execute:
        print(json.dumps({"mode": "configuration_preview_only", "agent": recipe["agent"], "verl_overrides": config}, indent=2))
        return
    if sys.platform != "linux":
        raise RuntimeError("GPU training requires the documented Linux/CUDA environment")
    if run.exists():
        raise FileExistsError("Choose a new run directory; existing checkpoints are never overwritten")
    if not (model / "config.json").is_file():
        raise FileNotFoundError("Download the pinned model snapshot first")
    tasks, val = load_tasks(data, "train"), load_tasks(data, "internal_dev")
    if len(tasks) != 6563 or len(val) != 434:
        raise ValueError("Data counts differ from the documented Spider preparation")
    if {r["db_id"] for r in tasks} & {r["db_id"] for r in val}:
        raise ValueError("Training and internal-dev databases overlap")
    gold = json.loads((data / "evaluator/gold.json").read_text(encoding="utf-8"))
    if any(r["source_id"] not in gold for r in tasks + val):
        raise ValueError("Evaluator references are incomplete")
    del gold
    tasks, padding = pad_training_rows(tasks)
    os.environ.update({
        "VERL_SPIDER_DATA_DIR": str(data), "SQL_AGENT_GOLD_PATH": str(data / "evaluator/gold.json"),
        "SQL_AGENT_ARTIFACT_DIR": str(run / "traces"), "SQL_AGENT_TOKENIZER_PATH": str(model),
        "WANDB_MODE": "disabled", "AGENTOPS_API_KEY": "local-dummy", "DO_NOT_TRACK": "1",
        "OPENAI_API_KEY": "local-dummy", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "2",
        "AGL_SERVER_HOST": "127.0.0.1", "OTEL_EXPORTER_OTLP_TIMEOUT": "60", "OTEL_EXPORTER_OTLP_TRACES_TIMEOUT": "60",
    })
    verify_runtime()
    import random
    import numpy as np
    import torch
    import ray
    import agentlightning as agl
    from omegaconf import OmegaConf
    from .agent import LitSQLAgent
    from .callbacks import AuditedDaemon, AuditedTrainer
    from .trace_adapter import CompleteTraceAdapter

    if torch.cuda.device_count() != 2:
        raise RuntimeError("Expose exactly two GPUs with CUDA_VISIBLE_DEVICES")
    seed = config["data"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    run.mkdir(parents=True, exist_ok=False)
    (run / "traces").mkdir()
    (run / "exposure_padding.json").write_text(json.dumps(padding), encoding="utf-8")
    (run / "recipe.json").write_text(json.dumps(recipe, indent=2), encoding="utf-8")
    runtime = args.runtime_dir.resolve() if args.runtime_dir else Path(tempfile.mkdtemp(prefix="sqlrl-"))
    if len(str(runtime).encode()) > 30:
        raise ValueError("Choose a Ray runtime path of at most 30 bytes")
    runtime.mkdir(parents=True, exist_ok=True)
    algorithm = agl.VERL(config, trainer_cls=AuditedTrainer, daemon_cls=AuditedDaemon)
    (run / "resolved_config.json").write_text(json.dumps(OmegaConf.to_container(algorithm.config, resolve=True), indent=2, default=str), encoding="utf-8")
    try:
        ray.init(address="local", _node_ip_address="127.0.0.1", include_dashboard=False, _temp_dir=str(runtime),
                 num_cpus=16, num_gpus=2, object_store_memory=4 * 1024**3)
        trainer = agl.Trainer(
            n_runners=recipe["agent"]["n_runners"], algorithm=algorithm,
            adapter=CompleteTraceAdapter(agent_match=recipe["agent"]["agent_match"], audit_dir=str(run / "traces/adapter")),
            strategy={"type": "cs", "server_host": "127.0.0.1", "server_port": args.port},
        )
        trainer.fit(LitSQLAgent(feedback_mode=recipe["agent"]["feedback_mode"], max_turns=3, val_temperature=0.0),
                    train_dataset=tasks, val_dataset=val)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
