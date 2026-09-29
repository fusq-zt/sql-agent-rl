"""Dependency-free configuration and task validation for the public entrypoint."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

DEFAULT_RECIPE = Path(__file__).resolve().parents[1] / "configs/grpo.json"
NODE_PATTERN = r"^(write_query|rewrite_query)$"


def load_recipe(path: Path = DEFAULT_RECIPE) -> dict:
    recipe = json.loads(path.read_text(encoding="utf-8"))
    if recipe["agent"]["agent_match"] != NODE_PATTERN:
        raise ValueError("This entrypoint trains the write/rewrite pair")
    if recipe["agent"]["max_turns"] != 3 or recipe["agent"]["n_runners"] != 8:
        raise ValueError("The documented recipe uses three SQL candidates and eight runners")
    cfg = recipe["verl"]
    if cfg["trainer"]["n_gpus_per_node"] != 2 or cfg["actor_rollout_ref"]["rollout"]["n"] != 4:
        raise ValueError("The documented recipe uses two GPUs and four rollouts per question")
    return recipe


def make_config(recipe: dict, data: Path, model: Path, run: Path, train_count: int = 6563) -> dict:
    cfg = copy.deepcopy(recipe["verl"])
    cfg["data"].update(train_files=str(data / "splits/train.parquet"), val_files=str(data / "splits/internal_dev.parquet"))
    cfg["actor_rollout_ref"]["model"]["path"] = str(model)
    frequency = math.ceil(math.ceil(train_count / cfg["data"]["train_batch_size"]) / 4)
    cfg["trainer"].update(default_local_dir=str(run / "checkpoints"), save_freq=frequency, test_freq=frequency)
    return cfg


def load_tasks(data: Path, split: str) -> list[dict]:
    if split not in {"train", "internal_dev", "heldout"}:
        raise ValueError("Unknown split")
    from .sql_safety import resolve_database

    rows = [json.loads(line) for line in (data / "splits" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Empty task split")
    identifiers = set()
    for row in rows:
        if "query" in row or "gold" in row:
            raise ValueError("Reference SQL must not enter model tasks")
        if row["source_id"] in identifiers or row["split"] != split or not row["question"].strip():
            raise ValueError("Invalid task identity/split/question")
        identifiers.add(row["source_id"])
        resolve_database(data, row)
    return rows


def pad_training_rows(rows: list[dict], batch_size: int = 32) -> tuple[list[dict], list[str]]:
    if not rows or batch_size <= 0:
        raise ValueError("Nonempty rows and a positive batch size are required")
    extra = (-len(rows)) % batch_size
    padding = (sorted(rows, key=lambda row: row["source_id"]) * math.ceil(extra / len(rows)))[:extra]
    return rows + padding, [row["source_id"] for row in padding]


def require_complete_actions(candidate_count: int, selected_names: list[str]) -> None:
    if not 1 <= candidate_count <= 3:
        raise ValueError("A rollout must contain one to three executed SQL candidates")
    expected = ["write_query"] + ["rewrite_query"] * (candidate_count - 1)
    if selected_names != expected:
        raise RuntimeError(f"Incomplete policy actions: expected {expected}, got {selected_names}")
