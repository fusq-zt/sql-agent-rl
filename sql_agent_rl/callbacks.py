# Copyright (c) Microsoft. All rights reserved.
# Adapted for SQL Agent RL: portable package imports/entrypoints; see THIRD_PARTY_NOTICES.md.
"""Observe the official trainer and retain best/last without replacing its loop."""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig
from verl import DataProto

from agentlightning.verl.daemon import AgentModeDaemon
from agentlightning.verl.trainer import AgentLightningTrainer


class AuditedDaemon(AgentModeDaemon):
    """Record the actual four-rollout groups before upstream builds transitions."""

    def get_train_data_batch(
        self, max_prompt_length: int, max_response_length: int, device: torch.device, global_steps: int
    ) -> tuple[DataProto, dict[str, Any]]:
        groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for rollout_id, rollout in self._completed_rollouts_v0.items():
            source = self._task_id_to_original_sample[rollout_id]
            triplets = rollout.triplets or []
            if any(not t.prompt.get("token_ids") or not t.response.get("token_ids") for t in triplets):
                raise RuntimeError("Selected training trace is missing actual prompt/response token IDs")
            groups[source["data_id"]].append(
                {
                    "source_id": source["source_id"],
                    "rollout_id": rollout_id,
                    "reward": rollout.final_reward,
                    "selected_nodes": len(triplets),
                    "prompt_tokens": sum(len(t.prompt["token_ids"]) for t in triplets),
                    "response_tokens": sum(len(t.response["token_ids"]) for t in triplets),
                }
            )
        if any(len(values) != 4 for values in groups.values()):
            raise RuntimeError("Refusing an incomplete GRPO group")
        records = [
            {"data_id": key, "rollouts": values, "reward_variance": float(np.var([v["reward"] for v in values]))}
            for key, values in groups.items()
        ]
        path = Path(os.environ["SQL_AGENT_ARTIFACT_DIR"]) / "reward_groups.jsonl"
        with path.open("a") as stream:
            stream.write(json.dumps({"global_step": global_steps, "groups": records}) + "\n")
        return super().get_train_data_batch(max_prompt_length, max_response_length, device, global_steps)


class AuditedTrainer(AgentLightningTrainer):
    """Delegate every update/validation/save/load to the pinned upstream trainer.

    Additional operations are local evidence recording, driver RNG persistence,
    and hardlink retention of the best *trained* checkpoint. No loss/mask,
    reward, optimizer or rollout algorithm is implemented here.
    """

    config: DictConfig

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # The upstream trainer forwards untyped backend-specific kwargs.
        super().__init__(*args, **kwargs)  # pyright: ignore[reportUnknownMemberType]
        seed = int(self.config.data.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
        self.audit_dir = Path(os.environ["SQL_AGENT_ARTIFACT_DIR"])
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_root = Path(self.config.trainer.default_local_dir).resolve()
        self.selection_file = self.checkpoint_root / "selection.json"
        self.selection: dict[str, Any] = (
            json.loads(self.selection_file.read_text()) if self.selection_file.exists() else {}
        )
        self.last_validation: dict[str, Any] | None = None

    def record(self, kind: str, values: dict[str, Any]) -> None:
        record = {"kind": kind, "global_step": getattr(self, "global_steps", 0), "timestamp": time.time(), **values}
        with (self.audit_dir / "training_metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record, default=float) + "\n")

    def _train_step(self, batch_dict: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        self.record("input_batch", {"source_ids": list(batch_dict["source_id"])})
        # The inherited trainer returns an unparameterized metric dictionary.
        metrics = cast(dict[str, Any], super()._train_step(batch_dict))  # pyright: ignore[reportUnknownMemberType]
        self.record("update", {"metrics": metrics, "wall_seconds": time.monotonic() - started})
        for key, value in metrics.items():
            if any(term in key for term in ("loss", "grad_norm", "reward")):
                if isinstance(value, (float, int, np.number)) and not math.isfinite(float(cast(Any, value))):
                    raise FloatingPointError(f"Non-finite actual training metric: {key}={value}")
        return metrics

    def _validate(self) -> dict[str, Any]:
        started = time.monotonic()
        metrics = super()._validate()
        self.last_validation = {"step": self.global_steps, "score": float(metrics["val/reward"])}
        self.record("validation", {"metrics": metrics, "wall_seconds": time.monotonic() - started})
        return metrics

    def _save_checkpoint(self) -> None:
        started = time.monotonic()
        super()._save_checkpoint()
        current = self.checkpoint_root / f"global_step_{self.global_steps}"
        torch.save(
            {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state()},
            current / "driver_rng.pt",
        )
        assert self.last_validation and self.last_validation["step"] == self.global_steps
        score = self.last_validation["score"]
        best = self.selection.get("best")
        if best is None or score > best["score"]:
            # A separate link protects files from upstream's rolling deletion.
            # Use unique step directories; do not overwrite an earlier artifact.
            destination = self.checkpoint_root / "best" / current.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(current, destination, copy_function=os.link)
            previous = Path(best["path"]) if best else None
            self.selection["best"] = {"step": self.global_steps, "score": score, "path": str(destination)}
            self._write_selection(current, score)
            if previous and previous != destination:
                previous = previous.resolve()
                assert previous.parent == (self.checkpoint_root / "best").resolve()
                assert previous.name.startswith("global_step_")
                shutil.rmtree(previous)
        else:
            self._write_selection(current, score)
        self.record("checkpoint", {**self.selection, "wall_seconds": time.monotonic() - started})

    def _write_selection(self, current: Path, score: float) -> None:
        self.selection.update(
            last={"step": self.global_steps, "score": score, "path": str(current)},
            rule="max internal-dev final execution accuracy; earliest trained step on ties",
        )
        temporary = self.selection_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.selection, indent=2))
        temporary.replace(self.selection_file)

    def _load_checkpoint(self) -> Any:
        result = super()._load_checkpoint()
        if self.global_steps:
            folder = (
                Path(self.config.trainer.resume_from_path)
                if self.config.trainer.resume_mode == "resume_path"
                else self.checkpoint_root / f"global_step_{self.global_steps}"
            )
            path = folder / "driver_rng.pt"
            state = torch.load(path, weights_only=False, map_location="cpu")
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(state["torch_cpu"])
            self.record(
                "resume", {"path": str(folder), "driver_rng_restored": True, "bitwise_continuation_claimed": False}
            )
        return result
