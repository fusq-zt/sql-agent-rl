# Copyright (c) Microsoft. All rights reserved.
# Adapted for SQL Agent RL: portable package imports/entrypoints; see THIRD_PARTY_NOTICES.md.
"""Deterministic presentation of observable SQL results; accepts no gold data."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

Status = Literal["success", "sql_error", "policy_violation", "execution_timeout"]


@dataclass(frozen=True)
class ExecutionObservation:
    """One bounded, model-visible execution observation shared by both conditions."""

    status: Status
    raw_message: str = ""
    result_preview: str | None = None
    result_truncated: bool = False


def classify_error(message: str) -> tuple[str, str | None]:
    """Classify only explicit SQLite diagnostics, without inferring a repair."""
    patterns = (
        (r"no such table:\s*(.+)", "UNKNOWN_TABLE"),
        (r"no such column:\s*(.+)", "UNKNOWN_COLUMN"),
        (r"ambiguous column name:\s*(.+)", "AMBIGUOUS_COLUMN"),
        (r"no such function:\s*(.+)", "TYPE_OR_FUNCTION_ERROR"),
    )
    for pattern, kind in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return kind, match.group(1).strip()
    lowered = message.lower()
    if "syntax error" in lowered or "incomplete input" in lowered or "unrecognized token" in lowered:
        return "SYNTAX_ERROR", None
    if any(s in lowered for s in ("wrong number of arguments", "datatype mismatch", "misuse of")):
        return "TYPE_OR_FUNCTION_ERROR", None
    return "OTHER", None


def render_feedback(observation: ExecutionObservation, mode: str) -> str:
    """Render identical observed content as raw text or structured JSON."""
    if mode not in ("raw", "structured"):
        raise ValueError(f"Unknown feedback mode: {mode}")
    if mode == "raw":
        if observation.status == "success":
            return (observation.result_preview or "[]") + (
                "\n... (result truncated)" if observation.result_truncated else ""
            )
        return f"{observation.status}: {observation.raw_message}"
    payload: dict[str, Any] = asdict(observation)
    if observation.status != "success":
        error_type, identifier = classify_error(observation.raw_message)
        payload["error_type"] = error_type
        if identifier is not None:
            payload["identifier"] = identifier
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
