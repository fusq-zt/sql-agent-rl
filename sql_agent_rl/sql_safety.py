# Copyright (c) Microsoft. All rights reserved.
# Adapted for SQL Agent RL: portable package imports/entrypoints; see THIRD_PARTY_NOTICES.md.
"""Read-only SQLite connections, real VM deadlines, and bounded observations.

The evaluator still compares complete results. The preview limit applies only
to what the agent sees. This is a SQLite policy boundary, not a code sandbox.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from .feedback import ExecutionObservation

SQL_TIMEOUT_SECONDS = 60.0
PREVIEW_ROWS = 50
PREVIEW_CHARS = 2048
_READ_PRAGMAS = frozenset({"table_info", "table_xinfo", "index_list", "index_info", "index_xinfo", "foreign_key_list"})
_ALLOWED = frozenset({sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE})
_FORBIDDEN_FUNCTIONS = frozenset({"load_extension", "readfile", "writefile", "fts3_tokenizer"})


def read_only_connection(path: str | Path, timeout: float = SQL_TIMEOUT_SECONDS) -> sqlite3.Connection:
    """Open an existing DB read-only and enforce permission and execution limits."""
    db = Path(path).resolve(strict=True)
    connection = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True, timeout=min(timeout, 5.0))
    connection.enable_load_extension(False)
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA trusted_schema=OFF")

    def decode_text(value: bytes) -> str:
        return value.decode(errors="ignore")

    connection.text_factory = decode_text
    connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 8 * 1024 * 1024)
    connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 100_000)
    deadline = time.monotonic() + timeout
    connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)

    def authorize(action: int, first: str | None, second: str | None, database: str | None, trigger: str | None) -> int:
        if action == sqlite3.SQLITE_PRAGMA:
            name = (first or "").lower()
            permitted = name in _READ_PRAGMAS or (name in {"read_uncommitted", "foreign_keys"} and second is None)
            return sqlite3.SQLITE_OK if permitted else sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION and (second or "").lower() in _FORBIDDEN_FUNCTIONS:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK if action in _ALLOWED else sqlite3.SQLITE_DENY

    connection.set_authorizer(authorize)
    return connection


def execute_observation(
    path: str | Path,
    query: str,
    *,
    timeout: float = SQL_TIMEOUT_SECONDS,
    preview_rows: int = PREVIEW_ROWS,
    preview_chars: int = PREVIEW_CHARS,
) -> ExecutionObservation:
    """Execute SQL and return observable evidence without any reference answer."""
    connection = read_only_connection(path, timeout)
    try:
        cursor = connection.execute(query)
        rows = cursor.fetchmany(preview_rows + 1)
        truncated = len(rows) > preview_rows
        preview = json.dumps(rows[:preview_rows], ensure_ascii=False, default=str, separators=(",", ":"))
        truncated |= len(preview) > preview_chars
        return ExecutionObservation("success", result_preview=preview[:preview_chars], result_truncated=truncated)
    except sqlite3.Error as exc:
        message = str(exc)
        if "interrupted" in message.lower() or "locked" in message.lower():
            status = "execution_timeout"
        elif any(s in message.lower() for s in ("not authorized", "readonly", "read-only", "authorization denied")):
            status = "policy_violation"
        else:
            status = "sql_error"
        return ExecutionObservation(status, raw_message=message[:preview_chars])
    finally:
        connection.close()


def resolve_database(data_dir: str | Path, task: dict[str, Any]) -> Path:
    """Resolve explicit source location, independent of train/validation runtime mode."""
    root = Path(data_dir).resolve(strict=True)
    relative = task.get("db_relative_path")
    if not relative:
        raise ValueError("Dataset manifest must supply db_relative_path")
    database = (root / relative).resolve(strict=True)
    if not database.is_relative_to(root) or database.suffix != ".sqlite":
        raise ValueError("Database path escapes the manifest root or is not SQLite")
    if database.stem != task["db_id"]:
        raise ValueError("Database id does not match filename")
    return database
