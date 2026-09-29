# Copyright (c) Microsoft. All rights reserved.
# Adapted for SQL Agent RL: portable package imports/entrypoints; see THIRD_PARTY_NOTICES.md.
"""Permission/deadline wrapper retaining upstream SQL comparison and flags."""

from __future__ import annotations

from pathlib import Path

from .spider_eval import exec_eval
from .sql_safety import SQL_TIMEOUT_SECONDS, read_only_connection


def _safe_cursor(path: str):
    return read_only_connection(path, SQL_TIMEOUT_SECONDS).cursor()


def evaluate_query(query: str, ground_truth: str, database: str, raise_on_error: bool = True) -> float:
    """Compare complete results using the unchanged upstream evaluator.

    The upstream async timeout cannot interrupt blocking sqlite execute/fetchall;
    the connection progress handler implements the advertised 60 s VM timeout.
    Gold execution exceptions remain infrastructure/data errors, never reward 0.
    """
    Path(database).resolve(strict=True)
    exec_eval.get_cursor_from_path = _safe_cursor
    return float(
        exec_eval.eval_exec_match(
            db=str(Path(database).resolve()),
            p_str=query,
            g_str=ground_truth,
            plug_value=False,
            keep_distinct=False,
            progress_bar_for_each_datapoint=False,
        )
    )
