# Copyright (c) Microsoft. All rights reserved.
# Adapted for SQL Agent RL: portable package imports/entrypoints; see THIRD_PARTY_NOTICES.md.
"""Build a source-audited, db-disjoint Spider manifest before any model evaluation.

Usage: python -m sql_agent_rl.prepare_data --archive spider-data.zip --output data
The supplied upstream archive is retained unchanged; only train_spider and dev
annotations participate. Gold is kept in a separate evaluator-only mapping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .sql_safety import read_only_connection


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def prepare(archive: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError("Choose a new data directory")
    expected = "e6d2efb262d9a8a57cd9ddddd63a9c9522bab71775013319b6aa41ea76a28d7f"
    if digest(archive) != expected:
        raise ValueError("Spider archive differs from the documented source")
    output.mkdir(parents=True, exist_ok=False)
    for directory in ("source", "evaluator", "splits"):
        (output / directory).mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        annotations = {name: json.loads(source.read(name)) for name in ("train_spider.json", "dev.json")}
        source_counts = {
            name: len(json.loads(source.read(name)))
            for name in ("train_spider.json", "train_others.json", "dev.json", "test.json")
        }
        databases = sorted({row["db_id"] for rows in annotations.values() for row in rows})
        for name in ("train_spider.json", "dev.json", "tables.json", "README.txt"):
            (output / "source" / name).write_bytes(source.read(name))
        for database in databases:
            assert database.replace("_", "").isalnum(), database
            directory = output / "database" / database
            directory.mkdir(parents=True, exist_ok=True)
            for suffix in (database + ".sqlite", "schema.sql"):
                name = f"database/{database}/{suffix}"
                if name in source.namelist():
                    (directory / suffix).write_bytes(source.read(name))

    integrity = []
    for database in databases:
        path = output / "database" / database / f"{database}.sqlite"
        try:
            connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
            try:
                status = connection.execute("PRAGMA quick_check").fetchall()
            finally:
                connection.close()
            integrity.append({"db_id": database, "sha256": digest(path), "quick_check": status})
            if status != [("ok",)]:
                raise ValueError(f"Integrity failure: {database}: {status}")
        except Exception as error:
            raise RuntimeError(f"Database infrastructure check failed: {database}") from error

    tasks, gold, checks = [], {}, []
    for origin, rows in annotations.items():
        for index, row in enumerate(rows):
            for field in ("question", "query", "db_id"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"Invalid source field: {origin}:{index}:{field}")
            source_id = f"{origin}:{index:05d}"
            tasks.append(
                {
                    "source_id": source_id,
                    "db_id": row["db_id"],
                    "question": row["question"],
                    "source": origin,
                    "source_row_sha256": canonical_hash(row),
                    "db_relative_path": f"database/{row['db_id']}/{row['db_id']}.sqlite",
                }
            )
            gold[source_id] = row["query"]

    def check_gold(task):
        connection = read_only_connection(output / task["db_relative_path"])
        try:
            cursor = connection.execute(gold[task["source_id"]])
            count = 0
            while rows := cursor.fetchmany(1024):
                count += len(rows)
            return {"source_id": task["source_id"], "status": "ok", "rows": count}
        except sqlite3.Error as error:
            return {"source_id": task["source_id"], "status": "quarantine", "reason": str(error)}
        finally:
            connection.close()

    with ThreadPoolExecutor(max_workers=8) as pool:
        checks = list(pool.map(check_gold, tasks))
    quarantined = {check["source_id"] for check in checks if check["status"] != "ok"}
    eligible = [task for task in tasks if task["source_id"] not in quarantined]
    train_rows = [task for task in eligible if task["source"] == "train_spider.json"]
    heldout = [task for task in eligible if task["source"] == "dev.json"]
    train_dbs = sorted({task["db_id"] for task in train_rows})
    assert set(train_dbs).isdisjoint(task["db_id"] for task in heldout)
    random.Random(42).shuffle(train_dbs)
    counts = Counter(task["db_id"] for task in train_rows)
    internal_dbs, internal_count = [], 0
    for db in train_dbs:
        if internal_count >= 300 and abs(internal_count - 400) <= abs(internal_count + counts[db] - 400):
            break
        internal_dbs.append(db)
        internal_count += counts[db]
    split_rows = {
        "train": [task for task in train_rows if task["db_id"] not in internal_dbs],
        "internal_dev": [task for task in train_rows if task["db_id"] in internal_dbs],
        "heldout": heldout,
    }
    import pandas as pd

    for split, rows in split_rows.items():
        for task in rows:
            task["split"] = split
        (output / "splits" / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8", newline="\n"
        )
        pd.DataFrame(rows).to_parquet(output / "splits" / f"{split}.parquet", index=False)

    pairs = Counter((task["db_id"], task["question"]) for task in tasks)
    duplicate_questions = [
        {"db_id": db, "question": question, "count": count} for (db, question), count in pairs.items() if count > 1
    ]
    with zipfile.ZipFile(archive) as source:
        import io

        parquet_rows = pd.read_parquet(io.BytesIO(source.read("train_spider.parquet"))).to_dict("records")
    originals = annotations["train_spider.json"]
    parquet_matches_json = len(parquet_rows) == len(originals) and all(
        all(a[key] == b[key] for key in ("db_id", "question", "query")) for a, b in zip(parquet_rows, originals)
    )
    report = {
        "archive_sha256": digest(archive),
        "source_counts": source_counts,
        "quality_rule": "quarantine only pre-model reference-SQL execution failures; preserve duplicate source IDs",
        "quarantined": [c for c in checks if c["status"] != "ok"],
        "gold_checks": checks,
        "database_checks": integrity,
        "duplicate_questions_preserved": duplicate_questions,
        "train_parquet_matches_source_json": parquet_matches_json,
        "excluded_sources": ["train_others.json", "test.json", "test_dev.parquet", "test_dev_500.parquet"],
    }
    (output / "evaluator" / "gold.json").write_text(json.dumps(gold, ensure_ascii=False), encoding="utf-8")
    (output / "evaluator" / "data_checks.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "dataset": "Spider 1.0",
        "license": "CC BY-SA 4.0",
        "archive_sha256": report["archive_sha256"],
        "source_url": "https://drive.google.com/file/d/1oi9J1jZP9TyM35L85CL3qeGWl2jqlnL6/view",
        "split_seed": 42,
        "internal_dev_db_ids": internal_dbs,
        "source_counts": source_counts,
        "quarantine_count": len(quarantined),
        "splits": {},
    }
    for split, rows in split_rows.items():
        manifest["splits"][split] = {
            "questions": len(rows),
            "databases": len({r["db_id"] for r in rows}),
            "sha256": digest(output / "splits" / f"{split}.jsonl"),
            "source_ids": [r["source_id"] for r in rows],
        }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "split_counts": {k: len(v) for k, v in split_rows.items()},
                "quarantined": report["quarantined"],
                "duplicates": len(duplicate_questions),
                "parquet_matches_json": parquet_matches_json,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.archive.resolve(strict=True), args.output.resolve())
