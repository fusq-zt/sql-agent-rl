# Copyright (c) Microsoft. All rights reserved.
# Adapted for SQL Agent RL: portable package imports/entrypoints; see THIRD_PARTY_NOTICES.md.
"""Keep upstream schema formatting while reading SQLite sample values verbatim."""

import sqlite3
from typing import Any, cast

from langchain_community.utilities import SQLDatabase
from sqlalchemy import Table, select
from sqlalchemy.exc import NoReferencedColumnError, NoReferencedTableError, NoSuchTableError


class SQLiteSchemaDatabase(SQLDatabase):
    """Avoid SQLAlchemy date coercion on SQLite's permissive DATE columns.

    Spider includes text such as ``28-AUG-2011`` in DATE columns. SQLite can
    query those values, but SQLAlchemy's result processors reject them. The
    raw DBAPI cursor preserves the database evidence, with upstream's exact
    row limit, column selection, formatting and 100-character value limit.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Defer reflection so the same public fallback covers malformed FK
        # declarations that SQLite itself permits and SELECT queries can use.
        kwargs["lazy_table_reflection"] = True
        self.schema_fallback_reason: str | None = None
        super().__init__(*args, **kwargs)  # pyright: ignore[reportUnknownMemberType]  # LangChain's schema metadata generic is unbound.

    def get_table_info(self, table_names: list[str] | None = None, get_col_comments: bool = False) -> str:
        try:
            return super().get_table_info(table_names, get_col_comments=get_col_comments)
        except (NoReferencedColumnError, NoReferencedTableError, NoSuchTableError) as error:
            self.schema_fallback_reason = type(error).__name__
            tables = sorted(table_names or self.get_usable_table_names())
            blocks: list[str] = []
            with self._engine.connect() as connection:
                cursor = cast(sqlite3.Cursor, connection.connection.cursor())
                try:
                    for name in tables:
                        ddl = cursor.execute(
                            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (name,)
                        ).fetchone()
                        if ddl is None:
                            raise ValueError(f"Missing table: {name}")
                        quoted = '"' + name.replace('"', '""') + '"'
                        cursor.execute(f"SELECT * FROM {quoted} LIMIT {int(self._sample_rows_in_table_info)}")
                        assert cursor.description is not None
                        columns = "\t".join(column[0] for column in cursor.description)
                        samples = "\n".join("\t".join(str(v)[:100] for v in row) for row in cursor.fetchall())
                        blocks.append(
                            f"{ddl[0]}\n\n/*\n{self._sample_rows_in_table_info} rows from {name} table:\n{columns}\n{samples}\n*/"
                        )
                finally:
                    cursor.close()
            return "\n\n".join(blocks)

    def _get_sample_rows(self, table: Table) -> str:
        command = select(table).limit(self._sample_rows_in_table_info)
        compiled = command.compile(self._engine, compile_kwargs={"literal_binds": True})
        with self._engine.connect() as connection:
            cursor = cast(sqlite3.Cursor, connection.connection.cursor())
            try:
                cursor.execute(str(compiled))
                rows = cursor.fetchall()
            finally:
                cursor.close()
        columns = "\t".join(col.name for col in table.columns)
        samples = "\n".join("\t".join(str(value)[:100] for value in row) for row in rows)
        return f"{self._sample_rows_in_table_info} rows from {table.name} table:\n{columns}\n{samples}"
