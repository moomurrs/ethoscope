"""SQL / type mapping utilities - pure functions, no I/O."""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=128)
def map_mysql_to_sqlite(mysql_type: str) -> str:
    """Map a MySQL column type string to its SQLite equivalent.

    Uses structural pattern matching (3.12) to keep the dispatch explicit and
    fast after the first call due to ``lru_cache``.
    """
    upper = mysql_type.upper().strip()
    match upper:
        case t if t in {"FLOAT", "DOUBLE"}:
            return "REAL"
        case t if t.startswith("INT"):
            return "INTEGER"
        case t if t.startswith(("CHAR", "VARCHAR", "TEXT")):
            return "TEXT"
        case _:
            return "TEXT"


@functools.lru_cache(maxsize=128)
def map_sql_data_type_to_sqlite(sql_type: str) -> str:
    """Map ``BaseIntVariable.sql_data_type`` values to SQLite storage types."""
    upper = sql_type.upper()
    match upper:
        case t if "INT" in t:
            return "INTEGER"
        case t if "FLOAT" in t or "DOUBLE" in t:
            return "REAL"
        case t if "TEXT" in t or "CHAR" in t or "VARCHAR" in t:
            return "TEXT"
        case _:
            return "TEXT"


def sqlite_placeholders(n_cols: int, n_rows: int = 1) -> str:
    """Build ``(?, ?, ...)`` placeholders for ``n_rows`` rows."""
    single = "(" + ", ".join(["?"] * n_cols) + ")"
    if n_rows == 1:
        return single
    return ", ".join([single] * n_rows)
