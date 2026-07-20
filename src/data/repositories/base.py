from __future__ import annotations

import logging
from datetime import date as datetime_date
from typing import Any, Dict, List, Optional, TypeVar, Union
import pandas as pd
from sqlalchemy import Table, select, text
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.dialects.postgresql import insert

from src.data.db import get_engine, get_db_session, upsert_rows

logger = logging.getLogger(__name__)

T = TypeVar("T")


def parse_date_arg(val: Optional[Any]) -> Optional[datetime_date]:
    if val is None:
        return None
    if isinstance(val, datetime_date):
        return val
    if isinstance(val, str) and len(val) >= 10:
        try:
            return datetime_date.fromisoformat(val[:10])
        except ValueError:
            return None
    return None


class BaseRepository:
    """Base repository incorporating SQLAlchemy Core data access utilities."""

    def __init__(self, engine: Optional[Engine] = None) -> None:
        self._engine = engine

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = get_engine()
        return self._engine

    def connect(self) -> Connection:
        return self.engine.connect()

    def fetch_dataframe(
        self,
        clause: Any,
        params: Optional[Dict[str, Any]] = None,
        index_col: Optional[Union[str, List[str]]] = None,
        conn: Optional[Connection] = None,
    ) -> pd.DataFrame:
        """Executes a SQLAlchemy Core clause or text query and returns a Pandas DataFrame."""
        if conn is not None:
            df = pd.read_sql(clause, conn, params=params)
        else:
            with self.connect() as connection:
                df = pd.read_sql(clause, connection, params=params)

        if df.empty:
            return df

        if index_col:
            if isinstance(index_col, list):
                for col in index_col:
                    if col in df.columns and "date" in col.lower():
                        df[col] = pd.to_datetime(df[col])
            elif isinstance(index_col, str) and index_col in df.columns and "date" in index_col.lower():
                df[index_col] = pd.to_datetime(df[index_col])
            df = df.set_index(index_col).sort_index()

        return df

    def bulk_upsert(self, table: Table, rows: List[Dict[str, Any]], batch_size: int = 1000) -> int:
        """Delegates to production-ready batched upsert in db.py."""
        return upsert_rows(self.engine, table, rows, batch_size=batch_size)

    def execute(self, clause: Any, params: Optional[Dict[str, Any]] = None) -> Any:
        with self.engine.begin() as connection:
            return connection.execute(clause, params or {})

    def scalar(self, clause: Any, params: Optional[Dict[str, Any]] = None) -> Any:
        with self.connect() as connection:
            return connection.execute(clause, params or {}).scalar()
