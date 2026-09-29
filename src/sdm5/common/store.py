"""版の管理(上位文書 3-3、ルール3)。

表は <root>/<table>/<YYYY-MM-DD>_<tag>.csv に保存し、同名の .meta.json に
パラメータ・データベース・モデルの版と内容のハッシュを記録する。保存済みの版は上書きしない(固定)。
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import ALL_TABLES, validate


class VersionExistsError(FileExistsError):
    pass


def content_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()[:16]


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, table: str, version: str) -> Path:
        return self.root / table / f"{version}.csv"

    def save(self, table: str, df: pd.DataFrame, tag: str = "v", meta: dict[str, Any] | None = None,
             date: str | None = None) -> str:
        if table in ALL_TABLES:
            validate(df, ALL_TABLES[table])
        date = date or _dt.date.today().isoformat()
        version = f"{date}_{tag}"
        p = self._path(table, version)
        if p.exists():
            raise VersionExistsError(f"版 {table}/{version} は固定済みです。tag を変えてください")
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        info = {"table": table, "version": version, "rows": len(df), "sha256_16": content_hash(df),
                "saved_at": _dt.datetime.now().isoformat(timespec="seconds"), **(meta or {})}
        p.with_suffix(".meta.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str),
                                               encoding="utf-8")
        return version

    def versions(self, table: str) -> list[str]:
        d = self.root / table
        return sorted(p.stem for p in d.glob("*.csv")) if d.exists() else []

    def load(self, table: str, version: str | None = None) -> pd.DataFrame:
        vs = self.versions(table)
        if not vs:
            raise FileNotFoundError(f"{table} の版がありません")
        version = version or vs[-1]
        return pd.read_csv(self._path(table, version))

    def meta(self, table: str, version: str) -> dict:
        return json.loads(self._path(table, version).with_suffix(".meta.json").read_text(encoding="utf-8"))
