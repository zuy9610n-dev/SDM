"""仮パラメータの読み込み(src/sdm5/data/params.yaml)。"""
from __future__ import annotations

import copy
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


@lru_cache(maxsize=None)
def _load_default() -> dict[str, Any]:
    text = resources.files("sdm5.data").joinpath("params.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


def load_params(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    """既定値を読み、任意の YAML と辞書で上書きした複製を返す。"""
    p = copy.deepcopy(_load_default())
    if path is not None:
        with open(path, encoding="utf-8") as f:
            _deep_update(p, yaml.safe_load(f) or {})
    if overrides:
        _deep_update(p, overrides)
    return p


def _deep_update(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
