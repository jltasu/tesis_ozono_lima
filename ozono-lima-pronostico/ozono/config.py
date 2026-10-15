"""Carga de configuración (YAML) con sobreescritura opcional para --fast."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path, fast: bool = False, stations: list[str] | None = None) -> dict:
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = str(path.parent)
    if fast:
        cfg = _deep_update(copy.deepcopy(cfg), cfg.get("fast", {}))
        cfg["_fast"] = True
    else:
        cfg["_fast"] = False
    if stations:
        cfg["data"]["stations"] = {k: v for k, v in cfg["data"]["stations"].items() if k in stations}
        if not cfg["data"]["stations"]:
            raise ValueError(f"Ninguna estación coincide con {stations}")
    root = Path(cfg["_root"])
    cfg["_data_dir"] = str((root / cfg["data"]["dir"]).resolve())
    out = root / cfg["output_dir"]
    if cfg["_fast"]:
        out = out.parent / (out.name + "_fast")
    cfg["_out"] = str(out.resolve())
    return cfg
