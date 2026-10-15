#!/usr/bin/env python3
"""Ejecuta el pipeline completo: datos -> descomposición -> modelos -> validación -> tablas/figuras.

Uso:
    python run_all.py                     # corrida completa (para el manuscrito)
    python run_all.py --fast              # prueba de humo (minutos), salida en outputs_fast/
    python run_all.py --stations ATE SB   # solo algunas estaciones
    python run_all.py --report-only       # rehace tablas/figuras desde predicciones guardadas
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

from ozono import __version__
from ozono.config import load_config
from ozono.forecast import run_station
from ozono.generalization import run_generalization
from ozono.report import apply_boosting_label, boosting_label, build_report, metric_frame
from ozono.report_extra import generalization_tables, write_paragraphs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--fast", action="store_true", help="grillas reducidas; solo para verificar que todo corre")
    ap.add_argument("--stations", nargs="*", default=None)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, fast=args.fast, stations=args.stations)
    out = Path(cfg["_out"])
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    stations = list(cfg["data"]["stations"])
    t0 = time.time()

    if args.report_only:
        preds = pd.concat([pd.read_csv(out / "predictions" / f"{s}.csv.gz", parse_dates=["time"]) for s in stations])
        metas = {s: joblib.load(out / "predictions" / f"{s}_meta.joblib") for s in stations}
    else:
        print(f"Pipeline ozono v{__version__} | estaciones: {stations} | modo: {'FAST' if cfg['_fast'] else 'COMPLETO'}")
        n_jobs = min(int(cfg.get("n_jobs", 1)), len(stations))
        results = joblib.Parallel(n_jobs=n_jobs, verbose=0)(joblib.delayed(run_station)(s, cfg) for s in stations)
        preds = pd.concat([r[0] for r in results], ignore_index=True)
        metas = {r[1]["station"]: r[1] for r in results}
        for s in stations:
            preds[preds.station == s].to_csv(out / "predictions" / f"{s}.csv.gz", index=False)
            joblib.dump(metas[s], out / "predictions" / f"{s}_meta.joblib")

    lab = boosting_label(cfg)
    preds = apply_boosting_label(preds, lab)
    summary = build_report(preds, metas, cfg)

    # generalización entre inviernos y entre estaciones (responde a "ampliar el dataset")
    gpath = out / "predictions" / "generalizacion.csv.gz"
    if cfg.get("generalization", {}).get("enabled"):
        if args.report_only and gpath.exists():
            G = pd.read_csv(gpath, parse_dates=["time"])
        else:
            G = run_generalization(cfg, metas)
            G.to_csv(gpath, index=False)
        res = generalization_tables(G, cfg, out / "tables")
    else:
        res = {}
    M = metric_frame(preds, metas, cfg["metrics"]["near_zero_threshold"])
    write_paragraphs(M, res, cfg, stations, out)
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "pipeline": __version__,
           "fast": cfg["_fast"], "runtime_total_s": round(time.time() - t0, 1)}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "statsmodels", "matplotlib", "xgboost", "torch"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception as exc:  # noqa: BLE001
            env[mod] = f"no disponible ({type(exc).__name__})"
    from ozono.models import XGB
    env["XGB_backend"] = XGB(backend=cfg.get("models", {}).get("XGB", {}).get("backend", "auto")).backend
    epath = out / "entorno.json"
    if args.report_only and epath.exists():
        # conserva el entorno de la corrida que entrenó los modelos; solo registra la versión del reporte
        prev = json.loads(epath.read_text(encoding="utf-8"))
        prev["pipeline_tablas_figuras"] = __version__
        env = prev
    epath.write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nResumen (h = 1, prueba completa):")
    for s, r in summary["stations"].items():
        print(f"  {s}: mejor={r['best_univariate_h1']:<5} MAE={r['MAE_best']:.3f} (Naive {r['MAE_naive']:.3f}) "
              f"SMAPE={r['SMAPE_best']:.2f} (Naive {r['SMAPE_naive']:.2f}) MAE rel.={r['RelMAE_best']:.3f}")
    print(f"\nSalida en: {out}\nRevise primero: {out / 'verificaciones.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
