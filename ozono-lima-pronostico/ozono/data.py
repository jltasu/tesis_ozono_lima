"""Carga, verificación e imputación (MICE) de los datos por estación."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_station(code: str, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Lee el CSV de una estación, verifica su estructura e imputa si hay faltantes.

    Devuelve el DataFrame (columnas: time, season, target, covariables, imputed)
    y un diccionario con el registro de calidad.
    """
    dc = cfg["data"]
    path = Path(cfg["_data_dir"]) / dc["stations"][code]
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df[dc["time_col"]], format=dc["time_format"])
    df = df.sort_values("time").reset_index(drop=True)
    tgt = dc["target"]
    covs = [c for c in dc["covariates"] if c in df.columns]

    log: dict = {"station": code, "file": path.name, "n": len(df),
                 "start": str(df.time.iloc[0]), "end": str(df.time.iloc[-1])}
    log["duplicated_times"] = int(df.time.duplicated().sum())
    df = df[~df.time.duplicated()].reset_index(drop=True)
    df["season"] = df.time.dt.year

    # huecos dentro de cada invierno (se espera paso horario exacto)
    gaps = 0
    for _, g in df.groupby("season"):
        d = g.time.diff().dropna()
        gaps += int((d != pd.Timedelta(hours=1)).sum())
    log["intra_season_gaps"] = gaps
    log["hours_per_season"] = df.groupby("season").size().to_dict()
    log["missing_before_impute"] = {c: int(df[c].isna().sum()) for c in [tgt] + covs}
    log["covariates_available"] = covs
    log["nonpositive_target"] = int((df[tgt] <= 0).sum())

    df["imputed"] = df[tgt].isna()
    if df[[tgt] + covs].isna().any().any():
        if not dc.get("impute_if_missing", True):
            raise ValueError(f"{code}: hay faltantes y impute_if_missing=false")
        df = _mice(df, tgt, covs, cfg["seed"])
        log["imputation"] = "MICE (IterativeImputer, BayesianRidge, 10 iteraciones) sobre objetivo + covariables + hora + día"
    else:
        log["imputation"] = ("El CSV no tiene faltantes: los datos ya vienen imputados y no traen marca de "
                             "imputación, por lo que la Tabla 1 (horas imputadas) no se puede regenerar desde este archivo.")
    log["n_imputed_target"] = int(df["imputed"].sum())

    # valores no positivos: necesarios para log
    if cfg["transform"] == "log" and (df[tgt] <= 0).any():
        floor = df.loc[df[tgt] > 0, tgt].min() / 2
        log["log_floor_applied"] = float(floor)
        df.loc[df[tgt] <= 0, tgt] = floor
    return df, log


def _mice(df: pd.DataFrame, tgt: str, covs: list[str], seed: int) -> pd.DataFrame:
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    X = df[[tgt] + covs].copy()
    X["hour_sin"] = np.sin(2 * np.pi * df.time.dt.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * df.time.dt.hour / 24)
    X["dow"] = df.time.dt.dayofweek
    imp = IterativeImputer(max_iter=10, random_state=seed, sample_posterior=False)
    Xi = imp.fit_transform(X)
    out = df.copy()
    for j, c in enumerate([tgt] + covs):
        out[c] = Xi[:, j]
    return out


def flatline_runs(y: np.ndarray, min_len: int = 3) -> tuple[int, int]:
    """Número de rachas de valores idénticos consecutivos (>= min_len) y horas totales en ellas."""
    runs, hours, cur = 0, 0, 1
    for i in range(1, len(y)):
        if y[i] == y[i - 1]:
            cur += 1
        else:
            if cur >= min_len:
                runs += 1
                hours += cur
            cur = 1
    if cur >= min_len:
        runs += 1
        hours += cur
    return runs, hours
