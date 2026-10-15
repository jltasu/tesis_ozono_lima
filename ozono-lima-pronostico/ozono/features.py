"""Matrices de rezagos para pronóstico directo a h pasos.

Convención: para el objetivo en el índice tau, el origen del pronóstico es tau - h.
La fila tau contiene [e_{tau-h}, e_{tau-h-1}, ..., e_{tau-h-p+1}] (todo conocido en el origen).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def lag_matrix(e: np.ndarray, p: int, h: int) -> np.ndarray:
    n = len(e)
    X = np.full((n, p), np.nan)
    for k in range(p):
        s = h + k
        if s < n:
            X[s:, k] = e[: n - s]
    return X


def covariate_matrix(df: pd.DataFrame, covs: list[str], lags: list[int], h: int,
                     wd_col: str | None) -> tuple[np.ndarray, list[str]]:
    """Covariables observadas hasta el origen: lag l=1 es el valor en tau-h."""
    blocks, names = [], []
    for c in covs:
        if c == wd_col:
            rad = np.deg2rad(df[c].to_numpy(float))
            series = {f"{c}_sin": np.sin(rad), f"{c}_cos": np.cos(rad)}
        else:
            series = {c: df[c].to_numpy(float)}
        for nm, v in series.items():
            for l in lags:
                s = h + l - 1
                col = np.full(len(v), np.nan)
                col[s:] = v[: len(v) - s]
                blocks.append(col)
                names.append(f"{nm}_lag{l}")
    return np.column_stack(blocks), names


def valid_mask(season: np.ndarray, h: int, maxlag: int, drop_cross: bool) -> np.ndarray:
    """True si toda la ventana [tau-h-maxlag+1, tau] existe y (opcional) cae en un mismo invierno."""
    n = len(season)
    first = np.arange(n) - h - maxlag + 1
    ok = first >= 0
    if drop_cross:
        same = np.zeros(n, bool)
        same[ok] = season[first[ok]] == season[ok]
        ok &= same
    return ok


def select_p_bic(e: np.ndarray, season: np.ndarray, train_end: int, max_p: int, drop_cross: bool) -> tuple[int, pd.DataFrame]:
    """Orden autorregresivo por BIC de un AR(p) lineal (OLS), con muestra común para todos los p."""
    mask = valid_mask(season, 1, max_p, drop_cross)
    idx = np.where(mask[:train_end])[0]
    rows = []
    Xall = lag_matrix(e, max_p, 1)
    y = e[idx]
    n = len(idx)
    for p in range(1, max_p + 1):
        A = np.column_stack([np.ones(n), Xall[idx, :p]])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        rss = float(((y - A @ coef) ** 2).sum())
        k = p + 1
        bic = n * np.log(rss / n) + k * np.log(n)
        aic = n * np.log(rss / n) + 2 * k
        rows.append({"p": p, "BIC": bic, "AIC": aic})
    tab = pd.DataFrame(rows)
    return int(tab.loc[tab.BIC.idxmin(), "p"]), tab
