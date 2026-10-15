"""Componente determinístico: tendencia + dummies de año, día de la semana y hora (OLS).

Corrección respecto al manuscrito original: con intercepto, las dummies de año deben
omitir una categoría de referencia (2017); de lo contrario la matriz es singular.
La tendencia se mide en tiempo real (años desde el inicio), no por índice de fila,
para respetar los ~9 meses de hueco entre inviernos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def design_matrix(time: pd.Series, dec: dict) -> tuple[np.ndarray, list[str]]:
    cols = [np.ones(len(time))]
    names = ["const"]
    if dec.get("trend", True):
        t = (time - time.iloc[0]).dt.total_seconds().to_numpy() / (3600 * 24 * 365.25)
        cols.append(t)
        names.append("trend_years")
    if dec.get("year_dummies", True):
        years = sorted(time.dt.year.unique())
        for y in years[1:]:
            cols.append((time.dt.year == y).to_numpy(float))
            names.append(f"year_{y}")
    if dec.get("dow_dummies", True):
        # domingo (6) = referencia -> lunes..sábado
        for d, lab in enumerate(["lun", "mar", "mie", "jue", "vie", "sab"]):
            cols.append((time.dt.dayofweek == d).to_numpy(float))
            names.append(f"dow_{lab}")
    if dec.get("hour_dummies", True):
        for h in range(1, 24):
            cols.append((time.dt.hour == h).to_numpy(float))
            names.append(f"hour_{h:02d}")
    return np.column_stack(cols), names


def fit_deterministic(ly: np.ndarray, X: np.ndarray, train_end: int) -> dict:
    """OLS sobre [0, train_end). Devuelve componente determinístico para toda la serie."""
    Xtr, ytr = X[:train_end], ly[:train_end]
    keep = Xtr.std(axis=0) > 0
    keep[0] = True  # intercepto
    rank = np.linalg.matrix_rank(Xtr[:, keep])
    if rank < keep.sum():
        raise np.linalg.LinAlgError("Matriz de diseño determinística con rango deficiente")
    beta = np.zeros(X.shape[1])
    beta[keep] = np.linalg.lstsq(Xtr[:, keep], ytr, rcond=None)[0]
    zd = X @ beta
    resid_tr = ytr - zd[:train_end]
    r2 = 1 - resid_tr.var() / ytr.var()
    return {"beta": beta, "zd": zd, "e": ly - zd, "r2": float(r2), "dropped": (~keep).sum()}


def ols_table(ly: np.ndarray, X: np.ndarray, names: list[str], train_end: int) -> pd.DataFrame:
    """Coeficientes con errores estándar HAC (Newey–West, 24 rezagos) para el material suplementario."""
    import statsmodels.api as sm

    res = sm.OLS(ly[:train_end], X[:train_end]).fit(cov_type="HAC", cov_kwds={"maxlags": 24})
    return pd.DataFrame({"term": names, "coef": res.params, "se_hac": res.bse, "p": res.pvalues})
