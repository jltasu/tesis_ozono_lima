"""ARIMA(p,0,q) sobre el residuo estocástico: selección por AIC/BIC, ajuste MLE,
pronóstico recursivo a h pasos con parámetros fijos y diagnósticos (Ljung–Box, raíces)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX


def _fit(e: np.ndarray, p: int, q: int):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = SARIMAX(e, order=(p, 0, q), trend="n",
                      enforce_stationarity=True, enforce_invertibility=True)
        return mod.fit(disp=False, maxiter=200)


def select_order(e_train: np.ndarray, max_p: int, max_q: int, criterion: str) -> tuple[tuple[int, int], pd.DataFrame]:
    rows = []
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                r = _fit(e_train, p, q)
                rows.append({"p": p, "q": q, "AIC": r.aic, "BIC": r.bic, "converged": bool(r.mle_retvals.get("converged", True))})
            except Exception as exc:  # noqa: BLE001
                rows.append({"p": p, "q": q, "AIC": np.inf, "BIC": np.inf, "converged": False, "error": str(exc)})
    tab = pd.DataFrame(rows)
    crit = criterion.upper()
    best = tab.loc[tab[crit].idxmin()]
    return (int(best.p), int(best.q)), tab


def fit(e_train: np.ndarray, order: tuple[int, int]) -> dict:
    r = _fit(e_train, *order)
    return {"order": order, "ar": np.asarray(r.arparams), "ma": np.asarray(r.maparams), "res": r}


def innovations(e: np.ndarray, ar: np.ndarray, ma: np.ndarray) -> np.ndarray:
    p, q = len(ar), len(ma)
    eps = np.zeros(len(e))
    start = max(p, q)
    for t in range(start, len(e)):
        v = e[t]
        for i in range(p):
            v -= ar[i] * e[t - 1 - i]
        for k in range(q):
            v -= ma[k] * eps[t - 1 - k]
        eps[t] = v
    return eps


def forecast(e: np.ndarray, ar: np.ndarray, ma: np.ndarray, targets: np.ndarray, h: int) -> np.ndarray:
    """Pronóstico a h pasos para cada objetivo tau (origen tau-h), usando la información hasta el origen."""
    eps = innovations(e, ar, ma)
    p, q = len(ar), len(ma)
    origins = targets - h
    F = np.zeros((len(targets), h))
    for j in range(1, h + 1):
        f = np.zeros(len(targets))
        for i in range(1, p + 1):
            t = j - i  # posición relativa al origen
            f += ar[i - 1] * (F[:, t - 1] if t >= 1 else e[origins + t])
        for k in range(1, q + 1):
            t = j - k
            if t <= 0:
                f += ma[k - 1] * eps[origins + t]
        F[:, j - 1] = f
    return F[:, h - 1]


def diagnostics(fitd: dict, lags=(10, 20, 24, 168)) -> dict:
    r = fitd["res"]
    p, q = fitd["order"]
    resid = np.asarray(r.resid)[max(p, q):]
    lb = acorr_ljungbox(resid, lags=list(lags), model_df=p + q, return_df=True)
    out = {"p": p, "q": q, "AIC": float(r.aic), "BIC": float(r.bic)}
    for L in lags:
        out[f"LB_p_{L}"] = float(lb.loc[L, "lb_pvalue"])
    arr, mar = np.asarray(r.arroots), np.asarray(r.maroots)
    out["min_ar_root_mod"] = float(np.abs(arr).min()) if len(arr) else np.nan
    out["min_ma_root_mod"] = float(np.abs(mar).min()) if len(mar) else np.nan
    if len(arr) and len(mar):
        out["min_ar_ma_root_distance"] = float(np.min(np.abs(arr[:, None] - mar[None, :])))
    else:
        out["min_ar_ma_root_distance"] = np.nan
    return out


def coef_table(fitd: dict) -> pd.DataFrame:
    r = fitd["res"]
    return pd.DataFrame({"param": r.param_names, "coef": r.params, "se": r.bse, "p": r.pvalues})


# ----------------------------------------------------------------------------- SARIMA
def fit_sarima(e_train: np.ndarray, order: tuple[int, int], seasonal: tuple[int, int], s: int = 24) -> dict:
    """SARIMA(p,0,q)(P,0,Q)_s sobre el residuo. Devuelve los polinomios expandidos
    (producto estacional x no estacional) para reutilizar `forecast` con parámetros fijos."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = SARIMAX(e_train, order=(order[0], 0, order[1]),
                      seasonal_order=(seasonal[0], 0, seasonal[1], s), trend="n",
                      enforce_stationarity=True, enforce_invertibility=True)
        r = mod.fit(disp=False, maxiter=200)
    ar_full = np.convolve(r.polynomial_ar, r.polynomial_seasonal_ar)   # [1, -a1, -a2, ...]
    ma_full = np.convolve(r.polynomial_ma, r.polynomial_seasonal_ma)   # [1, m1, m2, ...]
    return {"order": order, "seasonal": seasonal, "s": s, "ar": -ar_full[1:], "ma": ma_full[1:], "res": r}
