"""Estadísticos descriptivos, estacionariedad (ADF + KPSS), estacionalidad remanente
en el residuo y calidad de datos por año."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from .data import flatline_runs


def describe(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    vals, counts = np.unique(np.round(x, 2), return_counts=True)
    return {
        "Mínimo": x.min(), "Primer cuartil": np.quantile(x, .25), "Mediana": np.median(x),
        "Media": x.mean(), "Moda": vals[counts.argmax()], "Varianza": x.var(ddof=1),
        "Desv. típica": x.std(ddof=1), "Asimetría": stats.skew(x),
        "Curtosis (exceso)": stats.kurtosis(x), "Tercer cuartil": np.quantile(x, .75),
        "Máximo": x.max(),
    }


def stationarity(x: np.ndarray) -> dict:
    """ADF (constante, rezagos por AIC hasta 12(n/100)^(1/4)) y KPSS (nivel, rezagos 'auto')."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf = adfuller(x, regression="c", autolag="AIC")
        kp = kpss(x, regression="c", nlags="auto")
    return {"ADF": adf[0], "ADF_p": adf[1], "ADF_lags": adf[2], "KPSS": kp[0], "KPSS_p": kp[1]}


def descriptives_table(series: dict[str, np.ndarray], log: bool) -> pd.DataFrame:
    """Tabla 2: descriptivos + estacionariedad para original y log por estación."""
    cols = {}
    for code, y in series.items():
        for lab, v in [(code, y), (f"log({code})", np.log(y))] if log else [(code, y)]:
            d = describe(v)
            s = stationarity(v)
            d.update({"ADF (estadístico)": s["ADF"], "ADF (valor p)": s["ADF_p"],
                      "ADF (rezagos, AIC)": s["ADF_lags"], "KPSS (estadístico)": s["KPSS"],
                      "KPSS (valor p)": s["KPSS_p"]})
            cols[lab] = d
    return pd.DataFrame(cols)


def residual_season_check(e: np.ndarray, lags=(24, 168)) -> dict:
    """¿Queda estacionalidad diaria/semanal tras la etapa determinística? (justifica ARIMA vs SARIMA)."""
    lb = acorr_ljungbox(e, lags=list(lags), return_df=True)
    r = acf(e, nlags=max(lags), fft=True)
    pr = pacf(e, nlags=max(lags), method="ols")
    out = {}
    for L in lags:
        out[f"ACF_{L}"] = float(r[L])
        out[f"PACF_{L}"] = float(pr[L])
        out[f"LB_p_{L}"] = float(lb.loc[L, "lb_pvalue"])
    out["ACF_band_95"] = 1.96 / np.sqrt(len(e))
    st = stationarity(e)
    out.update({"ADF_resid": st["ADF"], "ADF_resid_p": st["ADF_p"], "KPSS_resid": st["KPSS"], "KPSS_resid_p": st["KPSS_p"]})
    return out


def quality_by_year(df: pd.DataFrame, tgt: str, near_zero: float) -> pd.DataFrame:
    rows = []
    for yr, g in df.groupby("season"):
        y = g[tgt].to_numpy()
        runs, hrs = flatline_runs(y, 3)
        rows.append({"Año": yr, "n": len(y), "Media": y.mean(), "Mediana": np.median(y),
                     "Desv. típica": y.std(ddof=1), "P95": np.quantile(y, .95), "Máximo": y.max(),
                     f"% < {near_zero:g}": 100 * np.mean(y < near_zero),
                     "Rachas planas (>=3 h)": runs, "Horas en rachas planas": hrs})
    tab = pd.DataFrame(rows)
    # prueba de Kruskal–Wallis de igualdad de niveles entre años
    groups = [g[tgt].to_numpy() for _, g in df.groupby("season")]
    kw = stats.kruskal(*groups)
    tab.attrs["kruskal_H"] = float(kw.statistic)
    tab.attrs["kruskal_p"] = float(kw.pvalue)
    tab.attrs["max_min_mean_ratio"] = float(tab["Media"].max() / tab["Media"].min())
    return tab
