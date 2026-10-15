"""Métricas de precisión y prueba de Diebold–Mariano (HLN y HAC) con corrección de Holm."""
from __future__ import annotations

import numpy as np
from scipy import stats


def scale_mase(y_train: np.ndarray, kind: str = "train_naive") -> float:
    """Denominador de MASE (Hyndman & Koehler, 2006): MAE in-sample del Naive (o SNaive24) en entrenamiento."""
    lag = 24 if kind == "train_snaive24" else 1
    return float(np.mean(np.abs(y_train[lag:] - y_train[:-lag])))


def all_metrics(y: np.ndarray, yhat: np.ndarray, mase_scale: float, near_zero: float) -> dict:
    e = y - yhat
    a = np.abs(e)
    return {
        "MAE": a.mean(),
        "RMSE": np.sqrt((e ** 2).mean()),
        "MDAE": np.median(a),
        "SMAPE": 100 * np.mean(a / ((np.abs(y) + np.abs(yhat)) / 2)),
        "MAPE": 100 * np.mean(a / np.abs(y)),
        "MASE": a.mean() / mase_scale,
        "R2": 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2),
        "share_near_zero": 100 * np.mean(y < near_zero),
        "n": len(y),
    }


def _loss(e, kind):
    return np.abs(e) if kind == "absolute" else e ** 2


def dm_test(e1: np.ndarray, e2: np.ndarray, h: int = 1, loss: str = "absolute", hac_lags: int = 24) -> dict:
    """d_t = L(e1_t) - L(e2_t). Estadístico negativo => el modelo 1 es más preciso.

    - HLN: Diebold–Mariano con corrección de Harvey, Leybourne y Newbold (1997), varianza con h-1 rezagos.
    - HAC: misma media, varianza Newey–West (Bartlett) con max(h-1, hac_lags) rezagos, robusta a la
      autocorrelación de los errores horarios (más conservadora).
    """
    d = _loss(e1, loss) - _loss(e2, loss)
    n = len(d)
    dbar = d.mean()
    dc = d - dbar

    def acov(k):
        return np.sum(dc[k:] * dc[: n - k]) / n

    # HLN
    v = acov(0) + 2 * sum(acov(k) for k in range(1, h))
    v = max(v, 1e-12)
    dm = dbar / np.sqrt(v / n)
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    hln = dm * corr
    p_hln = 2 * stats.t.sf(abs(hln), df=n - 1)
    # HAC
    L = max(h - 1, hac_lags)
    vh = acov(0) + 2 * sum((1 - k / (L + 1)) * acov(k) for k in range(1, L + 1))
    vh = max(vh, 1e-12)
    hac = dbar / np.sqrt(vh / n)
    p_hac = 2 * stats.norm.sf(abs(hac))
    return {"DM_HLN": hln, "p_HLN": p_hln, "DM_HAC": hac, "p_HAC": p_hac, "mean_loss_diff": dbar}


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
