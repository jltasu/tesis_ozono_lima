"""Generalización temporal y espacial con la data disponible.

No se pueden añadir años, estaciones del año ni estaciones de monitoreo, pero sí probar
si el marco generaliza fuera de la muestra en las dos dimensiones que pide el revisor:

1. ENTRE INVIERNOS (temporal): se entrena con los inviernos previos y se pronostica un invierno
   completo no visto (2017 -> 2018; 2017-2018 -> 2019). Como un año no visto no tiene dummy de año,
   la componente determinística usa solo calendario intra-anual (día de la semana y hora); el nivel
   del nuevo invierno lo aporta la dinámica de rezagos.
2. ENTRE ESTACIONES (espacial): el modelo estocástico se entrena en una estación (o en las cuatro
   combinadas) y se aplica a otra. La componente determinística de la estación destino se estima
   con su propio período de entrenamiento (perfil de calendario). Se compara con el modelo local.

Todas las líneas base usan la misma regla de información (origen tau - h).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import arima as A
from .data import load_station
from .decomposition import design_matrix, fit_deterministic
from .features import lag_matrix, valid_mask
from .models import build, grid


def _hp(meta: dict, cfg: dict, h: int, m: str) -> dict:
    hp = meta["hp"].get(f"h{h}|uni|{m}")
    if hp is None:  # h no evaluado en la corrida principal: primer valor de la grilla
        hp = grid(m, cfg["models"][m])[0]
    return hp


def _rows(exp, st, h, te, df, y, yhat, model, **extra):
    d = pd.DataFrame({"experiment": exp, "station": st, "h": h, "tau": te,
                      "time": df["time"].to_numpy()[te], "y": y[te], "yhat": yhat, "model": model})
    for k, v in extra.items():
        d[k] = v
    return d


def _baselines(exp, st, h, te, df, y, **extra):
    out = [_rows(exp, st, h, te, df, y, y[te - h], "Naive", **extra),
           _rows(exp, st, h, te, df, y, y[te - 24 * int(np.ceil(h / 24))], "SNaive24", **extra)]
    return out


# ----------------------------------------------------------------------------- entre inviernos
def cross_winter(st: str, df: pd.DataFrame, meta: dict, cfg: dict, h: int) -> list[pd.DataFrame]:
    g = cfg["generalization"]
    y = df[cfg["data"]["target"]].to_numpy(float)
    ly = np.log(y) if cfg["transform"] == "log" else y.copy()
    inv = np.exp if cfg["transform"] == "log" else (lambda v: v)
    season = df["season"].to_numpy()
    X, _ = design_matrix(df["time"], g["cross_winter_decomposition"])
    p = meta["p"]
    seed = int(cfg["seed"])
    out = []
    for Y in sorted(np.unique(season))[1:]:
        start = int(np.argmax(season == Y))
        det = fit_deterministic(ly, X, start)
        e, zd = det["e"], det["zd"]
        maxlag = max(p, 24 * int(np.ceil(h / 24)))
        mask = valid_mask(season, h, maxlag, True)
        tr = np.where(mask[:start])[0]
        te = np.where(mask & (season == Y))[0]
        L = lag_matrix(e, p, h)
        extra = {"test_winter": int(Y), "train_winters": "-".join(map(str, sorted(np.unique(season[:start]))))}
        for m in g["models"]:
            mdl = build(m, _hp(meta, cfg, h, m), cfg["models"][m], seed).fit(L[tr], e[tr])
            out.append(_rows("cross_winter", st, h, te, df, y, inv(zd[te] + mdl.predict(L[te])), m, **extra))
        af = A.fit(e[:start], tuple(meta["arima_order"]))
        out.append(_rows("cross_winter", st, h, te, df, y,
                         inv(zd[te] + A.forecast(e[: te.max() + 1], af["ar"], af["ma"], te, h)), "ARIMA", **extra))
        out.append(_rows("cross_winter", st, h, te, df, y, inv(zd[te]), "Determ", **extra))
        out += _baselines("cross_winter", st, h, te, df, y, **extra)
    return out


# ----------------------------------------------------------------------------- entre estaciones
def cross_station(data: dict[str, pd.DataFrame], metas: dict, cfg: dict, h: int) -> list[pd.DataFrame]:
    g = cfg["generalization"]
    stations = list(data)
    seed = int(cfg["seed"])
    p = int(round(np.median([metas[s]["p"] for s in stations])))  # p común para que las entradas sean compatibles
    a0 = metas[stations[0]]["a0"]
    inv = np.exp if cfg["transform"] == "log" else (lambda v: v)
    prep = {}
    for s, df in data.items():
        y = df[cfg["data"]["target"]].to_numpy(float)
        ly = np.log(y) if cfg["transform"] == "log" else y.copy()
        X, _ = design_matrix(df["time"], cfg["decomposition"])
        det = fit_deterministic(ly, X, a0)
        season = df["season"].to_numpy()
        mask = valid_mask(season, h, max(p, 24 * int(np.ceil(h / 24))), True)
        L = lag_matrix(det["e"], p, h)
        prep[s] = {"y": y, "e": det["e"], "zd": det["zd"], "L": L,
                   "tr": np.where(mask[:a0])[0], "te": np.where(mask & (np.arange(len(y)) >= a0))[0]}
    out = []
    for m in g["models"]:
        mcfg = cfg["models"][m]
        for tgt in stations:
            T = prep[tgt]
            te = T["te"]
            hp = _hp(metas[tgt], cfg, h, m)
            sources = {"Local": [tgt], "Combinado": stations, **{s: [s] for s in stations if s != tgt}}
            for lab, srcs in sources.items():
                Xtr = np.vstack([prep[s]["L"][prep[s]["tr"]] for s in srcs])
                ytr = np.concatenate([prep[s]["e"][prep[s]["tr"]] for s in srcs])
                mdl = build(m, hp, mcfg, seed).fit(Xtr, ytr)
                yhat = inv(T["zd"][te] + mdl.predict(T["L"][te]))
                out.append(_rows("cross_station", tgt, h, te, data[tgt], T["y"], yhat, m, source=lab, p_common=p))
    for tgt in stations:
        T = prep[tgt]
        out += _baselines("cross_station", tgt, h, T["te"], data[tgt], T["y"], source="—", p_common=p)
    return out


def run_generalization(cfg: dict, metas: dict, verbose: bool = True) -> pd.DataFrame:
    g = cfg.get("generalization", {})
    if not g.get("enabled"):
        return pd.DataFrame()
    stations = list(metas)
    data = {s: load_station(s, cfg)[0] for s in stations}
    frames = []
    for h in g["horizons"]:
        if g.get("cross_winter"):
            for s in stations:
                frames += cross_winter(s, data[s], metas[s], cfg, h)
            if verbose:
                print(f"[generalización] entre inviernos h={h} listo", flush=True)
        if g.get("cross_station") and len(stations) > 1:
            frames += cross_station(data, metas, cfg, h)
            if verbose:
                print(f"[generalización] entre estaciones h={h} listo", flush=True)
    return pd.concat(frames, ignore_index=True)
