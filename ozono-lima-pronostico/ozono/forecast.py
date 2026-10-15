"""Motor de evaluación con origen móvil (rolling origin, ventana expansiva).

Para cada estación, horizonte h y ventana k = [a, b) del período de prueba:
  1. Se reajusta la componente determinística (OLS) con datos [0, a).
  2. Se construyen los rezagos del residuo e = log(y) - z_det (pronóstico directo a h pasos).
  3. Se reentrenan ELM, MLP, NPAR (y sus variantes con meteorología) y ARIMA con datos < a.
  4. Se pronostica cada objetivo tau en [a, b) usando solo información hasta tau - h.
  5. Se retransforma: y_hat = exp(z_det + e_hat).
Las líneas base (Naive, SNaive24, SNaive168) usan la MISMA regla de información (origen tau - h),
de modo que todas las comparaciones son a igual horizonte.
Los hiperparámetros se eligen una sola vez por (estación, h, modelo, variante) en un hold-out
temporal al final del primer conjunto de entrenamiento, minimizando SMAPE en escala original.
"""
from __future__ import annotations

import time as _time

import numpy as np
import pandas as pd

from . import arima as A
from .data import load_station
from .decomposition import design_matrix, fit_deterministic, ols_table
from .diagnostics import quality_by_year, residual_season_check
from .features import covariate_matrix, lag_matrix, select_p_bic, valid_mask
from .metrics import scale_mase
from .models import build, grid

ML_MODELS_DEFAULT = ["ELM", "MLP", "NPAR", "RF", "XGB", "GRU", "LSTM"]


def _smape(y, yh):
    return 100 * np.mean(np.abs(y - yh) / ((np.abs(y) + np.abs(yh)) / 2))


def run_station(code: str, cfg: dict, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    t0 = _time.time()
    say = (lambda *m: print(f"[{code}]", *m, flush=True)) if verbose else (lambda *m: None)
    tgt = cfg["data"]["target"]
    df, qlog = load_station(code, cfg)
    y = df[tgt].to_numpy(float)
    n = len(y)
    is_log = cfg["transform"] == "log"
    ly = np.log(y) if is_log else y.copy()
    inv = np.exp if is_log else (lambda v: v)
    season = df["season"].to_numpy()
    seed = int(cfg["seed"])
    drop_cross = bool(cfg["lags"]["drop_cross_season"])

    Xdet, det_names = design_matrix(df["time"], cfg["decomposition"])
    a0 = n - int(cfg["split"]["test_hours"])
    win = [(int(w[0]), int(w[-1]) + 1) for w in np.array_split(np.arange(a0, n), int(cfg["split"]["n_windows"]))]
    dets = [fit_deterministic(ly, Xdet, a) for a, _ in win]
    det0 = dets[0]
    say(f"n={n}, entrenamiento inicial={a0}, ventanas={[b - a for a, b in win]}, R2 determinístico={det0['r2']:.3f}")

    # --- orden autorregresivo común (ELM, MLP, NPAR) ---
    if cfg["lags"]["select"] == "bic":
        p, bic_tab = select_p_bic(det0["e"], season, a0, int(cfg["lags"]["max_p"]), drop_cross)
    else:
        p, bic_tab = int(cfg["lags"]["fixed_p"]), None
    say(f"orden de rezagos p={p}")

    # --- ARIMA: selección de orden en el primer entrenamiento ---
    ac = cfg["models"]["ARIMA"]
    order0, arima_grid = A.select_order(det0["e"][:a0], ac["max_p"], ac["max_q"], ac["criterion"])
    arima_fits = []
    for k, (a, _) in enumerate(win):
        order = order0
        if ac.get("reselect_each_origin") and k > 0:
            order, _ = A.select_order(dets[k]["e"][:a], ac["max_p"], ac["max_q"], ac["criterion"])
        arima_fits.append(A.fit(dets[k]["e"][:a], order))
    arima_diag = A.diagnostics(arima_fits[0])
    arima_coefs = A.coef_table(arima_fits[0])
    say(f"ARIMA{(order0[0], 0, order0[1])} seleccionado por {ac['criterion'].upper()}")

    # --- SARIMA (contraste estacional, Obs. Soria-01) ---
    sc = cfg["models"].get("SARIMA", {})
    sarima_fits, sarima_diag, sarima_coefs = [], None, None
    if sc.get("enabled", False):
        so, ss, sper = tuple(sc["order"]), tuple(sc["seasonal_order"]), int(sc.get("period", 24))
        f0 = A.fit_sarima(det0["e"][:a0], so, ss, sper)
        for k, (a, _) in enumerate(win):
            if k > 0 and sc.get("refit_each_origin", False):
                sarima_fits.append(A.fit_sarima(dets[k]["e"][:a], so, ss, sper))
            else:
                sarima_fits.append(f0)
        sarima_diag = A.diagnostics({"order": (so[0] + ss[0], so[1] + ss[1]), "res": f0["res"]})
        sarima_diag.update({"label": f"SARIMA({so[0]},0,{so[1]})({ss[0]},0,{ss[1]})$_{{{sper}}}$"})
        sarima_coefs = A.coef_table(f0)
        say(f"SARIMA{so}x{ss}_{sper} ajustado (AIC={f0['res'].aic:.1f})")

    mase_den = scale_mase(y[:a0], cfg["metrics"]["mase_scale"])
    mv = cfg.get("multivariate", {})
    variants = [("uni", [m for m in cfg.get("stochastic_models", ML_MODELS_DEFAULT) if m in cfg["models"]])]
    if mv.get("enabled"):
        variants.append(("met", list(mv["models"])))
    max_cov_lag = max(mv.get("cov_lags", [1])) if mv.get("enabled") else 1

    records, hp_sel, sel_logs = [], {}, []
    for h in cfg["horizons"]:
        C = None
        if mv.get("enabled"):
            C, cnames = covariate_matrix(df, mv["covariates"], mv["cov_lags"], h, mv.get("wind_direction_col"))
        for k, (a, b) in enumerate(win):
            det = dets[k]
            e, zd = det["e"], det["zd"]
            te = np.arange(a, b)
            L = lag_matrix(e, p, h)
            for var, models in variants:
                maxlag = max(p, max_cov_lag) if var == "met" else p
                mask = valid_mask(season, h, maxlag, drop_cross)
                if not mask[te].all():
                    raise RuntimeError(f"{code}: objetivos de prueba con rezagos fuera del invierno (h={h})")
                tr = np.where(mask[:a])[0]
                X = L if var == "uni" else np.hstack([L, C])
                for m in models:
                    key = (h, var, m)
                    mcfg = cfg["models"][m]
                    if key not in hp_sel:
                        hp_sel[key], slog = _select_hp(m, mcfg, ly, y, Xdet, season, a0, p, h, var, C,
                                                        drop_cross, maxlag, cfg, seed, inv)
                        sel_logs.extend({"station": code, "h": h, "variant": var, "model": m, **r} for r in slog)
                    mdl = build(m, hp_sel[key], mcfg, seed).fit(X[tr], e[tr])
                    ehat = mdl.predict(X[te])
                    if cfg["bias_correction"] and is_log:
                        s2 = np.var(e[tr] - mdl.predict(X[tr]))
                        ehat = ehat + s2 / 2
                    label = m if var == "uni" else f"{m}+met"
                    records.append(_rec(code, h, k, te, df, y, inv(zd[te] + ehat), label))
            # ARIMA (univariado)
            af = arima_fits[k]
            ehat = A.forecast(e[:b], af["ar"], af["ma"], te, h)
            if cfg["bias_correction"] and is_log:
                ehat = ehat + float(af["res"].params[-1]) / 2
            records.append(_rec(code, h, k, te, df, y, inv(zd[te] + ehat), "ARIMA"))
            if sarima_fits:
                sf = sarima_fits[k]
                ehat = A.forecast(e[:b], sf["ar"], sf["ma"], te, h)
                records.append(_rec(code, h, k, te, df, y, inv(zd[te] + ehat), "SARIMA"))
            # componente determinística sola
            records.append(_rec(code, h, k, te, df, y, inv(zd[te]), "Determ"))
            # líneas base con la misma regla de información (origen tau-h)
            for bname in cfg["baselines"]:
                if bname == "Naive":
                    yh = y[te - h]
                elif bname == "SNaive24":
                    yh = y[te - 24 * int(np.ceil(h / 24))]
                elif bname == "SNaive168":
                    yh = y[te - 168 * int(np.ceil(h / 168))]
                else:
                    raise ValueError(bname)
                records.append(_rec(code, h, k, te, df, y, yh, bname))
        say(f"h={h} listo ({_time.time() - t0:.0f} s)")

    preds = pd.concat(records, ignore_index=True)
    meta = {
        "station": code, "quality_log": qlog, "n": n, "a0": a0, "windows": win, "p": p,
        "bic_table": bic_tab, "arima_order": order0, "arima_grid": arima_grid,
        "arima_diag": arima_diag, "arima_coefs": arima_coefs,
        "sarima_diag": sarima_diag, "sarima_coefs": sarima_coefs,
        "arima_orders_by_window": [f["order"] for f in arima_fits],
        "hp": {f"h{k[0]}|{k[1]}|{k[2]}": v for k, v in hp_sel.items()},
        "hp_search": pd.DataFrame(sel_logs),
        "det_r2": [d["r2"] for d in dets], "det_names": det_names,
        "ols_table": ols_table(ly, Xdet, det_names, a0),
        "resid_season": residual_season_check(det0["e"][:a0]),
        "quality_year": quality_by_year(df, tgt, cfg["metrics"]["near_zero_threshold"]),
        "mase_den": mase_den,
        "series": pd.DataFrame({"time": df["time"], "y": y, "ly": ly, "zd": det0["zd"], "e": det0["e"],
                                "season": season}),
        "y_all": y, "runtime_s": _time.time() - t0,
        "vars_summary": {c: {"mean": (float(np.rad2deg(np.arctan2(np.sin(np.deg2rad(df[c])).mean(),
                                                                   np.cos(np.deg2rad(df[c])).mean())) % 360)
                                      if c == "dirvie" else float(df[c].mean())),
                             "min": float(df[c].min()), "max": float(df[c].max()),
                             "missing": int(qlog["missing_before_impute"].get(c, 0))}
                         for c in [tgt] + qlog["covariates_available"]},
        "diurnal": _diurnal(df, tgt, qlog["covariates_available"]),
        "acf": _acf_block(ly[:a0], det0["e"][:a0]),
    }
    say(f"terminado en {meta['runtime_s']:.0f} s")
    return preds, meta


def _diurnal(df, tgt, covs):
    """Perfil horario medio de ozono y meteorología (dirección del viento: media circular)."""
    g = df.groupby(df["time"].dt.hour)
    out = pd.DataFrame({"hour": sorted(df["time"].dt.hour.unique())})
    out[tgt] = g[tgt].mean().to_numpy()
    for c in covs:
        if c == "dirvie":
            rad = np.deg2rad(df[c])
            out[c] = (np.rad2deg(np.arctan2(np.sin(rad).groupby(df["time"].dt.hour).mean(),
                                            np.cos(rad).groupby(df["time"].dt.hour).mean())) % 360).to_numpy()
        else:
            out[c] = g[c].mean().to_numpy()
    return out


def _acf_block(ly_tr, e_tr, nlags=168):
    """ACF de la serie logarítmica y ACF/PACF del residuo (identificación Box–Jenkins)."""
    from statsmodels.tsa.stattools import acf, pacf
    return {"acf_y": acf(ly_tr, nlags=nlags, fft=True), "acf_e": acf(e_tr, nlags=nlags, fft=True),
            "pacf_e": pacf(e_tr, nlags=nlags, method="ols"), "band": 1.96 / np.sqrt(len(e_tr))}


def _rec(code, h, k, te, df, y, yhat, model):
    return pd.DataFrame({"station": code, "h": h, "window": k, "tau": te,
                         "time": df["time"].to_numpy()[te], "hour": df["time"].dt.hour.to_numpy()[te],
                         "y": y[te], "yhat": yhat, "model": model})


def _select_hp(m, mcfg, ly, y, Xdet, season, a0, p, h, var, C, drop_cross, maxlag, cfg, seed, inv):
    """Hold-out temporal: últimas val_fraction horas válidas del primer entrenamiento."""
    mask = valid_mask(season, h, maxlag, drop_cross)
    tr_all = np.where(mask[:a0])[0]
    nv = int(round(cfg["split"]["val_fraction"] * a0))
    val = tr_all[-nv:]
    cut = int(val[0])
    det = fit_deterministic(ly, Xdet, cut)  # sin mirar el bloque de validación
    e, zd = det["e"], det["zd"]
    L = lag_matrix(e, p, h)
    X = L if var == "uni" else np.hstack([L, C])
    fit_idx = tr_all[tr_all < cut - h + 1]  # el objetivo de entrenamiento debe ser observable antes de la validación
    rows = []
    for hp in grid(m, mcfg):
        mdl = build(m, hp, mcfg, seed).fit(X[fit_idx], e[fit_idx])
        yh = inv(zd[val] + mdl.predict(X[val]))
        rows.append({**hp, "val_SMAPE": _smape(y[val], yh), "val_MAE": float(np.mean(np.abs(y[val] - yh))),
                     "n_val": len(val)})
    best = min(rows, key=lambda r: r["val_SMAPE"])
    return {k: v for k, v in best.items() if k in ("n_hidden", "alpha", "ridge", "max_depth", "min_samples_leaf", "learning_rate", "hidden")}, rows
