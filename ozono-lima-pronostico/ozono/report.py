"""Tablas (CSV + LaTeX booktabs listas para Overleaf), figuras (PDF) y verificaciones automáticas."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .diagnostics import descriptives_table  # noqa: E402
from .metrics import all_metrics, dm_test, holm, stars  # noqa: E402

UNI = ["ELM", "MLP", "NPAR", "RF", "XGB", "GRU", "LSTM", "ARIMA", "SARIMA"]
BASE = ["Naive", "SNaive24", "SNaive168"]
ORDER = ["ELM", "MLP", "NPAR", "RF", "XGB", "GRU", "LSTM", "ARIMA", "SARIMA", "ELM+met", "MLP+met", "NPAR+met",
         "Determ", "Naive", "SNaive24", "SNaive168"]
# color fijo por entidad (paleta categórica validada; nunca por rango)
COLORS = {"ELM": "#2a78d6", "MLP": "#eb6834", "NPAR": "#1baf7a", "ARIMA": "#eda100",
          "Naive": "#e87ba4", "RF": "#008300", "SNaive24": "#6250d6", "XGB": "#e34948", "Observado": "#0b0b0b"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STATION_NAMES = {"ATE": "ATE", "CDM": "Campo de Marte", "SB": "San Borja", "STA": "Santa Anita"}


def boosting_label(cfg: dict | None = None) -> str:
    """Rótulo honesto del modelo de boosting: 'XGB' si se usó XGBoost; 'HGB' si se usó
    HistGradientBoosting de scikit-learn (p. ej., por falta de la librería OpenMP)."""
    import os
    if os.environ.get("OZONO_BOOST_LABEL"):
        return os.environ["OZONO_BOOST_LABEL"]
    from .models import XGB
    backend = ((cfg or {}).get("models", {}).get("XGB", {}) or {}).get("backend", "auto")
    return "XGB" if XGB(backend=backend).backend == "xgboost" else "HGB"


def apply_boosting_label(preds: pd.DataFrame, lab: str) -> pd.DataFrame:
    if lab == "XGB":
        return preds
    for lst in (UNI, ORDER):
        for i, m in enumerate(lst):
            if m == "XGB":
                lst[i] = lab
    COLORS[lab] = COLORS.get("XGB", "#e34948")
    return preds.replace({"model": {"XGB": lab}})


# ----------------------------------------------------------------------------- utilidades
def _style():
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def _tex_escape(s) -> str:
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("%", r"\%"), ("_", r"\_"), ("&", r"\&"), ("#", r"\#")]:
        s = s.replace(a, b)
    for a, b in [("λ", r"$\lambda$"), ("α", r"$\alpha$"), ("∞", r"$\infty$"), ("≥", r"$\geq$"), ("≤", r"$\leq$"),
                 ("±", r"$\pm$"), ("μ", r"$\mu$")]:
        s = s.replace(a, b)
    return s


def write_table(df: pd.DataFrame, out: Path, name: str, caption: str, label: str,
                note: str | None = None, resize: bool = True, raw_cols: tuple = ()) -> None:
    """Guarda CSV y un .tex con entorno table + booktabs (incluir en Overleaf con \\input)."""
    df.to_csv(out / f"{name}.csv", index=False)
    cols = list(df.columns)
    fmt = "l" * min(2, len(cols)) + "c" * (len(cols) - min(2, len(cols)))
    lines = [r"\begin{table}[H]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}"]
    if resize:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines += [rf"\begin{{tabular}}{{{fmt}}}", r"\toprule",
              " & ".join(rf"\textbf{{{_tex_escape(c)}}}" for c in cols) + r" \\", r"\midrule"]
    prev_first = None
    for _, row in df.iterrows():
        cells = []
        for j, c in enumerate(cols):
            v = row[c]
            if j == 0 and v == prev_first:
                cells.append("")
                continue
            cells.append(str(v) if c in raw_cols else _tex_escape(v))
        if prev_first is not None and row[cols[0]] != prev_first:
            lines.append(r"\midrule")
        prev_first = row[cols[0]]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}" + ("}" if resize else "")]
    if note:
        lines.append(rf"\par\smallskip{{\footnotesize {note}}}")
    lines.append(r"\end{table}")
    (out / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _f(x, d=3):
    return "" if pd.isna(x) else f"{x:.{d}f}"


def _pfmt(p):
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


# ----------------------------------------------------------------------------- métricas
def metric_frame(preds: pd.DataFrame, metas: dict, near_zero: float) -> pd.DataFrame:
    """Métricas por (estación, h, modelo, ventana) y agregadas (window = 'all')."""
    rows = []
    for (st, h, m), g in preds.groupby(["station", "h", "model"]):
        den = metas[st]["mase_den"]
        for w, gw in list(g.groupby("window")) + [("all", g)]:
            r = all_metrics(gw.y.to_numpy(), gw.yhat.to_numpy(), den, near_zero)
            rows.append({"station": st, "h": h, "model": m, "window": w, **r})
    M = pd.DataFrame(rows)
    naive = M[M.model == "Naive"][["station", "h", "window", "MAE"]].rename(columns={"MAE": "MAE_naive"})
    M = M.merge(naive, on=["station", "h", "window"], how="left")
    M["RelMAE"] = M["MAE"] / M["MAE_naive"]
    return M


def best_model(M: pd.DataFrame, st: str, h: int = 1, pool=UNI) -> str:
    g = M[(M.station == st) & (M.h == h) & (M.window == "all") & (M.model.isin(pool))]
    return g.loc[g.MAE.idxmin(), "model"]


# ----------------------------------------------------------------------------- reporte principal
def build_report(preds: pd.DataFrame, metas: dict, cfg: dict) -> dict:
    out = Path(cfg["_out"])
    tdir, fdir = out / "tables", out / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    for old in ("fig_niveles_por_anio", "fig_descomposicion", "fig_observado_vs_pronostico",
                "fig_error_por_hora", "fig_horizontes"):  # nombres antiguos en español
        (fdir / f"{old}.pdf").unlink(missing_ok=True)
    _style()
    nz = cfg["metrics"]["near_zero_threshold"]
    stations = list(metas)
    M = metric_frame(preds, metas, nz)
    M.to_csv(tdir / "metricas_completas.csv", index=False)
    summary: dict = {"stations": {}}

    # ---------- T01 calidad de datos por año
    rows = []
    for st in stations:
        q = metas[st]["quality_year"]
        for _, r in q.iterrows():
            rows.append({"Estación": st, "Año": int(r["Año"]), "Media": _f(r["Media"], 1), "Mediana": _f(r["Mediana"], 1),
                         "D.E.": _f(r["Desv. típica"], 1), "P95": _f(r["P95"], 1), "Máx.": _f(r["Máximo"], 1),
                         f"% < {nz:g}": _f(r[f"% < {nz:g}"], 1), "Horas en rachas planas": int(r["Horas en rachas planas"])})
    kw_note = "; ".join(f"{st}: H={metas[st]['quality_year'].attrs['kruskal_H']:.0f}, p={_pfmt(metas[st]['quality_year'].attrs['kruskal_p'])}"
                        for st in stations)
    write_table(pd.DataFrame(rows), tdir, "T01_calidad_por_anio",
                r"Nivel y calidad de la serie de ozono ($\mu$g/m$^3$) por invierno y estación.",
                "tab_calidad", note=f"Kruskal--Wallis entre años: {kw_note}. Rachas planas: $\\geq$3 h consecutivas con el mismo valor.")

    # ---------- T02 descriptivos + estacionariedad
    series = {st: metas[st]["y_all"] for st in stations}
    D = descriptives_table(series, cfg["transform"] == "log")
    D2 = D.copy()
    for c in D2.columns:
        D2[c] = [(_f(v, 0) if "rezagos" in i else _pfmt(v) if "valor p" in i else _f(v, 2)) for i, v in D2[c].items()]
    D2.insert(0, "Medida", D2.index)
    write_table(D2.reset_index(drop=True), tdir, "T02_descriptivos",
                "Estadísticos descriptivos y pruebas de estacionariedad (ADF: $H_0$ raíz unitaria; KPSS: $H_0$ estacionariedad) "
                "para las series original y logarítmica.", "tab2",
                note="ADF con constante y rezagos elegidos por AIC; KPSS en nivel con ancho de banda automático. Curtosis en exceso (normal = 0).")

    # ---------- T03 estacionalidad remanente en el residuo
    rows = []
    for st in stations:
        r = metas[st]["resid_season"]
        rows.append({"Estación": st, "ACF(24)": _f(r["ACF_24"]), "PACF(24)": _f(r["PACF_24"]),
                     "ACF(168)": _f(r["ACF_168"]), "PACF(168)": _f(r["PACF_168"]),
                     "Banda 95%": _f(r["ACF_band_95"]), "ADF residuo": _f(r["ADF_resid"], 2),
                     "ADF p": _pfmt(r["ADF_resid_p"]), "KPSS p": _pfmt(r["KPSS_resid_p"]),
                     "R2 determ.": _f(metas[st]["det_r2"][0])})
    write_table(pd.DataFrame(rows), tdir, "T03_residuo_estacionalidad",
                "Estacionalidad remanente y estacionariedad del residuo estocástico $e_h$ (primer conjunto de entrenamiento).",
                "tab_resid", note="La ACF en los rezagos 24 y 168 también refleja la persistencia autorregresiva; la PACF aísla la dependencia "
                                  "estacional directa. Si la PACF(24) queda dentro de la banda, un término estacional adicional (SARIMA) sería redundante; si queda fuera, se justifica el SARIMA de contraste (Tabla~\\ref{tab_sarima}).")

    # ---------- T04 ARIMA (principal), T04b SARIMA y S1 (coeficientes completos)
    rows, srows = [], []
    for st in stations:
        d, co = metas[st]["arima_diag"], metas[st]["arima_coefs"]
        a1 = co[co.param == "ar.L1"]
        t1 = co[co.param == "ma.L1"]
        rows.append({"Estación": st, "p": d["p"], "d": 0, "q": d["q"],
                     "AR1 (EE)": f"{a1.coef.iloc[0]:+.4f} ({a1.se.iloc[0]:.4f})" if len(a1) else "---",
                     "MA1 (EE)": f"{t1.coef.iloc[0]:+.4f} ({t1.se.iloc[0]:.4f})" if len(t1) else "---",
                     "AIC": _f(d["AIC"], 1), "BIC": _f(d["BIC"], 1),
                     "LB p (24)": _pfmt(d["LB_p_24"]), "LB p (168)": _pfmt(d["LB_p_168"]),
                     "Dist. mín. raíces AR-MA": _f(d["min_ar_ma_root_distance"])})
        for lab, cot in [("ARIMA", co), ("SARIMA", metas[st].get("sarima_coefs"))]:
            if cot is None:
                continue
            for _, r in cot.iterrows():
                srows.append({"Estación": st, "Modelo": lab, "Parámetro": r.param, "Coef.": f"{r.coef:+.4f}{stars(r.p)}",
                              "EE": _f(r.se, 4), "p": _pfmt(r.p)})
    crit = cfg["models"]["ARIMA"]["criterion"].upper()
    write_table(pd.DataFrame(rows), tdir, "T04_arima",
                f"Modelo ARIMA($p$,0,$q$) del componente estocástico: orden por {crit} sobre $p,q\\in\\{{0,\\ldots,"
                f"{cfg['models']['ARIMA']['max_p']}\\}}$, coeficientes principales y diagnóstico de residuos (Ljung--Box).",
                "tab_arima", note="Distancia mínima entre raíces AR y MA < 0.1 indica cancelación aproximada (sobreparametrización).")
    srows_s = []
    for st in stations:
        d = metas[st].get("sarima_diag")
        co = metas[st].get("sarima_coefs")
        if d is None:
            continue
        get = lambda nm: co[co.param == nm]
        cell = lambda nm: (f"{get(nm).coef.iloc[0]:+.3f}{stars(get(nm).p.iloc[0])}" if len(get(nm)) else "---")
        srows_s.append({"Estación": st, "Modelo": d["label"], "AR1": cell("ar.L1"), "MA1": cell("ma.L1"),
                        "AR estacional (24)": cell("ar.S.L24"), "MA estacional (24)": cell("ma.S.L24"),
                        "AIC": _f(d["AIC"], 1), "BIC": _f(d["BIC"], 1),
                        "LB p (24)": _pfmt(d["LB_p_24"]), "LB p (168)": _pfmt(d["LB_p_168"])})
    if srows_s:
        write_table(pd.DataFrame(srows_s), tdir, "T04b_sarima",
                    "Modelo SARIMA del residuo estocástico (contraste estacional con periodo de 24 h): coeficientes, "
                    "criterios de información y diagnóstico de residuos.", "tab_sarima", raw_cols=("Modelo",),
                    note=r"$^{***}p<0.001$; $^{**}p<0.01$; $^{*}p<0.05$.")
    write_table(pd.DataFrame(srows), tdir, "S1_arima_coeficientes",
                "Coeficientes completos de los modelos ARIMA y SARIMA del residuo estocástico.", "tab_arima_full",
                resize=False, note=r"$^{***}p<0.001$; $^{**}p<0.01$; $^{*}p<0.05$.")
    for st in stations:
        metas[st]["ols_table"].to_csv(tdir / f"S2_ols_determinístico_{st}.csv", index=False)
        metas[st]["hp_search"].to_csv(tdir / f"S3_busqueda_hiperparametros_{st}.csv", index=False)

    # ---------- T05 hiperparámetros seleccionados
    def _hp(hp, h, var, m, keys):
        d = hp.get(f"h{h}|{var}|{m}", {})
        vals = [("∞" if (k == "max_depth" and d.get(k, "") is None) else d.get(k, "")) for k in keys]
        return " / ".join(str(v) for v in vals) if d else ""
    rows = []
    for st in stations:
        hp = metas[st]["hp"]
        for h in cfg["horizons"]:
            rows.append({"Estación": st, "h": h, "p": metas[st]["p"],
                         "ARIMA": f"({metas[st]['arima_order'][0]},0,{metas[st]['arima_order'][1]})",
                         "ELM L / λ": _hp(hp, h, "uni", "ELM", ["n_hidden", "ridge"]),
                         "MLP nodos": _hp(hp, h, "uni", "MLP", ["n_hidden"]),
                         "NPAR α": _hp(hp, h, "uni", "NPAR", ["alpha"]),
                         "RF prof. / hoja": _hp(hp, h, "uni", "RF", ["max_depth", "min_samples_leaf"]),
                         f"{'XGB' if 'XGB' in ORDER else 'HGB'} prof. / tasa": _hp(hp, h, "uni", "XGB", ["max_depth", "learning_rate"]),
                         "GRU unid.": _hp(hp, h, "uni", "GRU", ["hidden"]),
                         "LSTM unid.": _hp(hp, h, "uni", "LSTM", ["hidden"]),
                         "ELM+met L / λ": _hp(hp, h, "met", "ELM", ["n_hidden", "ridge"]),
                         "MLP+met nodos": _hp(hp, h, "met", "MLP", ["n_hidden"])})
    nv = int(round(cfg["split"]["val_fraction"] * (metas[stations[0]]["a0"])))
    write_table(pd.DataFrame(rows), tdir, "T05_hiperparametros",
                f"Hiperparámetros seleccionados por estación y horizonte: $p$ por {cfg['lags']['select'].upper()}; el resto, por "
                f"mínimo SMAPE en un hold-out temporal de {nv} h al final del primer conjunto de entrenamiento. ELM: "
                "activación sigmoide, pesos $\\mathcal{U}(-1,1)$, $L$ neuronas y regularización $\\lambda$; MLP: activación "
                "logística, L-BFGS, L2 $=10^{-4}$; RF: 300 árboles (profundidad máxima / mínimo por hoja); "
                f"{'XGB' if 'XGB' in ORDER else 'HGB'}: 400 iteraciones de boosting "
                "(profundidad / tasa de aprendizaje); GRU y LSTM: una capa recurrente, Adam, parada temprana.", "tab_hp")

    # ---------- T06 validación por ventanas (h=1), formato ancho
    models_present = [m for m in ORDER if m in preds.model.unique()]
    W = M[(M.h == 1) & (M.window != "all")]
    rows = []
    for m in models_present:
        r = {"Modelo": m}
        for st in stations:
            g = W[(W.station == st) & (W.model == m)]
            r[f"{st} MAE"] = f"{g.MAE.mean():.2f} ({g.MAE.std(ddof=1):.2f})"
            r[f"{st} SMAPE"] = f"{g.SMAPE.mean():.1f} ({g.SMAPE.std(ddof=1):.1f})"
        rows.append(r)
    sizes = [b - a for a, b in metas[stations[0]]["windows"]]
    write_table(pd.DataFrame(rows), tdir, "T06_validacion_ventanas_h1",
                f"Validación con origen móvil, $h=1$: media (desviación estándar) de MAE ($\\mu$g/m$^3$) y SMAPE (\\%) sobre "
                f"{len(sizes)} ventanas ({', '.join(map(str, sizes))} h; total {sum(sizes)} h). Todos los modelos, incluidas las "
                "líneas base, pronostican con la misma información (origen $\\tau-1$).", "tab_cv")

    # ---------- T07 prueba agregada (h=1): formato ancho (principal) y largo (apéndice)
    T = M[(M.h == 1) & (M.window == "all")]
    best_by = {st: best_model(M, st, 1) for st in stations}
    rows_w, rows_l = [], []
    for m in models_present:
        r = {"Modelo": m}
        for st in stations:
            g = T[(T.station == st) & (T.model == m)].iloc[0]
            b = (lambda s: f"\\textbf{{{s}}}") if m == best_by[st] else (lambda s: s)
            r[f"{st} MAE"] = b(_f(g.MAE, 2))
            r[f"{st} SMAPE"] = _f(g.SMAPE, 1)
            r[f"{st} MASE"] = _f(g.MASE, 2)
            r[f"{st} R²"] = _f(g.R2, 2)
            rows_l.append({"Estación": st, "Modelo": m, "MAE": _f(g.MAE), "RMSE": _f(g.RMSE), "SMAPE (%)": _f(g.SMAPE, 2),
                           "MASE": _f(g.MASE), "R²": _f(g.R2), "MAE rel. Naive": _f(g.RelMAE), "MAPE (%)": _f(g.MAPE, 1)})
        rows_w.append(r)
    rows_l = sorted(rows_l, key=lambda r: stations.index(r["Estación"]))
    write_table(pd.DataFrame(rows_w), tdir, "T07_prueba_h1",
                f"Desempeño en el período de prueba completo ({sum(sizes)} h), $h=1$, escala original. MAE en $\\mu$g/m$^3$; "
                "SMAPE en \\%; MASE escalado con el MAE del Naive en entrenamiento; R$^2$ de pronóstico. Negrita: mejor modelo "
                "del componente estocástico (excluye líneas base) por MAE.", "tab3",
                raw_cols=tuple(f"{st} MAE" for st in stations))
    write_table(pd.DataFrame(rows_l), tdir, "S4_prueba_h1_completa",
                f"Métricas completas en el período de prueba ({sum(sizes)} h), $h=1$, incluidos RMSE, MAE relativo al Naive y MAPE.",
                "tab3_full")

    # ---------- T08 Diebold–Mariano (h=1): largo (CSV/verificaciones) y ancho (manuscrito)
    rows = []
    P1 = preds[preds.h == 1]
    for st in stations:
        best = best_by[st]
        piv = P1[P1.station == st].pivot_table(index="tau", columns="model", values="yhat")
        yy = P1[(P1.station == st) & (P1.model == best)].set_index("tau").y.loc[piv.index]
        comps = [m for m in models_present if m not in (best, "Determ")]
        res = [dm_test((yy - piv[best]).to_numpy(), (yy - piv[m]).to_numpy(), 1, cfg["dm"]["loss"], cfg["dm"]["hac_lags"])
               for m in comps]
        adj = holm([r["p_HAC"] for r in res]) if cfg["dm"]["holm"] else [r["p_HAC"] for r in res]
        for m, r, pa in zip(comps, res, adj):
            rows.append({"Estación": st, "Mejor": best, "Rival": m, "Comparación": f"{best} vs. {m}",
                         "DM (HLN)": f"{r['DM_HLN']:+.3f}{stars(r['p_HLN'])}", "p (HLN)": _pfmt(r["p_HLN"]),
                         "DM (HAC)": f"{r['DM_HAC']:+.3f}", "p (HAC)": _pfmt(r["p_HAC"]),
                         "p (HAC, Holm)": _pfmt(pa) + stars(pa), "_dm": r["DM_HAC"], "_p": pa})
    dm_df = pd.DataFrame(rows)
    dm_df.drop(columns=["_dm", "_p"]).to_csv(tdir / "T08_diebold_mariano_h1_largo.csv", index=False)
    wide = []
    for m in [m for m in models_present if m != "Determ"]:
        r = {"Modelo rival": m}
        for st in stations:
            sub = dm_df[(dm_df["Estación"] == st) & (dm_df.Rival == m)]
            r[f"{st} (vs. {best_by[st]})"] = "---" if sub.empty else f"{sub._dm.iloc[0]:+.2f}{stars(sub._p.iloc[0])}"
        wide.append(r)
    write_table(pd.DataFrame(wide), tdir, "T08_diebold_mariano_h1",
                "Prueba de Diebold--Mariano ($h=1$, pérdida absoluta): estadístico del mejor modelo de cada estación (entre "
                "paréntesis) frente a cada rival. Negativo: el mejor modelo es más preciso.", "tab_dm",
                note="Varianza Newey--West con " f"{cfg['dm']['hac_lags']}" " rezagos (HAC), robusta a la autocorrelación horaria; "
                     "significancia con valores $p$ ajustados por Holm: $^{***}p<0.001$; $^{**}p<0.01$; $^{*}p<0.05$. "
                     "Versión HLN y valores $p$ completos en T08\\_diebold\\_mariano\\_h1\\_largo.csv.")

    # ---------- T09 desempeño diurno vs resto (h=1), formato ancho
    d0, d1 = cfg["diurnal"]["day_hours"]
    P1 = P1.assign(period=np.where(P1.hour.between(d0, d1), "Diurno", "Resto"))
    hour_rows, rows = [], []
    for (st, m), g in P1.groupby(["station", "model"]):
        for hr, gh in g.groupby("hour"):
            hour_rows.append({"station": st, "model": m, "hour": hr, "MAE": float((gh.y - gh.yhat).abs().mean()),
                              "mean_obs": float(gh.y.mean())})
    for m in [m for m in models_present if m != "Determ"]:
        r = {"Modelo": m}
        for st in stations:
            g = P1[(P1.station == st) & (P1.model == m)]
            mae = g.assign(a=(g.y - g.yhat).abs()).groupby("period").a.mean()
            r[f"{st} {d0}-{d1} h"] = _f(mae.get("Diurno"), 2)
            r[f"{st} resto"] = _f(mae.get("Resto"), 2)
        rows.append(r)
    write_table(pd.DataFrame(rows), tdir, "T09_diurno_nocturno_h1",
                f"MAE ($\\mu$g/m$^3$, $h=1$) en la franja fotoquímica ({d0}:00--{d1}:00) y en el resto de horas, por estación.",
                "tab_diurnal")
    H = pd.DataFrame(hour_rows)
    H.to_csv(tdir / "T09b_mae_por_hora.csv", index=False)

    # ---------- T10 horizontes
    rows = []
    for st in stations:
        for h in cfg["horizons"]:
            g = M[(M.station == st) & (M.h == h) & (M.window == "all")].set_index("model")
            r = {"Estación": st, "h": h}
            for m in [m for m in models_present if m not in ("Determ", "SNaive168")]:
                r[m] = f"{g.loc[m, 'MAE']:.2f} ({g.loc[m, 'RelMAE']:.2f})"
            rows.append(r)
    write_table(pd.DataFrame(rows), tdir, "T10_horizontes",
                "MAE ($\\mu$g/m$^3$) por horizonte de pronóstico y, entre paréntesis, MAE relativo al Naive del mismo horizonte "
                "(estrategia directa).", "tab_horizons")

    # ---------- T11 univariado vs multivariado
    rows = []
    if cfg.get("multivariate", {}).get("enabled"):
        for st in stations:
            for h in cfg["horizons"]:
                Ph = preds[(preds.station == st) & (preds.h == h)]
                piv = Ph.pivot_table(index="tau", columns="model", values="yhat")
                yy = Ph[Ph.model == "Naive"].set_index("tau").y.loc[piv.index]
                for m in cfg["multivariate"]["models"]:
                    mm = f"{m}+met"
                    if mm not in piv:
                        continue
                    r = dm_test((yy - piv[mm]).to_numpy(), (yy - piv[m]).to_numpy(), h, cfg["dm"]["loss"], cfg["dm"]["hac_lags"])
                    mae_u = float((yy - piv[m]).abs().mean())
                    mae_m = float((yy - piv[mm]).abs().mean())
                    rows.append({"Estación": st, "h": h, "Modelo": m, "MAE univariado": _f(mae_u), "MAE + meteorología": _f(mae_m),
                                 "Cambio (%)": _f(100 * (mae_m - mae_u) / mae_u, 1), "DM (HAC)": f"{r['DM_HAC']:+.3f}",
                                 "p (HAC)": _pfmt(r["p_HAC"])})
        covs = ", ".join(cfg["multivariate"]["covariates"])
        write_table(pd.DataFrame(rows), tdir, "T11_univariado_vs_meteorologia",
                    f"Efecto de añadir covariables meteorológicas rezagadas ({covs}; rezagos {cfg['multivariate']['cov_lags']}) "
                    "a ELM y MLP. DM negativo: la variante con meteorología es más precisa.", "tab_multivar")
    mv_df = pd.DataFrame(rows)

    # ---------- T12 diagnóstico de MAPE
    rows = []
    for st in stations:
        g = T[T.station == st].set_index("model")
        best = best_by[st]
        rows.append({"Estación": st, f"% horas de prueba < {nz:g}": _f(g.loc["Naive", "share_near_zero"], 1),
                     "MAPE Naive (%)": _f(g.loc["Naive", "MAPE"], 1), "SMAPE Naive (%)": _f(g.loc["Naive", "SMAPE"], 2),
                     "Mejor modelo": best, "MAPE mejor (%)": _f(g.loc[best, "MAPE"], 1),
                     "SMAPE mejor (%)": _f(g.loc[best, "SMAPE"], 2)})
    write_table(pd.DataFrame(rows), tdir, "T12_diagnostico_mape",
                f"Diagnóstico de la inestabilidad del MAPE: proporción de horas de prueba con concentración < {nz:g} $\\mu$g/m$^3$.",
                "tab_mape", resize=True)

    # ---------- T15 variables del estudio (Obs. Soria-02)
    rows = []
    VARS = [("ozono", "Ozono troposférico (O$_3$)", "$\\mu$g/m$^3$", "Variable respuesta"),
            ("temp", "Temperatura del aire", "°C", "Covariable (variante +met)"),
            ("humrel", "Humedad relativa", "\\%", "Covariable (variante +met)"),
            ("velvie", "Velocidad del viento", "m/s", "Covariable (variante +met)"),
            ("dirvie", "Dirección del viento", "° (seno/coseno)", "Covariable (variante +met)")]
    for col, name, unit, use in VARS:
        vals = [metas[st]["vars_summary"].get(col) for st in stations]
        rng = [v for v in vals if v]
        rows.append({"Variable": name, "Unidad": unit, "Fuente": "SENAMHI (4 estaciones)",
                     "Media (rango entre estaciones)": (f"{np.mean([v['mean'] for v in rng]):.1f} "
                                                        f"({min(v['mean'] for v in rng):.1f}--{max(v['mean'] for v in rng):.1f})") if rng else "---",
                     "Faltantes": "0 \\%" if rng and all(v["missing"] == 0 for v in rng) else "ver texto", "Uso": use})
    for name in ["Radiación solar", "NO$_x$ y NO", "Compuestos orgánicos volátiles (COV)", "Flujo vehicular"]:
        rows.append({"Variable": name, "Unidad": "---", "Fuente": "No disponible", "Media (rango entre estaciones)": "---",
                     "Faltantes": "100 \\%", "Uso": "No incluida (limitación declarada)"})
    write_table(pd.DataFrame(rows), tdir, "T15_variables",
                "Variables consideradas en el estudio, disponibilidad en el período 2017--2019 (inviernos) y uso en los modelos.",
                "tab_vars", raw_cols=("Variable", "Unidad", "Faltantes"))

    # ---------- figuras
    _fig_years(metas, stations, fdir)
    _fig_decomp(metas, stations, fdir)
    _fig_obs_pred(preds, M, stations, fdir)
    _fig_hourly(H, M, stations, fdir)
    _fig_horizons(M, stations, cfg, fdir)
    _fig_acf(metas, stations, fdir)
    _fig_diurnal_met(metas, stations, fdir)

    # ---------- verificaciones y resumen
    checks = _checks(preds, M, metas, stations, cfg, dm_df, mv_df)
    (out / "verificaciones.md").write_text(checks, encoding="utf-8")
    for st in stations:
        g = T[T.station == st].set_index("model")
        best = best_model(M, st, 1)
        summary["stations"][st] = {
            "best_univariate_h1": best,
            "best_any_h1": best_model(M, st, 1, pool=[m for m in models_present if m not in BASE + ["Determ"]]),
            "MAE_best": float(g.loc[best, "MAE"]), "MAE_naive": float(g.loc["Naive", "MAE"]),
            "SMAPE_best": float(g.loc[best, "SMAPE"]), "SMAPE_naive": float(g.loc["Naive", "SMAPE"]),
            "MASE_best": float(g.loc[best, "MASE"]), "RelMAE_best": float(g.loc[best, "RelMAE"]),
            "p": metas[st]["p"], "arima_order": list(metas[st]["arima_order"]),
            "runtime_s": metas[st]["runtime_s"],
        }
    (out / "resumen_resultados.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


# ----------------------------------------------------------------------------- figuras
def _grid4(sharex=False):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=sharex)
    return fig, axes.ravel()


def _fig_years(metas, stations, fdir):
    fig, axes = plt.subplots(1, len(stations), figsize=(7.2, 2.6), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, st in zip(axes, stations):
        s = metas[st]["series"]
        yrs = sorted(s.season.unique())
        ax.boxplot([s.loc[s.season == y, "y"] for y in yrs], tick_labels=[str(y) for y in yrs], widths=0.5,
                   showfliers=False, medianprops={"color": COLORS["ELM"], "linewidth": 2},
                   boxprops={"color": INK2}, whiskerprops={"color": INK2}, capprops={"color": INK2})
        ax.set_title(STATION_NAMES.get(st, st), color=INK)
    axes[0].set_ylabel(r"O$_3$ ($\mu$g m$^{-3}$)")
    fig.suptitle("Ozone level by winter (outliers not shown)", color=INK, fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(fdir / "fig_levels_by_year.pdf")
    plt.close(fig)


def _fig_decomp(metas, stations, fdir):
    fig, axes = plt.subplots(len(stations), 2, figsize=(7.2, 1.7 * len(stations)), squeeze=False)
    for i, st in enumerate(stations):
        s = metas[st]["series"]
        x = np.arange(len(s))
        a0 = metas[st]["a0"]
        ax = axes[i, 0]
        ax.plot(x, s.ly, color="#c3c2b7", lw=0.4, label="Observed log(O$_3$)")
        ax.plot(x, s.zd, color=COLORS["ELM"], lw=0.8, label="Deterministic component")
        ax.axvline(a0, color=INK2, lw=0.8, ls="--")
        ax.set_ylabel(STATION_NAMES.get(st, st), color=INK)
        ax2 = axes[i, 1]
        ax2.plot(x, s.e, color=COLORS["MLP"], lw=0.4)
        ax2.axvline(a0, color=INK2, lw=0.8, ls="--")
        for a in (ax, ax2):
            b = np.where(np.diff(s.season.to_numpy()) != 0)[0]
            a.set_xticks(np.r_[0, b + 1])
            a.set_xticklabels([str(y) for y in sorted(s.season.unique())])
    axes[0, 0].set_title("log(O$_3$) and deterministic component", color=INK)
    axes[0, 1].set_title("Stochastic residual $e_h$", color=INK)
    axes[0, 0].legend(loc="upper right", fontsize=7)
    fig.text(0.5, -0.01, "Dashed line: start of the test period", ha="center", color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(fdir / "fig_decomposition.pdf")
    plt.close(fig)


def _fig_obs_pred(preds, M, stations, fdir, last=168):
    fig, axes = _grid4()
    for ax, st in zip(axes, stations):
        best = best_model(M, st, 1)
        g = preds[(preds.station == st) & (preds.h == 1)]
        tail = sorted(g.tau.unique())[-last:]
        ob = g[(g.model == best) & g.tau.isin(tail)].sort_values("tau")
        nv = g[(g.model == "Naive") & g.tau.isin(tail)].sort_values("tau")
        ax.plot(ob.time, ob.y, color=COLORS["Observado"], lw=1.2, label="Observed")
        ax.plot(ob.time, ob.yhat, color=COLORS["ELM"], lw=1.2, label=best)
        ax.plot(nv.time, nv.yhat, color=COLORS["Naive"], lw=0.9, ls=(0, (3, 2)), label="Naive")
        ax.set_title(STATION_NAMES.get(st, st), color=INK)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    handles = [plt.Line2D([], [], color=COLORS["Observado"], lw=1.2, label="Observed"),
               plt.Line2D([], [], color=COLORS["ELM"], lw=1.2, label="Best model at each station (see panel title)"),
               plt.Line2D([], [], color=COLORS["Naive"], lw=0.9, ls=(0, (3, 2)), label="Naive")]
    for ax, st in zip(axes, stations):
        ax.set_title(f"{STATION_NAMES.get(st, st)} — {best_model(M, st, 1)}", color=INK)
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.04))
    axes[0].set_ylabel(r"O$_3$ ($\mu$g m$^{-3}$)")
    axes[2].set_ylabel(r"O$_3$ ($\mu$g m$^{-3}$)")
    fig.suptitle(f"Observed vs. 1-h-ahead forecast (last {last} h of the test period)", color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(fdir / "fig_observed_vs_forecast.pdf")
    plt.close(fig)


def _fig_hourly(H, M, stations, fdir):
    fig, axes = _grid4(sharex=True)
    for ax, st in zip(axes, stations):
        best = best_model(M, st, 1)
        show = [best] + [m for m in ["ARIMA", "Naive"] if m != best]
        g = H[H.station == st]
        for m in show:
            gm = g[g.model == m].sort_values("hour")
            ax.plot(gm.hour, gm.MAE, color=COLORS.get(m, COLORS["ELM"]), lw=2, marker="o", ms=3, label=m)
        ax.set_title(STATION_NAMES.get(st, st), color=INK)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.legend(fontsize=7)
    for i in (2, 3):
        axes[i].set_xlabel("Local time (h)")
    for i in (0, 2):
        axes[i].set_ylabel(r"MAE ($\mu$g m$^{-3}$)")
    fig.suptitle("Mean absolute error by hour of day, $h=1$", color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(fdir / "fig_error_by_hour.pdf")
    plt.close(fig)


def _fig_horizons(M, stations, cfg, fdir):
    if len(cfg["horizons"]) < 2:
        return
    fig, axes = _grid4(sharex=True)
    for ax, st in zip(axes, stations):
        g = M[(M.station == st) & (M.window == "all")]
        for m in ["ELM", "MLP", "NPAR", next((x for x in ("XGB", "HGB") if x in set(g.model)), "XGB"), "ARIMA", "SNaive24"]:
            gm = g[g.model == m].sort_values("h")
            if len(gm):
                ax.plot(gm.h, gm.RelMAE, color=COLORS[m], lw=2, marker="o", ms=4, label=m)
        ax.axhline(1.0, color=INK2, lw=0.8, ls="--")
        ax.set_title(STATION_NAMES.get(st, st), color=INK)
        ax.set_xticks(cfg["horizons"])
    axes[0].legend(fontsize=7)
    for i in (2, 3):
        axes[i].set_xlabel("Forecast horizon (h)")
    for i in (0, 2):
        axes[i].set_ylabel("MAE relative to naive")
    fig.suptitle("Performance by horizon (< 1: better than the naive forecast at the same horizon)", color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(fdir / "fig_horizons.pdf")
    plt.close(fig)


def _fig_acf(metas, stations, fdir):
    """Identificación Box–Jenkins: ACF de log(O3) y ACF/PACF del residuo, rezagos 1–168 h."""
    fig, axes = plt.subplots(len(stations), 3, figsize=(7.2, 1.55 * len(stations)), sharex=True, squeeze=False)
    titles = ["ACF of log(O$_3$)", "ACF of residual $e_h$", "PACF of residual $e_h$"]
    for i, st in enumerate(stations):
        a = metas[st]["acf"]
        for j, key in enumerate(["acf_y", "acf_e", "pacf_e"]):
            ax = axes[i, j]
            v = np.asarray(a[key])[1:]
            lags = np.arange(1, len(v) + 1)
            ax.vlines(lags, 0, v, color=COLORS["ELM"], lw=0.8)
            ax.axhline(0, color=INK2, lw=0.6)
            ax.axhspan(-a["band"], a["band"], color=GRID, alpha=0.8, lw=0)
            for L in (24, 48, 72, 96, 120, 144, 168):
                ax.axvline(L, color="#c3c2b7", lw=0.5, ls=":", zorder=0)
            ax.set_xticks([24, 72, 120, 168])
            if i == 0:
                ax.set_title(titles[j], color=INK)
            if j == 0:
                ax.set_ylabel(STATION_NAMES.get(st, st), color=INK)
    for j in range(3):
        axes[-1, j].set_xlabel("Lag (h)")
    fig.text(0.5, -0.01, "Shaded band: 95% confidence limits; dotted lines: multiples of 24 h", ha="center", color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(fdir / "fig_acf_pacf.pdf")
    plt.close(fig)


def _fig_diurnal_met(metas, stations, fdir):
    """Perfil horario medio de ozono, velocidad y dirección del viento, por estación."""
    st_colors = dict(zip(stations, [COLORS["ELM"], COLORS["MLP"], COLORS["NPAR"], COLORS["ARIMA"]]))
    panels = [("ozono", r"O$_3$ ($\mu$g m$^{-3}$)"), ("velvie", "Wind speed (m s$^{-1}$)"),
              ("dirvie", "Wind direction (°)"), ("humrel", "Relative humidity (%)")]
    panels = [p for p in panels if all(p[0] in metas[s]["diurnal"] for s in stations)]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2, 2.4))
    axes = np.atleast_1d(axes)
    for ax, (col, lab) in zip(axes, panels):
        for st in stations:
            d = metas[st]["diurnal"]
            ax.plot(d["hour"], d[col], color=st_colors[st], lw=2, label=STATION_NAMES.get(st, st))
        ax.set_title(lab, color=INK, fontsize=8)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_xlabel("Local time (h)")
    h, l = axes[0].get_legend_handles_labels()
    fig.suptitle("Mean diurnal cycle of ozone and meteorology (winters 2017–2019)", color=INK, fontsize=9, y=1.02)
    fig.tight_layout()
    fig.legend(h, l, loc="lower center", ncol=len(stations), fontsize=7, bbox_to_anchor=(0.5, -0.08))
    fig.savefig(fdir / "fig_diurnal_met.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------- verificaciones
def _checks(preds, M, metas, stations, cfg, dm_df, mv_df) -> str:
    L = ["# Verificaciones automáticas", "",
         "Generado por el pipeline. Use estas frases para que el texto del manuscrito coincida con las tablas.", ""]
    st0 = stations[0]
    sizes = [b - a for a, b in metas[st0]["windows"]]
    L += ["## Partición", f"- n = {metas[st0]['n']}; entrenamiento inicial = {metas[st0]['a0']}; prueba = {sum(sizes)} h "
          f"= {len(sizes)} ventanas de {sizes} h (suma exacta).",
          f"- Hold-out de hiperparámetros = {int(round(cfg['split']['val_fraction'] * metas[st0]['a0']))} h "
          f"({100 * cfg['split']['val_fraction']:.0f} % del primer entrenamiento).", ""]
    for st in stations:
        if cfg["lags"]["select"] == "bic" and metas[st]["p"] >= cfg["lags"]["max_p"]:
            L.append(f"- **ALERTA** {st}: el BIC eligió p = {metas[st]['p']}, el tope de la grilla (max_p). Aumente max_p.")
    L.append("")
    L += ["## Datos"]
    for st in stations:
        q = metas[st]["quality_log"]
        qy = metas[st]["quality_year"]
        L.append(f"- {st}: faltantes antes de imputar = {q['missing_before_impute']}; covariables disponibles = "
                 f"{q['covariates_available']}. {q['imputation']}")
        if qy.attrs["max_min_mean_ratio"] > 2:
            L.append(f"  - **ALERTA**: la media anual varía {qy.attrs['max_min_mean_ratio']:.1f}× entre inviernos "
                     f"({', '.join(f'{int(r.Año)}: {r.Media:.1f}' for _, r in qy.iterrows())}). Discutir posible cambio "
                     "de régimen o de calibración del analizador.")
    L.append("")
    L += ["## ¿El mejor modelo supera al Naive? (h = 1, prueba completa)"]
    T = M[(M.h == 1) & (M.window == "all")]
    for st in stations:
        g = T[T.station == st].set_index("model")
        b = best_model(M, st, 1)
        beats_mae = g.loc[b, "MAE"] < g.loc["Naive", "MAE"]
        beats_smape = g.loc[b, "SMAPE"] < g.loc["Naive", "SMAPE"]
        sub = dm_df[(dm_df["Estación"] == st) & (dm_df["Comparación"] == f"{b} vs. Naive")]
        p_adj = sub["p (HAC, Holm)"].iloc[0] if len(sub) else "n/d"
        sig = isinstance(p_adj, str) and "*" in p_adj
        if beats_mae and beats_smape:
            verdict = "SÍ, significativo" if sig else "SÍ en puntos, pero NO significativo (DM)"
        else:
            verdict = "PARCIAL" if beats_mae or beats_smape else "NO"

        L.append(f"- {st}: mejor univariado = {b}; MAE {g.loc[b, 'MAE']:.3f} vs Naive {g.loc['Naive', 'MAE']:.3f} "
                 f"(MAE rel. {g.loc[b, 'RelMAE']:.3f}); SMAPE {g.loc[b, 'SMAPE']:.2f} vs {g.loc['Naive', 'SMAPE']:.2f}; "
                 f"DM p (HAC, Holm) = {p_adj} → **{verdict}**")
    L.append("- Si alguna estación sale NO, PARCIAL o no significativa, el resumen y las conclusiones no pueden afirmar que ELM/MLP "
             "superan a las líneas base en todas las estaciones.")
    L.append("")
    L += ["## MAPE"]
    for st in stations:
        g = T[T.station == st].set_index("model")
        L.append(f"- {st}: {g.loc['Naive', 'share_near_zero']:.1f} % de horas de prueba < {cfg['metrics']['near_zero_threshold']:g}; "
                 f"MAPE Naive = {g.loc['Naive', 'MAPE']:.1f} %.")
    L.append("- Atribuya MAPE extremos a valores cercanos a cero solo en estaciones con proporción alta de horas bajo el umbral.")
    L.append("")
    L += ["## ARIMA"]
    for st in stations:
        d = metas[st]["arima_diag"]
        flag = []
        if d["LB_p_24"] < 0.05:
            flag.append("Ljung–Box rechaza ruido blanco en el rezago 24")
        if d["min_ar_ma_root_distance"] < 0.1:
            flag.append("raíces AR y MA casi se cancelan (sobreparametrización)")
        L.append(f"- {st}: ARIMA({d['p']},0,{d['q']}); " + ("; ".join(flag) if flag else "diagnósticos sin alertas") + ".")
    L.append("")
    L += ["## Estacionalidad remanente (justificación ARIMA vs SARIMA)"]
    for st in stations:
        r = metas[st]["resid_season"]
        out24 = abs(r["PACF_24"]) > r["ACF_band_95"]
        out168 = abs(r["PACF_168"]) > r["ACF_band_95"]
        L.append(f"- {st}: PACF(24) = {r['PACF_24']:.3f}, PACF(168) = {r['PACF_168']:.3f} (banda ±{r['ACF_band_95']:.3f}; "
                 f"ACF(24) = {r['ACF_24']:.3f}) → "
                 + ("queda dependencia estacional directa: justificar la exclusión de SARIMA con un SARIMA de contraste "
                    "o decir que los rezagos 24 los capturan ELM/MLP/NPAR" if out24 or out168
                    else "sin estacionalidad remanente directa: SARIMA es redundante"))
    L.append("")
    if len(mv_df):
        L += ["## Meteorología (h = 1)"]
        for _, r in mv_df[mv_df.h == 1].iterrows():
            L.append(f"- {r['Estación']} {r['Modelo']}: MAE {r['MAE univariado']} → {r['MAE + meteorología']} "
                     f"({r['Cambio (%)']} %), p (HAC) = {r['p (HAC)']}.")
        L.append("")
    L += ["## Consistencia de signos en la prueba DM",
          "- Convención: DM negativo ⇒ el primer modelo de la comparación es más preciso. "
          "En T08 el primer modelo es siempre el mejor por MAE, por lo que un DM positivo y significativo sería una incoherencia.",
          ""]
    bad = dm_df[dm_df["DM (HAC)"].str.startswith("+") & dm_df["p (HAC, Holm)"].str.contains(r"\*")]
    L.append("- Incoherencias detectadas: " + ("ninguna." if bad.empty else bad.to_string(index=False)))
    return "\n".join(L) + "\n"
