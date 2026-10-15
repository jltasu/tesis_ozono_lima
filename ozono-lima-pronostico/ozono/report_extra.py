"""Tablas de generalización (T13, T14) y párrafos de resultados redactados con las cifras reales
(horizontes y generalización), en español e inglés, listos para \\input en Overleaf."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import dm_test, stars
from .report import BASE, ORDER, UNI, _f, _pfmt, best_model, write_table

NAMES_ES = {"ATE": "ATE", "CDM": "Campo de Marte", "SB": "San Borja", "STA": "Santa Anita"}


def _mae(g):
    return float((g.y - g.yhat).abs().mean())


def _smape(g):
    return float(100 * np.mean((g.y - g.yhat).abs() / ((g.y.abs() + g.yhat.abs()) / 2)))


# ----------------------------------------------------------------------------- tablas
def generalization_tables(G: pd.DataFrame, cfg: dict, tdir: Path) -> dict:
    res = {}
    hac = cfg["dm"]["hac_lags"]
    # ---------- T13 entre inviernos
    W = G[G.experiment == "cross_winter"]
    rows, cw = [], []
    for (st, h, Y), g in W.groupby(["station", "h", "test_winter"]):
        nv = g[g.model == "Naive"].set_index("tau")
        mae_nv = _mae(nv)
        models = [m for m in ORDER if m in g.model.unique()]
        best = min([m for m in UNI if m in models], key=lambda m: _mae(g[g.model == m]))
        for m in models:
            gm = g[g.model == m].set_index("tau").loc[nv.index]
            r = {"Estación": st, "h": h, "Invierno de prueba": f"{int(Y)} (entren. {g.train_winters.iloc[0]})", "Modelo": m,
                 "MAE": _f(_mae(gm)), "SMAPE (%)": _f(_smape(gm), 2), "MAE rel. Naive": _f(_mae(gm) / mae_nv)}
            if m != "Naive":
                d = dm_test((gm.y - gm.yhat).to_numpy(), (nv.y - nv.yhat).to_numpy(), h, cfg["dm"]["loss"], hac)
                r["DM vs Naive (HAC)"] = f"{d['DM_HAC']:+.2f}"
                r["p"] = _pfmt(d["p_HAC"])
            else:
                r["DM vs Naive (HAC)"], r["p"] = "---", "---"
            rows.append(r)
            cw.append({"station": st, "h": h, "winter": int(Y), "model": m, "MAE": _mae(gm),
                       "rel": _mae(gm) / mae_nv, "best": m == best, "n": len(gm),
                       "p": (d["p_HAC"] if m != "Naive" else np.nan),
                       "train": g.train_winters.iloc[0]})
    T13 = pd.DataFrame(rows)
    if len(T13):
        n_h = int(W.groupby(["station", "h", "test_winter"]).tau.nunique().median())
        # versión larga (todas las métricas y DM) solo en CSV; versión compacta para el manuscrito
        T13.to_csv(tdir / "T13_generalizacion_entre_inviernos_largo.csv", index=False)
        C = pd.DataFrame(cw)
        mods = [m for m in ORDER if m in set(C.model) and m != "Naive"]
        wide = []
        for (st, Y, h), g in C.sort_values(["station", "winter", "h"]).groupby(["station", "winter", "h"], sort=False):
            g = g.set_index("model")
            r = {"Estación": st, "Invierno de prueba (entren.)": f"{Y} ({g.train.iloc[0]})", "h": int(h),
                 "MAE Naive": _f(g.loc["Naive", "MAE"])}
            for m in mods:
                txt = f"{g.loc[m, 'rel']:.2f}{stars(g.loc[m, 'p'])}"
                r[m] = rf"\textbf{{{txt}}}" if (m in UNI and g.loc[m, "best"]) else txt
            wide.append(r)
        write_table(pd.DataFrame(wide), tdir, "T13_generalizacion_entre_inviernos",
                    f"Generalización temporal: entrenamiento con los inviernos previos y pronóstico de un invierno completo "
                    f"no visto (~{n_h} h por invierno). Se muestra el MAE ($\\mu$g/m$^3$) del Naive y el MAE de cada modelo "
                    f"relativo al Naive (< 1: mejor que el Naive). Componente determinística sin tendencia ni dummies de año.",
                    "tab_crosswinter", raw_cols=tuple(mods),
                    note="Negrita: mejor modelo del componente estocástico. Prueba DM frente al Naive con varianza HAC: "
                         "$^{***}p<0.001$; $^{**}p<0.01$; $^{*}p<0.05$ (el signo se lee en la razón: < 1 mejora, > 1 empeora). "
                         "MAE, SMAPE y estadísticos DM completos en T13\\_generalizacion\\_entre\\_inviernos\\_largo.csv.")
    res["cw"] = pd.DataFrame(cw)

    # ---------- T14 entre estaciones
    S = G[G.experiment == "cross_station"] if "source" in G.columns else G.iloc[0:0]
    rows, cs = [], []
    for (tgt, h, m), g in (S[S.source != "—"].groupby(["station", "h", "model"]) if len(S) else []):
        local = _mae(g[g.source == "Local"])
        nv = _mae(S[(S.station == tgt) & (S.h == h) & (S.model == "Naive")])
        r = {"Estación destino": tgt, "h": h, "Modelo": m, "Naive": _f(nv), "Local": _f(local)}
        for src in ["Combinado"] + sorted(set(g.source) - {"Local", "Combinado"}):
            v = _mae(g[g.source == src])
            r[f"desde {src}"] = f"{v:.3f} ({v / local:.2f})"
            cs.append({"target": tgt, "h": h, "model": m, "source": src, "MAE": v, "rel_local": v / local, "local": local,
                       "naive": nv})
        rows.append(r)
    T14 = pd.DataFrame(rows)
    if len(T14):
        for c in [c for c in T14.columns if c.startswith("desde ")]:
            T14[c] = T14[c].fillna("---")
        p = int(S.p_common.iloc[0])
        write_table(T14, tdir, "T14_transferencia_entre_estaciones",
                    "Generalización espacial ($h$ según fila): MAE en la estación destino cuando el modelo estocástico se "
                    "entrena en ella misma (Local), en las cuatro estaciones combinadas o en otra estación; entre paréntesis, "
                    f"MAE relativo al modelo local. Rezagos comunes $p={p}$; período de prueba común de "
                    f"{cfg['split']['test_hours']} h.", "tab_crossstation")
    res["cs"] = pd.DataFrame(cs)
    return res


# ----------------------------------------------------------------------------- párrafos
def _horizon_facts(M: pd.DataFrame, cfg: dict, stations: list[str]) -> list[dict]:
    pool = [m for m in ORDER if m not in BASE + ["Determ"]]
    facts = []
    for st in stations:
        for h in cfg["horizons"]:
            g = M[(M.station == st) & (M.h == h) & (M.window == "all")].set_index("model")
            cand = [m for m in pool if m in g.index]
            b = g.loc[cand, "MAE"].idxmin()
            facts.append({"st": st, "h": h, "best": b, "MAE": g.loc[b, "MAE"], "rel": g.loc[b, "RelMAE"],
                          "rel_s24": g.loc[b, "MAE"] / g.loc["SNaive24", "MAE"], "naive": g.loc["Naive", "MAE"],
                          "s24": g.loc["SNaive24", "MAE"]})
    return facts


def horizon_paragraphs(M: pd.DataFrame, cfg: dict, stations: list[str]) -> tuple[str, str]:
    F = pd.DataFrame(_horizon_facts(M, cfg, stations))
    hs = cfg["horizons"]
    h1, hmax = min(hs), max(hs)
    es, en = [], []
    es.append(f"Para evaluar la utilidad del marco más allá de la hora siguiente, se repitió el protocolo de origen móvil "
              f"con pronóstico directo a $h \\in \\{{{', '.join(map(str, hs))}\\}}$ horas (Tabla~\\ref{{tab_horizons}}, "
              f"Figura~\\ref{{fig_horizons}}). Cada modelo se entrenó específicamente para cada horizonte y las líneas base "
              f"utilizaron la misma información disponible en el origen.")
    en.append(f"To assess the usefulness of the framework beyond the next hour, the rolling-origin protocol was repeated "
              f"with direct forecasting at $h \\in \\{{{', '.join(map(str, hs))}\\}}$ hours (Table~\\ref{{tab_horizons}}, "
              f"Figure~\\ref{{fig_horizons}}). A separate model was trained for each horizon, and the baselines used the "
              f"same information available at the forecast origin.")
    for st in stations:
        a = F[(F.st == st) & (F.h == h1)].iloc[0]
        b = F[(F.st == st) & (F.h == hmax)].iloc[0]
        inc = 100 * (b.MAE - a.MAE) / a.MAE
        beats = F[(F.st == st) & (F.rel < 1)].h.tolist()
        beats24 = F[(F.st == st) & (F.rel_s24 < 1)].h.tolist()
        es.append(f"En {NAMES_ES.get(st, st)}, el MAE del mejor modelo pasó de {a.MAE:.2f} ({a.best}, $h={h1}$) a "
                  f"{b.MAE:.2f}\\,$\\mu$g/m$^3$ ({b.best}, $h={hmax}$), un aumento de {inc:.0f}\\%; "
                  + (f"superó al Naive en $h \\in \\{{{', '.join(map(str, beats))}\\}}$" if beats else "no superó al Naive en ningún horizonte")
                  + (f" y al Naive estacional de 24 h en $h \\in \\{{{', '.join(map(str, beats24))}\\}}$."
                     if beats24 else " y no superó al Naive estacional de 24 h."))
        en.append(f"At {NAMES_ES.get(st, st)}, the MAE of the best model rose from {a.MAE:.2f} ({a.best}, $h={h1}$) to "
                  f"{b.MAE:.2f}\\,$\\mu$g/m$^3$ ({b.best}, $h={hmax}$), an increase of {inc:.0f}\\%; it "
                  + (f"outperformed the naive forecast at $h \\in \\{{{', '.join(map(str, beats))}\\}}$" if beats else "did not outperform the naive forecast at any horizon")
                  + (f" and the 24-h seasonal naive at $h \\in \\{{{', '.join(map(str, beats24))}\\}}$."
                     if beats24 else " and did not outperform the 24-h seasonal naive."))
    long = F[F.h >= 12]
    if long.empty:
        return " ".join(es), " ".join(en)
    share = 100 * (long.rel_s24 < 1).mean()
    es.append(f"En los horizontes de 12 a {hmax} h, el mejor modelo superó al Naive estacional de 24 h en "
              f"{(long.rel_s24 < 1).sum()} de {len(long)} combinaciones estación--horizonte ({share:.0f}\\%). "
              "Este resultado delimita el alcance operativo de la formulación univariada: su ventaja se concentra en el "
              "corto plazo, mientras que a horizontes diarios el perfil horario persistente captura buena parte de la "
              "información predecible.")
    en.append(f"For horizons of 12 to {hmax} h, the best model outperformed the 24-h seasonal naive in "
              f"{(long.rel_s24 < 1).sum()} of {len(long)} station--horizon combinations ({share:.0f}\\%). "
              "This result delimits the operational scope of the univariate formulation: its advantage is concentrated in "
              "the short term, whereas at daily horizons the persistent diurnal profile captures much of the predictable "
              "information.")
    return " ".join(es), " ".join(en)


def generalization_paragraphs(res: dict, cfg: dict) -> tuple[str, str]:
    es, en = [], []
    cw, cs = res.get("cw", pd.DataFrame()), res.get("cs", pd.DataFrame())
    if len(cw):
        b = cw[cw.best]
        h0 = int(b.h.min())
        bb = b[b.h == h0]
        n_beat = int((bb.rel < 1).sum())
        rng = (bb.rel.min(), bb.rel.max())
        winters = sorted(cw.winter.unique())
        es.append(f"\\textit{{Generalización temporal}}. Para verificar que el desempeño no depende de una partición particular "
                  f"dentro de 2019, el marco se entrenó con los inviernos previos y se aplicó a un invierno completo no visto "
                  f"({', '.join(map(str, winters))}; Tabla~\\ref{{tab_crosswinter}}). Con $h={h0}$, el mejor modelo superó al "
                  f"Naive en {n_beat} de {len(bb)} combinaciones estación--invierno, con un MAE relativo entre "
                  f"{rng[0]:.2f} y {rng[1]:.2f}.")
        en.append(f"\\textit{{Temporal generalization}}. To verify that performance does not depend on a particular split within "
                  f"2019, the framework was trained on the preceding winters and applied to an entire unseen winter "
                  f"({', '.join(map(str, winters))}; Table~\\ref{{tab_crosswinter}}). At $h={h0}$, the best model outperformed the "
                  f"naive forecast in {n_beat} of {len(bb)} station--winter combinations, with relative MAE between "
                  f"{rng[0]:.2f} and {rng[1]:.2f}.")
    if len(cs):
        h0 = int(cs.h.min())
        c = cs[cs.h == h0]
        tr = c[c.source != "Combinado"]
        pooled = c[c.source == "Combinado"]
        es.append(f"\\textit{{Generalización espacial}}. Al entrenar el componente estocástico en una estación y aplicarlo a otra "
                  f"(Tabla~\\ref{{tab_crossstation}}, $h={h0}$), el MAE fue en promedio {tr.rel_local.mean():.2f} veces el del "
                  f"modelo local (rango {tr.rel_local.min():.2f}--{tr.rel_local.max():.2f}); el modelo transferido superó al "
                  f"Naive de la estación destino en {(tr.MAE < tr.naive).sum()} de {len(tr)} casos. El modelo entrenado con las "
                  f"cuatro estaciones combinadas obtuvo un MAE relativo medio de {pooled.rel_local.mean():.2f} frente al local. "
                  "Esto indica en qué medida la dinámica de corto plazo del residuo es compartida por la red y, por tanto, "
                  "transferible a estaciones sin historial suficiente.")
        en.append(f"\\textit{{Spatial generalization}}. When the stochastic component was trained at one station and applied to "
                  f"another (Table~\\ref{{tab_crossstation}}, $h={h0}$), the MAE averaged {tr.rel_local.mean():.2f} times that "
                  f"of the local model (range {tr.rel_local.min():.2f}--{tr.rel_local.max():.2f}); the transferred model "
                  f"outperformed the target station's naive forecast in {(tr.MAE < tr.naive).sum()} of {len(tr)} cases. The model "
                  f"trained on the four stations pooled achieved a mean relative MAE of {pooled.rel_local.mean():.2f} with respect "
                  "to the local model. This indicates the extent to which the short-term residual dynamics are shared across "
                  "the network and are therefore transferable to stations with insufficient history.")
    return " ".join(es), " ".join(en)


def write_paragraphs(M: pd.DataFrame, res: dict, cfg: dict, stations: list[str], out: Path) -> None:
    tdir = out / "texto"
    tdir.mkdir(exist_ok=True)
    es_h, en_h = horizon_paragraphs(M, cfg, stations)
    (tdir / "resultados_horizontes_ES.tex").write_text(es_h + "\n", encoding="utf-8")
    (tdir / "resultados_horizontes_EN.tex").write_text(en_h + "\n", encoding="utf-8")
    es_g, en_g = generalization_paragraphs(res, cfg)
    if es_g:
        (tdir / "resultados_generalizacion_ES.tex").write_text(es_g + "\n", encoding="utf-8")
        (tdir / "resultados_generalizacion_EN.tex").write_text(en_g + "\n", encoding="utf-8")
