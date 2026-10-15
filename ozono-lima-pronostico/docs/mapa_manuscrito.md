# Correspondencia entre las salidas del pipeline y el manuscrito

Numeración según el manuscrito revisado (versión en español). Cada tabla está en `outputs/tables/`
en formato `.csv` y `.tex`; cada figura, en `outputs/figures/` en `.pdf`.

## Tablas

| Manuscrito | Archivo | Contenido |
|---|---|---|
| Tabla 2  | `T15_variables` | Variables del estudio, disponibilidad y uso |
| Tabla 3  | `T02_descriptivos` | Descriptivos y pruebas ADF/KPSS por estación |
| Tabla 4  | `T01_calidad_por_anio` | Nivel por invierno y prueba de Kruskal–Wallis |
| Tabla 5  | `T03_residuo_estacionalidad` | Estacionalidad remanente y estacionariedad del residuo |
| Tabla 6  | `T04_arima` | ARIMA: órdenes, coeficientes principales y Ljung–Box |
| Tabla 7  | `T04b_sarima` | SARIMA(1,0,1)(1,0,1)₂₄ |
| Tabla 8  | `T07_prueba_h1` | Desempeño en prueba, h = 1 (MAE, SMAPE, MASE, R²) |
| Tabla 9  | `T08_diebold_mariano_h1` | Prueba de Diebold–Mariano (HAC + Holm) |
| Tabla 10 | `T12_diagnostico_mape` | Diagnóstico del MAPE |
| Tabla 11 | `T10_horizontes` | MAE por horizonte (1, 6, 12, 24, 48 h) |
| Tabla 12 | `T13_generalizacion_entre_inviernos` | Generalización a inviernos no vistos |
| Tabla 13 | `T14_transferencia_entre_estaciones` | Transferencia entre estaciones |
| Tabla A1 | `T05_hiperparametros` | Hiperparámetros seleccionados |
| Tabla A2 | `S1_arima_coeficientes` | Coeficientes completos ARIMA y SARIMA |
| Tabla A3 | `T06_validacion_ventanas_h1` | Validación cruzada con origen móvil (5 ventanas) |
| Tabla A4 | `S4_prueba_h1_completa` | Métricas completas, incl. RMSE y MAPE |
| Tabla A5 | `T09_diurno_nocturno_h1` | Error diurno frente a nocturno |
| Tabla A6 | `T11_univariado_vs_meteorologia` | Aporte de la meteorología |

Archivos complementarios (solo CSV): `S2_ols_determinístico_*` (coeficientes OLS),
`S3_busqueda_hiperparametros_*` (búsqueda completa), `T08_diebold_mariano_h1_largo` (DM con HLN y HAC),
`T09b_mae_por_hora`, `T13_generalizacion_entre_inviernos_largo`, `metricas_completas`.

## Figuras

| Manuscrito | Archivo |
|---|---|
| Figura 4  | `fig_levels_by_year.pdf` |
| Figura 5  | `fig_decomposition.pdf` |
| Figura 6  | `fig_acf_pacf.pdf` |
| Figura 7  | `fig_observed_vs_forecast.pdf` |
| Figura 8  | `fig_diurnal_met.pdf` |
| Figura 9  | `fig_error_by_hour.pdf` |
| Figura 10 | `fig_horizons.pdf` |

Las Figuras 1 a 3 (esquema metodológico, arquitectura del ELM y series horarias) no las genera el
pipeline.

## Texto

`outputs/texto/resultados_horizontes_{ES,EN}.tex` y `resultados_generalizacion_{ES,EN}.tex` contienen
los párrafos de resultados redactados con las cifras de las tablas, incluidos en el manuscrito con
`\input`.
