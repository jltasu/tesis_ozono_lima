# Pronóstico horario de ozono en Lima Metropolitana

Código y datos para reproducir todos los resultados del artículo:

> Tarrillo, J., Casas, C., Pacheco, D. y López-Gonzales, J. L. *Hourly Ozone Forecasting in Metropolitan
> Lima with a Hybrid Deterministic–Stochastic Framework: Multi-Horizon and Cross-Station Evaluation*.

Un único comando regenera todas las tablas y figuras del artículo a partir de los datos de `data/`,
junto con un registro de verificaciones automáticas de coherencia (`outputs/verificaciones.md`).

*English summary.* This repository contains the data and Python code that reproduce every table and
figure of the article. Hourly ozone at four SENAMHI stations in Lima (winters 2017–2019) is
log-transformed and split into a deterministic component (trend and year, weekday and hour dummies,
OLS) and a stochastic residual modelled with ELM, MLP, additive non-parametric AR, random forest,
histogram gradient boosting, GRU, LSTM, ARIMA and SARIMA. Evaluation uses rolling-origin validation,
persistence baselines with the same information set, forecast horizons of 1–48 h, Diebold–Mariano
tests (HAC variance, Holm correction), and cross-winter and cross-station generalization. Run
`python run_all.py`.

## Contenido

```
data/                 series horarias de las 4 estaciones (ver data/README.md)
config.yaml           todas las decisiones metodológicas (modelos, rejillas, validación, horizontes)
run_all.py            punto de entrada
ozono/                código del pipeline
  data.py             carga y verificación de los datos
  decomposition.py    componente determinística (MCO)
  features.py         rezagos, covariables y orden p por BIC
  models.py           ELM, MLP, NPAR, RF, HGB, GRU y LSTM
  arima.py            ARIMA y SARIMA: selección, ajuste, pronóstico y diagnóstico
  forecast.py         validación con origen móvil, horizontes y variante meteorológica
  generalization.py   generalización entre inviernos y entre estaciones
  metrics.py          MAE, RMSE, SMAPE, MASE, R², MAPE; Diebold–Mariano (HLN/HAC) y Holm
  diagnostics.py      descriptivos, ADF/KPSS, ACF/PACF
  report.py, report_extra.py   tablas (CSV y LaTeX), figuras y textos
outputs/              resultados publicados en el artículo
  tables/  figures/  texto/    tablas, figuras y párrafos de resultados
  predictions/        todas las predicciones (permite rehacer tablas sin reentrenar)
  verificaciones.md   chequeos automáticos de coherencia
  entorno.json        versiones de Python y librerías usadas
docs/mapa_manuscrito.md   correspondencia entre archivos y tablas/figuras del artículo
```

## Instalación

Requiere Python 3.10 o superior (los resultados se generaron con Python 3.13).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

```bash
python run_all.py --report-only   # rehace tablas y figuras desde outputs/predictions (segundos)
python run_all.py --fast          # prueba rápida con rejillas mínimas (salida en outputs_fast/)
python run_all.py                 # corrida completa: reentrena todos los modelos (~12 min, 4 núcleos)
```

La corrida completa sobrescribe `outputs/`. Con las versiones de `requirements.txt` reproduce las
cifras publicadas; con otras versiones de las librerías pueden aparecer diferencias en los últimos
decimales de los modelos con componentes aleatorios.

## Diseño metodológico

1. **Transformación:** logaritmo natural; todas las métricas se calculan en µg/m³ tras retransformar.
2. **Componente determinística (MCO):** intercepto, tendencia en tiempo real, dummies de año (2018,
   2019; referencia 2017), día de la semana (referencia domingo) y hora (referencia 0 h): 33 parámetros.
3. **Componente estocástica:** ELM (capa oculta sigmoide, pesos U(−1, 1), solución analítica,
   20 semillas), MLP (L-BFGS), NPAR (splines cúbicas aditivas), RF, HGB (gradient boosting por
   histogramas de scikit-learn), GRU y LSTM (PyTorch) sobre los p rezagos del residuo (p por BIC);
   ARIMA(p,0,q) por AIC y SARIMA(1,0,1)(1,0,1)₂₄.
4. **Validación:** últimas 1368 h divididas en 5 ventanas (274, 274, 274, 273, 273 h), con reajuste de
   todos los modelos en cada origen (ventana expansiva). Hiperparámetros elegidos en un hold-out
   temporal de 540 h. Las líneas base (Naive, Naive estacional de 24 y 168 h) usan la misma
   información que los modelos.
5. **Horizontes:** pronóstico directo a 1, 6, 12, 24 y 48 h.
6. **Pruebas:** Diebold–Mariano con pérdida absoluta, versiones HLN y HAC (Newey–West, 24 rezagos) y
   ajuste de Holm.
7. **Generalización:** entrenamiento con inviernos previos y pronóstico de un invierno completo no visto;
   transferencia del modelo entre estaciones y modelo combinado de las cuatro estaciones.

## Datos

Los datos provienen del SENAMHI (<https://www.senamhi.gob.pe/site/descarga-datos/>). Consulte
`data/README.md` para la descripción de las variables, el procesamiento y las condiciones de uso.

## Licencia y cita

El código se distribuye bajo licencia MIT (`LICENSE`). Para citar el código use `CITATION.cff` o el DOI
de Zenodo de la versión archivada.
