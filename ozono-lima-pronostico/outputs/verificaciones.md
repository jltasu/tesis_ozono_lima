# Verificaciones automáticas

Generado por el pipeline. Use estas frases para que el texto del manuscrito coincida con las tablas.

## Partición
- n = 6768; entrenamiento inicial = 5400; prueba = 1368 h = 5 ventanas de [274, 274, 274, 273, 273] h (suma exacta).
- Hold-out de hiperparámetros = 540 h (10 % del primer entrenamiento).


## Datos
- ATE: faltantes antes de imputar = {'ozono': 0, 'temp': 0, 'humrel': 0, 'velvie': 0, 'dirvie': 0}; covariables disponibles = ['temp', 'humrel', 'velvie', 'dirvie']. El CSV no tiene faltantes: los datos ya vienen imputados y no traen marca de imputación, por lo que la Tabla 1 (horas imputadas) no se puede regenerar desde este archivo.
  - **ALERTA**: la media anual varía 4.1× entre inviernos (2017: 55.7, 2018: 15.8, 2019: 13.6). Discutir posible cambio de régimen o de calibración del analizador.
- CDM: faltantes antes de imputar = {'ozono': 0, 'temp': 0, 'humrel': 0, 'velvie': 0, 'dirvie': 0}; covariables disponibles = ['temp', 'humrel', 'velvie', 'dirvie']. El CSV no tiene faltantes: los datos ya vienen imputados y no traen marca de imputación, por lo que la Tabla 1 (horas imputadas) no se puede regenerar desde este archivo.
  - **ALERTA**: la media anual varía 4.0× entre inviernos (2017: 23.6, 2018: 12.1, 2019: 48.7). Discutir posible cambio de régimen o de calibración del analizador.
- SB: faltantes antes de imputar = {'ozono': 0, 'temp': 0, 'humrel': 0, 'velvie': 0, 'dirvie': 0}; covariables disponibles = ['temp', 'humrel', 'velvie', 'dirvie']. El CSV no tiene faltantes: los datos ya vienen imputados y no traen marca de imputación, por lo que la Tabla 1 (horas imputadas) no se puede regenerar desde este archivo.
- STA: faltantes antes de imputar = {'ozono': 0, 'temp': 0, 'humrel': 0, 'velvie': 0, 'dirvie': 0}; covariables disponibles = ['temp', 'humrel', 'velvie', 'dirvie']. El CSV no tiene faltantes: los datos ya vienen imputados y no traen marca de imputación, por lo que la Tabla 1 (horas imputadas) no se puede regenerar desde este archivo.

## ¿El mejor modelo supera al Naive? (h = 1, prueba completa)
- ATE: mejor univariado = HGB; MAE 3.123 vs Naive 4.472 (MAE rel. 0.698); SMAPE 17.41 vs 21.60; DM p (HAC, Holm) = <0.0001*** → **SÍ, significativo**
- CDM: mejor univariado = RF; MAE 6.099 vs Naive 6.117 (MAE rel. 0.997); SMAPE 20.76 vs 18.51; DM p (HAC, Holm) = 0.9238 → **PARCIAL**
- SB: mejor univariado = MLP; MAE 3.224 vs Naive 3.594 (MAE rel. 0.897); SMAPE 19.35 vs 21.11; DM p (HAC, Holm) = 0.0002*** → **SÍ, significativo**
- STA: mejor univariado = MLP; MAE 3.374 vs Naive 4.057 (MAE rel. 0.832); SMAPE 47.19 vs 53.10; DM p (HAC, Holm) = <0.0001*** → **SÍ, significativo**
- Si alguna estación sale NO, PARCIAL o no significativa, el resumen y las conclusiones no pueden afirmar que ELM/MLP superan a las líneas base en todas las estaciones.

## MAPE
- ATE: 1.2 % de horas de prueba < 2; MAPE Naive = 38.6 %.
- CDM: 2.9 % de horas de prueba < 2; MAPE Naive = 38.5 %.
- SB: 0.1 % de horas de prueba < 2; MAPE Naive = 23.0 %.
- STA: 31.9 % de horas de prueba < 2; MAPE Naive = 83.6 %.
- Atribuya MAPE extremos a valores cercanos a cero solo en estaciones con proporción alta de horas bajo el umbral.

## ARIMA
- ATE: ARIMA(4,0,4); Ljung–Box rechaza ruido blanco en el rezago 24; raíces AR y MA casi se cancelan (sobreparametrización).
- CDM: ARIMA(4,0,4); Ljung–Box rechaza ruido blanco en el rezago 24; raíces AR y MA casi se cancelan (sobreparametrización).
- SB: ARIMA(3,0,4); Ljung–Box rechaza ruido blanco en el rezago 24; raíces AR y MA casi se cancelan (sobreparametrización).
- STA: ARIMA(4,0,3); Ljung–Box rechaza ruido blanco en el rezago 24; raíces AR y MA casi se cancelan (sobreparametrización).

## Estacionalidad remanente (justificación ARIMA vs SARIMA)
- ATE: PACF(24) = -0.077, PACF(168) = 0.007 (banda ±0.027; ACF(24) = 0.446) → queda dependencia estacional directa: justificar la exclusión de SARIMA con un SARIMA de contraste o decir que los rezagos 24 los capturan ELM/MLP/NPAR
- CDM: PACF(24) = 0.048, PACF(168) = -0.008 (banda ±0.027; ACF(24) = 0.504) → queda dependencia estacional directa: justificar la exclusión de SARIMA con un SARIMA de contraste o decir que los rezagos 24 los capturan ELM/MLP/NPAR
- SB: PACF(24) = 0.054, PACF(168) = 0.017 (banda ±0.027; ACF(24) = 0.359) → queda dependencia estacional directa: justificar la exclusión de SARIMA con un SARIMA de contraste o decir que los rezagos 24 los capturan ELM/MLP/NPAR
- STA: PACF(24) = -0.001, PACF(168) = -0.003 (banda ±0.027; ACF(24) = 0.364) → sin estacionalidad remanente directa: SARIMA es redundante

## Meteorología (h = 1)
- ATE ELM: MAE 3.279 → 3.846 (17.3 %), p (HAC) = <0.0001.
- ATE MLP: MAE 3.283 → 3.380 (3.0 %), p (HAC) = 0.0666.
- CDM ELM: MAE 8.422 → 8.216 (-2.4 %), p (HAC) = 0.0800.
- CDM MLP: MAE 6.352 → 6.744 (6.2 %), p (HAC) = 0.0275.
- SB ELM: MAE 3.595 → 3.964 (10.3 %), p (HAC) = <0.0001.
- SB MLP: MAE 3.224 → 3.231 (0.2 %), p (HAC) = 0.8761.
- STA ELM: MAE 4.338 → 4.647 (7.1 %), p (HAC) = 0.0004.
- STA MLP: MAE 3.374 → 3.361 (-0.4 %), p (HAC) = 0.8799.

## Consistencia de signos en la prueba DM
- Convención: DM negativo ⇒ el primer modelo de la comparación es más preciso. En T08 el primer modelo es siempre el mejor por MAE, por lo que un DM positivo y significativo sería una incoherencia.

- Incoherencias detectadas: ninguna.
