# Datos

Series horarias de cuatro estaciones de monitoreo de la calidad del aire de Lima Metropolitana,
inviernos 2017, 2018 y 2019 (21 de junio al 22 de septiembre de cada año).

| Archivo | Estación | Registros |
|---|---|---|
| `ozono_ate.csv` | ATE | 6768 |
| `ozono_cdm.csv` | Campo de Marte (CDM) | 6768 |
| `ozono_sb.csv`  | San Borja (SB) | 6768 |
| `ozono_sta.csv` | Santa Anita (STA) | 6768 |

6768 = 3 inviernos × 94 días × 24 horas por estación (27 072 registros en total).

## Columnas

| Columna | Descripción | Unidad |
|---|---|---|
| `time` | Fecha y hora local (`dd/mm/aaaa hh:mm`) | — |
| `ozono` | Concentración de ozono troposférico (O₃) | µg/m³ |
| `temp` | Temperatura del aire | °C |
| `humrel` | Humedad relativa | % |
| `velvie` | Velocidad del viento | m/s |
| `dirvie` | Dirección del viento (de donde sopla) | grados |

## Fuente y procesamiento

- **Fuente:** Servicio Nacional de Meteorología e Hidrología del Perú (SENAMHI),
  <https://www.senamhi.gob.pe/site/descarga-datos/>. Datos validados por el SENAMHI.
- Los valores faltantes de ozono (1.68–2.29 % por estación) se imputaron con imputación múltiple por
  ecuaciones encadenadas (MICE), como se describe en el artículo (Tabla 1). Los archivos publicados
  son la versión ya imputada y no contienen horas faltantes ni duplicadas.
- Radiación solar, NOₓ, compuestos orgánicos volátiles y flujo vehicular no estaban disponibles para
  estas estaciones y este período.

## Condiciones de uso

Los datos originales pertenecen al SENAMHI. Se redistribuyen aquí únicamente para permitir la
reproducción de los resultados del artículo. Cualquier otro uso debe citar al SENAMHI como fuente y
respetar sus condiciones de uso.
