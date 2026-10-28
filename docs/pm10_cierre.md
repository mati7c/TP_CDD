# Tercera entrega OMS: PM10 final con cierre previo

La notebook `notebooks/tercera_entrega_oms_pm10_cierre.ipynb` conserva la estructura
de `tercera_entrega_oms_modificado.ipynb`, los gráficos observados por año, mes y
día de la semana, y el NO₂ original. Actualiza PM10 con el modelo reducido elegido.
También se incluye su HTML ejecutado.

## Ejecutar desde VS Code o PowerShell

Abrir la carpeta del proyecto y usar un entorno con `requirements-modeling.txt`.
Para crear ese entorno por primera vez:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-modeling.txt
```

Para reconstruir la variable, ajustar **sólo PM10**, calibrar su umbral, reproducir
los tres controles temporales y generar los resultados y la notebook:

```powershell
.\run_pm10_cierre.ps1
```

La alternativa equivalente es `python -m aire_caba.pm10_cierre` desde el entorno.
Para regenerar sólo la notebook usando los resultados ya incluidos:

```powershell
.\run_pm10_cierre.ps1 --report-only
```

La notebook, como la anterior, lee resultados persistidos; no reentrena modelos
al ejecutar sus celdas. El código de entrenamiento está en `aire_caba/pm10_cierre.py`
y la configuración está en `config/pm10_cierre.json`.

## Qué se conserva y qué cambia

NO₂ mantiene el archivo nativo, los metadatos, las predicciones, las métricas, las
clases, los parámetros y los cortes temporales originales. No se reentrena. Se
verifican sus archivos y resultados al generar la entrega.

PM10 conserva las **55 entradas originales** y agrega **una variable**:

| Entrada histórica | Significado |
|---|---|
| `valor_dia_anterior` | Promedio de los registros válidos publicados ayer; ya existía. |
| `pm_close_lag1` | Último registro publicado al cierre de ayer; única variable nueva. |

El último registro de PM10 ya es un promedio móvil de 24 horas. Aporta el nivel
con el que terminó la jornada, mientras que el promedio anterior resume todos
los registros móviles publicados durante ella. Se usa exactamente el día
calendario anterior por estación. Si falta su cierre, queda NaN: no se utiliza
el de otro día ni el valor del día objetivo. XGBoost maneja esos faltantes.

## Parámetros y regla de decisión

Se mantienen los parámetros del experimental seleccionado y se reduce su lista
de entradas a 56. No se incluyen los modelos ni los resultados de las variantes
descartadas en esta carpeta.

| Parámetro | Valor |
|---|---:|
| `n_estimators` | 200 |
| `max_depth` | 2 |
| `min_child_weight` | 10 |
| `learning_rate` | 0,05 |
| `reg_lambda` | 20 |
| `subsample` / `colsample_bytree` | 0,8 / 0,8 |
| Método / semilla | hist / 42 |
| Exponente de ponderación | 0,5 |
| Umbral de decisión | 0,49709615111351013 |

Los árboles poco profundos, el mayor peso mínimo de hijo y la penalización L2
restringen la complejidad. Los pesos suaves son la frecuencia inversa elevada a
0,5 y normalizada a media 1; se calculan sólo con entrenamiento y se pasan como
`sample_weight`. No se remuestrea.

El umbral maximiza las detecciones en validación sin superar un presupuesto fijo
de **14 falsas alarmas**. Ese presupuesto proviene del protocolo del proyecto y
no limita las falsas alarmas futuras. La elección de entradas combinó simplicidad,
validación y controles temporales anteriores; no se eligió por el resultado en
test. El test ya se había consultado en el proyecto: la evaluación es exploratoria.

No se atribuye la mejora frente al original únicamente a la variable nueva:
también cambian los parámetros, los pesos y el umbral.

## Datos, etiqueta y cortes temporales

Se utilizan los CSV de la segunda entrega ya incluidos en el repositorio:
`data/exports/dataset_objetivo_listo_modelar.csv` y
`data/exports/dataset_objetivo.csv`. Se verifican sus hashes contra la entrega
original. Se excluyen objetivos interpolados y valores fuera de los límites
plausibles de la transformación. Se requiere al menos 18 registros válidos del
contaminante y 24 filas de predictores diarios, igual que antes.

La etiqueta de PM10 conserva el redondeo al entero y el corte 45 µg/m³:
`Cumple guía OMS` / `Supera guía OMS`. El promedio diario de registros que ya son
móviles de 24 horas es una aproximación heredada; no equivale exactamente al
promedio del día calendario ni constituye un dictamen de cumplimiento sanitario.

- Entrenamiento: anterior a 2024, 10.881 estación-días.
- Validación: 2024, 886 estación-días; 32 excedencias.
- Prueba: desde 2025, 1.536 estación-días; 48 excedencias.
- Se conservan dos días de purga entre bloques.

Los controles adicionales entrenan antes de 2020/2021/2022, calibran en ese año y
evalúan 2021/2022/2023. Sus presupuestos de calibración son 7, 7 y 28 falsas
alarmas, respectivamente. Agregados: 2.278 casos, 150 excedencias, 105 detectadas,
54 falsas alarmas; precision 66,04 %, recall 70,00 %.

## Resultado de PM10 en prueba

| Métrica | Valor |
|---|---:|
| Accuracy | 95,90 % |
| Precision de excedencias | 40,51 % |
| Recall de excedencias | 66,67 % |
| F1 de excedencias | 50,39 % |
| F1 macro | 74,13 % |
| Tasa de falsos positivos | 3,16 % |
| TP / FP / FN / TN | 32 / 47 / 16 / 1.441 |

La matriz usa filas reales y columnas predichas. Precision, recall, F1 y FPR
de excedencias corresponden a la clase positiva. Los promedios macro se muestran
por separado. También se incluyen average precision y ROC-AUC en el CSV.

## Archivos para utilizar el modelo

Todo está en `outputs/clasificacion_oms_pm10_cierre/`:

- `modelos/PM10_XGBoost_cierre.ubj` y `.meta.json`: modelo final y columnas ordenadas,
  parámetros, clases, regla de decisión y restricciones de disponibilidad.
- `modelos/NO2_XGBoost_rezago.ubj` y `.meta.json`: copia idéntica del NO₂ original.
- CSV de métricas, métricas por clase, predicciones y controles de PM10.
- `matrices_confusion.json`, `importancia_shap.csv`, figuras y resumen.
- `auditoria.json`: comprobaciones de datos y conservación de NO₂.

Para inferir desde una tabla diaria que ya tenga las mismas entradas:

```python
import json
from pathlib import Path
from xgboost import XGBClassifier
from aire_caba.pm10_cierre import predict_daily

path = Path("outputs/clasificacion_oms_pm10_cierre/modelos/PM10_XGBoost_cierre.ubj")
metadata = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
model = XGBClassifier()
model.load_model(path)
scores, classes = predict_daily(model, tabla_diaria, metadata)
```

`predict_daily` aplica el umbral guardado. `model.predict()` utiliza su regla
predeterminada y no reproduce necesariamente estas clases. El score no está
calibrado como probabilidad sanitaria y la salida es una clase, no una
concentración numérica.

## Condiciones pendientes para uso operativo

Para pronosticar el día d deben estar publicados el promedio y el cierre de d−1.
No se confirmó una ingesta confiable con esa latencia; el cierre completo de hoy
no existe antes de que hoy termine. Para emitir antes, habría que adaptar y
reevaluar el entrenamiento según los datos realmente disponibles.

La meteorología de entrenamiento y evaluación sigue siendo ERA5 histórica. Falta
medir el efecto de utilizar pronósticos meteorológicos emitidos y evaluar datos
futuros independientes. El modelo corresponde a días por estación monitoreada,
sin garantizar el estado de cualquier ubicación de CABA.
