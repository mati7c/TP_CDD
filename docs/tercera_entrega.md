# Tercera entrega: modelado reproducible

Se utilizan el CSV con one-hot y la trazabilidad del ETL de segunda entrega. Los originales
no se modifican. Los experimentos comparan tres técnicas: referencia simple, Random Forest
y XGBoost. Se entrenan tres regresiones por técnica (CO, NO2 y PM10), y un clasificador
diario conjunto por técnica. Cada modelo comparte información de las tres estaciones.

## Ejecutar

Desde la raíz del proyecto, con Python 3.12 y un entorno .venv:

    .\.venv\Scripts\python.exe -m pip install -r requirements-modeling.txt
    .\run.ps1 model

La configuración de fechas, cobertura e hiperparámetros está en config/modeling.json.
Para regenerar las figuras, el notebook ejecutado y el HTML desde las métricas:

    .\run.ps1 model-report

El notebook también se puede abrir en VS Code, seleccionando el intérprete .venv.
Sus celdas leen los resultados persistidos: ejecutarlas no repite el entrenamiento.
El entrenamiento completo se reproduce con el comando model.

## Objetivos y entradas

La regresión estima la concentración publicada en cada estación-hora. Entradas:
meteorología ERA5, año, calendario y one-hot de estación, día, mes y hora. Se excluyen
las tres concentraciones de las entradas; una de ellas se elige como salida. Las fechas
sólo identifican registros y definen las particiones.

El modelo diario usa estación, año, calendario, día de semana y mes; para cada variable
meteorológica se calculan media, mínimo y máximo, y se suma la precipitación. No entran
las concentraciones, categorías observadas, cobertura ni conteos de sensores como
predictores.

Se conserva la convención temporal del ETL: HORA 24 se representa como 00 del día
siguiente; la fecha de medición original identifica el día de la jornada y sus
variables de calendario. La API futura deberá construir las entradas con esa convención.

## Límites y objetivos faltantes

Se reutilizan los máximos de plausibilidad definidos en transform.py y en el notebook
de segunda entrega: CO <= 10 ppm, NO2 <= 400 ppb y PM10 <= 600 µg/m³; se excluyen
también negativos y PM10 igual a cero. Son reglas del proyecto, no máximos universales
físicos ni umbrales de seguridad sanitaria.

El original contiene 321 mediciones por encima de esos máximos: CO Centenario 103,
CO Córdoba 96 y CO La Boca 122. Corresponden a 146 filas originales, ya que una fila
puede contener anomalías en más de una estación. NO2 y PM10 no exceden sus máximos.
El CSV con one-hot ya no tiene esas concentraciones.

Se recupera la columna de estado de cada contaminante desde dataset_objetivo.csv,
verificando la correspondencia uno a uno y la igualdad de concentraciones. Se admiten
objetivos medidos y censurados (LD/2, criterio del ETL). Los interpolados se excluyen
del ajuste y de la evaluación: no se reintroducen sustitutos de mediciones inválidas
ni se presenta una interpolación como observación real. La exclusión se aplica a cada
objetivo, sin eliminar datos útiles de otro contaminante.

## Etiquetas EPA adaptadas

Fuente: [EPA, Technical Assistance Document for the Reporting of Daily Air Quality](https://document.airnow.gov/technical-assistance-document-for-the-reporting-of-daily-air-quailty.pdf).

| Categoría del proyecto | CO ppm | NO2 ppb | PM10 µg/m³ |
|---|---|---|---|
| Buena | 0–4,4 | 0–53 | 0–54 |
| Moderada | 4,5–9,4 | 54–100 | 55–154 |
| Desfavorable, agrupada | >=9,5 | >=101 | >=155 |

Se trunca CO a un decimal y NO2/PM10 a enteros antes de aplicar los cortes. Las clases
EPA de AQI >=101 se agrupan como Desfavorable; no se usa el término Dañina para afirmar
el mismo riesgo en toda la población.

Se usan las concentraciones tal como fueron publicadas. El [catálogo oficial](https://data.buenosaires.gob.ar/es_AR/dataset/calidad-aire/resource/juqdkmgo-294-resource)
define CO como promedio móvil de 8 horas, NO2 como promedio de una hora y PM10 como
promedio móvil de 24 horas. No se recalculan esas ventanas.

La etiqueta diaria es la peor clase entre los registros disponibles de los tres
contaminantes durante el día de medición. Se exigen 18 horas válidas de cada uno y
24 filas con meteorología. Es una regla académica explícita, no el AQI diario oficial
de EPA, no una clasificación de OMS y no un dictamen de cumplimiento legal de CABA.
No se etiqueta como Buena a una jornada con cobertura insuficiente. El alcance se
limita a los contaminantes presentes; no se incluyen PM2.5, O3 ni SO2.

## Validación y parámetros

Entrenamiento anterior a 2024, validación en 2024 y prueba desde 2025. Se purgan dos
días antes de cada corte para separar ventanas vecinas; los cortes son comunes para
todas las estaciones. Los hiperparámetros y la técnica ganadora se eligen por
validación, nunca por la prueba.

La referencia de regresión usa la mediana de entrenamiento por estación. La de
clasificación predice la clase más frecuente de entrenamiento.
Random Forest y XGBoost usan los valores de config/modeling.json. Se comparan dos
configuraciones XGBoost. La selección usa MAE en regresión y F1 macro en clasificación.
La clasificación usa pesos para las clases únicamente en entrenamiento; no se
balancean ni se remuestrean validación y prueba. Semilla fija 42.

## Métricas

Regresión: MAE, RMSE, R² y MAE para valores altos, definidos por el percentil 90
del entrenamiento. Se informan por contaminante y estación, en unidades originales.
Las predicciones negativas se limitan a cero y se registra cuántas necesitaron
esa corrección. No se recortan ni se eliminan errores grandes de prueba.

Clasificación: matriz de confusión (filas reales, columnas predichas), accuracy,
precision, recall, F1-score y FPR. Para una clase contra el resto:

    FPR = FP / (FP + TN)

El FPR no es FP / total de predicciones de esa clase; esa segunda cantidad
está relacionada con precision. El detalle conserva TP, FP, FN, TN y soporte.

F1 macro promedia las tres etiquetas fijas con cero para clases no evaluables.
Se informa además F1 macro_supported, calculado sólo con clases de soporte positivo,
balanced accuracy y F1 ponderado. Una clase sin soporte tiene recall no evaluable
y queda vacía en el detalle. Una clase que tiene casos reales y ningún acierto
tiene F1 y recall cero.

La ausencia de Desfavorable en un período no demuestra capacidad de detectarla.
Accuracy y FPR pueden resultar favorables por el desbalance de clases, por lo que
siempre deben leerse junto con soporte, recall y la matriz.

## Archivos y límites de interpretación

- notebooks/tercera_entrega_modelado.ipynb y .html: notebook ejecutado y vista exportada.
- outputs/tercera_entrega/reporte.json: configuración, parámetros, selección y versiones.
- auditoria.json: hashes de fuentes y exclusiones.
- metricas_regresion.csv, metricas_clasificacion.csv y metricas_por_clase.csv.
- matrices_confusion.json y figuras/: matrices y gráficos.
- predicciones_CO.csv, predicciones_NO2.csv, predicciones_PM10.csv y predicciones_diarias.csv.
- dataset_diario_clasificacion.csv: vista derivada para la tarea diaria, sin modificar el CSV objetivo.
- models/: modelos y esquemas de entrada, disponibles localmente y regenerables.

Las importancias expresan asociaciones usadas por los modelos, no causalidad.
ERA5 es reanálisis histórico. El experimento mide capacidad de estimar concentraciones
condicionadas a esa meteorología y no valida por sí solo un pronóstico operativo
a 24/48 horas. La futura evaluación con pronósticos emitidos deberá medir ese error
adicional. No se desarrolla todavía frontend ni backend.
