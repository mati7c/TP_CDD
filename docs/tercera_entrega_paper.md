# Tercera entrega — texto para el paper

Borrador de las secciones de modelado para la plantilla de la cátedra. Las cifras
provienen de `outputs/tercera_entrega` (regresión) y `outputs/clasificacion_oms`
(clasificación). Figuras sugeridas entre corchetes.

## Metodología de modelado

**Problema.** Se plantean dos tareas sobre el dataset objetivo de la segunda entrega
(398.961 registros estación-hora de Centenario, Córdoba y La Boca, 2009–2026):
(i) estimar la concentración horaria de CO, NO₂ y PM10 a partir de meteorología ERA5 y
calendario (regresión); (ii) determinar si en una estación y un día la calidad del aire
cumple la guía sanitaria de la OMS (clasificación).

**Partición temporal.** Para evitar fuga de información entre días consecutivos, los
datos se dividen por fecha y no al azar: entrenamiento hasta 2023, validación en 2024 y
prueba desde 2025, con dos días de purga entre períodos. Los parámetros y la técnica se
eligen sólo con validación; la prueba se usa una única vez para informar el desempeño.

**Algoritmos.** En ambas tareas se comparan una referencia simple (mediana por estación
o clase mayoritaria), Random Forest y XGBoost. Random Forest promedia árboles
independientes y es robusto al ruido; XGBoost construye árboles secuenciales que
corrigen el error anterior y suele captar mejor interacciones no lineales, como la que
existe entre el viento, la hora y la estación del año. Se descartan los objetivos
interpolados y no se generan muestras sintéticas.

**Etiqueta de clasificación.** Una primera versión usó los cortes EPA (Buena /
Moderada / Desfavorable). En prueba, el 94% de los días resultó *Buena* y no hubo ningún
*Desfavorable*: la clase mayoritaria lograba 94% de accuracy sin detectar episodios. Se
adoptaron entonces las guías de calidad del aire de la OMS (2021), basadas en efectos
sobre la salud y definidas sobre promedios de 24 horas. NO₂ se clasifica en tres
niveles (≤25, 25–50 y >50 µg/m³: guía y objetivo intermedio 2) y PM10 en dos (≤45 y
>45 µg/m³). CO no superó la guía (4 mg/m³) en ningún día del período, por lo que sólo
se modela como regresión. [Figura: distribucion_oms.png]

**Parámetros.** La clasificación se seleccionó con una grilla explícita: Random Forest
con 300 árboles, profundidad 8, 12 o sin límite y mínimo de 5 o 20 casos por hoja;
XGBoost con tasa de aprendizaje 0,05, 300 o 600 árboles y profundidad 3, 4 o 6,
manteniendo submuestreo de filas y columnas de 0,8 y regularización L2 de 5. Cada
combinación se probó sin pesos de clase, con pesos suaves (frecuencia inversa elevada a
0,5) y con pesos balanceados, y con dos conjuntos de entradas: meteorología y calendario,
o éstas más el promedio del día anterior del contaminante. El criterio de selección fue
el F1 macro de validación, que pondera por igual a todas las clases y penaliza ignorar
las minoritarias. [Figura: seleccion_validacion.png]

**Métricas.** Regresión: MAE, RMSE y R², y el MAE en concentraciones altas (sobre el
percentil 90 de entrenamiento). Clasificación: matriz de confusión, accuracy, precision,
recall, F1 por clase y macro, balanced accuracy y tasa de falsos positivos por clase
(FP/(FP+TN)).

## Resultados

**Regresión horaria.** XGBoost fue elegido para CO y NO₂ y redujo el MAE de la
referencia en 20% (CO: 0,13 ppm, R² = 0,51) y 25% (NO₂: 6,10 ppb, R² = 0,42). Para PM10
ningún modelo superó a la mediana por estación (MAE 7,49 µg/m³). Esto es consistente con
que PM10 se publica como promedio móvil de 24 h y depende de fuentes no representadas en
las entradas.

**Clasificación OMS.** En ambos contaminantes, validación eligió XGBoost con rezago.

| | NO₂ (3 clases) | PM10 (2 clases) |
|---|---|---|
| Estación-días de prueba | 1.664 | 1.536 |
| F1 macro (modelo / clase mayoritaria) | **0,735** / 0,246 | **0,608** / 0,492 |
| Balanced accuracy | 72,7% | 62,0% |
| Accuracy (modelo / clase mayoritaria) | 76,8% / 58,3% | 94,7% / 96,9% |

En NO₂ el modelo reconoce el 82% de los días que superan el objetivo intermedio y el 85%
de los días intermedios; sólo 1 día que cumple la guía se clasificó dos niveles arriba.
En PM10 se detectaron 13 de 48 excedencias (recall 27%, precisión 22%). La accuracy de
PM10 es menor que la de la clase mayoritaria, lo que muestra que en datos desbalanceados
esa métrica no mide la detección de episodios. [Figura: matrices_oms.png]

El promedio del día anterior fue la entrada más útil: con ella, el F1 macro de
validación pasó de 0,715 a 0,776 en NO₂ y de 0,604 a 0,773 en PM10.

## Patrones e interpretación

Con TreeSHAP se descompuso cada predicción en la contribución de cada variable.
[Figuras: shap_importancia.png, shap_dependencia.png]

- **Persistencia.** El nivel del día anterior explica el 25–33% de las contribuciones:
  los episodios duran varios días porque dependen de situaciones meteorológicas
  persistentes.
- **Dispersión por viento.** En NO₂, la contribución a la clase más alta disminuye casi
  linealmente entre 1,5 y 7 m/s. Para PM10, los días con viento débil (< 2 m/s) y aire
  seco (humedad < 50%) son los de mayor riesgo. Esto confirma las correlaciones de la
  segunda entrega.
- **Actividad humana.** El año, la estación de medición y el fin de semana influyen en
  NO₂, cuyo origen principal es el tránsito.

En cuanto a las tendencias observadas [Figura: patrones_excedencias.png], en la mayoría
de los años más del 60% de los días supera la guía OMS de NO₂, con un máximo en junio.
Las excedencias de PM10 bajaron de alrededor del 25% de los días en 2011 al 2–5% en los
últimos años, con picos en 2017–2018 y 2022 en La Boca.

## Discusión y conclusiones

- La elección de la etiqueta fue la decisión más importante del modelado. Con los cortes
  EPA, el problema era trivial y la accuracy engañaba. Con la guía OMS, la tarea pasa a
  ser informativa: el modelo de NO₂ triplica el F1 macro de la referencia.
- Con el rezago, XGBoost superó a Random Forest en validación en ambos contaminantes. Sin
  rezago, Random Forest fue mejor en NO₂ (0,715 contra 0,694): la ventaja de XGBoost
  aparece cuando hay una señal fuerte que combinar con la meteorología. En PM10, más
  iteraciones empeoraron la validación (0,773 → 0,679), una señal de sobreajuste con
  pocos episodios.
- Para CABA, el contaminante crítico frente a la referencia sanitaria es el NO₂ y no
  PM10 ni CO. Sus excedencias se pueden anticipar con el nivel del día anterior y el
  viento previsto.
- **Limitaciones.** La meteorología ERA5 es histórica: un pronóstico real sumaría el
  error de la previsión meteorológica. PM10 ya se publica suavizado. Las importancias
  son asociaciones, no causas. Ninguna etiqueta constituye un dictamen normativo.
- **Recomendación.** Usar el clasificador de NO₂ como aviso diario por estación,
  alimentado con el pronóstico de viento y el dato medido del día anterior, y
  validarlo con pronósticos emitidos antes de usarlo operativamente.
