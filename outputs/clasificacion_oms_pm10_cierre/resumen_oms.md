# Clasificación diaria con las guías OMS 2021

- NO2: validación eligió XGBoost_rezago. En prueba, F1 macro 0.735 frente a 0.246 de la clase mayoritaria; balanced accuracy 72.7% y accuracy 76.8%. Variables más usadas (SHAP): valor_dia_anterior (25%), wind_speed_10m (10%), anio (9%), estacion (8%).
- PM10: XGBoost con 56 entradas, pesos suaves y una variable nueva: el cierre previo. En prueba detecta 32 de 48 excedencias, con 47 falsas alarmas. Precision de excedencias 40.51%, recall 66.67%, F1 50.39%; accuracy 95.90% y F1 macro 0.7413.
- CO: 0 estación-días superan la guía OMS de 4 mg/m³ en todo el período; con una sola clase no hay problema de clasificación y CO se analiza sólo con la regresión de la entrega.


## Método y alcance


- NO2 conserva el modelo, los resultados, los parámetros y las predicciones originales. No se reentrena.
- PM10 mantiene las 55 entradas originales (incluido el promedio anterior) y agrega únicamente pm_close_lag1: el último registro publicado del día anterior, que ya es un móvil de 24 h. No es una medición instantánea.
- El cierre agrega información de cómo terminó la jornada frente al promedio de todos sus registros. Se seleccionó una versión pequeña mediante validación y controles temporales, sin exigir que toda variable con poca importancia fuera inútil.
- Parámetros de PM10: 200 árboles, profundidad 2, min_child_weight 10, learning_rate 0.05, reg_lambda 20, subsample y colsample_bytree 0.8, método hist, semilla 42. Se restringe la complejidad de los árboles y se conservan los parámetros del experimental elegido.
- Ponderación suave: frecuencia inversa elevada a 0.5 y normalizada a media 1, calculada sólo con entrenamiento. No se duplican filas ni se balancean los datos de validación o prueba.
- Umbral de PM10: 0.497096151114. Se maximizaron detecciones en validación con hasta 14 falsas alarmas. Este presupuesto no garantiza la misma tasa futura y el score no está calibrado como probabilidad sanitaria.
- Entrenamiento anterior a 2024, validación 2024 y prueba desde 2025, con purga de dos días. Se conservan las filas, los filtros de calidad, la cobertura mínima de 18 registros y el redondeo de la entrega original.
- Los controles temporales entrenan antes de 2020/2021/2022, calibran en ese año y evalúan el siguiente. Sólo se reproducen los controles de este modelo final.
- La selección de entradas y umbral no se recalcula con test. El test ya fue consultado en experimentos del proyecto: estos resultados siguen siendo exploratorios y requieren una evaluación futura independiente.
- Para pronosticar el día d se requieren el promedio y el cierre del día d-1 ya publicados. La disponibilidad y latencia del proveedor aún deben verificarse. No permite usar el cierre completo de hoy antes de que hoy termine.
- La meteorología del día objetivo es ERA5 histórico. El desempeño con pronósticos meteorológicos emitidos no se midió. Las predicciones son diarias por estación, no para cualquier punto de CABA.
- PM10 se etiqueta con el promedio diario de sus registros móviles de 24 h: es una aproximación heredada y no equivale exactamente al promedio del día calendario ni a un dictamen de cumplimiento de la OMS.
- TreeSHAP describe asociaciones del modelo y no demuestra causalidad. La interpretación de NO2 mantiene el modelo original; la de PM10 corresponde al modelo reducido.


## Modelos finales en prueba


| pollutant | model | n | accuracy | precision_macro | recall_macro | f1_macro | balanced_accuracy | false_positive_rate_macro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NO2 | XGBoost_rezago | 1664 | 0.7680 | 0.7691 | 0.7274 | 0.7348 | 0.7274 | 0.1515 |
| PM10 | XGBoost_cierre | 1536 | 0.9590 | 0.6970 | 0.8175 | 0.7413 | 0.8175 | 0.1825 |


## Métricas por clase


| pollutant | category | support | precision | recall | f1 | false_positive_rate | tp | fp | fn | tn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NO2 | Cumple guía OMS | 372 | 0.8155 | 0.5108 | 0.6281 | 0.0333 | 190 | 43 | 182 | 1249 |
| NO2 | Entre guía y objetivo intermedio | 970 | 0.7757 | 0.8485 | 0.8104 | 0.3429 | 823 | 238 | 147 | 456 |
| NO2 | Supera objetivo intermedio | 322 | 0.7162 | 0.8230 | 0.7659 | 0.0782 | 265 | 105 | 57 | 1237 |
| PM10 | Cumple guía OMS | 1488 | 0.9890 | 0.9684 | 0.9786 | 0.3333 | 1441 | 16 | 47 | 32 |
| PM10 | Supera guía OMS | 48 | 0.4051 | 0.6667 | 0.5039 | 0.0316 | 32 | 47 | 16 | 1441 |
