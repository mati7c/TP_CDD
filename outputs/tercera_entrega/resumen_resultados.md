# Tercera entrega: resultados e interpretación

Se compararon una referencia simple, Random Forest y XGBoost. Los modelos se eligieron por validación, antes de evaluar la prueba.

Se identificaron 321 valores originales por encima de los máximos de plausibilidad, todos de CO. Ya estaban invalidados en el ETL. No se reintrodujeron valores interpolados como objetivos.

## Regresión de concentraciones

| task | model | mae | rmse | r2 | mae_high |
| --- | --- | --- | --- | --- | --- |
| CO | Referencia | 0.1676 | 0.2426 | 0.1716 | 0.6042 |
| CO | RandomForest | 0.1473 | 0.2015 | 0.4284 | 0.4070 |
| CO | XGBoost | 0.1345 | 0.1875 | 0.5050 | 0.3699 |
| NO2 | Referencia | 8.1325 | 10.5209 | -0.0924 | 24.4998 |
| NO2 | RandomForest | 6.7934 | 8.4912 | 0.2884 | 9.7389 |
| NO2 | XGBoost | 6.1043 | 7.6353 | 0.4246 | 8.6286 |
| PM10 | Referencia | 7.4916 | 10.1136 | 0.0102 | 26.2095 |
| PM10 | RandomForest | 8.4899 | 11.0740 | -0.1867 | 21.1134 |
| PM10 | XGBoost | 7.9442 | 10.1626 | 0.0006 | 20.5624 |

- CO: validación seleccionó XGBoost. MAE de prueba 0.1345; diferencia respecto de la referencia: 19.8% de reducción del MAE.
  El MAE varía entre 0.0957 en la_boca y 0.1702 en cordoba.
- NO2: validación seleccionó XGBoost. MAE de prueba 6.1043; diferencia respecto de la referencia: 24.9% de reducción del MAE.
  El MAE varía entre 5.9583 en centenario y 6.3946 en la_boca.
- PM10: validación seleccionó Referencia. MAE de prueba 7.4916; diferencia respecto de la referencia: 0.0% de reducción del MAE.
  El MAE varía entre 6.7739 en cordoba y 8.6753 en la_boca.

## Clasificación diaria

| model | n | accuracy | precision_macro | recall_macro | f1_macro | f1_macro_supported | false_positive_rate_macro |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Referencia | 1502 | 0.9447 | 0.3149 | 0.3333 | 0.3239 | 0.4858 | 0.3333 |
| RandomForest | 1502 | 0.9108 | 0.3894 | 0.3932 | 0.3912 | 0.5868 | 0.2735 |
| XGBoost | 1502 | 0.8848 | 0.3804 | 0.4105 | 0.3896 | 0.5844 | 0.2562 |

Validación seleccionó XGBoost; accuracy de prueba 0.885 y F1 macro fijo de tres clases 0.390. La referencia obtuvo F1 macro 0.324.

F1 macro_supported y balanced_accuracy promedian sólo las clases con casos reales; el F1 macro principal mantiene las tres etiquetas y asigna cero a las clases no evaluables. El detalle deja recall/precisión no definidos como vacíos, no como éxitos.

La clase Desfavorable tiene 0 casos reales en prueba. No se puede estimar su sensibilidad con este período; los falsos positivos sí se pueden contar. Una accuracy alta no demuestra detección de episodios desfavorables.

| category | support | precision | recall | f1 | false_positive_rate | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Buena | 1419 | 0.9581 | 0.9183 | 0.9377 | 0.6867 | 57 | 116 |
| Moderada | 83 | 0.1831 | 0.3133 | 0.2311 | 0.0817 | 116 | 57 |
| Desfavorable | 0 | no evaluable | no evaluable | no evaluable | 0.0000 | 0 | 0 |

## Conocimiento obtenido y alcance

- La clasificación está muy desbalanceada; la referencia mayoritaria puede tener accuracy alta sin detectar días moderados.
- Los errores de concentraciones altas se analizan por separado con un umbral percentil 90 calculado exclusivamente en entrenamiento.
- Las importancias describen asociaciones útiles para los modelos, no causas de contaminación. La estación y las variables meteorológicas pueden compartir información.
- Las métricas por estación permiten identificar dónde empeora cada modelo; el modelo sigue siendo conjunto para las tres estaciones.
- La evaluación es temporal con dos días de purga entre conjuntos. No se mezclaron estaciones de la misma fecha entre entrenamiento y prueba.
- Las categorías se basan en cortes EPA, aplicados a los registros publicados y resumidos por la peor clase del día. Es una adaptación del proyecto, no el AQI diario oficial.
- El conjunto sólo incluye CO, NO2 y PM10; una etiqueta Buena no describe contaminantes no medidos aquí.
- ERA5 describe meteorología histórica. Estos resultados no validan todavía el error operativo a 24/48 horas con pronósticos meteorológicos.

Fuentes: [EPA, AQI técnico](https://document.airnow.gov/technical-assistance-document-for-the-reporting-of-daily-air-quailty.pdf); [definición de contaminantes de Buenos Aires Data](https://data.buenosaires.gob.ar/es_AR/dataset/calidad-aire/resource/juqdkmgo-294-resource).

En días moderados, XGBoost logra recall 0.313 y precisión 0.183; pierde 57 de 83 días moderados y genera 116 falsas alarmas de esa categoría. Es una primera aproximación con limitaciones para detectar episodios, no un sistema de alerta validado.

En el clasificador elegido, las entradas de mayor importancia agrupada son: mes (23.7%), dia_semana (11.2%), anio (9.9%), estacion (8.0%), relative_humidity_2m (6.2%). Estas importancias son internas al modelo y no identifican causas.