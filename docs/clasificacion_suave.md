# XGBoost con ponderación suave para los tres contaminantes

Se entrena un clasificador horario por contaminante, compartido entre las tres
estaciones. Se conservan la regresión previa, el piloto NO₂ y los CSV de segunda
entrega. Esta experiencia no produce una categoría general de CABA.

## Reproducir

```powershell
.\run.ps1 classify-soft
```

Instalar primero las dependencias de `requirements-modeling.txt`. La configuración
queda en `config/clasificacion_suave.json`; admite `--model-config` y `--output-dir`.
El comando no descarga meteorología ni consume cuota de API.

Se aplica a CO y PM10 la configuración de XGBoost elegida por validación en el piloto
NO₂: 300 iteraciones, profundidad 4, tasa de aprendizaje 0.05, submuestreo y columnas
0.8, regularización L2 de 5 y método hist. La ponderación es fija para los tres:
frecuencia inversa elevada a **0.5**, normalizada a media uno con entrenamiento.
No se prueban otras ponderaciones ni se afirma que sea la mejor para CO o PM10.

El clasificador de NO₂ se reutiliza si coinciden datos, entradas, escala, semilla y
parámetros; se comprueba que reproduce las etiquetas del piloto. En una instalación
sin el binario local del piloto, se entrena nuevamente la misma configuración fija.

## Datos y evaluación

- Entradas: las mismas 59 variables meteorológicas, temporales y de estación.
  Las tres concentraciones están excluidas de los predictores.
- Etiquetas: CO con seis categorías de India; NO₂ y PM10 con diez niveles británicos.
  Se conservan las conversiones y reglas documentadas en `comparacion_indices.md`.
- Objetivos: sólo medidos/censurados válidos. Se excluyen interpolaciones, faltantes
  y valores fuera de plausibilidad. Se mantiene la trazabilidad del CSV original.
- Entrenamiento anterior a 2024, validación en 2024 y prueba desde 2025, con la misma
  purga de dos días. Los pesos y clases aprendidas se calculan sólo en entrenamiento.
- Comparaciones: categoría mayoritaria del entrenamiento y categoría obtenida del
  regresor seleccionado previamente por validación: XGBoost para CO/NO₂ y referencia
  de mediana por estación para PM10.

Se informan accuracy, precision, recall, F1, FPR, balanced accuracy, F1 ponderado y
error ordinal, además de matrices y detalle TP/FP/FN/TN. Los promedios macro fijos
usan todas las clases declaradas, con cero para las no evaluables. Las variantes
`macro_supported` usan sólo clases reales presentes. Comparar los F1 entre
contaminantes exige considerar sus escalas, cobertura y número de categorías.

El FPR es FP/(FP+TN) de cada categoría contra el resto; su promedio no es la fracción
de predicciones que resultaron equivocadas. Se mantienen las cuatro bandas
británicas como evaluación adicional, separadas de los diez niveles.

CO sólo tiene tres de sus seis categorías en test; NO₂ sólo tiene niveles 1–3 de
contaminación baja. PM10 contiene todos los niveles, con pocos ejemplos elevados.
Las clases no aprendidas conservan probabilidad cero. No se concluye detección
de episodios que no existen en prueba ni se crean ejemplos artificiales.

Los test ya fueron inspeccionados durante las experiencias anteriores: esta es una
comparación exploratoria, sin selección nueva por test. Se requiere confirmación
independiente y evaluar meteorología pronosticada antes del uso operativo a 24/48 h.

## Archivos para revisar y reutilizar

En `outputs/clasificacion_suave`:

- `resumen_metricas.md` y `.html`: método, interpretación, tablas y figuras.
- `metricas_resumen_test.csv`: todas las métricas de los tres XGBoost suaves,
  agregadas entre estaciones, con soporte y cobertura de categorías.
- `metricas_clasificacion.csv`: validación y prueba; modelos de referencia y nuevos;
  resultados por estación y resolución de la escala.
- `metricas_por_clase.csv`, `matrices_confusion.json`, `distribucion_categorias.csv`.
- `predicciones_CO.csv`, `predicciones_NO2.csv`, `predicciones_PM10.csv`: etiquetas
  y probabilidades de cada categoría, sin calibración adicional.
- `modelos/*_XGBoost_suave.ubj` y sus `.meta.json`: los tres modelos entrenados en
  formato nativo XGBoost, con orden de columnas, clases, escala y parámetros.
  Se incluyen en Git por su tamaño reducido. No requieren cargar un archivo pickle.
- `auditoria.json` y `reporte.json`: versiones, configuración, procedencia y hashes.

Al cargar el modelo, se deben ordenar las entradas según `features` del JSON.
La columna de probabilidad `i` del modelo corresponde a `original_classes[i]`,
no necesariamente al código `i` de DAQI: existen clases ausentes en entrenamiento.
El helper `aligned_probabilities` devuelve probabilidades alineadas a la escala
completa. En los CSV, los códigos parten de cero; el código cero británico es nivel 1.

Las convenciones temporales y las unidades de la segunda entrega siguen vigentes.
