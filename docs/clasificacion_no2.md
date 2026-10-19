# Piloto: clasificación horaria directa de NO₂

Se entrena un clasificador compartido entre estaciones para predecir directamente
el nivel británico DAQI de NO₂. Se mantienen las 59 entradas del modelo horario,
incluida la estación, sin incorporar concentraciones observadas como predictores.
Los modelos anteriores de concentración y el clasificador diario EPA se conservan.

## Reproducir

```powershell
.\run.ps1 classify-no2
```

Requiere `requirements-modeling.txt`, los CSV de segunda entrega y las predicciones
de regresión existentes en `outputs/tercera_entrega`. La configuración del piloto
está en `config/clasificacion_no2.json`; los cortes temporales se recuperan del
experimento original. Admite `--model-config` y `--output-dir`.

No descarga datos ni consume cuota meteorológica. Los resultados quedan en
`outputs/clasificacion_no2`: informe HTML y Markdown, candidatos de validación,
métricas generales y por clase/estación, matrices, figuras, probabilidades y
predicciones. Los modelos `.joblib` con el orden de entradas, clases y parámetros
se conservan en la subcarpeta local `models/`, ignorada por Git.

## Diseño

- NO₂: mismo objetivo medido/censurado y mismos descartes de plausibilidad e
  interpolaciones que en la tercera entrega. Se convierte de ppb a µg/m³ con
  factor 1.9125 y se redondea una única vez al entero antes de etiquetar.
  Se reutilizan los diez niveles de `config/indices_alternativos.json`.
- Se comparan Random Forest y XGBoost para clasificación. Cada técnica prueba
  tres ponderaciones: sin balancear, raíz de frecuencia inversa y frecuencia
  inversa completa. Se calculan sólo con entrenamiento y se normalizan a media uno.
- Las etiquetas presentes se codifican consecutivamente para entrenar y luego se
  decodifican a DAQI. Los niveles ausentes en entrenamiento conservan probabilidad
  cero, sin asignarles ejemplos ficticios ni desplazar etiquetas.
- Se elige la ponderación y el clasificador con F1 macro de validación. La selección
  se escribe en `seleccion_validacion.json` antes de calcular métricas de test.
- Se compara con la clase mayoritaria del entrenamiento y con el XGBoost de
  regresión anterior convertido a niveles. Se cotejan claves y objetivos para
  asegurar que la comparación utiliza exactamente las mismas filas.
- Entrenamiento anterior a 2024, validación en 2024, prueba desde 2025 y purga de
  dos días antes de cada corte. Se mantiene la distribución natural en evaluación.

Los códigos exportados se numeran desde cero: `label=0` representa el nivel 1
británico. Las probabilidades se identifican con el nivel humano 1–10. Son salidas
del modelo sin calibración adicional; no equivalen a certeza clínica o sanitaria.

## Interpretación

Se informan accuracy, precision, recall, F1, FPR, balanced accuracy y matriz de
confusión. Macro considera los diez niveles fijos; `macro_supported` considera
sólo clases reales presentes. Se conserva el detalle de soporte, TP, FP, FN y TN.
El FPR se calcula por clase contra el resto, FP/(FP+TN).

En validación y prueba sólo existen niveles 1, 2 y 3, todos dentro de contaminación
baja. Los niveles elevados son escasos o ausentes en entrenamiento. No se puede
validar detección de contaminación moderada o alta con este test. El nivel 3 tiene
sólo 19 ejemplos en prueba, por lo que sus métricas son especialmente inestables.

El test ya se había inspeccionado al decidir este experimento: sus resultados
son una comparación exploratoria y futuras decisiones necesitan una confirmación
independiente. Tampoco se mide todavía el error adicional de usar pronósticos
meteorológicos a 24/48 horas, porque las entradas históricas siguen siendo ERA5.

Las fuentes oficiales y conversiones están en [comparacion_indices.md](comparacion_indices.md).
