# Comparación ampliada de clasificadores

Se amplían las pruebas de CO y PM10 al mismo diseño del piloto NO₂: Random Forest
y XGBoost, cada uno sin ponderación, con ponderación suave y con ponderación
completa. Se reutiliza la configuración de árboles del piloto; no se busca una
nueva escala ni se cambia el dataset para mejorar métricas.

## Reproducir

```powershell
.\run.ps1 classify-compare
```

La configuración está en `config/comparacion_clasificadores.json`; admite
`--model-config` y `--output-dir`. Requiere las dependencias de modelado y los
resultados anteriores de tercera entrega, clasificación suave y piloto NO₂.
Esos CSV, manifiestos y los modelos XGBoost suaves están incluidos en la rama.

Se cargan los XGBoost suaves nativos de CO/PM10 tras verificar hash, dataset,
orden de entradas, escala, semilla e hiperparámetros. Esa variante se evalúa sin
reentrenarla. Se ajustan las otras cinco variantes por contaminante. NO₂ integra
los seis ensayos y las métricas de su piloto, sin repetir selección usando test.

## Selección y comparación

- Mismas 59 entradas y mismos objetivos medidos/censurados, filtros y cortes de fecha.
- Pesos derivados sólo de entrenamiento. Exponentes 0, 0.5 y 1, normalizados a media uno.
- Por técnica se elige la ponderación con mayor F1 macro en validación; luego se
  elige el clasificador con el mismo criterio. Los empates conservan el primer ensayo.
- Se guarda `seleccion_validacion.json` antes de evaluar test de cada contaminante.
- Se comparan los finalistas con la referencia de mayoría, la regresión previa
  convertida y el XGBoost suave fijo. No se cambia la elección mirando test.
- Se guardan los modelos elegidos por validación para facilitar su revisión y uso.
  Los binarios intermedios de finalistas quedan localmente en las subcarpetas
  `CO/models` y `PM10/models`, ignoradas por Git.

El resumen de XGBoost suave solicitado se conserva aunque alguna otra variante
gane la comparación de validación. Un clasificador elegido no implica que supere
la regresión previa en prueba ni que detecte niveles sin casos evaluables.

## Resultados

En `outputs/comparacion_clasificadores`:

- `resumen_comparacion.md` y `.html`: resumen de XGBoost suave, comparación ampliada,
  métricas de los 18 ensayos en validación, finalistas en test y límites de interpretación.
- `metricas_xgboost_suave_test.csv`: todas las métricas de los tres modelos suaves.
- `metricas_elegidos_test.csv`: resultados de los tres clasificadores elegidos por validación.
- `metricas_clasificacion.csv`, `metricas_por_clase.csv` y `matrices_confusion.json`:
  validación/prueba, estaciones, niveles y bandas.
- `candidatos_validacion.csv`: los 18 ensayos y sus ponderaciones.
- `predicciones_CO.csv` y `predicciones_PM10.csv`: finalistas, referencias y
  probabilidades del elegido. Las predicciones NO₂ se conservan en su piloto anterior.
- `modelos/`: modelos elegidos y metadatos de entradas, clases y parámetros.
- `selecciones_validacion.json`, `auditoria.json` y `reporte.json`: decisiones y trazabilidad.

Macro considera las seis categorías de CO o los diez niveles británicos, asignando
cero a métricas no evaluables. Macro_supported promedia sólo clases reales presentes.
FPR significa FP/(FP+TN) por categoría frente al resto. Se informan soporte y
TP/FP/FN/TN para evitar interpretar accuracy o FPR aisladamente.

La ampliación fue propuesta después de inspeccionar resultados anteriores de test.
Por lo tanto, es exploratoria: requiere confirmación independiente. Las limitaciones
de clases escasas, unidades, promedios publicados y uso de ERA5 histórico siguen
vigentes; ver `clasificacion_suave.md` y `comparacion_indices.md`.
