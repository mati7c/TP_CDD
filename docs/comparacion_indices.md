# Comparación de índices por contaminante

Se prueba la selección acordada: NO₂ y PM10 con DAQI británico y CO con los
rangos de India/CPCB. Se recategorizan los objetivos y las concentraciones predichas
de la tercera entrega. No se vuelve a entrenar ni se modifican el dataset, el
experimento EPA o su clasificador diario.

## Ejecutar y revisar

```powershell
.\run.ps1 compare-indices
```

Requiere las dependencias de `requirements-modeling.txt` y las predicciones existentes
en `outputs/tercera_entrega`. No descarga meteorología ni consume cuota de API.
Opcionalmente admite `--output-dir CARPETA`; debe ser distinta de la carpeta original.

Resultados en `outputs/comparacion_indices`:

- `comparacion_indices.html` y `resumen_comparacion.md`: método, resultados y gráficos.
- `metricas_clasificacion.csv`: accuracy, precision, recall, F1, FPR, balanced accuracy,
  cobertura de clases y error ordinal; por modelo, estación, escala y período.
- `metricas_por_clase.csv` y `matrices_confusion.json`: soporte, TP, FP, FN, TN y métricas
  para cada categoría, incluidas las que no tienen ejemplos.
- `distribucion_categorias.csv`: entrenamiento, validación, prueba y días de purga.
- `predicciones_indices_CO.csv`, `predicciones_indices_NO2.csv` y
  `predicciones_indices_PM10.csv`: concentraciones y etiquetas antiguas/nuevas.
  Los códigos se numeran desde cero; el código cero de DAQI representa el nivel 1.
- `reporte.json`: fuentes, límites, conversiones, reglas, modelos previos seleccionados
  y hashes de procedencia.
- `sensibilidad_conversion_CO.csv`: cambios de etiqueta al usar el factor de
  referencia a 20 °C en lugar de la convención principal a 25 °C.

## Escalas, unidades y precisión

La configuración versionada está en `config/indices_alternativos.json`.

- **NO₂:** DAQI con diez niveles, sobre el promedio publicado de una hora. Se convierte
  ppb a µg/m³ multiplicando por **1.9125**, factor DEFRA a 20 °C y 1013 mb. Se redondea
  una sola vez al entero antes de comparar con los límites.
- **PM10:** DAQI con diez niveles, sobre el promedio móvil publicado de 24 horas,
  sin volver a promediarlo. Se redondea una sola vez al entero. También se informan
  las cuatro bandas oficiales DAQI: baja, moderada, alta y muy alta. Los niveles
  1–3 siguen siendo contaminación baja; no se renombran como riesgo moderado.
- **CO:** seis categorías CPCB sobre el promedio publicado de ocho horas. Se adopta
  **1 ppm = 1.15 mg/m³**, factor de referencia OMS a 25 °C y 1013 mb recogido por DEFRA.
  Es una convención declarada de esta comparación, no una condición que se haya
  verificado en el sensor CABA ni una exigencia de CPCB. Se redondea a una decimal
  con mitades hacia arriba para resolver las separaciones de la tabla CPCB.
  Esa regla de redondeo es del experimento: el documento CPCB consultado no la especifica.

Los rangos de CO en mg/m³ son: buena hasta 1.0; satisfactoria hasta 2.0; moderada
hasta 10.0; mala hasta 17.0; muy mala hasta 34.0; severa por encima. Los límites
superiores se consideran incluidos después del redondeo declarado.

No se usa la temperatura o presión ERA5 para modificar la concentración medida.
No se imputa ni etiqueta un objetivo faltante. Se conserva el filtro de plausibilidad
y la exclusión de objetivos interpolados del experimento original. No se certifica
equivalencia del método instrumental PM10 con el método de referencia europeo.
Esta es una comparación académica de etiquetas, no una certificación de cumplimiento.

## Cómo interpretar la prueba

Los modelos siguen elegidos por el MAE de validación de la regresión original.
La referencia adicional `Mayoria_train` predice la categoría más frecuente sólo
en entrenamiento, sin mirar la distribución de test. Se verifica además que
las etiquetas EPA reproducen exactamente las métricas originales.

`precision_macro`, `recall_macro` y `f1_macro` promedian todas las clases declaradas
y usan cero cuando no se puede evaluar una clase. Las variantes `macro_supported`
promedian sólo categorías reales presentes: se muestran ambas y el número de
categorías con ejemplos. `balanced_accuracy` es el promedio del recall de clases
reales presentes. Una categoría no predicha puede tener FPR cero y recall cero;
FPR bajo por sí solo no demuestra detección.

Cambiar el objetivo no mejora por sí mismo las concentraciones: MAE, RMSE y R²
quedan iguales. Más clases pueden revelar variaciones antes ocultas, pero también
errores y desequilibrios. No se comparan los F1 entre distintas escalas como si
midieran la misma tarea. Como test ya fue inspeccionado, futuras elecciones
basadas en esta comparación requieren otra evaluación independiente.

Las escalas se mantienen separadas. No se calcula una categoría general de estación
o CABA, ni se reemplaza la etiqueta diaria EPA, porque no se acordó una equivalencia
entre niveles DAQI y categorías India.

## Fuentes verificadas el 18/10/2026

- [Límites DAQI, DEFRA, actualizados el 13/04/2026](https://www.gov.uk/government/publications/health-effects-of-air-pollution/pollutant-concentrations-for-the-daily-air-quality-index-daqi).
- [Implementación DAQI: promedios y redondeo único, DEFRA, páginas 2–3](https://uk-air.defra.gov.uk/reports/cat14/1304251155_Update_on_Implementation_of_the_DAQI_April_2013_Final.pdf).
- [Factores de conversión de gases, DEFRA](https://uk-air.defra.gov.uk/reports/cat06/0502160851_Conversion_Factors_Between_ppb_and.pdf).
- [National Air Quality Index, CPCB, tabla 3.11, página PDF 37](https://www.cpcb.gov.in/displaypdf.php?id=bmF0aW9uYWwtYWlyLXF1YWxpdHktaW5kZXgvRklOQUwtUkVQT1JUX0FRSV8ucGRm).
