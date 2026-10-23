# Clasificación diaria con las guías OMS 2021

Continúa la clasificación de la tercera entrega. Se conservan los datos, los cortes
temporales, las entradas diarias y las métricas de los experimentos anteriores; sólo
cambia la definición de la etiqueta y se agrega una selección de parámetros explícita.

## Reproducir

```powershell
.\run.ps1 classify-oms
```

Instalar primero `requirements-modeling.txt`. Configuración: `config/clasificacion_oms.json`
(grilla, ponderaciones y entradas) y `config/indices_oms.json` (cortes y conversiones).
Admite `--model-config` y `--output-dir`. Resultados en `outputs/clasificacion_oms`;
el notebook ejecutado queda en `notebooks/tercera_entrega_oms.ipynb` y su HTML.
Tarda unos 10 minutos (144 ajustes).

## Por qué cambiar la etiqueta

Con los cortes EPA, el 94% de los estación-días de prueba son *Buena*, no hay ningún
*Desfavorable* y la clase mayoritaria ya alcanza 94% de accuracy. El clasificador no
tiene episodios que aprender ni evaluar. Las escalas de India y Reino Unido mejoraron
algo, pero mezclan estándares por contaminante y se probaron después de mirar la prueba.

Las guías de la OMS 2021 son el estándar sanitario internacional vigente. Se basan en
efectos sobre la salud a exposiciones bajas y se definen sobre **promedios de 24 h**.
La pregunta del modelo pasa a ser: *¿el aire de esta estación cumple hoy la guía OMS?*

| Contaminante | Clases | Cortes (24 h) | Prueba (estación-días) |
|---|---|---|---|
| NO₂ | Cumple / Entre guía y objetivo intermedio / Supera objetivo intermedio | 25 y 50 µg/m³ (guía y OI-2) | 372 / 970 / 322 |
| PM10 | Cumple / Supera | 45 µg/m³ (guía) | 1.488 / 48 |
| CO | Cumple / Supera | 4 mg/m³ (guía) | sin excedencias |

CO no supera la guía en ningún estación-día de todo el período, así que no hay un
problema de clasificación y se analiza sólo con la regresión. El objetivo intermedio 1
de NO₂ (120 µg/m³) tiene 58 días en 16 años y se agrupa con la clase superior.

## Etiqueta

- Promedio de los registros publicados del día de medición, con al menos 18 horas
  válidas del contaminante y las 24 filas meteorológicas (la misma regla de la
  clasificación diaria original). Se excluyen los objetivos interpolados.
- Conversión con los factores del experimento de índices alternativos: NO₂
  1 ppb = 1,9125 µg/m³; CO 1 ppm = 1,15 mg/m³. Redondeo al entero (NO₂, PM10) o a un
  decimal (CO) antes de comparar; un valor igual a la guía la cumple.
- NO₂ se publica como promedio horario: el promedio diario es la media de 24 h.
  PM10 ya se publica como móvil de 24 h, por lo que su promedio diario es una
  aproximación. No es un dictamen de cumplimiento normativo.

## Entradas

- `meteo`: las mismas entradas de la clasificación diaria original (estación, año,
  calendario, día de semana, mes, media/mínimo/máximo de cada variable meteorológica
  y precipitación acumulada).
- `rezago`: lo anterior más el promedio del día anterior del mismo contaminante en la
  estación (vacío si ese día no tiene cobertura). Resuelve el pendiente de la segunda
  entrega. Es información disponible al estimar el día siguiente; no usa el objetivo.

## Algoritmos y justificación de parámetros

Se comparan una referencia (clase mayoritaria de entrenamiento), Random Forest y
XGBoost. Cada combinación se ajusta con tres ponderaciones de clase y se elige por
**F1 macro en validación (2024)**. La elección se guarda en
`outputs/clasificacion_oms/<contaminante>/seleccion_validacion.json` antes de calcular
la prueba (2025–2026).

| Parámetro | Valores | Justificación |
|---|---|---|
| RF `n_estimators` | 300 | Con más árboles el error de un bosque se estabiliza; aumentar sólo suma tiempo. |
| RF `max_features` | 0,8 | Igual que el modelado original: decorrelaciona árboles sin perder variables meteorológicas clave. |
| RF `max_depth` | 8, 12, sin límite | Controla la complejidad con ~11.000 estación-días de entrenamiento. |
| RF `min_samples_leaf` | 5, 20 | Hojas más grandes suavizan probabilidades y evitan memorizar días aislados. |
| XGB `learning_rate` | 0,05 | Aprendizaje lento y estable, como en el modelado original. |
| XGB `n_estimators` | 300, 600 | Compensa la tasa baja; se compara si más iteraciones ayudan o sobreajustan. |
| XGB `max_depth` | 3, 4, 6 | Orden de las interacciones permitidas (por ejemplo viento × estación × mes). |
| XGB `subsample`, `colsample_bytree`, `reg_lambda` | 0,8; 0,8; 5 | Regularización fija heredada del modelado original. |
| Exponente de pesos | 0; 0,5; 1 | Sin pesos, suave o balanceado. Los pesos se calculan sólo con entrenamiento; no se remuestrea. |

Resultado de la grilla (mejor F1 macro de validación por configuración): en NO₂, XGBoost
mejora con más profundidad e iteraciones (0,755 → 0,776). En PM10, XGBoost con
profundidad 6 y 300 árboles es el mejor (0,773), pero con 600 árboles cae a 0,679,
señal de sobreajuste con pocos días de excedencia. Con rezago, Random Forest queda por
debajo en ambos contaminantes; sin rezago, en NO₂ supera a XGBoost (0,715 contra 0,694). El rezago mejora a todas las familias: NO₂ pasa de 0,715 a 0,776 y PM10
de 0,604 a 0,773.

## Resultados en prueba (2025–2026)

| Contaminante | Modelo elegido | Accuracy | F1 macro | Balanced accuracy | F1 macro de la mayoría |
|---|---|---|---|---|---|
| NO₂ | XGBoost · rezago (prof. 6, 600 árboles, pesos 0,5) | 76,8% | 0,735 | 72,7% | 0,246 |
| PM10 | XGBoost · rezago (prof. 6, 300 árboles, pesos 1) | 94,7% | 0,608 | 62,0% | 0,492 |

- **NO₂**: recall de 82% en *Supera objetivo intermedio* y 85% en la clase central. La
  clase *Cumple* se confunde con la central (recall 51%, precisión 82%). Casi no hay
  errores de dos clases: sólo 1 día *Cumple* se predijo como *Supera*. El modelo
  ordena bien los niveles.
- **PM10**: la accuracy alta se explica por el desbalance (la mayoría ya tiene 96,9%).
  Detecta 13 de 48 excedencias (recall 27%, precisión 22%). En validación el F1 era
  0,77: la prueba muestra que los episodios de PM10 son más difíciles de generalizar.
  Es una alerta de baja sensibilidad, no un sistema de alerta validado.
- Por estación, NO₂ funciona mejor en La Boca (F1 0,71) y peor en Centenario (0,61).

## Interpretación (SHAP)

TreeSHAP nativo de XGBoost descompone cada predicción de prueba en contribuciones de
las variables (log-odds), agrupando columnas one-hot y resúmenes diarios.

- **Persistencia**: el valor del día anterior es la variable más usada (25% en NO₂, 33%
  en PM10). La contaminación depende de situaciones que duran varios días.
- **Viento**: es la variable meteorológica principal en NO₂. La contribución a
  *Supera objetivo intermedio* cae de forma casi lineal entre 1,5 y 7 m/s, en línea con
  la correlación de la segunda entrega (ρ ≈ −0,29). En PM10, el viento débil (< 2 m/s)
  eleva la probabilidad de excedencia.
- **Humedad**: con más de 85% de humedad media baja la chance de NO₂ alto. En PM10, los
  días secos (< 50%) favorecen la excedencia, lo que es compatible con polvo
  resuspendido.
- **Calendario y estación**: año, estación de medición y fin de semana aparecen en NO₂,
  coherente con el origen vehicular y con los niveles menores sábados y domingos.

Son asociaciones aprendidas por el modelo, no causas.

## Patrones observados

- La mayoría de los días supera la guía OMS de NO₂: en casi todos los años más del 60%
  de los estación-días, con mínimos de 20–40% en algunos años (La Boca 2016, Córdoba
  2014, Centenario 2024). La excedencia es máxima en junio (≈90–95%).
- En PM10, la excedencia bajó de 23–26% en 2011 a 2–5% en los últimos años, con picos
  en 2017–2018 y 2022 en La Boca. Hay estacionalidad invernal (junio–julio).

## Límites

- Promedios diarios de series publicadas (PM10 ya suavizado); no es un dictamen legal.
- ERA5 es meteorología histórica: un pronóstico operativo agregaría el error de la
  previsión meteorológica.
- La prueba de NO₂ es independiente de la elección. El conjunto de prueba de la
  clasificación EPA ya había sido inspeccionado en experimentos anteriores, pero
  esta etiqueta y su grilla se eligieron sólo con validación.

Fuentes: [WHO global air quality guidelines 2021](https://www.who.int/publications/i/item/9789240034228);
[resumen OMS](https://www.who.int/news-room/fact-sheets/detail/ambient-(outdoor)-air-quality-and-health);
[factores DEFRA](https://uk-air.defra.gov.uk/reports/cat06/0502160851_Conversion_Factors_Between_ppb_and.pdf).
