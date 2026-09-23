# Calendario de feriados y días no laborables

## Ejecutar

```powershell
.\run.ps1 calendar
```

En este equipo las dependencias ya están instaladas en `.venv`. En otro equipo:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-calendar.txt
.\run.ps1 calendar
```

No consulta Open-Meteo ni consume su cuota. El comando reconstruye el período exacto del CSV original de aire y reemplaza estas exportaciones:

- `data/exports/calendario.csv`: una fila por fecha civil, del 01/10/2009 al 24/08/2026, incluidos días sin eventos.
- `data/exports/feriados.csv`: una fila por evento, con fecha efectiva, fecha original, nombre, clasificación, fuente y nivel de verificación. Un día puede contener varios eventos.
- `data/exports/calendario.meta.json`: método, alcance, hashes y conteos.

Resultado inicial: 6.172 días, 327 eventos (299 eventos de feriado nacional en 297 fechas distintas y 28 eventos de día no laborable). Los indicadores pueden coincidir: por ejemplo, Jueves Santo y otro feriado nacional pueden caer el mismo día.

## Datos diarios

`FECHA` tiene formato YYYY-MM-DD. `dia_semana` usa lunes=1 y domingo=7. Los indicadores 0/1 separan fin de semana, feriado nacional, día no laborable y turístico. `presente_en_csv_aire` indica si esa fecha aparece en el CSV, sin afirmar que todas sus mediciones sean válidas. No se calcula un supuesto «día trabajado», porque un día no laborable o un fin de semana no prueba cierre de todas las actividades.

El calendario es independiente de estaciones y horas. No altera `00` ni `24`, ni hace todavía la unión con meteorología. Al construir la tabla final habrá que decidir si asociar el calendario a la fecha etiquetada o al instante representado por `24`.

## Fuente y alcance de la revisión

Es una **reconstrucción retrospectiva**, no una descarga de una API oficial con todo el histórico. Se fija [python-holidays 0.105](https://pypi.org/project/holidays/0.105/) y sus dependencias. Los resultados de la biblioteca antes de ajustes se conservan por año en `data/derived/calendar/holidays_0.105/`. No se guardan bajo `raw`, porque fueron calculados.

La [documentación del proveedor](https://holidays.readthedocs.io/en/latest/auto_gen_docs/argentina/) contiene referencias históricas. Se usan sus reglas de fechas observadas; `fecha_original` procede de la variante sin traslados cuando es posible. Se contrastaron casos históricos y excepciones, no cada fecha del período. `verificacion` distingue los eventos reconstruidos de los ajustes contrastados.

La [API de ArgentinaDatos](https://argentinadatos.com/docs/operations/get-feriados) documenta cobertura desde 2016 y no cubre 2009–2015; no se utilizó como fuente de esta exportación.

### Ajustes sobre la biblioteca

- Jueves Santo se clasifica como día no laborable y se completa antes de 2011 con fecha calculada de Pascua. Fuentes: [Ley 21.329](https://biblioteca.afip.gob.ar/dcp/LEY_C_021329_1976_06_09) y [Ley 27.399](https://www.argentina.gob.ar/normativa/nacional/ley-27399-281835/texto).
- Los puentes de 2018–2019, 2025 y 2026 se clasifican como días no laborables. Fuentes: [Decreto 923/2017](https://www.argentina.gob.ar/normativa/nacional/decreto-923-2017-287145/texto), [Decreto 1027/2024](https://www.argentina.gob.ar/normativa/nacional/decreto-1027-2024-406417/texto) y [Resolución 164/2025](https://www.argentina.gob.ar/normativa/nacional/norma-421799/texto).
- Se incorpora el 02/09/2022: [Decreto 573/2022](https://www.argentina.gob.ar/normativa/nacional/decreto-573-2022-370795/texto).
- El feriado del 12/10/2025 se mueve al 10/10/2025: [Resolución 139/2025](https://www.argentina.gob.ar/normativa/nacional/resoluci%C3%B3n-139-2025-417061/texto).

El alcance inicial es nacional y de aplicación general. No incluye asuetos administrativos/bancarios, días de gremios o comunidades religiosas, ni feriados locales como el G20 de CABA. Los ceros se interpretan dentro de este alcance. Estas ampliaciones pueden incorporarse como fuentes separadas si resultan útiles para el proyecto.

Para entrenamiento retrospectivo, los feriados extraordinarios requieren comprobar cuándo se anunciaron: conocer hoy su fecha no significa que fuera conocida al emitir un pronóstico 24–48 horas antes. La exportación no resuelve esa disponibilidad histórica.

Las pruebas comprueban Carnaval antes y después de 2011, coincidencias de eventos, cambios en la clasificación turística, excepciones, años bisiestos y ausencia de duplicados. No reemplazan una auditoría normativa exhaustiva.
