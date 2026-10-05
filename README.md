# Calidad del aire en CABA

Proyecto de Ciencia de Datos, UTN FRC, Grupo 12. Primera etapa: **extracción de meteorología horaria basada en el CSV original de contaminación**. El archivo `calidad-aire.csv` ya fue incorporado: 145.980 filas, 14 columnas, del 01/10/2009 al 24/08/2026. Incluye Centenario, Av. Córdoba, La Boca y Palermo (esta última sólo en 2009–2010). No limpia, imputa ni construye todavía la tabla de entrenamiento.

## Ejecutar en Windows

Desde esta carpeta en PowerShell:

```powershell
# Ya se ejecutó para el archivo actual. Repetir si se recibe otro CSV.
.\run.ps1 import-air --csv "C:\Users\matic\Downloads\calidad-aire.csv"

# Ver solicitudes pendientes y consumo estimado; no consulta la API.
.\run.ps1 plan

# Prueba corta, para las tres estaciones.
.\run.ps1 download --start 2021-01-01 --end 2021-01-02

# Descargar todo lo pendiente; repetir este comando para reanudar.
.\run.ps1 download

# Exportar los JSON descargados a un CSV meteorológico (sin usar la API).
.\run.ps1 export-weather

# Unir los CSV existentes para entrada de transformación.
.\run.ps1 join

# Limpiar y generar el dataset objetivo estación-hora (requiere requirements-etl.txt).
.\run.ps1 transform

# Generar el CSV con one-hot para modelado futuro.
.\run.ps1 prepare-model

# Ver progreso y verificar integridad de los archivos descargados.
.\run.ps1 status
```

El lanzador busca `.venv`, Python del sistema o el runtime de Codex de este equipo. No cambia la política de ejecución de PowerShell. Si el sistema bloquea scripts, usar directamente `python -m aire_caba plan` y los mismos argumentos.

Requiere Python >=3.10. La meteorología no necesita dependencias externas ni clave de API. El calendario usa las versiones fijas de `requirements-calendar.txt`, ya instaladas en `.venv` en este equipo. También funciona en Linux/macOS con `python -m aire_caba`. Instalación opcional del paquete: `python -m pip install -e .`; después se puede usar `aire-caba` (con `--config` explícito si se instala fuera del repositorio).

```powershell
# Trabajar de a un bloque anual nuevo.
.\run.ps1 download --max-requests 1

# Elegir período y estación.
.\run.ps1 download --start 2020-01-01 --end 2020-12-31 --stations centenario
```

Las fechas son inclusivas. Un bloque parcial tiene su propia caché y no sustituye un año completo: la prueba corta y la descarga completa pueden contener horas repetidas en archivos distintos. El reporte de cada ejecución identifica exactamente los bloques de esa selección; no concatenar indiscriminadamente todos los JSON.

## Cómo determina la cobertura a partir del CSV

`import-air` guarda una copia idéntica de los bytes, identificada por SHA-256, en `data/raw/air/`. Audita todas las filas, independientemente del orden, y actualiza `config/open_meteo.json` con períodos por estación y año. Cada período abarca desde la primera hasta la última fecha con al menos un valor numérico o censurado (`<0.05`, por ejemplo). Conserva ceros y valores anómalos; no decide su validez científica.

Los textos `s/d`, `x/d`, errores de planilla y celdas vacías se cuentan y conservan en la copia original, pero no se consideran mediciones para ampliar la cobertura. Estaciones desconocidas detienen la importación hasta verificar sus coordenadas. Las columnas enteramente vacías de NO2 y PM10 en Palermo no eliminan sus mediciones de CO.

El CSV usa principalmente HORA 1–24. También contiene 147 filas con HORA=0 y 135 con valores mayores que 24, todas sin valores numéricos/censurados. Se registran con número de línea en `data/reports/air_latest.json`; no se eliminan del original ni se corrigen. **La convención de HORA=24 y la zona horaria del CSV deben confirmarse antes de unir por hora.** El extractor incluye un día de margen cuando una medición lleva HORA=24, sólo para cubrir un posible cambio de fecha; no convierte ese registro a un instante.

La extracción continúa por intervalos anuales para ser eficiente. Puede incluir días sin mediciones dentro de un intervalo. No equivale a una unión exacta por observación. Un archivo meteorológico anterior más amplio se reutiliza sólo si coincide en coordenadas, modelo, variables, unidades y zona, y supera las verificaciones de integridad. El manifiesto distingue `requested_parameters` del intervalo realmente almacenado, y registra el hash del CSV base. `status` distingue horas de los intervalos solicitados de horas de la caché, que puede ser más amplia.

El plan actual tiene 56 bloques: 54 reutilizados y 2 de Palermo descargados. Para nuevas selecciones, repetir `import-air`, `plan` y `download`. Si la copia del CSV se altera, el programa se detiene. Respalda la carpeta `data`, además del código, para conservar los originales.

## Fuente seleccionada

El archivo `data/exports/meteorologia.csv` se genera con `export-weather`. Contiene una fila por estación/fecha/hora dentro de los intervalos del CSV base, con las ocho variables meteorológicas. FECHA usa `YYYY-MM-DD`; HORA se escribe con dos dígitos (`00`–`24`). La exportación no añade contaminantes ni reproduce cada fila del CSV original: es una tabla meteorológica separada.

Por solicitud del usuario, `00` y `24` quedan separados. Las horas `00`–`23` son originales de Open-Meteo. La fila con FECHA D/HORA `24` copia provisionalmente D+1/`00`; `fecha_hora_fuente` y `tipo_hora` hacen explícita esa correspondencia. No se deduplican ni se promedian. Si la hora del día siguiente no está descargada, la fila `24` queda con valores vacíos y estado `sin_hora_siguiente_descargada`; no se inventan valores ni se hace una consulta adicional. La convención de las horas de contaminación sigue pendiente de revisión.

`meteorologia.meta.json` documenta unidades, zona horaria, archivos originales, hash del CSV base y conteos. La codificación es UTF-8 con BOM, separador coma y decimal punto. Al importar en Excel, seleccionar HORA como texto si se desea conservar visualmente `00`. El comando reemplaza la exportación anterior y no consume cuota de API.

Se utiliza [Open-Meteo Historical Weather](https://open-meteo.com/en/docs/historical-weather-api) con `models=era5` explícito, para mantener consistencia durante todo el histórico. Son datos de reanálisis de grilla (~0,25°), no sensores instalados junto a las estaciones de aire. No se usa `best_match` ni se cambia automáticamente de proveedor.

Las coordenadas fueron verificadas el 22/09/2026 en el [catálogo oficial de estaciones de Buenos Aires Data](https://data.buenosaires.gob.ar/dataset/calidad-aire/resource/juqdkmgo-295-resource). La configuración conserva la fuente y la fecha. La celda, elevación y coordenadas devueltas por la API quedan registradas por separado. Estaciones cercanas pueden compartir celda; el proveedor también aplica ajustes por elevación. No asumir independencia espacial.

| Variable API | Unidad solicitada/publicada |
|---|---|
| temperature_2m | °C |
| relative_humidity_2m | % |
| surface_pressure | hPa |
| wind_speed_10m | m/s |
| wind_direction_10m | grados |
| precipitation | mm |
| cloud_cover | % |
| shortwave_radiation | W/m² |

Se preserva `hourly_units` de cada respuesta. La documentación describe precipitación acumulada y radiación media de la hora precedente. La zona solicitada es `America/Argentina/Buenos_Aires`; para el período del proyecto se valida UTC−3 y una secuencia de 24 horas por día. No se convierten todavía los tiempos a UTC. Los valores nulos se conservan y cuentan, sin rellenarlos.

## Límites y reanudación

[Límites oficiales consultados el 22/09/2026](https://open-meteo.com/en/pricing): 600 unidades/minuto, 5.000/hora, 10.000/día y 300.000/mes, para uso gratuito no comercial. Una solicitud larga cuenta como varias unidades: la duración mayor a 14 días y más de 10 variables aumentan el consumo.

El plan se deriva de las estaciones y fechas con mediciones del CSV. Se reserva por intento una estimación conservadora: `ceil(max(1, días/14) × max(1, variables/10) × 1,10)` para una ubicación y un modelo. `plan` informa las solicitudes pendientes y su consumo; los bloques reutilizados no consumen cuota. Es una estimación local, no una lectura de saldo del proveedor.

El contador SQLite persiste antes de enviar cada intento, incluso si falla. Usa ventanas móviles de 60 s, 1 h, 24 h y 31 días, con topes internos de 450, 4.000, 8.000 y 240.000. Para el tope por minuto espera automáticamente en pausas de hasta 60 segundos; para los demás termina con un mensaje y se puede ejecutar nuevamente más tarde. Esto evita ráfagas cuando todas las respuestas son muy rápidas.

HTTP 429 detiene inmediatamente la ejecución y persiste `Retry-After`; sin ese encabezado, aplica una pausa conservadora de 24 horas. Fallas de red y HTTP 5xx admiten hasta cuatro intentos con espera creciente. Los HTTP 4xx restantes no se reintentan. No hay descargas paralelas: un bloqueo del sistema operativo impide dos escritores en la misma carpeta de datos.

**El presupuesto sólo conoce este proyecto y esta carpeta de datos.** No ve consumos de otras aplicaciones/equipos con la misma IP. No borrar `data/state` ni usar otra carpeta para eludir límites. Se deja margen, pero un 429 puede ocurrir por consumo externo o cambios del servicio. La primera consulta manual de verificación, anterior al contador, no está registrada en SQLite.

## Archivos

```text
config/open_meteo.json       Cobertura derivada del CSV, variables y estaciones
aire_caba/                  Cliente, control de cuota y CLI
tests/                      Pruebas sin acceso a internet
data/raw/open_meteo/         Estación/año/intervalo-hash.json + .meta.json
data/raw/air/                Copia original del CSV por SHA-256
data/state/                 Cuotas, pausa del proveedor y bloqueo
data/reports/               Último manifiesto de bloques consultados
docs/                       Decisiones y fuentes
```

Los JSON de respuesta se guardan como bytes originales. Cada metadato registra parámetros, URL, fecha UTC, encabezados, versión del extractor, coordenadas solicitadas/devueltas, unidades, conteo de nulos y SHA-256. El hash de parámetros evita reutilizar respuestas de consultas diferentes. Escritura mediante archivo temporal y reemplazo atómico; sólo el metadato final marca un bloque completo.

La reanudación comprueba hash y estructura antes de omitir bloques. Una caché alterada detiene el proceso para revisión. Una respuesta inválida queda conservada, pero sin marca de completitud. El reporte más reciente se sobrescribe; los metadatos por bloque conservan la procedencia. `status` comprueba la selección pedida (por defecto, todo el histórico), no todos los posibles experimentos en disco.

## Verificar el proyecto

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Las pruebas cubren cortes por año, años bisiestos, cuotas persistentes, 429, reintentos, reanudación sin red, corrupción y rechazo de horas duplicadas. Las verificaciones estructurales no reemplazan la auditoría científica posterior.

## Segunda entrega: ETL y dataset objetivo

`notebooks/segunda_entrega_etl.ipynb` (y su exportación `.html`) contiene el análisis de calidad, las visualizaciones, la estrategia por caso y la descripción del dataset final. La lógica de limpieza está en `aire_caba/transform.py` y genera `data/exports/dataset_objetivo.csv` + `.meta.json` (una fila por estación-hora; Palermo excluida). Instalar dependencias: `.\.venv\Scripts\python.exe -m pip install -r requirements-etl.txt`.

`prepare-model` genera `data/exports/dataset_objetivo_listo_modelar.csv` desde ese dataset, con una fila por estación-hora. Codifica con one-hot las tres estaciones, los siete días de semana, los 12 meses y las 24 horas (`00`–`23`); mantiene año, banderas de calendario y meteorología numérica. `fecha_hora` y `fecha` identifican filas y períodos, pero no son predictores directos. `CO`, `NO2` y `PM10` son posibles salidas: sus faltantes permanecen vacíos y se filtran por separado al entrenar cada modelo. Los estados de medición, la estación del año y `hora_original` quedan en `dataset_objetivo.csv` para auditoría; no se usan como entradas. Ninguna otra columna de contaminante de la misma hora debe entrar como predictor. El archivo `.meta.json` registra el esquema, los conteos de faltantes y la procedencia.

La convención temporal sigue siendo la de la transformación previa: `fecha_hora` expresa la hora de cierre. La hora `00` procede de `HORA=24` de la fecha anterior; `fecha`, `anio`, `mes`, día de semana y banderas de calendario conservan esa **fecha de medición**. En una predicción operativa hay que construir esas entradas con la misma regla. La meteorología del CSV es ERA5 histórico; al predecir se utilizará un pronóstico meteorológico con las mismas variables y unidades. La evaluación con ERA5 refleja un escenario con meteorología histórica conocida y no mide directamente el error adicional del pronóstico.

## Uso posterior

1. Resolver la convención temporal del CSV y auditar sus valores anómalos.
2. Revisar el calendario reconstruido y las fechas de anuncio antes de usarlo en pronósticos históricos.
3. Construir la tabla por estación/hora en una etapa de transformación separada.

Para pronósticos, no usar el tiempo meteorológico observado del futuro como entrada. El reanálisis tampoco reproduce exactamente la información disponible en tiempo real. Las predicciones operativas requerirán verificar disponibilidad y pronósticos históricos emitidos previamente.

Atribución: datos meteorológicos de Open-Meteo y Copernicus/ECMWF ERA5, CC BY 4.0; coordenadas de Buenos Aires Data/APrA. Esta extracción no modifica los valores recibidos. ERA5 reciente puede ser provisional y revisarse: la caché conserva la versión descargada, sin actualización silenciosa.
