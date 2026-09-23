# Decisión inicial de fuente — 22/09/2026

| Fuente | Adecuación a esta primera etapa | Decisión |
|---|---|---|
| [Open-Meteo / ERA5](https://open-meteo.com/en/docs/historical-weather-api) | API sin credenciales, variables y período requeridos; selección explícita de ERA5 | Implementada |
| [Copernicus ERA5 time-series](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-timeseries?tab=overview) | Producto optimizado para series largas en puntos, CSV/NetCDF; candidato si crece el volumen | Alternativa, sin benchmark local |
| [NASA POWER](https://power.larc.nasa.gov/docs/services/api/temporal/hourly/) | Horarios desde 2001; UTC o tiempo solar local, que no equivale a hora civil | Alternativa, sin benchmark local |

Copernicus ofrece acceso eficiente a series temporales, pero su [API requiere configuración de cuenta/clave y aceptación de términos](https://cds.climate.copernicus.eu/how-to-api). NASA POWER tiene una [grilla más gruesa para meteorología y radiación](https://power.larc.nasa.gov/docs/tutorials/service-data-request/api/). No se afirma que alguna alternativa sea más rápida sin medir solicitudes equivalentes; el volumen de tres estaciones no justifica cambiar de fuente por el límite diario de Open-Meteo.

Estaciones: [catálogo oficial](https://data.buenosaires.gob.ar/dataset/calidad-aire/resource/juqdkmgo-295-resource), [CSV original](https://data.buenosaires.gob.ar/dataset/calidad-aire/resource/juqdkmgo-295-resource/download). Las coordenadas fueron transcritas de la vista previa oficial; no se infirieron a partir de direcciones. Tras inspeccionar el CSV del usuario, se incorporó Palermo para el período con mediciones de CO en 2009–2010.

El catálogo indica desactivación de Palermo desde 30/06/2010, pero el CSV del usuario contiene valores hasta 19/09/2010. Se conserva la selección del usuario y se registra la discrepancia para la auditoría posterior; no se recorta por la fecha del catálogo.

La comparación es documental y una prueba de conectividad a Open-Meteo; no es una medición comparativa de rendimiento. No se incorporan datos de distintas fuentes como si fueran intercambiables.
