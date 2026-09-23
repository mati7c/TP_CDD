# Extracción basada en calidad-aire.csv

Se incorporó el archivo `C:/Users/matic/Downloads/calidad-aire.csv` como fuente de la selección. La copia dentro del proyecto conserva exactamente sus bytes.

SHA-256: `2e7f655c2bbe802e53c9b0fe9428150cf0ff71435ee8e9b0a75e29832fd513f9`.

## Auditoría del original

- 145.980 filas y 14 columnas.
- FECHA mínima: 01/10/2009; máxima: 24/08/2026.
- 6.149 fechas distintas. El archivo no está ordenado cronológicamente.
- Cero claves FECHA/HORA duplicadas en sus valores originales. Esto no prueba unicidad después de una futura normalización horaria.
- 1.000.658 celdas numéricas; no se certifica todavía su validez científica.
- 10.459 valores censurados, como `<0.05`, conservados como texto.
- 465.480 vacíos, 274.875 marcadores de faltante y 288 errores `#REF!`.
- 324 filas no contienen mediciones numéricas ni censuradas en ninguna estación.
- 147 filas tienen HORA=0; 135 tienen HORA mayor que 24. Ninguna de esas 282 filas contiene valores numéricos o censurados.

| Estación | Filas con alguna medición numérica o censurada | Primera fecha | Última fecha |
|---|---:|---|---|
| Centenario | 136.303 | 01/10/2009 | 24/08/2026 |
| Av. Córdoba | 127.796 | 01/10/2009 | 24/08/2026 |
| La Boca | 134.742 | 01/10/2009 | 24/08/2026 |
| Palermo | 5.942 | 01/10/2009 | 19/09/2010 |

Las filas por estación no son observaciones independientes entre sí. Palermo sólo tiene valores de CO; NO2 y PM10 están completamente vacíos.

## Cambios en la extracción

El plan se calcula por estación y año a partir de las fechas con al menos una medición numérica o censurada. Incluye un margen de un día ante HORA=24, sin asignar aún un instante a esas mediciones. La cobertura anual puede contener huecos internos de contaminación.

Se reutilizaron los 54 bloques de las tres estaciones principales. Cuando abarcan más fechas que las seleccionadas, el manifiesto registra ambos intervalos por separado. Se agregaron dos bloques de Palermo: 01/10/2009–31/12/2009 y 01/01/2010–19/09/2010. No se descargaron años posteriores para Palermo.

Las dos descargas nuevas reservan 29 unidades estimadas de cuota. Los archivos reutilizados no generan consumo nuevo. Todos conservan coordenadas solicitadas y devueltas, modelo ERA5, variables, unidades y fecha de extracción.

## Pendiente para la unión horaria

Confirmar con la fuente el significado de HORA 1–24 y la zona horaria. No aplicar automáticamente `%24`, sumar horas mayores que 24 a FECHA ni desplazar todos los datos una hora. La meteorología está extraída; todavía no se construyó la tabla de entrenamiento ni se hizo una unión por hora.

El catálogo oficial de Palermo declara desactivación desde 30/06/2010, mientras el CSV contiene valores hasta septiembre. Se conservaron para respetar la selección aportada; la discrepancia queda pendiente de revisión.

Reportes reproducibles: `data/reports/air_latest.json` contiene la auditoría por columna y las líneas con problemas; `data/reports/open_meteo_latest.json` vincula cada bloque meteorológico con el hash del CSV y los parámetros solicitados. El comando `import-air` recalcula la selección al recibir un nuevo archivo.
