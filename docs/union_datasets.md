# CSV de entrada para transformación

```powershell
.\run.ps1 join
```

Genera `data/exports/entrada_transformacion.csv` a partir de la copia original de aire registrada en la configuración y los CSV existentes `meteorologia.csv`, `calendario.csv` y `feriados.csv`. No vuelve a descargar ni regenerar esas fuentes. Tráfico queda fuera.

La tabla base es calidad del aire, en su formato ancho original: una fila de FECHA/HORA con los contaminantes de las cuatro estaciones. La salida mantiene su orden, número de filas y todos sus textos. No se cambia el formato original de FECHA ni HORA, ni valores como `s/d`, `<0.05`, `#REF!`, vacíos o ceros.

Se anexan columnas:

- `METEO_CENTENARIO_*`, `METEO_CORDOBA_*`, `METEO_LA_BOCA_*`, `METEO_PALERMO_*`: variables y campos de procedencia meteorológica por estación.
- `CAL_*`: campos existentes del calendario por fecha.
- `FERIADO_1_*`, `FERIADO_2_*`, etc.: todos los campos del primer, segundo y sucesivos eventos de la fecha, en el orden del CSV de feriados. Así no se multiplican filas por fechas con dos eventos. La numeración no es una categoría del evento.

Para comparar claves únicamente, se lee la fecha SAS del CSV de aire y se representa como ISO; una hora `0` busca `00`, `1` busca `01`, etc. Las celdas originales no se reescriben. `24` se vincula exclusivamente con la fila `24` ya existente en meteorología; no se fusiona con `00`. El calendario se asocia a la fecha etiquetada, incluso para HORA=24.

Sin correspondencia, las columnas anexadas quedan vacías. Esto incluye horas inválidas y períodos de estaciones sin meteorología seleccionada, especialmente Palermo después de 2010. Esos vacíos no eliminan registros ni se imputan. Los indicadores y campos derivados que ya existían en los CSV se copian; la unión no calcula nuevos atributos para modelar.

Las claves duplicadas en meteorología o calendario provocan un error para impedir multiplicar las filas. Los duplicados originales de aire, si existieran, se preservan. Una celda ausente al final de una fila CSV se serializa vacía y se registra en el reporte; no se le asigna un valor. En el archivo actual de calendario falta el campo final `metodo` de 2009-10-12. Se respeta esa ausencia y el archivo fuente permanece intacto.

`entrada_transformacion.meta.json` registra hashes de las cuatro entradas y de la salida, columnas, conteos de coincidencias y advertencias de lectura. Antes de publicar el archivo se verifica cada celda de las columnas originales y el orden completo de las filas, además de que ninguna fuente haya cambiado durante la ejecución.

La transformación posterior podrá resolver horas, faltantes, errores, valores censurados, tipos y formato de modelado. Este archivo sólo reúne los datos.
