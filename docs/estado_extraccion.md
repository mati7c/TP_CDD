# Estado de la extracción inicial

**Registro histórico de la primera ejecución, basada en el resumen.** Fue reemplazado como estado vigente por `auditoria_csv.md`: el CSV original ya está incorporado y Palermo ya fue descargado. Se conserva este documento para explicar las 54 descargas reutilizadas.

Ejecución del 22/09/2026 (hora de Argentina).

- Fuente: Open-Meteo, modelo ERA5; ocho variables horarias.
- Período completo: 2009-10-01 a 2026-08-24, inclusive.
- Estaciones: Centenario, Av. Córdoba y La Boca.
- Resultado: **54/54 bloques anuales completos**, sin pendientes.
- **148.128 horas por estación; 444.384 registros estación-hora** en total.
- Ocho variables sin nulos en las respuestas descargadas: 3.555.072 valores.
- Ejes horarios completos, únicos y ordenados, verificados por bloque; hashes SHA-256 comprobados.
- No se imputaron, limpiaron ni modificaron valores meteorológicos.
- La prueba corta adicional contiene 48 horas de enero de 2021 por estación y queda fuera de estos totales. No concatenarla con los años completos.
- Contador local: 1.479 unidades estimadas reservadas (1.473 del histórico + 6 de la prueba corta). La consulta manual previa de conectividad queda fuera del contador.
- Una segunda ejecución del comando completo encontró cero pendientes y no hizo nuevas consultas.
- Nueve pruebas automatizadas aprobaron; incluyen cuotas, reintentos, 429, integridad, eje temporal y bloqueo de escritores simultáneos.

En la prueba de enero de 2021, Centenario y Córdoba devolvieron la celda (-34,5; -58,5), con elevaciones de 18 y 28 metros, respectivamente. La Boca devolvió (-34,75; -58,25), con elevación de 12 metros. No interpretar estos datos como tres sensores meteorológicos independientes.

El manifiesto `data/reports/open_meteo_latest.json` contiene los 54 bloques de la selección completa. Cada JSON original tiene un metadato adyacente con URL, parámetros, instante de extracción y unidades. Los archivos de datos y el estado local están excluidos de Git; conservarlos al respaldar el proyecto.

Pendiente: incorporar el CSV original de contaminación, auditar su cobertura y preparar las siguientes fuentes. La completitud estructural del reanálisis no garantiza representatividad meteorológica en cada estación ni disponibilidad de esas observaciones en tiempo real.
