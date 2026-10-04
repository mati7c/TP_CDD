"""Prepara una tabla estación-hora con entradas numéricas para modelado futuro."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path


STATIONS = ("centenario", "cordoba", "la_boca")
WEEKDAYS = tuple(range(1, 8))
HOURS = tuple(range(24))
TARGETS = ("CO", "NO2", "PM10")
IDENTIFIERS = ("fecha_hora", "fecha")
NUMERIC_FEATURES = (
    "anio", "mes", "es_fin_de_semana", "es_feriado_nacional",
    "es_dia_no_laborable", "temperature_2m", "relative_humidity_2m",
    "surface_pressure", "wind_speed_10m", "precipitation", "cloud_cover",
    "shortwave_radiation", "viento_u", "viento_v",
)
ONE_HOT_COLUMNS = (
    tuple(f"estacion_{station}" for station in STATIONS)
    + tuple(f"dia_semana_{day}" for day in WEEKDAYS)
    + tuple(f"hora_dia_{hour:02d}" for hour in HOURS)
)
OUTPUT_COLUMNS = IDENTIFIERS + NUMERIC_FEATURES + ONE_HOT_COLUMNS + TARGETS
INPUT_COLUMNS = IDENTIFIERS + NUMERIC_FEATURES + ("estacion", "dia_semana", "hora_dia") + TARGETS


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_row(row, line):
    if any(row[column] is None for column in INPUT_COLUMNS):
        raise ValueError(f"Fila {line}: faltan campos")
    station = row["estacion"]
    if station not in STATIONS:
        raise ValueError(f"Fila {line}: estación desconocida: {station!r}")
    try:
        weekday = int(row["dia_semana"])
        hour = int(row["hora_dia"])
    except ValueError as exc:
        raise ValueError(f"Fila {line}: día de semana u hora no numéricos") from exc
    if weekday not in WEEKDAYS or hour not in HOURS:
        raise ValueError(f"Fila {line}: día de semana u hora fuera de rango")
    for column in IDENTIFIERS + NUMERIC_FEATURES:
        if not row[column].strip():
            raise ValueError(f"Fila {line}: {column} está vacío")
    for column in NUMERIC_FEATURES + TARGETS:
        if row[column].strip():
            try:
                value = float(row[column])
            except ValueError as exc:
                raise ValueError(f"Fila {line}: {column} no es numérico") from exc
            if not math.isfinite(value):
                raise ValueError(f"Fila {line}: {column} no es finito")

    result = [row[column] for column in IDENTIFIERS + NUMERIC_FEATURES]
    result.extend("1" if station == value else "0" for value in STATIONS)
    result.extend("1" if weekday == value else "0" for value in WEEKDAYS)
    result.extend("1" if hour == value else "0" for value in HOURS)
    result.extend(row[column] for column in TARGETS)
    return result


def prepare_files(source, output):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("La entrada y la salida deben ser archivos distintos")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".tmp")
    rows = 0
    missing_targets = {column: 0 for column in TARGETS}
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as incoming, temp.open("w", encoding="utf-8-sig", newline="") as outgoing:
            reader = csv.DictReader(incoming)
            if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError("Encabezado de entrada vacío o con columnas duplicadas")
            missing = sorted(set(INPUT_COLUMNS) - set(reader.fieldnames))
            if missing:
                raise ValueError(f"Faltan columnas de entrada: {', '.join(missing)}")
            writer = csv.writer(outgoing)
            writer.writerow(OUTPUT_COLUMNS)
            for line, row in enumerate(reader, start=2):
                if None in row:
                    raise ValueError(f"Fila {line}: más campos que el encabezado")
                writer.writerow(prepare_row(row, line))
                rows += 1
                for column in TARGETS:
                    missing_targets[column] += not row[column].strip()
        os.replace(temp, output)
    finally:
        temp.unlink(missing_ok=True)

    meta = {
        "input": {"path": str(source), "sha256": sha256(source)},
        "output": {"path": str(output), "sha256": sha256(output), "rows": rows, "columns": list(OUTPUT_COLUMNS)},
        "missing_targets": missing_targets,
        "missing_predictors": 0,
        "one_hot": {"estacion": list(STATIONS), "dia_semana": list(WEEKDAYS), "hora_dia": list(HOURS)},
        "identifiers_not_predictors": list(IDENTIFIERS),
        "targets_not_same_hour_predictors": list(TARGETS),
        "time_note": "La hora 00 corresponde a HORA 24 del día de medición anterior; fecha, anio, mes y los indicadores de calendario conservan ese día de medición.",
        "weather_note": "Las variables meteorológicas históricas provienen de ERA5; en uso operativo deberán tener el mismo esquema y proceder de pronósticos disponibles antes de la hora estimada.",
    }
    output.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def prepare_dataset(data_dir):
    data_dir = Path(data_dir)
    return prepare_files(data_dir / "exports/dataset_objetivo.csv", data_dir / "exports/dataset_objetivo_listo_modelar.csv")
