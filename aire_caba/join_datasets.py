"""Unión de CSV existentes, preservando filas, orden y textos del archivo de aire."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path

from .air_source import parse_date
from .open_meteo import write_json


def file_hash(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_csv(path, warnings=None):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames
        if not headers or len(headers) != len(set(headers)):
            raise ValueError(f"Encabezados ausentes o repetidos: {path}")
        rows = []
        for line, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Fila de longitud incorrecta en {path}")
            missing = [field for field, value in row.items() if value is None]
            if missing:
                if warnings is not None:
                    warnings.append({"path": str(path), "line": line, "absent_trailing_fields": missing})
                # Ausencia de celda al final de una fila: se serializa vacía, no se estima.
                row.update({field: "" for field in missing})
            rows.append(row)
    return headers, rows


def hour_key(value):
    # Sólo representación de la clave: 0 y 00 coinciden; 24 sigue siendo 24.
    return value.zfill(2) if value.isdigit() else value


def unique_index(rows, key, label):
    index = {}
    for row in rows:
        value = key(row)
        if value in index:
            raise ValueError(f"Clave duplicada en {label}: {value}. Resolver antes de unir.")
        index[value] = row
    return index


def join_files(air_path, weather_path, calendar_path, holidays_path, output):
    inputs = {"aire": air_path, "meteorologia": weather_path, "calendario": calendar_path, "feriados": holidays_path}
    provenance = {name: {"path": str(path.resolve()), "sha256": file_hash(path)} for name, path in inputs.items()}
    warnings = []
    ah, air = read_csv(air_path, warnings)
    wh, weather = read_csv(weather_path, warnings)
    ch, calendar = read_csv(calendar_path, warnings)
    fh, holidays = read_csv(holidays_path, warnings)
    for headers, required, label in ((ah, {"FECHA", "HORA"}, "aire"), (wh, {"FECHA", "HORA", "estacion"}, "meteorología"), (ch, {"FECHA"}, "calendario"), (fh, {"fecha"}, "feriados")):
        if not required <= set(headers):
            raise ValueError(f"Faltan claves en {label}: {required - set(headers)}")
    stations = list(dict.fromkeys(r["estacion"] for r in weather))
    if not stations:
        raise ValueError("CSV meteorológico sin filas.")
    for station in stations:
        if not any(field.endswith("_" + station.upper()) for field in ah):
            raise ValueError(f"Estación meteorológica sin columnas de aire: {station}")
    wi = unique_index(weather, lambda r: (r["FECHA"], hour_key(r["HORA"]), r["estacion"]), "meteorología")
    ci = unique_index(calendar, lambda r: r["FECHA"], "calendario")
    hi = defaultdict(list)
    for event in holidays:
        hi[event["fecha"]].append(event)
    max_events = max((len(v) for v in hi.values()), default=0)
    wf = [f for f in wh if f not in ("FECHA", "HORA", "estacion")]
    cf = [f for f in ch if f != "FECHA"]
    columns = ah + [f"METEO_{s.upper()}_{f}" for s in stations for f in wf] + [f"CAL_{f}" for f in cf] + [f"FERIADO_{i}_{f}" for i in range(1, max_events+1) for f in fh]
    if len(columns) != len(set(columns)):
        raise ValueError("Colisión entre nombres de columnas originales y anexadas.")
    counts = Counter()
    matches = Counter({s: 0 for s in stations})
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.part")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for original in air:
                row = dict(original)
                try:
                    day = parse_date(original["FECHA"]).isoformat()
                except ValueError:
                    day = None
                    counts["unparseable_date_rows"] += 1
                hour = hour_key(original["HORA"])
                for station in stations:
                    point = wi.get((day, hour, station))
                    matches[station] += point is not None
                    for field in wf:
                        row[f"METEO_{station.upper()}_{field}"] = point[field] if point is not None else ""
                cal = ci.get(day)
                counts["calendar_matched"] += cal is not None
                for field in cf:
                    row[f"CAL_{field}"] = cal[field] if cal is not None else ""
                events = hi.get(day, [])
                counts["rows_with_holiday_events"] += bool(events)
                for i in range(1, max_events+1):
                    for field in fh:
                        row[f"FERIADO_{i}_{field}"] = events[i-1][field] if len(events) >= i else ""
                writer.writerow(row)
        # Comprobar todas las filas y todas las columnas originales, no sólo muestras.
        with temporary.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            verified = 0
            for original, joined in zip_longest(air, reader):
                if original is None or joined is None or any(original[f] != joined[f] for f in ah):
                    raise ValueError("La unión cambió filas, orden o valores originales.")
                if None in joined or any(v is None for v in joined.values()):
                    raise ValueError("Salida CSV con estructura inválida.")
                verified += 1
        for name, path in inputs.items():
            if file_hash(path) != provenance[name]["sha256"]:
                raise ValueError(f"El archivo {name} cambió durante la unión.")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    report = {"output": str(output.resolve()), "output_sha256": file_hash(output),
              "rows": verified, "columns": len(columns), "original_columns_preserved": ah,
              "inputs": provenance, "input_warnings": warnings, "weather_matches": dict(matches),
              "weather_unmatched": {s: verified - matches[s] for s in stations},
              "calendar_unmatched": verified - counts["calendar_matched"],
              "max_holiday_events_per_date": max_events, "counts": dict(counts),
              "join_policy": "Left join desde aire. Mismo número/orden de filas y textos originales. Claves auxiliares: FECHA SAS a ISO, HORA rellenada con cero a la izquierda. 00 y 24 no se fusionan. Calendario por fecha etiquetada. Detalles de feriados en columnas numeradas conservando su orden en el CSV.",
              "missing_policy": "Sin correspondencia: campos anexados vacíos. No imputación, limpieza, deduplicación, agregación, escalado ni nuevas variables calculadas."}
    write_json(output.with_suffix(".meta.json"), report)
    return report


def join_datasets(config, config_path, root):
    if "air_source" not in config:
        raise ValueError("Primero ejecutá import-air.")
    exports = Path(root) / "exports"
    return join_files(config_path.parent / config["air_source"]["path"],
                      exports / "meteorologia.csv", exports / "calendario.csv", exports / "feriados.csv",
                      exports / "entrada_transformacion.csv")
