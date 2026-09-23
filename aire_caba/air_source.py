"""Audita el CSV sin modificar observaciones y deriva la cobertura de extracción."""

import csv
import io
import json
import math
import os
import re
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .open_meteo import atomic_bytes, digest, write_json

MONTHS = {name: i + 1 for i, name in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}
POLLUTANTS = ("CO", "NO2", "PM10")
PALERMO = {"id": "palermo", "name": "PALERMO", "latitude": -34.5834529467442, "longitude": -58.4053598727298}


def parse_date(value):
    match = re.fullmatch(r"(\d{2})([A-Z]{3})(\d{4}):00:00:00", value)
    if not match or match[2] not in MONTHS:
        raise ValueError(f"FECHA no reconocida: {value!r}")
    return date(int(match[3]), MONTHS[match[2]], int(match[1]))


def classify(value):
    value = value.strip()
    if not value:
        return "empty"
    if value.lower() in ("s/d", "x/d"):
        return "missing_marker"
    if value.startswith("#"):
        return "error_marker"
    if re.fullmatch(r"<\s*\d+(?:\.\d+)?", value):
        return "censored"
    try:
        if math.isfinite(float(value)):
            return "numeric"
    except ValueError:
        pass
    return "unrecognized"


def audit(content, stations):
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""))
    headers = reader.fieldnames
    if not headers or len(headers) != len(set(headers)) or not {"FECHA", "HORA"} <= set(headers):
        raise ValueError("Encabezados inválidos: se requieren FECHA, HORA y columnas únicas.")
    known = {s["id"].upper(): s for s in stations}
    columns = {}
    for field in headers:
        if field in ("FECHA", "HORA"):
            continue
        match = re.fullmatch(r"(CO|NO2|PM10)_(.+)", field)
        if not match or match[2] not in known:
            raise ValueError(f"Columna/estación sin coordenadas verificadas: {field}")
        columns[field] = known[match[2]]["id"]
    coverage = {s: {"rows_with_measurements": 0, "dates": set(), "period_dates": set(), "columns": {}} for s in columns.values()}
    counts = {field: Counter() for field in columns}
    tokens = Counter()
    keys = set()
    dates = set()
    hours = Counter()
    issues = []
    duplicates = empty = invalid_hours = hour_zero = n = 0
    previous = None
    inversions = 0
    for n, row in enumerate(reader, start=1):
        if None in row or any(v is None for v in row.values()):
            raise ValueError(f"Fila {n+1}: cantidad incorrecta de columnas.")
        day = parse_date(row["FECHA"])
        dates.add(day)
        hours[row["HORA"]] += 1
        key = (row["FECHA"], row["HORA"])
        duplicates += key in keys
        keys.add(key)
        try:
            hour = int(row["HORA"])
        except ValueError:
            hour = -1
        order = (day, hour)
        inversions += previous is not None and order < previous
        previous = order
        active = set()
        for field, sid in columns.items():
            kind = classify(row[field])
            counts[field][kind] += 1
            if kind in ("numeric", "censored"):
                active.add(sid)
            elif kind != "empty":
                tokens[row[field]] += 1
            if kind == "unrecognized":
                issues.append({"source_line": n+1, "field": field, "value": row[field], "reason": "unrecognized_value"})
        empty += not active
        invalid_hours += not 0 <= hour <= 24
        hour_zero += hour == 0
        if not 1 <= hour <= 24:
            issues.append({"source_line": n+1, "FECHA": row["FECHA"], "HORA": row["HORA"],
                           "reason": "hour_zero" if hour == 0 else "invalid_hour", "has_measurements": bool(active)})
        for sid in active:
            info = coverage[sid]
            info["rows_with_measurements"] += 1
            info["dates"].add(day)
            info["period_dates"].add(day)
            # Sólo margen de extracción. NO asigna un instante a HORA=24.
            if hour == 24:
                info["period_dates"].add(day + timedelta(days=1))
    if n == 0:
        raise ValueError("CSV sin registros.")
    result = {
        "sha256": digest(content), "bytes": len(content), "rows": n, "headers": headers,
        "date_min": min(dates).isoformat(), "date_max": max(dates).isoformat(),
        "distinct_dates": len(dates), "duplicate_date_hour_keys": duplicates,
        "out_of_order_transitions": inversions, "hour_counts": dict(sorted(hours.items())),
        "rows_without_numeric_or_censored_measurements": empty,
        "rows_hour_zero": hour_zero, "rows_invalid_hour": invalid_hours,
        "column_counts": counts, "non_numeric_tokens": tokens, "issues": issues, "stations": {},
        "time_join_status": "pending_source_hour_convention",
        "coverage_policy": "Por estación/año, desde primera hasta última fecha con algún valor numérico o censurado. HORA=24 agrega un día de margen de extracción, sin normalizar la hora. No se alteran los datos.",
    }
    periods = {}
    for sid, info in coverage.items():
        if not info["dates"]:
            continue
        years = {}
        for day in info["period_dates"]:
            years.setdefault(day.year, []).append(day)
        periods[sid] = [{"start_date": min(ds).isoformat(), "end_date": max(ds).isoformat()} for year, ds in sorted(years.items())]
        result["stations"][sid] = {
            "rows_with_measurements": info["rows_with_measurements"],
            "first_measurement_date": min(info["dates"]).isoformat(),
            "last_measurement_date": max(info["dates"]).isoformat(),
            "distinct_measurement_dates": len(info["dates"]), "extraction_periods": periods[sid],
        }
    if any(i.get("has_measurements") for i in issues if i["reason"] in ("hour_zero", "invalid_hour")):
        result["requires_time_review"] = True
    return result, periods


def import_air(csv_path, config_path, root):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not any(s["id"] == "palermo" for s in config["stations"]):
        config["stations"].append(PALERMO.copy())
    content = csv_path.read_bytes()
    report, periods = audit(content, config["stations"])
    if not periods:
        raise ValueError("El CSV no contiene mediciones numéricas ni censuradas.")
    sha = report["sha256"]
    snapshot = root / "raw/air" / sha / "calidad-aire.csv"
    if snapshot.exists() and snapshot.read_bytes() != content:
        raise ValueError("La copia original del CSV fue alterada.")
    atomic_bytes(snapshot, content)
    report["original_path"] = str(csv_path.resolve())
    report["imported_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(root / "reports" / f"air_{sha}.json", report)
    write_json(root / "reports/air_latest.json", report)
    config["air_source"] = {"path": os.path.relpath(snapshot, config_path.parent), "sha256": sha}
    config["station_periods"] = periods
    config["start_date"] = min(p["start_date"] for ps in periods.values() for p in ps)
    config["end_date"] = max(p["end_date"] for ps in periods.values() for p in ps)
    write_json(config_path, config)
    return report


def verify_source(config, config_path):
    if "air_source" not in config:
        return
    source = config["air_source"]
    path = config_path.parent / source["path"]
    if digest(path.read_bytes()) != source["sha256"]:
        raise ValueError("El CSV base cambió o está corrupto. Volvé a ejecutar import-air.")
