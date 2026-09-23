"""Exporta meteorología; conserva 00 y un alias explícito 24 sin deduplicarlos."""

import csv
import hashlib
import json
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .open_meteo import cached, jobs, paths, resolve_cached_job, write_json


def weather_rows(index, periods, fields):
    for sid, intervals in periods.items():
        days = set()
        for interval in intervals:
            day = date.fromisoformat(interval["start_date"])
            end = date.fromisoformat(interval["end_date"])
            while day <= end:
                days.add(day)
                day += timedelta(days=1)
        for day in sorted(days):
            for hour in range(25):
                timestamp = (datetime.combine(day, datetime.min.time()) + timedelta(hours=hour)).isoformat(timespec="minutes")
                point = index.get((sid, timestamp))
                if point is None and hour != 24:
                    raise ValueError(f"Falta meteorología horaria requerida: {sid} {timestamp}")
                yield {
                    "estacion": sid, "FECHA": day.isoformat(), "HORA": f"{hour:02d}",
                    "fecha_hora_fuente": timestamp,
                    "tipo_hora": "alias_24_provisional" if hour == 24 else "hora_original",
                    "estado": "disponible" if point is not None else "sin_hora_siguiente_descargada",
                    **{field: point[i] if point is not None else None for i, field in enumerate(fields)},
                }


def export_weather(config, root):
    if "air_source" not in config or "station_periods" not in config:
        raise ValueError("Primero ejecutá import-air para definir la cobertura.")
    fields = config["hourly"]
    index = {}
    sources = {}
    units = None
    for requested in jobs(config):
        job = resolve_cached_job(root, requested)
        metadata = cached(root, job)
        if metadata is None:
            raise ValueError(f"Bloque pendiente: {job['key']}. Ejecutá download.")
        if job["key"] in sources:
            continue
        raw, _ = paths(root, job)
        payload = json.loads(raw.read_bytes())
        selected_units = {f: payload["hourly_units"][f] for f in fields}
        if units is not None and units != selected_units:
            raise ValueError("Unidades incompatibles entre archivos meteorológicos.")
        units = selected_units
        sources[job["key"]] = metadata["sha256"]
        for i, timestamp in enumerate(payload["hourly"]["time"]):
            key = (job["station"]["id"], timestamp)
            values = tuple(payload["hourly"][f][i] for f in fields)
            if key in index and index[key] != values:
                raise ValueError(f"Bloques meteorológicos con valores contradictorios: {key}")
            index[key] = values
    output = Path(root) / "exports/meteorologia.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.part")
    counts, per_station, nulls = Counter(), Counter(), Counter()
    columns = ["estacion", "FECHA", "HORA", "fecha_hora_fuente", "tipo_hora", "estado", *fields]
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row in weather_rows(index, config["station_periods"], fields):
                writer.writerow(row)
                counts["rows"] += 1
                counts[row["tipo_hora"]] += 1
                counts[row["estado"]] += 1
                per_station[row["estacion"]] += 1
                for field in fields:
                    nulls[field] += row[field] is None
        # Volver a leer el CSV para comprobar estructura y conteo exportado.
        with temporary.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            verified = 0
            for row in reader:
                if None in row or any(v is None for v in row.values()):
                    raise ValueError("Estructura de CSV inválida después de exportar.")
                verified += 1
        if verified != counts["rows"]:
            raise ValueError("Conteo de CSV no coincide con los registros exportados.")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    hasher = hashlib.sha256()
    with output.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    report = {
        "output": str(output.resolve()), "sha256": hasher.hexdigest(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "air_source": config["air_source"], "timezone": config["timezone"],
        "model": config["model"], "units": units, "counts": dict(counts),
        "rows_per_station": dict(per_station), "null_counts": dict(nulls),
        "raw_blocks_sha256": sources, "station_periods": config["station_periods"],
        "format": "UTF-8 BOM, separador coma, decimal punto; HORA texto de dos dígitos 00–24",
        "hour_policy": "00–23 son horas originales. FECHA D/HORA 24 es alias provisional de D+1/00; ambas filas se conservan. Si no está descargada esa hora, los valores quedan vacíos. No confirma la convención horaria del CSV de aire.",
        "scope": "Intervalos meteorológicos por estación/año derivados del CSV. Incluye días sin contaminación dentro de cada intervalo. No es una unión con las filas de contaminación.",
    }
    write_json(output.with_suffix(".meta.json"), report)
    return report
