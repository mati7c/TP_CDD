"""Calendario reconstruido con versión fija y ajustes oficiales documentados."""

import csv
import io
import json
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .air_source import parse_date
from .open_meteo import atomic_bytes, digest, write_json

VERSION = "0.105"
BASE = "https://pypi.org/project/holidays/0.105/"
SOURCES = {
    "jueves_historico": "https://biblioteca.afip.gob.ar/dcp/LEY_C_021329_1976_06_09",
    "ley_actual": "https://www.argentina.gob.ar/normativa/nacional/ley-27399-281835/texto",
    "puentes_2018_2019": "https://www.argentina.gob.ar/normativa/nacional/decreto-923-2017-287145/texto",
    "puentes_2025": "https://www.argentina.gob.ar/normativa/nacional/decreto-1027-2024-406417/texto",
    "puentes_2026": "https://www.argentina.gob.ar/normativa/nacional/norma-421799/texto",
    "extraordinario_2022": "https://www.argentina.gob.ar/normativa/nacional/decreto-573-2022-370795/texto",
    "traslado_2025": "https://www.argentina.gob.ar/normativa/nacional/resoluci%C3%B3n-139-2025-417061/texto",
}
NONWORKING_BRIDGES = {
    "2018-04-30": "puentes_2018_2019", "2018-12-24": "puentes_2018_2019", "2018-12-31": "puentes_2018_2019",
    "2019-07-08": "puentes_2018_2019", "2019-08-19": "puentes_2018_2019", "2019-10-14": "puentes_2018_2019",
    "2025-05-02": "puentes_2025", "2025-08-15": "puentes_2025", "2025-11-21": "puentes_2025",
    "2026-03-23": "puentes_2026", "2026-07-10": "puentes_2026", "2026-12-07": "puentes_2026",
}
WEEKDAYS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")


def build_events(years):
    try:
        import holidays
        from dateutil.easter import easter
    except ImportError as exc:
        raise ValueError("Instalá requirements-calendar.txt en .venv para generar el calendario.") from exc
    if holidays.__version__ != VERSION:
        raise ValueError(f"Se requiere holidays=={VERSION} para reproducir este calendario.")
    years = list(years)
    if not years or min(years) < 2009 or max(years) > 2026:
        raise ValueError("Calendario revisado para 2009–2026. Revisar fuentes antes de ampliar años.")
    observed = holidays.AR(years=years, observed=True, language="es", categories="public", expand=False)
    nominal = holidays.AR(years=years, observed=False, language="es", categories="public", expand=False)
    baseline = [{"fecha": d.isoformat(), "nombres": observed.get_list(d)} for d in sorted(observed)]
    originals = {(d.year, name): d.isoformat() for d in nominal for name in nominal.get_list(d)}
    events = []
    for item in baseline:
        day = item["fecha"]
        for name in item["nombres"]:
            original_name = name
            kind, subtype, source, review = "feriado_nacional", "general", BASE, "reconstruido_biblioteca"
            if name == "Jueves Santo":
                kind, subtype, source, review = "dia_no_laborable", "jueves_santo", SOURCES["ley_actual"], "clasificacion_contrastada"
            elif "fines turísticos" in name:
                subtype = "turistico"
                if day in NONWORKING_BRIDGES:
                    kind, source, review = "dia_no_laborable", SOURCES[NONWORKING_BRIDGES[day]], "fecha_y_tipo_contrastados"
                    name = "Día no laborable con fines turísticos"
            original = originals.get((date.fromisoformat(day).year, original_name), day)
            # Corrección de un traslado excepcional ausente en la versión fijada.
            if day == "2025-10-12" and "Diversidad Cultural" in name:
                day_actual, original = "2025-10-10", "2025-10-12"
                source, review = SOURCES["traslado_2025"], "fecha_y_tipo_contrastados"
            else:
                day_actual = day
            events.append({"fecha": day_actual, "nombre": name, "tipo": kind, "subtipo": subtype,
                           "fecha_original": original, "es_trasladado": int(original != day_actual),
                           "ambito": "nacional", "fuente": source, "verificacion": review})
    # Jueves Santo existía como no laborable antes de 2011 y no está en PUBLIC.
    for year in years:
        thursday = (easter(year) - timedelta(days=3)).isoformat()
        if not any(e["fecha"] == thursday and e["nombre"] == "Jueves Santo" for e in events):
            events.append({"fecha": thursday, "nombre": "Jueves Santo", "tipo": "dia_no_laborable", "subtipo": "jueves_santo",
                           "fecha_original": thursday, "es_trasladado": 0, "ambito": "nacional",
                           "fuente": SOURCES["jueves_historico"], "verificacion": "regla_contrastada_fecha_calculada"})
    if 2022 in years:
        events.append({"fecha": "2022-09-02", "nombre": "Feriado extraordinario - Decreto 573/2022",
                       "tipo": "feriado_nacional", "subtipo": "extraordinario", "fecha_original": "2022-09-02",
                       "es_trasladado": 0, "ambito": "nacional", "fuente": SOURCES["extraordinario_2022"],
                       "verificacion": "fecha_y_tipo_contrastados"})
    return baseline, sorted(events, key=lambda e: (e["fecha"], e["tipo"], e["nombre"]))


def daily_rows(first, last, events, source_dates):
    grouped = defaultdict(list)
    for event in events:
        grouped[event["fecha"]].append(event)
    day = first
    while day <= last:
        items = grouped[day.isoformat()]
        public = [e for e in items if e["tipo"] == "feriado_nacional"]
        optional = [e for e in items if e["tipo"] == "dia_no_laborable"]
        yield {"FECHA": day.isoformat(), "anio": day.year, "mes": day.month,
               "dia_semana": day.isoweekday(), "nombre_dia": WEEKDAYS[day.weekday()],
               "es_fin_de_semana": int(day.weekday() >= 5), "es_feriado_nacional": int(bool(public)),
               "es_dia_no_laborable": int(bool(optional)), "es_turistico": int(any(e["subtipo"] == "turistico" for e in items)),
               "nombres_feriados": " | ".join(e["nombre"] for e in public),
               "nombres_no_laborables": " | ".join(e["nombre"] for e in optional),
               "presente_en_csv_aire": int(day in source_dates),
               "fuentes_eventos": " | ".join(sorted({e["fuente"] for e in items})),
               "metodo": "reconstruccion_holidays_0.105_con_ajustes_documentados"}
        day += timedelta(days=1)


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def export_calendar(config, config_path, root):
    if "air_source" not in config:
        raise ValueError("Primero ejecutá import-air.")
    source = config_path.parent / config["air_source"]["path"]
    with source.open(encoding="utf-8-sig", newline="") as stream:
        source_dates = {parse_date(row["FECHA"]) for row in csv.DictReader(stream)}
    first, last = min(source_dates), max(source_dates)
    baseline, full_events = build_events(range(first.year, last.year + 1))
    # Se conserva la salida de la biblioteca aparte; NO se presenta como API oficial.
    generated = Path(root) / "derived/calendar/holidays_0.105"
    for year in range(first.year, last.year + 1):
        path = generated / f"{year}.json"
        data = {"provider": "python-holidays", "version": VERSION, "year": year,
                "observed": True, "category": "PUBLIC", "source": BASE,
                "events_before_adjustments": [e for e in baseline if e["fecha"].startswith(str(year))]}
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != data:
            raise ValueError("La salida de la biblioteca cambió pese a conservar la versión. Revisar antes de exportar.")
        write_json(path, data)
    events = [e for e in full_events if first.isoformat() <= e["fecha"] <= last.isoformat()]
    rows = list(daily_rows(first, last, events, source_dates))
    if len(rows) != (last-first).days + 1 or len({r["FECHA"] for r in rows}) != len(rows):
        raise ValueError("Calendario incompleto o con fechas repetidas.")
    outputs = {}
    for name, content in (("calendario.csv", rows), ("feriados.csv", events)):
        target = Path(root) / "exports" / name
        body = csv_bytes(content)
        atomic_bytes(target, body)
        with target.open(encoding="utf-8-sig", newline="") as stream:
            if len(list(csv.DictReader(stream))) != len(content):
                raise ValueError("Conteo incorrecto después de exportar.")
        outputs[name] = {"path": str(target.resolve()), "sha256": digest(body), "rows": len(content)}
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "source_csv": config["air_source"],
              "start_date": first.isoformat(), "end_date": last.isoformat(), "outputs": outputs,
              "event_types": dict(Counter(e["tipo"] for e in events)),
              "days_public_holiday": sum(r["es_feriado_nacional"] for r in rows),
              "days_non_working": sum(r["es_dia_no_laborable"] for r in rows),
              "method": "Reconstrucción local, no descarga de un calendario oficial completo. Revisión focalizada de clasificaciones y excepciones.",
              "sources": {"library": BASE, **SOURCES},
              "excluded": ["Feriados provinciales/locales, incluido G20 de CABA", "Asuetos administrativos y bancarios", "Días restringidos a comunidades religiosas o sectores laborales"],
              "future_information": "Calendario retrospectivo: no prueba disponibilidad del dato antes de cada predicción. Excepcionales requieren fecha de anuncio para evaluación prospectiva.",
              "time_policy": "Una fila por fecha civil. No modifica las etiquetas horarias 00/24 ni une datos meteorológicos."}
    write_json(Path(root) / "exports/calendario.meta.json", report)
    return report
