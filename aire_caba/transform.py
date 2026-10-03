"""Transformación de entrada_transformacion.csv al dataset objetivo estación-hora."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

STATIONS = ("CENTENARIO", "CORDOBA", "LA_BOCA")
EXCLUDED_STATIONS = {"PALERMO": "Sólo CO entre 10/2009 y 09/2010 (4% de las filas); NO2 y PM10 vacíos; desactivada según el catálogo oficial."}
POLLUTANTS = {"CO": "ppm", "NO2": "ppb", "PM10": "µg/m³"}
# Rango físicamente plausible para promedios horarios urbanos; fuera de él se considera error de registro.
VALID_RANGE = {"CO": (0.0, 10.0), "NO2": (0.0, 400.0), "PM10": (0.0, 600.0)}
ZERO_INVALID = {"PM10"}
METEO_RANGE = {
    "temperature_2m": (-15.0, 50.0),
    "relative_humidity_2m": (0.0, 100.0),
    "surface_pressure": (950.0, 1060.0),
    "wind_speed_10m": (0.0, 60.0),
    "wind_direction_10m": (0.0, 360.0),
    "precipitation": (0.0, 200.0),
    "cloud_cover": (0.0, 100.0),
    "shortwave_radiation": (0.0, 1400.0),
}
CALENDAR = ("es_fin_de_semana", "es_feriado_nacional", "es_dia_no_laborable")
MISSING_MARKERS = {"s/d", "x/d"}
MAX_GAP_HOURS = 3
FLATLINE_HOURS = 24
SEASONS = {12: "verano", 1: "verano", 2: "verano", 3: "otono", 4: "otono", 5: "otono",
           6: "invierno", 7: "invierno", 8: "invierno", 9: "primavera", 10: "primavera", 11: "primavera"}


def classify(raw):
    """Devuelve (valor numérico, estado) para cada celda de contaminante tal como figura en el CSV."""
    text = raw.fillna("").astype(str).str.strip()
    value = pd.to_numeric(text, errors="coerce").astype(float)
    state = pd.Series("error", index=text.index, dtype=object)
    state[value.notna()] = "medido"
    state[text == ""] = "vacio"
    state[text.str.lower().isin(MISSING_MARKERS)] = "sin_dato"
    censored = text.str.fullmatch(r"<\s*\d+(\.\d+)?")
    limit = pd.to_numeric(text.str.lstrip("<").str.strip(), errors="coerce")
    state[censored] = "censurado"
    value = value.where(~censored, limit / 2)
    return value, state


def flatline(value, state, hours=FLATLINE_HOURS):
    """Marca mediciones idénticas durante `hours` horas consecutivas o más (sensor trabado)."""
    measured = value.where(state == "medido")
    run = (measured.ne(measured.shift()) | measured.isna()).cumsum()
    size = measured.groupby(run).transform("size")
    return measured.notna() & (size >= hours)


def fill_short_gaps(value, max_hours=MAX_GAP_HOURS):
    """Interpola linealmente sólo huecos internos de hasta `max_hours` horas en una grilla horaria completa."""
    missing = value.isna()
    gap_length = missing.groupby((~missing).cumsum()).transform("sum")
    inside = value.ffill().notna() & value.bfill().notna()
    fill = missing & inside & (gap_length <= max_hours)
    return value.where(~fill, value.interpolate(limit_area="inside")), fill


def build_target(entrada):
    report = {"filas_entrada": len(entrada)}
    fecha = pd.to_datetime(entrada["FECHA"], format="%d%b%Y:%H:%M:%S")
    hora = pd.to_numeric(entrada["HORA"], errors="coerce")
    valid_hour = hora.between(1, 24)
    pollutant_cols = [f"{p}_{s}" for s in STATIONS + tuple(EXCLUDED_STATIONS) for p in POLLUTANTS]
    invalid_with_data = sum((classify(entrada.loc[~valid_hour, c])[0].notna()).sum() for c in pollutant_cols)
    if invalid_with_data:
        raise ValueError(f"Hay {invalid_with_data} mediciones en filas con HORA fuera de 1-24; revisar antes de descartarlas.")
    report["filas_hora_invalida_descartadas"] = int((~valid_hour).sum())

    base = entrada[valid_hour].copy()
    # HORA 1-24 = promedio de la hora que termina en FECHA + HORA (HORA 24 cierra a las 00 del día siguiente).
    base.index = pd.DatetimeIndex(fecha[valid_hour] + pd.to_timedelta(hora[valid_hour], unit="h"), name="fecha_hora")
    duplicated = int(base.index.duplicated().sum())
    if duplicated:
        raise ValueError(f"{duplicated} instantes duplicados tras normalizar FECHA/HORA.")
    base = base.sort_index()
    grid = pd.date_range(base.index.min(), base.index.max(), freq="h", name="fecha_hora")
    report["horas_grilla"] = len(grid)
    report["horas_ausentes_en_csv"] = len(grid) - len(base)

    frames, stations_report = [], {}
    for station in STATIONS:
        df = pd.DataFrame(index=base.index)
        df["estacion"] = station.lower()
        df["fecha"] = pd.to_datetime(base["FECHA"], format="%d%b%Y:%H:%M:%S").dt.strftime("%Y-%m-%d")
        df["hora_original"] = pd.to_numeric(base["HORA"]).astype(int)
        for col in CALENDAR:
            df[col] = pd.to_numeric(base[f"CAL_{col}"]).astype("int8")
        df["dia_semana"] = pd.to_numeric(base["CAL_dia_semana"]).astype("int8")
        for var in METEO_RANGE:
            df[var] = pd.to_numeric(base[f"METEO_{station}_{var}"], errors="coerce")
        info = {}
        for pol in POLLUTANTS:
            value, state = classify(base[f"{pol}_{station}"])
            df[pol], df[f"{pol}_estado"] = value, state
        df = df.reindex(grid)
        df["en_csv"] = df["estacion"].notna()

        for pol in POLLUTANTS:
            value, state = df[pol], df[f"{pol}_estado"].fillna("hora_ausente")
            counts = {"inicial": state[df["en_csv"]].value_counts().to_dict()}
            lo, hi = VALID_RANGE[pol]
            out = value.notna() & ((value < lo) | (value > hi) | ((value == 0) & (pol in ZERO_INVALID)))
            state[out] = "fuera_rango"
            stuck = flatline(value.where(~out), state)
            state[stuck] = "congelado"
            value = value.where(~(out | stuck))
            value, filled = fill_short_gaps(value)
            state[filled] = "interpolado"
            counts.update(fuera_rango=int(out.sum()), congelado=int(stuck.sum()), interpolado=int((filled & df["en_csv"]).sum()))
            df[pol], df[f"{pol}_estado"] = value, state
            info[pol] = counts

        meteo_missing = {}
        for var, (lo, hi) in METEO_RANGE.items():
            value = df[var]
            out = value.notna() & ((value < lo) | (value > hi))
            filled_value, filled = fill_short_gaps(value.where(~out))
            df[var] = filled_value
            meteo_missing[var] = {"fuera_rango": int(out.sum()), "interpolado": int((filled & df["en_csv"]).sum())}

        has_pollutant = df[list(POLLUTANTS)].notna().any(axis=1)
        keep = df["en_csv"] & has_pollutant
        no_meteo = keep & df[list(METEO_RANGE)].isna().any(axis=1)
        info["filas_csv"] = int(df["en_csv"].sum())
        info["filas_sin_contaminantes_descartadas"] = int((df["en_csv"] & ~has_pollutant).sum())
        info["filas_sin_meteorologia_descartadas"] = int(no_meteo.sum())
        info["meteorologia"] = meteo_missing
        df = df[keep & ~no_meteo]
        info["filas_finales"] = len(df)
        info["faltantes_finales"] = {p: int(df[p].isna().sum()) for p in POLLUTANTS}
        stations_report[station.lower()] = info
        frames.append(df)

    data = pd.concat(frames).reset_index()
    for col in ("hora_original", "dia_semana", *CALENDAR):
        data[col] = data[col].astype(int)
    direction = np.deg2rad(data["wind_direction_10m"])
    # Componentes del vector viento (convención meteorológica: dirección desde donde sopla).
    data["viento_u"] = (-data["wind_speed_10m"] * np.sin(direction)).round(3)
    data["viento_v"] = (-data["wind_speed_10m"] * np.cos(direction)).round(3)
    fecha_label = pd.to_datetime(data["fecha"])
    data["anio"] = fecha_label.dt.year
    data["mes"] = fecha_label.dt.month
    data["estacion_anio"] = data["mes"].map(SEASONS)
    data["hora_dia"] = data["fecha_hora"].dt.hour
    data["fecha_hora"] = data["fecha_hora"].dt.strftime("%Y-%m-%d %H:%M")
    columns = (["estacion", "fecha_hora", "fecha", "hora_original", "anio", "mes", "dia_semana", "hora_dia", "estacion_anio", *CALENDAR]
               + [c for p in POLLUTANTS for c in (p, f"{p}_estado")]
               + [v for v in METEO_RANGE if v != "wind_direction_10m"] + ["viento_u", "viento_v"])
    data = data[columns].sort_values(["estacion", "fecha_hora"], ignore_index=True).round(3)
    report["estaciones"] = stations_report
    report["estaciones_excluidas"] = EXCLUDED_STATIONS
    report["filas_salida"] = len(data)
    return data, report


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transform_files(source, output):
    source, output = Path(source), Path(output)
    entrada = pd.read_csv(source, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    data, report = build_target(entrada)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".tmp")
    data.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, output)
    meta = {
        "input": {"path": str(source), "sha256": sha256(source)},
        "output": {"path": str(output), "sha256": sha256(output), "rows": len(data), "columns": list(data.columns)},
        "unidades": POLLUTANTS,
        "decisiones": {
            "hora": "HORA 1-24 se interpreta como hora de cierre del promedio: fecha_hora = FECHA + HORA. Coincide con la unión meteorológica (HORA 24 = 00 del día siguiente).",
            "horas_invalidas": "HORA 0 y > 24 no contienen mediciones: se descartan.",
            "marcadores": "s/d, x/d, vacíos y #REF! se convierten en faltantes, con su motivo en *_estado.",
            "censurados": "Valores '<LD' se reemplazan por LD/2.",
            "rangos": {k: {"min": v[0], "max": v[1]} for k, v in VALID_RANGE.items()} | {"PM10_cero": "inválido"},
            "congelados": f"{FLATLINE_HOURS} h o más con el mismo valor medido se invalidan.",
            "imputacion": f"Interpolación lineal sólo en huecos internos de hasta {MAX_GAP_HOURS} h; huecos mayores quedan vacíos.",
            "filas": "Se conservan estación-horas con al menos un contaminante y meteorología completa.",
            "columnas": "Se descartan procedencia meteorológica, detalle de feriados y wind_direction_10m (reemplazada por viento_u/viento_v).",
        },
        "reporte": report,
    }
    output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=int), encoding="utf-8")
    return meta


def transform_dataset(data_dir):
    data_dir = Path(data_dir)
    return transform_files(data_dir / "exports/entrada_transformacion.csv", data_dir / "exports/dataset_objetivo.csv")
