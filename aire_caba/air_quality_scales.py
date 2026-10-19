"""Escalas alternativas por contaminante, con unidades y convenciones explícitas."""

import json
from pathlib import Path

import numpy as np


DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config/indices_alternativos.json"


def load_scales(path=None):
    config = json.loads(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    for pollutant, spec in config["pollutants"].items():
        bounds = np.asarray(spec["upper_bounds"], dtype=float)
        if len(spec["classes"]) != len(bounds) + 1 or not np.all(np.diff(bounds) > 0):
            raise ValueError(f"{pollutant}: límites o clases inconsistentes")
        if not np.isfinite(bounds).all() or not np.isfinite(spec["factor"]) or spec["factor"] <= 0:
            raise ValueError(f"{pollutant}: límites o conversión inválidos")
        if spec["rounding"] != "half_up" or spec["decimals"] not in (0, 1):
            raise ValueError(f"{pollutant}: redondeo no admitido")
        if "level_to_band" in spec:
            mapping = spec["level_to_band"]
            if len(mapping) != len(spec["classes"]) or not np.isin(mapping, range(len(spec["band_classes"]))).all():
                raise ValueError(f"{pollutant}: bandas inconsistentes")
    return config


def convert_concentration(values, spec):
    """Convierte unidades; conserva NaN y marca inválidos como NaN, sin imputación."""
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values) & (values >= 0), values * spec["factor"], np.nan)


def classify_concentration(values, spec, resolution="level"):
    """Etiquetas desde cero; -1 para faltantes/negativos/no finitos."""
    converted = convert_concentration(values, spec)
    valid = np.isfinite(converted)
    labels = np.full(converted.shape, -1, dtype=np.int8)
    precision = 10 ** spec["decimals"]
    rounded = np.floor(converted[valid] * precision + 0.5 + 1e-10) / precision
    labels[valid] = np.searchsorted(spec["upper_bounds"], rounded, side="left")
    if resolution == "band":
        if "level_to_band" not in spec:
            raise ValueError("Esta escala no define una segunda agrupación en bandas")
        labels[valid] = np.asarray(spec["level_to_band"], dtype=np.int8)[labels[valid]]
    elif resolution != "level":
        raise ValueError("La resolución debe ser level o band")
    return labels


def views(spec):
    result = [("level", spec["classes"])]
    if "band_classes" in spec:
        result.append(("band", spec["band_classes"]))
    return result
