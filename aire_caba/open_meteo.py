"""Cliente de archivo histórico; sólo biblioteca estándar de Python."""

import hashlib
import json
import math
import re
import sqlite3
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from . import __version__

ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"
VARIABLES = {
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "wind_speed_10m", "wind_direction_10m", "precipitation",
    "cloud_cover", "shortwave_radiation",
}
# Ventanas móviles y margen respecto de 600/min, 5000/h, 10000/día, 300000/mes.
LIMITS = ((60, 450), (3600, 4000), (86400, 8000), (31 * 86400, 240000))


def atomic_bytes(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(content)
    temporary.replace(path)


def write_json(path, value):
    atomic_bytes(path, json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8"))


def digest(content):
    return hashlib.sha256(content).hexdigest()


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config["model"] != "era5" or config["timezone"] != "America/Argentina/Buenos_Aires":
        raise ValueError("Esta versión valida exclusivamente ERA5 y la zona horaria de CABA.")
    fields = config["hourly"]
    if not fields or len(set(fields)) != len(fields) or not set(fields) <= VARIABLES:
        raise ValueError("Variables vacías, repetidas o no soportadas.")
    ids = set()
    for station in config["stations"]:
        sid = station["id"]
        if not re.fullmatch(r"[a-z][a-z0-9_]*", sid) or sid in ids:
            raise ValueError("Identificador de estación inválido o repetido.")
        ids.add(sid)
        if not (-90 <= station["latitude"] <= 90 and -180 <= station["longitude"] <= 180):
            raise ValueError("Coordenadas inválidas.")
    if not ids:
        raise ValueError("Faltan estaciones.")
    return config


def jobs(config, start=None, end=None, stations=None):
    first = date.fromisoformat(start or config["start_date"])
    last = date.fromisoformat(end or config["end_date"])
    if first < date(2009, 10, 1) or last < first:
        raise ValueError("Rango inválido: se admite desde 2009-10-01 y fin >= inicio.")
    known = {s["id"] for s in config["stations"]}
    if stations and not set(stations) <= known:
        raise ValueError("Estación desconocida: " + str(set(stations) - known))
    for station in config["stations"]:
        if stations and station["id"] not in stations:
            continue
        if "station_periods" in config:
            periods = config["station_periods"].get(station["id"], [])
        else:
            periods = [{"start_date": date(year, 1, 1).isoformat(), "end_date": date(year, 12, 31).isoformat()}
                       for year in range(first.year, last.year + 1)]
        for period in periods:
            a = max(first, date.fromisoformat(period["start_date"]))
            b = min(last, date.fromisoformat(period["end_date"]))
            if a > b:
                continue
            year = a.year
            params = {
                "latitude": station["latitude"], "longitude": station["longitude"],
                "start_date": a.isoformat(), "end_date": b.isoformat(),
                "hourly": ",".join(config["hourly"]), "models": config["model"],
                "timezone": config["timezone"], "timeformat": "iso8601",
                "temperature_unit": "celsius", "wind_speed_unit": "ms",
                "precipitation_unit": "mm", "cell_selection": "land",
            }
            fingerprint = digest(json.dumps(params, sort_keys=True).encode())[:16]
            yield {
                "station": station, "params": params,
                "key": f"{station['id']}/{year}/{a}_{b}_{fingerprint}",
                # Estimación conservadora, redondeada y con margen del 10%.
                "cost": math.ceil(max(1, (b-a).days / 14 + 1/14)
                                  * max(1, len(config["hourly"]) / 10) * 1.10),
            }


def validate(payload, params):
    if payload.get("error"):
        raise ValueError(str(payload.get("reason", "Error de API")))
    for key in ("latitude", "longitude", "elevation", "timezone", "utc_offset_seconds", "hourly_units"):
        if key not in payload:
            raise ValueError(f"Respuesta sin {key}")
    if payload["timezone"] != params["timezone"] or payload["utc_offset_seconds"] != -10800:
        raise ValueError("Zona horaria inesperada.")
    hourly = payload["hourly"]
    start = datetime.fromisoformat(params["start_date"])
    finish = datetime.fromisoformat(params["end_date"]) + timedelta(days=1)
    count = int((finish - start).total_seconds() / 3600)
    expected = [(start + timedelta(hours=i)).isoformat(timespec="minutes") for i in range(count)]
    if hourly.get("time") != expected:
        raise ValueError("Serie horaria incompleta, duplicada o fuera del intervalo solicitado.")
    nulls = {}
    for field in params["hourly"].split(","):
        values = hourly.get(field)
        if not isinstance(values, list) or len(values) != count or field not in payload["hourly_units"]:
            raise ValueError(f"Variable incompleta: {field}")
        if any(v is not None and (type(v) not in (int, float) or not math.isfinite(v)) for v in values):
            raise ValueError(f"Valor no numérico: {field}")
        nulls[field] = values.count(None)
    return {"hours": count, "null_counts": nulls}


def paths(root, job):
    base = Path(root) / "raw" / "open_meteo" / job["key"]
    return base.with_suffix(".json"), base.with_suffix(".meta.json")


def cached(root, job):
    raw, meta = paths(root, job)
    if not meta.exists():
        return None
    metadata = json.loads(meta.read_text(encoding="utf-8"))
    if metadata["parameters"] != job["params"] or digest(raw.read_bytes()) != metadata["sha256"]:
        raise ValueError(f"Caché alterada: {raw}. Revisar antes de volver a descargar.")
    validate(json.loads(raw.read_bytes()), job["params"])
    return metadata


def resolve_cached_job(root, job):
    """Reutiliza un bloque más amplio sólo si todos los demás parámetros coinciden."""
    if cached(root, job):
        return job
    def signature(params):
        return {k: v for k, v in params.items() if k not in ("start_date", "end_date")}
    candidates = []
    base = Path(root) / "raw/open_meteo"
    for path in (base / job["station"]["id"]).glob("*/*.meta.json"):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        params = metadata["parameters"]
        if signature(params) != signature(job["params"]):
            continue
        if params["start_date"] <= job["params"]["start_date"] and params["end_date"] >= job["params"]["end_date"]:
            candidates.append((metadata["validation"]["hours"], path, params))
    for _, path, params in sorted(candidates, key=lambda item: (item[0], str(item[1]))):
        candidate = {**job, "params": params, "key": path.relative_to(base).as_posix().removesuffix(".meta.json")}
        if cached(root, candidate):
            return candidate
    return job


class BudgetExceeded(RuntimeError):
    def __init__(self, message, retry_seconds=None):
        super().__init__(message)
        self.retry_seconds = retry_seconds


class Ledger:
    """Reserva cada intento antes de la red; persiste entre ejecuciones."""

    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.execute("CREATE TABLE IF NOT EXISTS attempts (at REAL, cost REAL, url TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS cooldown (until REAL)")
        self.db.commit()

    def reserve(self, cost, url, now=None):
        now = time.time() if now is None else now
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            until = self.db.execute("SELECT MAX(until) FROM cooldown").fetchone()[0] or 0
            if now < until:
                raise BudgetExceeded(f"Pausa por HTTP 429 hasta {datetime.fromtimestamp(until, timezone.utc).isoformat()}")
            for window, limit in LIMITS:
                used = self.db.execute("SELECT COALESCE(SUM(cost),0) FROM attempts WHERE at > ?", (now-window,)).fetchone()[0]
                if used + cost > limit:
                    retry_seconds = None
                    if window == 60 and cost <= limit:
                        oldest = self.db.execute("SELECT MIN(at) FROM attempts WHERE at > ?", (now-window,)).fetchone()[0]
                        retry_seconds = min(60, max(1, oldest + window - now + 0.1))
                    raise BudgetExceeded(f"Presupuesto local agotado: {used}+{cost} > {limit} en {window}s.", retry_seconds)
            self.db.execute("INSERT INTO attempts VALUES (?, ?, ?)", (now, cost, url))

    def pause(self, seconds):
        with self.db:
            self.db.execute("INSERT INTO cooldown VALUES (?)", (time.time() + seconds,))

    def usage(self):
        now = time.time()
        return {str(window): self.db.execute("SELECT COALESCE(SUM(cost),0) FROM attempts WHERE at > ?", (now-window,)).fetchone()[0] for window, _ in LIMITS}


@contextmanager
def download_lock(root):
    """Bloqueo del SO: se libera incluso al matar el proceso."""
    path = Path(root) / "state" / "download.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        import os
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Ya hay otra descarga usando esta carpeta de datos.") from exc
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("Ya hay otra descarga usando esta carpeta de datos.") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def retry_delay(value):
    from email.utils import parsedate_to_datetime
    try:
        return max(60, float(value))
    except (TypeError, ValueError):
        try:
            return max(60, parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError, AttributeError):
            return 86400


def fetch(root, config, job, ledger, opener=urlopen, sleep=time.sleep):
    previous = cached(root, job)
    if previous:
        return "cached", previous
    url = ENDPOINT + "?" + urlencode(job["params"])
    raw, meta = paths(root, job)
    for attempt in range(4):
        while True:
            try:
                ledger.reserve(job["cost"], url)
                break
            except BudgetExceeded as exc:
                if exc.retry_seconds is None:
                    raise
                print(f"Pausa de cuota por minuto: {exc.retry_seconds:.1f}s", flush=True)
                sleep(exc.retry_seconds)
        try:
            with opener(Request(url, headers={"User-Agent": "caba-academic-etl/" + __version__}), timeout=90) as response:
                content = response.read()
                headers = dict(response.headers)
            # Guardar respuesta intacta ANTES de validar. Metadatos marcan completitud.
            atomic_bytes(raw, content)
            quality = validate(json.loads(content), job["params"])
            payload = json.loads(content)
            metadata = {
                "source": "Open-Meteo / Copernicus ERA5", "license": "CC BY 4.0",
                "extractor_version": __version__, "url": url,
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "parameters": job["params"], "station": job["station"],
                "stations_source": config["stations_source"],
                "stations_verified_on": config["stations_verified_on"],
                "response_headers": headers, "sha256": digest(content),
                "estimated_quota_per_attempt": job["cost"],
                "returned": {k: payload[k] for k in ("latitude", "longitude", "elevation", "timezone", "utc_offset_seconds", "hourly_units")},
                "validation": quality,
            }
            write_json(meta, metadata)
            return "downloaded", metadata
        except HTTPError as exc:
            error = raw.with_suffix(f".http{exc.code}.{time.time_ns()}.txt")
            atomic_bytes(error, exc.read())
            if exc.code == 429:
                ledger.pause(retry_delay(exc.headers.get("Retry-After")))
                raise BudgetExceeded("Open-Meteo devolvió 429. Pausa persistida; no se seguirá consultando.") from exc
            if exc.code < 500 or attempt == 3:
                raise
            # Respetar Retry-After en fallas transitorias; no esperar indefinidamente.
            delay = retry_delay(exc.headers["Retry-After"]) if "Retry-After" in exc.headers else 2 ** (attempt + 1)
            if delay > 60:
                ledger.pause(delay)
                raise BudgetExceeded("Servidor pidió una pausa prolongada. Reejecutar más tarde.") from exc
            sleep(delay)
        except (URLError, TimeoutError, ConnectionError):
            if attempt == 3:
                raise
            sleep(2 ** (attempt + 1))
    raise RuntimeError("Descarga sin resultado")
