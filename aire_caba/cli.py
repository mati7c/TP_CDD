import argparse
import json
import sys
from datetime import date
from pathlib import Path

from .open_meteo import Ledger, cached, download_lock, fetch, jobs, load_config, resolve_cached_job, write_json
from .air_source import import_air, verify_source
from .export_weather import export_weather

PROJECT = Path(__file__).resolve().parent.parent


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extracción de meteorología horaria para CABA")
    parser.add_argument("command", choices=["import-air", "plan", "download", "status", "export-weather", "calendar"])
    parser.add_argument("--csv", type=Path, help="CSV original para import-air")
    parser.add_argument("--config", type=Path, default=PROJECT / "config/open_meteo.json")
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "data")
    parser.add_argument("--start", help="Fecha inicial inclusiva YYYY-MM-DD")
    parser.add_argument("--end", help="Fecha final inclusiva YYYY-MM-DD")
    parser.add_argument("--stations", nargs="+", help="centenario cordoba la_boca palermo")
    parser.add_argument("--max-requests", type=int, help="Máximo de bloques nuevos en esta ejecución")
    args = parser.parse_args(argv)
    try:
        if args.max_requests is not None and args.max_requests < 1:
            raise ValueError("--max-requests debe ser positivo")
        if args.command == "import-air":
            if not args.csv:
                raise ValueError("import-air requiere --csv RUTA")
            with download_lock(args.data_dir):
                report = import_air(args.csv, args.config, args.data_dir)
            overview = {k: report[k] for k in ("sha256", "rows", "date_min", "date_max", "rows_hour_zero", "rows_invalid_hour")}
            overview["stations"] = {sid: {k: v for k, v in info.items() if k != "extraction_periods"} for sid, info in report["stations"].items()}
            print(json.dumps(overview, indent=2))
            return 0
        config = load_config(args.config)
        verify_source(config, args.config)
        if args.command == "calendar":
            from .calendar_source import export_calendar
            if args.start or args.end or args.stations or args.max_requests or args.csv:
                raise ValueError("calendar usa el período completo del CSV importado.")
            with download_lock(args.data_dir):
                report = export_calendar(config, args.config, args.data_dir)
            print(json.dumps({k: report[k] for k in ("start_date", "end_date", "outputs", "event_types", "days_public_holiday", "days_non_working")}, indent=2))
            return 0
        if args.command == "export-weather":
            if args.start or args.end or args.stations or args.max_requests or args.csv:
                raise ValueError("export-weather usa toda la selección importada; no admite filtros adicionales.")
            with download_lock(args.data_dir):
                report = export_weather(config, args.data_dir)
            print(json.dumps({k: report[k] for k in ("output", "counts", "rows_per_station", "null_counts")}, indent=2))
            return 0
        requested = list(jobs(config, args.start, args.end, args.stations))
        if not requested:
            raise ValueError("No hay mediciones seleccionadas para ese rango/estación.")
        planned = [resolve_cached_job(args.data_dir, j) for j in requested]
        completed = [(j, cached(args.data_dir, j)) for j in planned]
        pending = [j for j, m in completed if not m]
        summary = {
            "blocks": len(planned), "complete": len(planned)-len(pending),
            "air_source_sha256": config.get("air_source", {}).get("sha256"),
            "pending": len(pending), "http_requests_without_retries": len(pending),
            "estimated_pending_quota_with_margin": sum(j["cost"] for j in pending),
            "validated_cached_station_hours": sum(m["validation"]["hours"] for j, m in completed if m),
            "requested_envelope_station_hours": sum((date.fromisoformat(j["params"]["end_date"]) - date.fromisoformat(j["params"]["start_date"])).days * 24 + 24 for j in requested),
            "null_counts": {
                field: sum(m["validation"]["null_counts"][field] for j, m in completed if m)
                for field in config["hourly"]
            },
        }
        print(json.dumps(summary, indent=2), flush=True)
        if args.command in ("plan", "status"):
            if args.command == "status":
                state = args.data_dir / "state/open_meteo.sqlite"
                if state.exists():
                    ledger = Ledger(state)
                    print("Cuota estimada local por ventana (segundos):", ledger.usage())
                    ledger.db.close()
            return 0
        with download_lock(args.data_dir):
            ledger = Ledger(args.data_dir / "state/open_meteo.sqlite")
            try:
                for job in pending[:args.max_requests]:
                    state, metadata = fetch(args.data_dir, config, job, ledger)
                    print(f"{state}: {job['key']} | {metadata['validation']['hours']} horas", flush=True)
            finally:
                ledger.db.close()
            entries = [{"key": j["key"], "requested_parameters": request["params"],
                        "air_source": config.get("air_source"), "metadata": cached(args.data_dir, j)}
                       for request, j in zip(requested, planned)]
            write_json(args.data_dir / "reports/open_meteo_latest.json", entries)
        return 0
    except KeyboardInterrupt:
        print("Interrumpido. Volvé a ejecutar el mismo comando para continuar.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
