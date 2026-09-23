import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from aire_caba.air_source import PALERMO, audit, import_air, parse_date, verify_source
from aire_caba.open_meteo import Ledger, fetch, jobs, load_config, resolve_cached_job
from test_extraction import CONFIG, Response


def content(rows, station="PALERMO"):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["FECHA", "HORA", "CO_" + station, "NO2_" + station, "PM10_" + station])
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


class AirSourceTests(unittest.TestCase):
    def test_locale_independent_date(self):
        self.assertEqual(parse_date("29FEB2012:00:00:00").isoformat(), "2012-02-29")
        with self.assertRaises(ValueError):
            parse_date("29FEB2013:00:00:00")

    def test_unsorted_censored_zero_and_invalid_hours_preserved(self):
        raw = content([
            ["19SEP2010:00:00:00", "12", "<0.05", "", ""],
            ["01OCT2009:00:00:00", "1", "0", "s/d", "#REF!"],
            ["01JAN2026:00:00:00", "96", "", "", ""],
        ])
        report, periods = audit(raw, [PALERMO])
        self.assertEqual(report["rows"], 3)
        self.assertEqual(report["stations"]["palermo"]["rows_with_measurements"], 2)
        self.assertEqual(report["rows_invalid_hour"], 1)
        self.assertEqual(report["column_counts"]["CO_PALERMO"]["censored"], 1)
        self.assertEqual(periods["palermo"][-1]["end_date"], "2010-09-19")
        self.assertEqual(len(periods["palermo"]), 2)

    def test_hour24_adds_buffer_across_year_without_interpreting_timestamp(self):
        report, periods = audit(content([["31DEC2009:00:00:00", "24", "1", "", ""]]), [PALERMO])
        self.assertEqual(periods["palermo"][-1], {"start_date": "2010-01-01", "end_date": "2010-01-01"})
        self.assertEqual(report["time_join_status"], "pending_source_hour_convention")

    def test_import_exact_bytes_and_no_later_palermo_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = root / "config.json"
            cfg.write_bytes(CONFIG.read_bytes())
            src = root / "source.csv"
            raw = content([["01OCT2009:00:00:00", "1", "2.3", "", ""], ["19SEP2010:00:00:00", "1", "1", "", ""]])
            src.write_bytes(raw)
            import_air(src, cfg, root / "data")
            config = load_config(cfg)
            verify_source(config, cfg)
            self.assertEqual(len(list(jobs(config))), 2)
            self.assertEqual(list(jobs(config, "2021-01-01", "2021-12-31")), [])
            snapshot = cfg.parent / config["air_source"]["path"]
            self.assertEqual(snapshot.read_bytes(), raw)
            snapshot.write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                verify_source(config, cfg)

    def test_unknown_station_stops_before_download(self):
        with self.assertRaisesRegex(ValueError, "coordenadas"):
            audit(content([["01OCT2009:00:00:00", "1", "2", "", ""]], "NUEVA"), [PALERMO])

    def test_superset_cache_requires_same_model_variables_and_location(self):
        from datetime import datetime, timedelta
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_config(CONFIG)
            config.pop("station_periods", None)
            broad = next(jobs(config, "2021-01-01", "2021-01-02", ["centenario"]))
            narrow = next(jobs(config, "2021-01-01", "2021-01-01", ["centenario"]))
            payload = {"latitude": -34.5, "longitude": -58.5, "elevation": 18,
                       "timezone": config["timezone"], "utc_offset_seconds": -10800,
                       "hourly_units": {v: "u" for v in config["hourly"]},
                       "hourly": {"time": [(datetime(2021,1,1)+timedelta(hours=i)).isoformat(timespec="minutes") for i in range(48)],
                                  **{v: [1]*48 for v in config["hourly"]}}}
            ledger = Ledger(root / "state/quota.sqlite")
            try:
                fetch(root, config, broad, ledger, opener=lambda *a, **k: Response(json.dumps(payload).encode()))
                self.assertEqual(resolve_cached_job(root, narrow)["key"], broad["key"])
                config["hourly"] = ["temperature_2m"]
                other = next(jobs(config, "2021-01-01", "2021-01-01", ["centenario"]))
                self.assertEqual(resolve_cached_job(root, other)["key"], other["key"])
            finally:
                ledger.db.close()
