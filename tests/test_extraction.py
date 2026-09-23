import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError

from aire_caba.open_meteo import BudgetExceeded, Ledger, cached, download_lock, fetch, jobs, load_config, paths

CONFIG = Path(__file__).resolve().parents[1] / "config/open_meteo.json"


class Response(io.BytesIO):
    headers = {"Content-Type": "application/json"}


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = load_config(CONFIG)
        self.config.pop("air_source", None)
        self.config.pop("station_periods", None)
        self.config["stations"] = [s for s in self.config["stations"] if s["id"] != "palermo"]
        self.job = next(jobs(self.config, "2021-01-01", "2021-01-02", ["centenario"]))
        self.ledger = Ledger(self.root / "state/budget.sqlite")
        self.addCleanup(self.ledger.db.close)

    def payload(self):
        fields = self.config["hourly"]
        return {
            "latitude": -34.5, "longitude": -58.5, "elevation": 18,
            "timezone": self.config["timezone"], "utc_offset_seconds": -10800,
            "hourly_units": {f: "unit" for f in fields},
            "hourly": {"time": [(datetime(2021, 1, 1) + timedelta(hours=h)).isoformat(timespec="minutes") for h in range(48)],
                       **{f: [None] + [1.0] * 47 for f in fields}},
        }

    def test_year_boundaries_leap_year_and_cost(self):
        work = list(jobs(self.config))
        self.assertEqual(len(work), 54)
        self.assertLess(sum(j["cost"] for j in work), 2000)
        leap = next(j for j in work if j["params"]["start_date"] == "2012-01-01")
        self.assertEqual(leap["params"]["end_date"], "2012-12-31")
        self.assertEqual(leap["cost"], 29)
        self.assertEqual(work[0]["params"]["start_date"], "2009-10-01")
        self.assertEqual(work[-1]["params"]["end_date"], "2026-08-24")

    def test_resume_avoids_network_and_detects_corruption(self):
        original = json.dumps(self.payload()).encode()
        fetch(self.root, self.config, self.job, self.ledger, opener=lambda *a, **k: Response(original))
        def no_network(*a, **k):
            self.fail("No debe llamar a la red al reanudar")
        state, meta = fetch(self.root, self.config, self.job, self.ledger, opener=no_network)
        self.assertEqual(state, "cached")
        self.assertEqual(meta["validation"]["null_counts"]["temperature_2m"], 1)
        raw, _ = paths(self.root, self.job)
        self.assertEqual(raw.read_bytes(), original)
        raw.write_bytes(b"{}")
        with self.assertRaises(ValueError):
            cached(self.root, self.job)

    def test_budget_persists_and_expires(self):
        self.ledger.reserve(450, "test", now=1000)
        second = Ledger(self.root / "state/budget.sqlite")
        try:
            with self.assertRaises(BudgetExceeded):
                second.reserve(1, "test", now=1059)
            second.reserve(1, "test", now=1061)
        finally:
            second.db.close()

    def test_daily_budget_independent_of_hourly_budget(self):
        for i in range(20):
            self.ledger.reserve(400, "test", now=10000 + i * 3700)
        with self.assertRaises(BudgetExceeded):
            self.ledger.reserve(1, "test", now=10000 + 20 * 3700)

    def test_429_stops_and_persists_cooldown(self):
        def limited(*a, **k):
            raise HTTPError("test", 429, "limit", {"Retry-After": "120"}, io.BytesIO(b"limit"))
        with self.assertRaises(BudgetExceeded):
            fetch(self.root, self.config, self.job, self.ledger, opener=limited)
        with self.assertRaises(BudgetExceeded):
            self.ledger.reserve(1, "test")
        self.assertEqual(self.ledger.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0], 1)

    def test_retry_charges_each_attempt(self):
        calls = []
        def flaky(*a, **k):
            calls.append(1)
            if len(calls) == 1:
                raise URLError("temporary")
            return Response(json.dumps(self.payload()).encode())
        fetch(self.root, self.config, self.job, self.ledger, opener=flaky, sleep=lambda _: None)
        self.assertEqual(self.ledger.db.execute("SELECT SUM(cost) FROM attempts").fetchone()[0], 2 * self.job["cost"])

    def test_bad_hour_axis_rejected_but_raw_preserved(self):
        payload = self.payload()
        payload["hourly"]["time"][2] = payload["hourly"]["time"][1]
        with self.assertRaises(ValueError):
            fetch(self.root, self.config, self.job, self.ledger, opener=lambda *a, **k: Response(json.dumps(payload).encode()))
        raw, meta = paths(self.root, self.job)
        self.assertTrue(raw.exists())
        self.assertFalse(meta.exists())

    def test_parameter_change_has_different_cache(self):
        self.config["hourly"] = ["temperature_2m"]
        modified = next(jobs(self.config, "2021-01-01", "2021-01-02", ["centenario"]))
        self.assertNotEqual(self.job["key"], modified["key"])

    def test_simultaneous_writer_is_blocked_then_released(self):
        with download_lock(self.root):
            with self.assertRaisesRegex(RuntimeError, "otra descarga"):
                with download_lock(self.root):
                    self.fail("No debe aceptar dos escritores")
        with download_lock(self.root):
            pass


if __name__ == "__main__":
    unittest.main()
