import unittest
from datetime import date

from aire_caba.calendar_source import build_events, daily_rows


class CalendarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.events = build_events(range(2009, 2027))

    def on(self, day):
        return [e for e in self.events if e["fecha"] == day]

    def test_historical_carnival_not_projected_backwards(self):
        self.assertFalse(any("Carnaval" in e["nombre"] for e in self.on("2010-02-15")))
        self.assertTrue(any("Carnaval" in e["nombre"] for e in self.on("2011-03-07")))

    def test_thursday_overlap_preserves_both_categories(self):
        rows = list(daily_rows(date(2016,3,24), date(2016,3,24), self.events, {date(2016,3,24)}))
        self.assertEqual(rows[0]["es_feriado_nacional"], 1)
        self.assertEqual(rows[0]["es_dia_no_laborable"], 1)
        self.assertEqual(self.on("2010-04-01")[0]["tipo"], "dia_no_laborable")

    def test_tourist_categories_change_with_year(self):
        self.assertEqual(self.on("2024-04-01")[0]["tipo"], "feriado_nacional")
        for day in ("2018-04-30", "2019-07-08", "2025-05-02", "2026-03-23"):
            self.assertEqual(self.on(day)[0]["tipo"], "dia_no_laborable")

    def test_exceptional_2022_and_2025_shift(self):
        self.assertEqual(len(self.on("2022-09-02")), 1)
        self.assertEqual(self.on("2022-09-02")[0]["tipo"], "feriado_nacional")
        self.assertFalse(self.on("2025-10-12"))
        self.assertEqual(self.on("2025-10-10")[0]["fecha_original"], "2025-10-12")

    def test_leap_day_continuous_calendar_not_only_source_dates(self):
        rows = list(daily_rows(date(2012,2,28), date(2012,3,1), self.events, {date(2012,2,28)}))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1]["FECHA"], "2012-02-29")
        self.assertEqual(rows[1]["presente_en_csv_aire"], 0)

    def test_no_duplicate_events_and_no_local_or_religious_events(self):
        keys = [(e["fecha"], e["nombre"], e["tipo"]) for e in self.events]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertFalse(self.on("2018-11-30"))
        self.assertFalse(any("bancario" in e["nombre"].lower() for e in self.events))
