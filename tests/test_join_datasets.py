import csv
import tempfile
import unittest
from pathlib import Path

from aire_caba.join_datasets import join_files


class JoinTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.air = self.write("air.csv", [
            ["FECHA", "HORA", "CO_CENTENARIO"],
            ["01JAN2021:00:00:00", "24", "<0.05"],
            ["01JAN2021:00:00:00", "0", "s/d"],
            ["01JAN2021:00:00:00", "96", "#REF!"],
            ["01JAN2021:00:00:00", "24", "0"],
        ])
        self.weather = self.write("weather.csv", [["estacion", "FECHA", "HORA", "temperature_2m"],
            ["centenario", "2021-01-01", "00", "12.00"], ["centenario", "2021-01-01", "24", "16.0"]])
        self.calendar = self.write("calendar.csv", [["FECHA", "es_feriado_nacional", "metodo"], ["2021-01-01", "1"]])
        self.holidays = self.write("holidays.csv", [["fecha", "nombre"], ["2021-01-01", "Evento A"], ["2021-01-01", "Evento B"]])
        self.output = self.root / "joined.csv"

    def write(self, name, rows):
        path = self.root / name
        with path.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)
        return path

    def join(self):
        return join_files(self.air, self.weather, self.calendar, self.holidays, self.output)

    def test_preserve_rows_duplicates_tokens_and_distinct_hours(self):
        before = [p.read_bytes() for p in (self.air, self.weather, self.calendar, self.holidays)]
        report = self.join()
        with self.output.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 4)
        self.assertEqual([r["CO_CENTENARIO"] for r in rows], ["<0.05", "s/d", "#REF!", "0"])
        self.assertEqual([r["HORA"] for r in rows], ["24", "0", "96", "24"])
        self.assertEqual([r["METEO_CENTENARIO_temperature_2m"] for r in rows], ["16.0", "12.00", "", "16.0"])
        self.assertEqual(rows[0]["FERIADO_2_nombre"], "Evento B")
        self.assertEqual(rows[0]["CAL_metodo"], "")
        self.assertEqual(report["weather_unmatched"]["centenario"], 1)
        self.assertEqual(len(report["input_warnings"]), 1)
        self.assertEqual(before, [p.read_bytes() for p in (self.air, self.weather, self.calendar, self.holidays)])

    def test_duplicate_weather_rejected_before_overwriting_output(self):
        self.output.write_text("previous", encoding="utf-8")
        with self.weather.open("a", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerow(["centenario", "2021-01-01", "0", "12"])
        with self.assertRaisesRegex(ValueError, "Clave duplicada"):
            self.join()
        self.assertEqual(self.output.read_text(), "previous")
