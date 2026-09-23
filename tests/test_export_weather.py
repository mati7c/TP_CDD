import unittest

from aire_caba.export_weather import weather_rows


class ExportWeatherTests(unittest.TestCase):
    def test_24_and_next_day_00_remain_distinct_with_identical_values(self):
        index = {("palermo", f"2010-01-{day:02d}T{hour:02d}:00"): (day * 100 + hour,)
                 for day in (1, 2) for hour in range(24)}
        periods = {"palermo": [{"start_date": "2010-01-01", "end_date": "2010-01-02"}]}
        rows = list(weather_rows(index, periods, ["temperature_2m"]))
        self.assertEqual(len(rows), 50)
        end_day_one, start_day_two = rows[24:26]
        self.assertEqual(end_day_one["HORA"], "24")
        self.assertEqual(start_day_two["HORA"], "00")
        self.assertNotEqual(end_day_one["FECHA"], start_day_two["FECHA"])
        self.assertEqual(end_day_one["temperature_2m"], start_day_two["temperature_2m"])
        self.assertEqual(end_day_one["fecha_hora_fuente"], start_day_two["fecha_hora_fuente"])
        self.assertIsNone(rows[-1]["temperature_2m"])
        self.assertEqual(rows[-1]["estado"], "sin_hora_siguiente_descargada")

    def test_missing_original_hour_is_not_silently_filled(self):
        with self.assertRaisesRegex(ValueError, "Falta meteorología"):
            list(weather_rows({}, {"palermo": [{"start_date": "2010-01-01", "end_date": "2010-01-01"}]}, ["temperature_2m"]))
