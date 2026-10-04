import csv
import tempfile
import unittest
from pathlib import Path

from aire_caba.model_dataset import (
    INPUT_COLUMNS, ONE_HOT_COLUMNS, OUTPUT_COLUMNS, TARGETS, prepare_files,
)


class ModelDatasetTests(unittest.TestCase):
    def sample(self):
        row = {column: "1" for column in INPUT_COLUMNS}
        row.update({
            "estacion": "la_boca", "fecha_hora": "2021-01-02 00:00",
            "fecha": "2021-01-01", "dia_semana": "5", "hora_dia": "0",
            "CO": "", "NO2": "12.5", "PM10": "30",
        })
        return row

    def write_input(self, path, row):
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=INPUT_COLUMNS)
            writer.writeheader()
            writer.writerow(row)

    def test_one_hot_and_missing_target_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.csv"
            output = Path(directory) / "output.csv"
            self.write_input(source, self.sample())
            meta = prepare_files(source, output)
            with output.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(tuple(reader.fieldnames), OUTPUT_COLUMNS)
                row = next(reader)
                self.assertEqual(row["CO"], "")
                self.assertEqual([row[target] for target in TARGETS], ["", "12.5", "30"])
                self.assertEqual(row["estacion_la_boca"], "1")
                self.assertEqual(row["dia_semana_5"], "1")
                self.assertEqual(row["hora_dia_00"], "1")
                self.assertEqual(row["fecha"], "2021-01-01")
                self.assertEqual(sum(int(row[col]) for col in ONE_HOT_COLUMNS), 3)
                self.assertEqual(meta["missing_targets"], {"CO": 1, "NO2": 0, "PM10": 0})
                self.assertEqual(meta["missing_predictors"], 0)

    def test_missing_predictor_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.csv"
            output = Path(directory) / "output.csv"
            row = self.sample()
            row["temperature_2m"] = ""
            self.write_input(source, row)
            with self.assertRaisesRegex(ValueError, "temperature_2m está vacío"):
                prepare_files(source, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
