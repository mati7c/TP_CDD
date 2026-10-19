import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from aire_caba.model_dataset import OUTPUT_COLUMNS
from aire_caba.modeling import daily_dataset, load_data, temporal_split

CONFIG = {
    "validation_start": "2024-01-01", "test_start": "2025-01-01", "gap_days": 2,
    "minimum_daily_hours": 18, "daily_predictor_hours": 24,
}


class ModelingTests(unittest.TestCase):
    def sample_rows(self):
        rows = []
        for day in ("2023-12-01", "2023-12-02"):
            for hour in range(24):
                row = {column: 0 for column in OUTPUT_COLUMNS}
                row.update({"fecha": day, "fecha_hora": f"{day} {hour:02d}:00", "anio": 2023,
                            "estacion_centenario": 1, "dia_semana_1": 1, "mes_12": 1,
                            f"hora_dia_{hour:02d}": 1, "CO": 0.5, "NO2": 20, "PM10": 30,
                            "estacion": "centenario"})
                rows.append(row)
        return pd.DataFrame(rows)

    def test_dates_are_split_with_shared_gap(self):
        dates = pd.Series(["2023-12-29", "2023-12-30", "2024-01-01", "2024-12-29", "2024-12-30", "2025-01-01"])
        self.assertEqual(temporal_split(dates, CONFIG).tolist(),
                         ["train", "purga", "validation", "validation", "purga", "test"])

    def test_daily_label_uses_worst_pollutant_and_requires_coverage(self):
        data = self.sample_rows()
        data.loc[3, "NO2"] = 150
        data.loc[24:40, "PM10"] = np.nan
        daily, features, report = daily_dataset(data, CONFIG)
        self.assertEqual(len(daily), 1)
        self.assertEqual(daily.iloc[0]["category"], 2)
        self.assertEqual(report["excluded_station_days"], 1)
        self.assertFalse(any("category" in feature or "_hours" in feature for feature in features))
        self.assertFalse(any(feature in ("CO", "NO2", "PM10") for feature in features))

    def test_high_and_interpolated_targets_do_not_enter_training(self):
        data = self.sample_rows().iloc[:3].drop(columns="estacion")
        data.loc[0, "CO"] = 12
        quality = data[["fecha_hora", "CO", "NO2", "PM10"]].copy()
        quality["estacion"] = "centenario"
        for pollutant in ("CO", "NO2", "PM10"):
            quality[f"{pollutant}_estado"] = "medido"
        quality.loc[1, "CO_estado"] = "interpolado"
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "exports"
            folder.mkdir()
            data.to_csv(folder / "dataset_objetivo_listo_modelar.csv", index=False)
            quality.to_csv(folder / "dataset_objetivo.csv", index=False)
            with patch("aire_caba.modeling.original_outliers", return_value={}):
                result, features, audit = load_data(Path(directory), CONFIG)
            self.assertEqual(result["CO"].notna().tolist(), [False, False, True])
            self.assertTrue(result["NO2"].notna().all())
            self.assertEqual(audit["targets"]["CO"]["invalid_in_input"], 1)
            self.assertNotIn("NO2", features)
            self.assertNotIn("fecha_hora", features)
