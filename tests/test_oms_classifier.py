import unittest

import numpy as np
import pandas as pd

from aire_caba.air_quality_scales import classify_concentration, load_scales
from aire_caba.model_dataset import OUTPUT_COLUMNS
from aire_caba.modeling import WEATHER
from aire_caba.oms_classifier import LAG, SCALES, daily_frame, feature_columns, select_by_validation


CONFIG = {"validation_start": "2024-01-01", "test_start": "2025-01-01", "gap_days": 2,
          "minimum_daily_hours": 18, "daily_predictor_hours": 24}


def hourly(days, hours_with_value=24, value=10.0):
    rows = []
    for day in days:
        for hour in range(24):
            row = {column: 0.0 for column in OUTPUT_COLUMNS if column not in ("fecha", "fecha_hora")}
            row.update({"estacion": "centenario", "estacion_centenario": 1.0, "dia_semana_1": 1.0, "mes_01": 1.0,
                        "anio": 2023.0, "fecha": pd.Timestamp(day),
                        "fecha_hora": pd.Timestamp(day) + pd.Timedelta(hours=hour + 1),
                        "NO2": value if hour < hours_with_value else np.nan})
            rows.append(row)
    return pd.DataFrame(rows)


class OmsClassifierTests(unittest.TestCase):
    def test_who_cuts_use_converted_daily_mean_and_inclusive_guideline(self):
        specs = load_scales(SCALES)["pollutants"]
        # NO2 en ppb: 13.1 -> 25.05 -> 25 cumple; 13.4 -> 25.6 -> 26 supera; 26.4 -> 50.49 -> 50; 26.5 -> 51.
        np.testing.assert_array_equal(classify_concentration([13.0, 13.1, 13.4, 26.4, 26.5], specs["NO2"]),
                                      [0, 0, 1, 1, 2])
        np.testing.assert_array_equal(classify_concentration([45.0, 45.4, 45.6], specs["PM10"]), [0, 0, 1])
        # CO en ppm: 3.5 -> 4.03 -> 4.0 cumple; 3.53 -> 4.06 -> 4.1 supera.
        np.testing.assert_array_equal(classify_concentration([3.5, 3.53], specs["CO"]), [0, 1])

    def test_daily_frame_requires_coverage_and_lags_only_covered_previous_days(self):
        data = pd.concat([hourly(["2023-03-01"], value=10.0), hourly(["2023-03-02"], hours_with_value=12),
                          hourly(["2023-03-03"], value=20.0), hourly(["2023-03-04"], value=30.0)], ignore_index=True)
        daily, features, report = daily_frame(data, "NO2", CONFIG)
        self.assertEqual(report["eligible_station_days"], 3)
        self.assertEqual(daily["fecha"].dt.day.tolist(), [1, 3, 4])
        self.assertTrue(np.isnan(daily.loc[0, LAG]))
        self.assertTrue(np.isnan(daily.loc[1, LAG]), "el 2 de marzo no tiene cobertura: no se usa como rezago")
        self.assertEqual(daily.loc[2, LAG], 20.0)
        self.assertEqual(daily["value"].tolist(), [10.0, 20.0, 30.0])
        self.assertNotIn(LAG, features)
        self.assertIn(LAG, feature_columns(features, "rezago"))
        self.assertTrue(all(f"{variable}_mean" in features for variable in WEATHER))

    def test_selection_uses_validation_scores_only(self):
        trials = [
            {"model": "RandomForest", "feature_set": "meteo", "validation_f1_macro": .4, "test_f1_macro": .9},
            {"model": "RandomForest", "feature_set": "meteo", "validation_f1_macro": .5, "test_f1_macro": .1},
            {"model": "XGBoost", "feature_set": "rezago", "validation_f1_macro": .6, "test_f1_macro": .2},
        ]
        best, winner = select_by_validation(trials, "f1_macro")
        self.assertEqual(len(best), 2)
        self.assertEqual(best.loc[best["model"].eq("RandomForest"), "validation_f1_macro"].item(), .5)
        self.assertEqual((winner["model"], winner["feature_set"]), ("XGBoost", "rezago"))


if __name__ == "__main__":
    unittest.main()
