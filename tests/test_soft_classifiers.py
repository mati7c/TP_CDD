from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from aire_caba.air_quality_scales import load_scales
from aire_caba.no2_classifier import aligned_probabilities, encode_targets
from aire_caba.soft_classifiers import matched_previous_predictions, score_predictions, SOFT


class SoftClassifiersTests(unittest.TestCase):
    def test_previous_pm10_uses_the_selected_reference_without_reordering(self):
        frame = pd.DataFrame({"estacion": ["a", "b"], "fecha_hora": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                              "split": ["test", "test"], "PM10": [20., 40.]})
        previous = frame.rename(columns={"PM10": "actual"}).iloc[::-1].copy()
        previous["prediction_Referencia"] = [31., 21.]
        previous["prediction_XGBoost"] = [99., 99.]
        np.testing.assert_equal(matched_previous_predictions(frame, previous, "PM10", "Referencia"), [21., 31.])
        with self.assertRaises(ValueError):
            matched_previous_predictions(frame, previous.iloc[:1], "PM10", "Referencia")

    def test_co_and_pm10_do_not_share_a_fixed_number_of_classes(self):
        specs = load_scales()["pollutants"]
        frame = pd.DataFrame({"estacion": ["a"] * 4, "split": ["validation", "validation", "test", "test"],
                              "label": [0, 1, 0, 1]})
        for pollutant, count in (("CO", 6), ("PM10", 10)):
            summaries, details, matrices = [], [], []
            score_predictions(frame, specs[pollutant], {SOFT: np.array([0, 1, 0, 0])},
                              summaries, details, matrices, pollutant)
            native = next(row for row in matrices if row["resolution"] == "level" and row["split"] == "test"
                          and row["station"] == "todas")
            self.assertEqual(np.shape(native["matrix"]), (count, count))
            self.assertEqual(np.asarray(native["matrix"]).sum(), 2)
            summary = next(row for row in summaries if row["resolution"] == "level" and row["split"] == "test"
                           and row["station"] == "todas")
            self.assertEqual(summary["accuracy"], .5)

    def test_native_xgboost_export_preserves_probabilities_and_level_mapping(self):
        x = np.arange(24, dtype=np.float32).reshape(12, 2)
        classes, encoded = encode_targets([0, 2, 8] * 4)
        model = XGBClassifier(n_estimators=3, max_depth=2, objective="multi:softprob", num_class=3,
                              n_jobs=1, random_state=42)
        model.fit(x, encoded)
        expected = aligned_probabilities(model, x, classes)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.ubj"
            model.save_model(path)
            restored = XGBClassifier()
            restored.load_model(path)
            actual = aligned_probabilities(restored, x, classes)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8)
        self.assertEqual(actual[:, [1, 3, 4, 5, 6, 7, 9]].sum(), 0)
