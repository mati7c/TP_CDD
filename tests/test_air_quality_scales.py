import unittest

import numpy as np
import pandas as pd

from aire_caba.air_quality_scales import classify_concentration, convert_concentration, load_scales
from aire_caba.aqi import classification_metrics
from aire_caba.compare_indices import score_saved_predictions


class AlternativeScalesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs = load_scales()["pollutants"]

    def test_pm10_rounding_is_once_and_boundaries_are_inclusive(self):
        spec = self.specs["PM10"]
        values = [16.49, 16.5, 33.49, 33.5, 50.486, 50.5, 100.49, 100.5]
        self.assertEqual(classify_concentration(values, spec).tolist(), [0, 1, 1, 2, 2, 3, 8, 9])
        self.assertEqual(classify_concentration(values, spec, "band").tolist(), [0, 0, 0, 0, 0, 1, 2, 3])

    def test_no2_conversion_happens_before_rounding(self):
        spec = self.specs["NO2"]
        values = np.array([67.49, 67.5, 134.49, 134.5, 600.49, 600.5]) / 1.9125
        self.assertEqual(classify_concentration(values, spec).tolist(), [0, 1, 1, 2, 8, 9])
        np.testing.assert_allclose(convert_concentration([10], spec), [19.125])

    def test_co_uses_mass_units_and_declared_decimal_convention(self):
        spec = self.specs["CO"]
        concentrations = np.array([1.0, 1.049, 1.05, 2.0, 2.049, 2.05, 10.0, 10.05, 17.0, 17.05, 34.0, 34.05])
        self.assertEqual(classify_concentration(concentrations / spec["factor"], spec).tolist(),
                         [0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5])
        self.assertEqual(classify_concentration([1], spec).tolist(), [1])

    def test_invalid_targets_are_unlabelled(self):
        for spec in self.specs.values():
            self.assertEqual(classify_concentration([np.nan, -1, np.inf], spec).tolist(), [-1, -1, -1])
            self.assertTrue(np.isnan(convert_concentration([np.nan, -1, np.inf], spec)).all())

    def test_ten_class_metrics_keep_absent_classes_visible(self):
        result = classification_metrics([0, 1], [0, 1], [str(i) for i in range(10)])
        self.assertEqual(np.shape(result["confusion"]), (10, 10))
        self.assertEqual(result["summary"]["f1_macro"], .2)
        self.assertEqual(result["summary"]["f1_macro_supported"], 1)
        self.assertEqual(result["summary"]["precision_macro_supported"], 1)
        self.assertIsNone(result["classes"][9]["recall"])
        with self.assertRaises(ValueError):
            classification_metrics([10], [0], [str(i) for i in range(10)])

    def test_relabelling_does_not_select_majority_using_test(self):
        frame = pd.DataFrame({"estacion": ["centenario", "centenario"], "split": ["test", "test"],
                              "actual": [60., 60.], "prediction_Referencia": [1., 1.],
                              "prediction_RandomForest": [50., 50.], "prediction_XGBoost": [55., 55.]})
        before = frame.copy(deep=True)
        summaries, details, matrices = [], [], []
        result = score_saved_predictions(frame, [1., 1., 60.], "NO2", self.specs["NO2"], "XGBoost",
                                         summaries, details, matrices)
        pd.testing.assert_frame_equal(frame, before)
        self.assertEqual(result["Mayoria_train_UK_DAQI_level"].tolist(), [0, 0])
        chosen = next(row for row in summaries if row["scheme"] == "UK_DAQI" and row["resolution"] == "level"
                      and row["station"] == "todas" and row["model"] == "Mayoria_train")
        self.assertEqual(chosen["accuracy"], 0)
        self.assertEqual(chosen["balanced_accuracy"], 0)
