import unittest

import numpy as np

from aire_caba.aqi import categorize, classification_metrics


class AQITests(unittest.TestCase):
    def test_boundaries_and_missing_values(self):
        self.assertEqual(categorize([4.499, 4.5, 9.499, 9.5, np.nan, -1, np.inf], "CO").tolist(),
                         [0, 1, 1, 2, -1, -1, -1])
        self.assertEqual(categorize([53.99, 54, 100.99, 101], "NO2").tolist(), [0, 1, 1, 2])
        self.assertEqual(categorize([54.99, 55, 154.99, 155], "PM10").tolist(), [0, 1, 1, 2])

    def test_confusion_and_false_positive_denominator(self):
        result = classification_metrics([0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 0])
        self.assertEqual(result["confusion"], [[1, 1, 0], [0, 1, 1], [1, 0, 1]])
        self.assertEqual(result["summary"]["accuracy"], 0.5)
        self.assertEqual(result["summary"]["f1_macro"], 0.5)
        for row in result["classes"]:
            self.assertEqual(row["false_positive_rate"], 0.25)
            self.assertEqual(row["precision"], 0.5)
            self.assertEqual(row["recall"], 0.5)

    def test_absent_class_is_not_claimed_successful(self):
        result = classification_metrics([0, 0], [0, 0])
        self.assertIsNone(result["classes"][2]["recall"])
        self.assertEqual(result["classes"][2]["support"], 0)
        self.assertEqual(result["summary"]["f1_macro"], 1 / 3)
        self.assertEqual(result["summary"]["f1_macro_supported"], 1)
