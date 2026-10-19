import unittest

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from aire_caba.no2_classifier import aligned_probabilities, encode_targets, reconcile_regression, training_weights


class NO2ClassifierTests(unittest.TestCase):
    def test_nonconsecutive_levels_are_decoded_without_inventing_classes(self):
        original, encoded = encode_targets([0, 0, 2, 8])
        self.assertEqual(original.tolist(), [0, 2, 8])
        self.assertEqual(encoded.tolist(), [0, 0, 1, 2])
        x = np.zeros((4, 1))
        model = DummyClassifier(strategy="prior").fit(x, encoded)
        probabilities = aligned_probabilities(model, x, original)
        np.testing.assert_allclose(probabilities[0, [0, 2, 8]], [.5, .25, .25])
        self.assertEqual(probabilities.shape, (4, 10))
        self.assertEqual(probabilities[:, [1, 3, 4, 5, 6, 7, 9]].sum(), 0)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)

    def test_weights_balance_training_and_allow_smoothing(self):
        labels = np.array([0, 0, 0, 1])
        np.testing.assert_allclose(training_weights(labels, 0), np.ones(4))
        balanced = training_weights(labels, 1)
        self.assertAlmostEqual(balanced[labels == 0].sum(), balanced[labels == 1].sum())
        softened = training_weights(labels, .5)
        self.assertGreater(softened[-1], 1)
        self.assertLess(softened[-1], balanced[-1])
        self.assertAlmostEqual(softened.mean(), 1)
        with self.assertRaises(ValueError):
            training_weights(labels, 2)

    def test_comparison_matches_rows_by_key_and_rejects_stale_targets(self):
        frame = pd.DataFrame({"estacion": ["a", "b"], "fecha_hora": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                              "split": ["test", "test"], "NO2": [10., 20.]})
        saved = frame.rename(columns={"NO2": "actual"}).iloc[::-1].copy()
        saved["prediction_XGBoost"] = [19., 9.]
        self.assertEqual(reconcile_regression(frame, saved).tolist(), [9., 19.])
        saved.loc[saved.index[0], "actual"] = 21.
        with self.assertRaises(ValueError):
            reconcile_regression(frame, saved)

    def test_single_training_class_is_not_claimed_multiclass(self):
        with self.assertRaises(ValueError):
            encode_targets([0, 0])
