from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from aire_caba.air_quality_scales import load_scales
from aire_caba.classifier_experiments import load_cached_soft
from aire_caba.modeling import write_json
from aire_caba.no2_classifier import fit_candidates


class FrozenClassifier:
    classes_ = np.array([0, 1])

    def fit(self, *args, **kwargs):
        raise AssertionError("Un clasificador reutilizado no debe entrenarse de nuevo")

    def predict_proba(self, x):
        return np.tile([.6, .4], (len(x), 1))


class ClassifierExperimentsTests(unittest.TestCase):
    def test_cached_candidates_are_evaluated_without_refitting_or_renaming_levels(self):
        train = pd.DataFrame({"f1": [0., 1., 2.], "label": [0, 0, 1]})
        validation = pd.DataFrame({"f1": [3., 4.], "label": [0, 1]})
        config = {"seed": 42, "threads": 1, "weight_powers": [.5], "selection_metric": "f1_macro",
                  "random_forest": {}, "xgboost": {}}
        spec = load_scales()["pollutants"]["CO"]
        cached = {("RandomForest", .5): FrozenClassifier(), ("XGBoost", .5): FrozenClassifier()}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "models").mkdir()
            selection, trials, classes = fit_candidates(train, validation, ["f1"], config, spec, output, "CO", cached)
            self.assertTrue(all(row["reused"] for row in trials))
            self.assertEqual(classes.tolist(), [0, 1])
            self.assertEqual(selection["selection_metric"], "f1_macro")
            self.assertTrue((output / "seleccion_validacion.json").exists())
            self.assertTrue((output / "models/CO_XGBoost.joblib").exists())
            self.assertFalse((output / "models/NO2_XGBoost.joblib").exists())

    def test_corrupt_cached_native_model_is_rejected_before_loading(self):
        audit = {"input_sha256": "input", "quality_sha256": "quality"}
        config = {"seed": 42, "threads": 1, "xgboost": {}}
        spec = load_scales()["pollutants"]["CO"]
        metadata = audit | {"features": ["f1"], "scale": spec, "model_sha256": "wrong",
                            "parameters": {"weight_power": .5, "random_state": 42}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "outputs/clasificacion_suave/modelos"
            folder.mkdir(parents=True)
            (folder / "CO_XGBoost_suave.ubj").write_bytes(b"not a model")
            write_json(folder / "CO_XGBoost_suave.meta.json", metadata)
            with patch("aire_caba.classifier_experiments.PROJECT", root):
                with self.assertRaisesRegex(ValueError, "cambió"):
                    load_cached_soft("CO", ["f1"], spec, audit, config)
