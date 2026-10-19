"""Experimentos reproducibles de regresión horaria y clasificación diaria."""

import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from .aqi import BREAKPOINTS, CLASS_NAMES, EPA_SOURCE, categorize, classification_metrics
from .model_dataset import IDENTIFIERS, OUTPUT_COLUMNS, TARGETS, sha256
from .transform import VALID_RANGE, ZERO_INVALID

PROJECT = Path(__file__).resolve().parent.parent
WEATHER = (
    "temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m",
    "precipitation", "cloud_cover", "shortwave_radiation", "viento_u", "viento_v",
)
MODEL_NAMES = ("Referencia", "RandomForest", "XGBoost")
ACCEPTED_STATES = ("medido", "censurado")


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def temporal_split(dates, config):
    """Cortes por fecha compartidos entre estaciones; purga de días entre períodos."""
    dates = pd.to_datetime(dates)
    validation = pd.Timestamp(config["validation_start"])
    test = pd.Timestamp(config["test_start"])
    gap = pd.Timedelta(days=config["gap_days"])
    if validation >= test or config["gap_days"] < 0:
        raise ValueError("Los cortes temporales no son válidos")
    split = pd.Series("purga", index=dates.index, dtype=object)
    split.loc[dates < validation - gap] = "train"
    split.loc[(dates >= validation) & (dates < test - gap)] = "validation"
    split.loc[dates >= test] = "test"
    return split


def original_outliers(data_dir):
    paths = list((Path(data_dir) / "raw/air").glob("*/calidad-aire.csv"))
    if len(paths) != 1:
        raise ValueError("Se necesita identificar una única copia original del CSV de aire")
    raw = pd.read_csv(paths[0], dtype=str, keep_default_na=False, encoding="utf-8-sig")
    rows = []
    flags = []
    for pollutant in TARGETS:
        for station in ("CENTENARIO", "CORDOBA", "LA_BOCA"):
            value = pd.to_numeric(raw[f"{pollutant}_{station}"], errors="coerce")
            above = value.gt(VALID_RANGE[pollutant][1])
            flags.append(above)
            rows.append({
                "pollutant": pollutant, "station": station.lower(),
                "maximum_allowed": VALID_RANGE[pollutant][1], "above_maximum": int(above.sum()),
                "maximum_observed": float(value.max()),
            })
    return {
        "source": str(paths[0].relative_to(PROJECT)), "sha256": sha256(paths[0]),
        "by_station": rows, "total_above_maximum": sum(row["above_maximum"] for row in rows),
        "distinct_original_rows": int(pd.concat(flags, axis=1).any(axis=1).sum()),
    }


def load_data(data_dir, config):
    source = Path(data_dir) / "exports/dataset_objetivo_listo_modelar.csv"
    quality_source = Path(data_dir) / "exports/dataset_objetivo.csv"
    data = pd.read_csv(source, encoding="utf-8-sig")
    if list(data.columns) != list(OUTPUT_COLUMNS):
        raise ValueError("El esquema de dataset_objetivo_listo_modelar.csv cambió")
    features = [column for column in OUTPUT_COLUMNS if column not in IDENTIFIERS + TARGETS]
    if not np.isfinite(data[features].to_numpy(dtype=float)).all():
        raise ValueError("Hay predictores vacíos o no finitos")
    station_columns = [column for column in features if column.startswith("estacion_")]
    if not data[station_columns].sum(axis=1).eq(1).all():
        raise ValueError("Cada registro debe tener una sola estación")
    data["estacion"] = data[station_columns].idxmax(axis=1).str.removeprefix("estacion_")
    quality = pd.read_csv(quality_source, encoding="utf-8-sig")
    quality_cols = ["estacion", "fecha_hora", *TARGETS, *(f"{p}_estado" for p in TARGETS)]
    quality = quality[quality_cols].rename(columns={p: f"{p}_clean" for p in TARGETS})
    data = data.merge(quality, on=["estacion", "fecha_hora"], how="left", validate="one_to_one", indicator=True)
    if not data["_merge"].eq("both").all():
        raise ValueError("Hay registros sin trazabilidad en dataset_objetivo.csv")
    data = data.drop(columns="_merge")
    audit = {
        "input_rows": len(data), "input_sha256": sha256(source),
        "quality_sha256": sha256(quality_source), "original_outliers": original_outliers(data_dir),
        "targets": {}, "accepted_states": list(ACCEPTED_STATES),
    }
    for pollutant in TARGETS:
        same = data[pollutant].eq(data[f"{pollutant}_clean"]) | (
            data[pollutant].isna() & data[f"{pollutant}_clean"].isna())
        if not same.all():
            raise ValueError(f"{pollutant} difiere de su fuente de calidad")
        value = data[pollutant]
        lo, hi = VALID_RANGE[pollutant]
        invalid = value.notna() & (~np.isfinite(value) | value.lt(lo) | value.gt(hi))
        if pollutant in ZERO_INVALID:
            invalid |= value.eq(0)
        observed = data[f"{pollutant}_estado"].isin(ACCEPTED_STATES)
        usable = value.notna() & ~invalid & observed
        audit["targets"][pollutant] = {
            "plausibility_range": [lo, hi],
            "missing_in_input": int(value.isna().sum()),
            "invalid_in_input": int(invalid.sum()),
            "interpolated_excluded": int(data[f"{pollutant}_estado"].eq("interpolado").sum()),
            "usable": int(usable.sum()),
        }
        # No se entrena/evalúa con objetivos reconstruidos mediante interpolación.
        data[pollutant] = value.where(usable)
        data = data.drop(columns=f"{pollutant}_clean")
    data["fecha"] = pd.to_datetime(data["fecha"])
    data["fecha_hora"] = pd.to_datetime(data["fecha_hora"])
    data["split"] = temporal_split(data["fecha"], config)
    data[features] = data[features].astype(np.float32)
    data = data.sort_values(["fecha", "fecha_hora", "estacion"], ignore_index=True)
    return data, features, audit


def daily_dataset(data, config):
    """Máxima categoría de los registros disponibles; adaptación, no AQI diario EPA exacto."""
    keys = ["estacion", "fecha"]
    static = [
        column for column in OUTPUT_COLUMNS if column.startswith(("estacion_", "dia_semana_", "mes_"))
    ] + ["anio", "es_fin_de_semana", "es_feriado_nacional", "es_dia_no_laborable"]
    grouped = data.groupby(keys, sort=True, observed=True)
    if grouped[static].nunique().gt(1).any().any():
        raise ValueError("Los indicadores diarios son inconsistentes dentro del mismo día")
    daily = grouped[static].first()
    weather = grouped[list(WEATHER)].agg(["mean", "min", "max"])
    weather.columns = [f"{variable}_{statistic}" for variable, statistic in weather.columns]
    daily = daily.join(weather)
    daily["precipitation_sum"] = grouped["precipitation"].sum()
    features = list(daily.columns)
    daily["predictor_hours"] = grouped.size()
    for pollutant in TARGETS:
        daily[f"{pollutant}_hours"] = grouped[pollutant].count()
        # El máximo de concentraciones tiene la misma etiqueta que el máximo de categorías.
        daily[f"{pollutant}_category"] = categorize(grouped[pollutant].max().to_numpy(), pollutant)
    counts = [f"{p}_hours" for p in TARGETS]
    eligible = daily[counts].ge(config["minimum_daily_hours"]).all(axis=1)
    eligible &= daily["predictor_hours"].eq(config["daily_predictor_hours"])
    report = {
        "station_days_available": len(daily), "eligible_station_days": int(eligible.sum()),
        "excluded_station_days": int((~eligible).sum()),
        "minimum_hours_per_pollutant": config["minimum_daily_hours"],
        "required_predictor_hours": config["daily_predictor_hours"],
    }
    daily = daily.loc[eligible].copy()
    daily["category"] = daily[[f"{p}_category" for p in TARGETS]].max(axis=1).astype(int)
    daily = daily.reset_index()
    daily["split"] = temporal_split(daily["fecha"], config)
    return daily, features, report


class StationMedianRegressor:
    def fit(self, X, y):
        station_cols = [c for c in X.columns if c.startswith("estacion_")]
        self.columns_ = station_cols
        self.fallback_ = float(np.median(y))
        self.medians_ = {
            column: float(np.median(np.asarray(y)[X[column].to_numpy() == 1]))
            for column in station_cols if X[column].eq(1).any()
        }
        return self

    def predict(self, X):
        result = np.full(len(X), self.fallback_)
        for column, value in self.medians_.items():
            result[X[column].to_numpy() == 1] = value
        return result


def reg_metrics(actual, predicted, high_threshold):
    high = actual >= high_threshold
    return {
        "n": len(actual), "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(root_mean_squared_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else None,
        "high_threshold_train_p90": float(high_threshold), "n_high": int(high.sum()),
        "mae_high": float(mean_absolute_error(actual[high], predicted[high])) if high.any() else None,
    }


def class_rows(actual, predicted, context):
    result = classification_metrics(actual, predicted)
    summary = context | result["summary"]
    details = [context | row for row in result["classes"]]
    matrix = context | {"matrix": result["confusion"]}
    return summary, details, matrix


def add_importance(model, features, context, output):
    if hasattr(model, "feature_importances_"):
        output.extend(context | {"feature": name, "importance": float(value)}
                      for name, value in zip(features, model.feature_importances_))


def save_model(model, output, task, name, features, params):
    folder = output / "models"
    folder.mkdir(exist_ok=True)
    joblib.dump({"model": model, "task": task, "features": features,
                 "parameters": params, "classes": CLASS_NAMES}, folder / f"{task}_{name}.joblib", compress=3)


def fit_regression(data, features, config, output, regression, categories, class_detail, matrices, importance):
    winners, trials, model_parameters = {}, [], {}
    for pollutant in TARGETS:
        subset = data.loc[data[pollutant].notna()].copy()
        splits = {name: subset.loc[subset["split"].eq(name)] for name in ("train", "validation", "test")}
        if any(frame.empty for frame in splits.values()):
            raise ValueError(f"{pollutant}: falta un período para entrenamiento o evaluación")
        train, validation = splits["train"], splits["validation"]
        xtrain, xval = train[features], validation[features]
        ytrain, yval = train[pollutant].to_numpy(), validation[pollutant].to_numpy()
        high_threshold = float(np.quantile(ytrain, 0.9))
        predictions = {
            name: frame[["estacion", "fecha", "fecha_hora", "split", pollutant]].rename(columns={pollutant: "actual"}).copy()
            for name, frame in splits.items() if name != "train"
        }
        candidates = [
            ("Referencia", StationMedianRegressor(), {"strategy": "median_per_station"}),
            ("RandomForest", RandomForestRegressor(
                **config["regression_forest"], random_state=config["seed"], n_jobs=config["threads"]),
             config["regression_forest"]),
            ("XGBoost", None, None),
        ]
        scores = {}
        for name, model, params in candidates:
            started = time.perf_counter()
            if name == "XGBoost":
                best_score, best_model, best_params = float("inf"), None, None
                for index, trial_params in enumerate(config["regression_xgboost_candidates"]):
                    trial = XGBRegressor(**trial_params, **config["xgboost_common"],
                                         objective="reg:squarederror", random_state=config["seed"],
                                         n_jobs=config["threads"])
                    trial.fit(xtrain, ytrain)
                    score = float(mean_absolute_error(yval, np.maximum(trial.predict(xval), 0)))
                    trials.append({"task": pollutant, "candidate": index, "validation_mae": score, "parameters": trial_params})
                    if score < best_score:
                        best_score, best_model, best_params = score, trial, trial_params
                model, params = best_model, best_params | config["xgboost_common"]
            else:
                model.fit(xtrain, ytrain)
            fit_seconds = time.perf_counter() - started
            model_parameters[f"{pollutant}_{name}"] = params
            for split_name in ("validation", "test"):
                frame = splits[split_name]
                predicted_raw = model.predict(frame[features])
                predicted = np.maximum(predicted_raw, 0)
                actual = frame[pollutant].to_numpy()
                context = {"task": pollutant, "model": name, "split": split_name, "station": "todas"}
                metrics = reg_metrics(actual, predicted, high_threshold)
                regression.append(context | metrics | {"fit_seconds": fit_seconds,
                                  "negative_predictions_clipped": int((predicted_raw < 0).sum())})
                if split_name == "validation":
                    scores[name] = metrics["mae"]
                actual_class, predicted_class = categorize(actual, pollutant), categorize(predicted, pollutant)
                summary, detail, matrix = class_rows(actual_class, predicted_class, context | {"level": "hourly_from_regression"})
                categories.append(summary)
                class_detail.extend(detail)
                matrices.append(matrix)
                predictions[split_name][f"prediction_{name}"] = predicted
                predictions[split_name][f"category_{name}"] = predicted_class
                for station in sorted(frame["estacion"].unique()):
                    mask = frame["estacion"].eq(station).to_numpy()
                    station_context = context | {"station": station}
                    regression.append(station_context | reg_metrics(actual[mask], predicted[mask], high_threshold))
                    summary, detail, matrix = class_rows(actual_class[mask], predicted_class[mask],
                                                         station_context | {"level": "hourly_from_regression"})
                    categories.append(summary)
                    class_detail.extend(detail)
                    matrices.append(matrix)
            add_importance(model, features, {"task": pollutant, "model": name}, importance)
            save_model(model, output, pollutant, name, features, params)
            print(f"{pollutant} / {name}: MAE validación={scores[name]:.4f}; {fit_seconds:.1f} s", flush=True)
            del model
            gc.collect()
        winners[pollutant] = min(scores, key=scores.get)
        pd.concat(predictions.values()).to_csv(output / f"predicciones_{pollutant}.csv", index=False, encoding="utf-8-sig")
    return winners, trials, model_parameters


def fit_daily(daily, features, config, output, summaries, details, matrices, importance):
    splits = {name: daily.loc[daily["split"].eq(name)] for name in ("train", "validation", "test")}
    if any(frame.empty for frame in splits.values()):
        raise ValueError("Falta algún período de clasificación diaria")
    train, validation = splits["train"], splits["validation"]
    xtrain, ytrain = train[features].astype(np.float32), train["category"].to_numpy()
    xval, yval = validation[features].astype(np.float32), validation["category"].to_numpy()
    if set(np.unique(ytrain)) != {0, 1, 2}:
        raise ValueError("El entrenamiento diario no contiene las tres categorías EPA")
    predictions = {
        name: frame[["estacion", "fecha", "split", "category"]].rename(columns={"category": "actual_category"}).copy()
        for name, frame in splits.items() if name != "train"
    }
    trials, parameters, scores = [], {}, {}
    sample_weight = compute_sample_weight("balanced", ytrain)
    candidates = [
        ("Referencia", DummyClassifier(strategy="most_frequent"), {"strategy": "most_frequent"}),
        ("RandomForest", RandomForestClassifier(
            **config["classification_forest"], random_state=config["seed"], n_jobs=config["threads"]),
         config["classification_forest"]),
        ("XGBoost", None, None),
    ]
    for name, model, params in candidates:
        started = time.perf_counter()
        if name == "XGBoost":
            best_score, best_model, best_params = -1.0, None, None
            for index, trial_params in enumerate(config["classification_xgboost_candidates"]):
                trial = XGBClassifier(**trial_params, **config["xgboost_common"],
                                      objective="multi:softprob", num_class=3,
                                      random_state=config["seed"], n_jobs=config["threads"])
                trial.fit(xtrain, ytrain, sample_weight=sample_weight)
                score = classification_metrics(yval, trial.predict(xval))["summary"]["f1_macro"]
                trials.append({"task": "daily", "candidate": index, "validation_f1_macro": score, "parameters": trial_params})
                if score > best_score:
                    best_score, best_model, best_params = score, trial, trial_params
            model, params = best_model, best_params | config["xgboost_common"] | {"sample_weight": "balanced_train"}
        else:
            model.fit(xtrain, ytrain)
        fit_seconds = time.perf_counter() - started
        parameters[name] = params
        for split_name in ("validation", "test"):
            frame = splits[split_name]
            actual = frame["category"].to_numpy()
            predicted = model.predict(frame[features].astype(np.float32)).astype(int)
            context = {"task": "daily", "model": name, "split": split_name, "station": "todas", "level": "daily_classifier"}
            summary, detail, matrix = class_rows(actual, predicted, context)
            summary["fit_seconds"] = fit_seconds
            summaries.append(summary)
            details.extend(detail)
            matrices.append(matrix)
            if split_name == "validation":
                scores[name] = summary["f1_macro"]
            predictions[split_name][f"prediction_{name}"] = predicted
            for station in sorted(frame["estacion"].unique()):
                mask = frame["estacion"].eq(station).to_numpy()
                summary, detail, matrix = class_rows(actual[mask], predicted[mask], context | {"station": station})
                summaries.append(summary)
                details.extend(detail)
                matrices.append(matrix)
        add_importance(model, features, {"task": "daily", "model": name}, importance)
        save_model(model, output, "daily", name, features, params)
        print(f"Clasificación diaria / {name}: F1 macro validación={scores[name]:.4f}; {fit_seconds:.1f} s", flush=True)
        del model
        gc.collect()
    pd.concat(predictions.values()).to_csv(output / "predicciones_diarias.csv", index=False, encoding="utf-8-sig")
    return max(scores, key=scores.get), trials, parameters


def distributions(data, daily):
    records = []
    for pollutant in TARGETS:
        frame = data.loc[data[pollutant].notna(), ["estacion", "split", pollutant]].copy()
        frame["category"] = categorize(frame[pollutant].to_numpy(), pollutant)
        for (station, split, label), size in frame.groupby(["estacion", "split", "category"]).size().items():
            records.append({"level": "hourly", "pollutant": pollutant, "station": station,
                            "split": split, "category": CLASS_NAMES[label], "count": int(size)})
    for (station, split, label), size in daily.groupby(["estacion", "split", "category"]).size().items():
        records.append({"level": "daily", "pollutant": "combined", "station": station,
                        "split": split, "category": CLASS_NAMES[label], "count": int(size)})
    return pd.DataFrame(records)


def run_modeling(data_dir, config_path=None, output=None):
    data_dir = Path(data_dir)
    config_path = Path(config_path or PROJECT / "config/modeling.json")
    output = Path(output or PROJECT / "outputs/tercera_entrega")
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    data, features, audit = load_data(data_dir, config)
    daily, daily_features, coverage = daily_dataset(data, config)
    write_json(output / "auditoria.json", audit)
    distributions(data, daily).to_csv(output / "distribucion_categorias.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(output / "dataset_diario_clasificacion.csv", index=False, encoding="utf-8-sig")
    print(f"Entrada: {len(data)} estación-horas; clasificación diaria: {len(daily)} estación-días", flush=True)
    print(daily.groupby(["split", "category"]).size().to_string(), flush=True)
    regression, summaries, details, matrices, importance = [], [], [], [], []
    reg_winners, reg_trials, reg_params = fit_regression(
        data, features, config, output, regression, summaries, details, matrices, importance)
    daily_winner, daily_trials, daily_params = fit_daily(
        daily, daily_features, config, output, summaries, details, matrices, importance)
    pd.DataFrame(regression).to_csv(output / "metricas_regresion.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(output / "metricas_clasificacion.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(details).to_csv(output / "metricas_por_clase.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(importance).to_csv(output / "importancia_variables.csv", index=False, encoding="utf-8-sig")
    write_json(output / "matrices_confusion.json", matrices)
    report = {
        "config": config, "parameters": {"regression": reg_params, "daily": daily_params},
        "selected_by_validation": {"regression": reg_winners, "daily": daily_winner},
        "tuning_trials": reg_trials + daily_trials, "coverage": coverage,
        "features": {"hourly": features, "daily": daily_features},
        "splits_hourly": data["split"].value_counts().to_dict(),
        "splits_daily": daily["split"].value_counts().to_dict(),
        "input_sha256": audit["input_sha256"], "quality_sha256": audit["quality_sha256"],
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source": EPA_SOURCE, "category_names": CLASS_NAMES, "breakpoints": BREAKPOINTS,
        "plausibility_limits": VALID_RANGE, "outliers_original": audit["original_outliers"]["total_above_maximum"],
        "versions": {package: importlib.metadata.version(package) for package in
                     ("numpy", "pandas", "scikit-learn", "xgboost", "matplotlib", "joblib")},
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "Etiquetas del proyecto: máximo diario de categorías por registro entre CO, NO2 y PM10; no es el AQI diario oficial EPA.",
            "Meteorología ERA5 conocida a posteriori: no mide por sí sola el error operativo a 24/48 h con pronósticos meteorológicos.",
            "Los objetivos interpolados se excluyen; los censurados se conservan como LD/2 según la segunda entrega.",
            "Buena se refiere sólo a los tres contaminantes disponibles; no se evalúan PM2.5, ozono ni SO2.",
            "Los máximos de plausibilidad son reglas del proyecto, no límites universales físicos ni sanitarios.",
            "F1 macro usa las tres etiquetas fijas; las clases sin soporte se muestran también como no evaluables en el detalle.",
        ],
    }
    write_json(output / "reporte.json", report)
    print(f"Completado en {report['elapsed_seconds']:.1f} s. Resultados: {output}", flush=True)
    return report
