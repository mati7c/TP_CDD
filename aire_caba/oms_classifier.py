"""Clasificación diaria con las guías OMS 2021: NO2 y PM10 por estación-día."""

from datetime import datetime, timezone
import gc
from html import escape
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import sys
import textwrap
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import DMatrix, XGBClassifier

from .air_quality_scales import classify_concentration, load_scales
from .compare_indices import distribution_rows, evaluate
from .model_dataset import OUTPUT_COLUMNS, TARGETS, sha256
from .modeling import PROJECT, WEATHER, load_data, temporal_split, write_json
from .no2_classifier import aligned_probabilities, encode_targets, training_weights


SCALES = PROJECT / "config/indices_oms.json"
MAJORITY = "Mayoria_train"
FAMILIES = ("RandomForest", "XGBoost")
LAG = "valor_dia_anterior"
COLORS = {"train": "#8193a6", "validation": "#147d92", "test": "#e28a35"}
STATION_COLORS = {"centenario": "#147d92", "cordoba": "#e28a35", "la_boca": "#6a4c93"}


def daily_frame(data, pollutant, config):
    """Promedio diario (ventana de 24 h de la OMS) con las entradas diarias del modelado original."""
    static = [column for column in OUTPUT_COLUMNS if column.startswith(("estacion_", "dia_semana_", "mes_"))] + [
        "anio", "es_fin_de_semana", "es_feriado_nacional", "es_dia_no_laborable"]
    grouped = data.groupby(["estacion", "fecha"], sort=True, observed=True)
    if grouped[static].nunique().gt(1).any().any():
        raise ValueError("Los indicadores diarios son inconsistentes dentro del mismo día")
    daily = grouped[static].first()
    weather = grouped[list(WEATHER)].agg(["mean", "min", "max"])
    weather.columns = [f"{variable}_{statistic}" for variable, statistic in weather.columns]
    daily = daily.join(weather)
    daily["precipitation_sum"] = grouped["precipitation"].sum()
    features = list(daily.columns)
    daily["predictor_hours"] = grouped.size()
    daily["hours"] = grouped[pollutant].count()
    daily["value"] = grouped[pollutant].mean()
    daily = daily.reset_index()
    covered = daily["hours"].ge(config["minimum_daily_hours"])
    # El día anterior ya fue medido al momento de estimar el día siguiente: no hay fuga del objetivo.
    previous = daily.loc[covered, ["estacion", "fecha", "value"]].rename(columns={"value": LAG})
    previous["fecha"] = previous["fecha"] + pd.Timedelta(days=1)
    daily = daily.merge(previous, on=["estacion", "fecha"], how="left", validate="one_to_one")
    eligible = covered & daily["predictor_hours"].eq(config["daily_predictor_hours"])
    report = {"station_days_available": len(daily), "eligible_station_days": int(eligible.sum()),
              "excluded_station_days": int((~eligible).sum()),
              "eligible_without_previous_day": int((eligible & daily[LAG].isna()).sum())}
    daily = daily.loc[eligible].reset_index(drop=True)
    daily["split"] = temporal_split(daily["fecha"], config)
    return daily, features, report


def feature_columns(base, feature_set):
    if feature_set == "meteo":
        return list(base)
    if feature_set == "rezago":
        return [*base, LAG]
    raise ValueError(f"Conjunto de entradas desconocido: {feature_set}")


def candidates(config):
    """Grilla chica y explícita; los valores fijos vienen del modelado original."""
    grid = config["random_forest_grid"]
    for depth, leaf in itertools.product(grid["max_depth"], grid["min_samples_leaf"]):
        yield "RandomForest", config["random_forest"] | {"max_depth": depth, "min_samples_leaf": leaf}
    grid = config["xgboost_grid"]
    for depth, trees in itertools.product(grid["max_depth"], grid["n_estimators"]):
        yield "XGBoost", config["xgboost"] | {"max_depth": depth, "n_estimators": trees}


def build_model(family, params, n_classes, config):
    if family == "RandomForest":
        return RandomForestClassifier(**params, random_state=config["seed"], n_jobs=config["threads"])
    if n_classes == 2:
        objective = {"objective": "binary:logistic", "eval_metric": "logloss"}
    else:
        objective = {"objective": "multi:softprob", "eval_metric": "mlogloss", "num_class": n_classes}
    return XGBClassifier(**params, **objective, random_state=config["seed"], n_jobs=config["threads"])


def select_by_validation(trials, metric):
    """Mejor ensayo por familia y conjunto de entradas, y ganador global; nunca mira test."""
    frame = pd.DataFrame(trials)
    column = f"validation_{metric}"
    best = frame.loc[frame.groupby(["model", "feature_set"], sort=False)[column].idxmax()]
    winner = best.loc[best[column].idxmax()]
    return best, winner


def shap_importance(model, X, features, positive_class):
    """Contribuciones SHAP nativas de XGBoost (TreeSHAP), en log-odds."""
    contributions = model.get_booster().predict(DMatrix(X, feature_names=list(features)), pred_contribs=True)
    if contributions.ndim == 3:
        overall = np.abs(contributions[:, :, :-1]).sum(axis=1)
        positive = contributions[:, positive_class, :-1]
    else:
        overall = np.abs(contributions[:, :-1])
        positive = contributions[:, :-1]
    return pd.Series(overall.mean(axis=0), index=features), pd.DataFrame(positive, columns=features, index=X.index)


def feature_group(feature):
    for prefix in ("estacion_", "dia_semana_", "mes_"):
        if feature.startswith(prefix):
            return prefix[:-1]
    for variable in WEATHER:
        if feature.startswith(variable + "_"):
            return variable
    return feature


def fit_pollutant(daily, base_features, pollutant, spec, config, output, records):
    daily = daily.copy()
    daily["label"] = classify_concentration(daily["value"], spec)
    parts = {name: daily.loc[daily["split"].eq(name)] for name in ("train", "validation", "test")}
    original_classes, encoded = encode_targets(parts["train"]["label"])
    if len(original_classes) != len(spec["classes"]):
        raise ValueError(f"{pollutant}: entrenamiento sin todas las clases OMS")
    n_classes = len(spec["classes"])
    yval = parts["validation"]["label"].to_numpy()
    trials, kept = [], {}
    for feature_set in config["feature_sets"]:
        features = feature_columns(base_features, feature_set)
        xtrain = parts["train"][features].astype(np.float32)
        xval = parts["validation"][features].astype(np.float32)
        for (family, params), power in itertools.product(candidates(config), config["weight_powers"]):
            started = time.perf_counter()
            model = build_model(family, params, n_classes, config)
            model.fit(xtrain, encoded, sample_weight=training_weights(parts["train"]["label"], power))
            predicted = aligned_probabilities(model, xval, original_classes, n_classes).argmax(axis=1)
            summaries, details, matrices = [], [], []
            evaluate(yval, predicted, spec["classes"], {}, summaries, details, matrices)
            trial = {"pollutant": pollutant, "model": family, "feature_set": feature_set, "weight_power": power,
                     "max_depth": params["max_depth"],
                     "min_samples_leaf": params.get("min_samples_leaf"), "n_estimators": params["n_estimators"],
                     "fit_seconds": time.perf_counter() - started,
                     **{f"validation_{key}": value for key, value in summaries[0].items()}}
            trials.append(trial)
            key = (family, feature_set)
            score = trial[f"validation_{config['selection_metric']}"]
            if key not in kept or score > kept[key][0]:
                kept[key] = (score, model, params | {"weight_power": power}, features)
            else:
                del model
            gc.collect()
        print(f"{pollutant} / {feature_set}: " + ", ".join(
            f"{family} F1 val={kept[(family, feature_set)][0]:.4f}" for family in FAMILIES), flush=True)
    best, winner = select_by_validation(trials, config["selection_metric"])
    winner_name = f"{winner['model']}_{winner['feature_set']}"
    # La selección se escribe antes de calcular cualquier métrica de prueba.
    selection = {"selected": winner_name, "metric": config["selection_metric"],
                 "validation_score": float(winner[f"validation_{config['selection_metric']}"]),
                 "finalists": {f"{family}_{fs}": {"validation_score": float(score), "parameters": params,
                                                   "features": features}
                               for (family, fs), (score, _, params, features) in kept.items()}}
    write_json(output / pollutant / "seleccion_validacion.json", selection)

    frame = pd.concat([parts["validation"], parts["test"]], ignore_index=True)
    majority = int(np.bincount(parts["train"]["label"], minlength=n_classes).argmax())
    predictions = {MAJORITY: np.full(len(frame), majority, dtype=int)}
    probabilities = {}
    for (family, feature_set), (_, model, _, features) in kept.items():
        name = f"{family}_{feature_set}"
        probabilities[name] = aligned_probabilities(model, frame[features].astype(np.float32),
                                                    original_classes, n_classes)
        predictions[name] = probabilities[name].argmax(axis=1)
    actual = frame["label"].to_numpy()
    for name, labels in predictions.items():
        for split in ("validation", "test"):
            for station in ("todas", *sorted(frame["estacion"].unique())):
                mask = frame["split"].eq(split).to_numpy()
                if station != "todas":
                    mask = mask & frame["estacion"].eq(station).to_numpy()
                context = {"pollutant": pollutant, "scheme": spec["scheme"], "resolution": "level", "model": name,
                           "selected_by_validation": name == winner_name, "split": split, "station": station}
                evaluate(actual[mask], labels[mask], spec["classes"], context, *records)

    exported = frame[["estacion", "fecha", "split", "value", LAG, "label"]].rename(
        columns={"value": f"{pollutant}_promedio_diario"})
    for name, labels in predictions.items():
        exported[f"prediction_{name}"] = labels
    for index in range(n_classes):
        exported[f"selected_probability_class_{index}"] = probabilities[winner_name][:, index]
    exported.to_csv(output / f"predicciones_{pollutant}.csv", index=False, encoding="utf-8-sig", float_format="%.9g")

    _, model, params, features = kept[(winner["model"], winner["feature_set"])]
    target = output / "modelos" / f"{pollutant}_{winner_name}"
    if winner["model"] == "XGBoost":
        target = target.with_suffix(".ubj")
        model.save_model(target)
    else:
        target = target.with_suffix(".joblib")
        joblib.dump(model, target, compress=3)
    metadata = {"pollutant": pollutant, "model": winner_name, "features": features, "scale": spec,
                "parameters": params | {"random_state": config["seed"]},
                "original_classes": original_classes.tolist(), "class_names": spec["classes"],
                "train_size": len(parts["train"]), "validation_size": len(parts["validation"]),
                "test_size": len(parts["test"]), "probabilities_calibrated": False, "model_sha256": sha256(target)}
    write_json(target.with_suffix(".meta.json"), metadata)

    # Interpretación con el mejor XGBoost del conjunto de entradas ganador (TreeSHAP exacto).
    shap_name = f"XGBoost_{winner['feature_set']}"
    _, shap_model, _, shap_features = kept[("XGBoost", winner["feature_set"])]
    test = parts["test"]
    importance, positive = shap_importance(shap_model, test[shap_features].astype(np.float32),
                                           shap_features, n_classes - 1)
    shap = {"model": shap_name, "positive_class": spec["classes"][-1],
            "importance": importance.groupby(feature_group).sum().sort_values(ascending=False),
            "positive": positive, "values": test[shap_features]}
    return best, trials, selection, metadata, shap, daily


def make_figures(output, report, daily_frames, distribution, matrices, trials, shaps):
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "tmp/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "white", "savefig.facecolor": "white"})
    folder = output / "figuras"
    folder.mkdir(exist_ok=True)
    scales = report["scales"]["pollutants"]
    pollutants = list(scales)

    fig, axes = plt.subplots(1, len(pollutants), figsize=(15, 4.5), layout="constrained")
    for ax, pollutant in zip(axes, pollutants):
        part = distribution.loc[distribution["pollutant"].eq(pollutant) & distribution["station"].eq("todas")]
        names = scales[pollutant]["classes"]
        positions = np.arange(len(names))
        for offset, split, title in zip((-.27, 0, .27), ("train", "validation", "test"),
                                        ("Entrenamiento", "Validación", "Prueba")):
            values = part.loc[part["split"].eq(split)].sort_values("label")["fraction"].to_numpy()
            ax.bar(positions + offset, values * 100, width=.27, color=COLORS[split], label=title)
        ax.set(xticks=positions, xticklabels=[name.replace(" ", "\n", 2) for name in names],
               ylabel="% de estación-días", ylim=(0, 105), title=pollutant)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Clases OMS 2021 sobre promedios diarios: distribución natural por período", fontweight="bold")
    fig.savefig(folder / "distribucion_oms.png", dpi=150)
    plt.close(fig)

    modeled = report["config"]["pollutants"]
    fig, axes = plt.subplots(1, len(modeled), figsize=(13, 4.5), layout="constrained")
    for ax, pollutant in zip(axes, modeled):
        part = trials.loc[trials["pollutant"].eq(pollutant)]
        for (family, feature_set), group in part.groupby(["model", "feature_set"]):
            curve = group.groupby("weight_power")["validation_f1_macro"].max()
            ax.plot(curve.index, curve.values, marker="o", label=f"{family} · {feature_set}",
                    ls="-" if feature_set == "rezago" else "--",
                    color="#147d92" if family == "RandomForest" else "#e28a35")
        ax.set(xlabel="Exponente de ponderación de clases", ylabel="Mejor F1 macro en validación",
               title=pollutant, xticks=report["config"]["weight_powers"])
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Selección por validación: técnica, entradas y ponderación", fontweight="bold")
    fig.savefig(folder / "seleccion_validacion.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(modeled), figsize=(12, 4.8), layout="constrained")
    for ax, pollutant in zip(axes, modeled):
        winner = report["selections"][pollutant]["selected"]
        row = next(item for item in matrices if item["pollutant"] == pollutant and item["model"] == winner
                   and item["split"] == "test" and item["station"] == "todas")
        matrix = np.asarray(row["matrix"])
        support = matrix.sum(axis=1, keepdims=True)
        fractions = np.divide(matrix, support, out=np.zeros_like(matrix, dtype=float), where=support != 0)
        ax.imshow(fractions, vmin=0, vmax=1, cmap="Blues")
        for (i, j), count in np.ndenumerate(matrix):
            ax.text(j, i, f"{count}\n{fractions[i, j]:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if fractions[i, j] > .5 else "#14263d")
        labels = [textwrap.fill(name, 14) for name in row["classes"]]
        ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
               xlabel="Clase predicha", ylabel="Clase real", title=f"{pollutant} — {winner}")
    fig.suptitle("Matrices de confusión en prueba (2025–2026) del modelo elegido por validación", fontweight="bold")
    fig.savefig(folder / "matrices_oms.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(modeled), figsize=(13, 4.8), layout="constrained")
    for ax, pollutant in zip(axes, modeled):
        top = shaps[pollutant]["importance"].head(10)[::-1]
        ax.barh(top.index, top.values / shaps[pollutant]["importance"].sum() * 100, color="#147d92")
        ax.set(xlabel="% de la contribución SHAP absoluta media", title=f"{pollutant} — {shaps[pollutant]['model']}")
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Qué variables usa el modelo (SHAP en prueba, agrupando one-hot y resúmenes diarios)",
                 fontweight="bold")
    fig.savefig(folder / "shap_importancia.png", dpi=150)
    plt.close(fig)

    weather = ("wind_speed_10m_mean", "temperature_2m_mean", "precipitation_sum", "relative_humidity_2m_mean")
    fig, axes = plt.subplots(len(modeled), len(weather), figsize=(16, 7.5), layout="constrained")
    for row_axes, pollutant in zip(axes, modeled):
        values, positive = shaps[pollutant]["values"], shaps[pollutant]["positive"]
        for ax, feature in zip(row_axes, weather):
            ax.scatter(values[feature], positive[feature], s=6, alpha=.35, color="#147d92")
            ax.axhline(0, color="#8193a6", lw=.8)
            ax.set(xlabel=feature, ylabel="SHAP (log-odds)" if feature == weather[0] else "",
                   title=f"{pollutant}: {shaps[pollutant]['positive_class']}" if feature == weather[0] else "")
            ax.grid(alpha=.2)
    fig.suptitle("Efecto de la meteorología sobre la clase más alta (cada punto es un estación-día de prueba)",
                 fontweight="bold")
    fig.savefig(folder / "shap_dependencia.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(len(modeled), 2, figsize=(15, 7.5), layout="constrained")
    for row_axes, pollutant in zip(axes, modeled):
        frame = daily_frames[pollutant].assign(supera=lambda d: d["label"].gt(0) * 100,
                                               year=lambda d: d["fecha"].dt.year,
                                               month=lambda d: d["fecha"].dt.month)
        for station, color in STATION_COLORS.items():
            subset = frame.loc[frame["estacion"].eq(station)]
            yearly = subset.groupby("year")["supera"].mean()
            row_axes[0].plot(yearly.index, yearly.values, marker="o", color=color, label=station)
            monthly = subset.groupby("month")["supera"].mean()
            row_axes[1].plot(monthly.index, monthly.values, marker="o", color=color, label=station)
        guide = report["scales"]["pollutants"][pollutant]["upper_bounds"][0]
        unit = report["scales"]["pollutants"][pollutant]["unit"]
        row_axes[0].set(title=f"{pollutant}: % de días sobre la guía OMS ({guide} {unit}) por año",
                        ylabel="% de estación-días")
        row_axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        row_axes[1].set(title=f"{pollutant}: estacionalidad (todos los años)", xticks=range(1, 13),
                        xlabel="Mes", ylabel="% de estación-días")
        for ax in row_axes:
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Patrones observados: tendencia anual y ciclo estacional de excedencias", fontweight="bold")
    fig.savefig(folder / "patrones_excedencias.png", dpi=150)
    plt.close(fig)


def md_table(frame):
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "| " + " | ".join("---" for _ in frame.columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = ["no evaluable" if pd.isna(v) else f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)
                  for v in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(output, report, metrics, details, finalists, shaps, distribution):
    columns = ["pollutant", "model", "n", "accuracy", "precision_macro", "recall_macro", "f1_macro",
               "balanced_accuracy", "false_positive_rate_macro", "f1_weighted"]
    test = metrics.loc[metrics["split"].eq("test") & metrics["station"].eq("todas")]
    observations = []
    for pollutant in report["config"]["pollutants"]:
        winner = report["selections"][pollutant]["selected"]
        chosen = test.loc[test["pollutant"].eq(pollutant) & test["model"].eq(winner)].iloc[0]
        base = test.loc[test["pollutant"].eq(pollutant) & test["model"].eq(MAJORITY)].iloc[0]
        top = ", ".join(f"{name} ({value / shaps[pollutant]['importance'].sum():.0%})"
                        for name, value in shaps[pollutant]["importance"].head(4).items())
        observations.append(
            f"{pollutant}: validación eligió {winner}. En prueba, F1 macro {chosen['f1_macro']:.3f} frente a "
            f"{base['f1_macro']:.3f} de la clase mayoritaria; balanced accuracy {chosen['balanced_accuracy']:.1%} "
            f"y accuracy {chosen['accuracy']:.1%}. Variables más usadas (SHAP): {top}.")
    co = distribution.loc[distribution["pollutant"].eq("CO") & distribution["station"].eq("todas")
                          & distribution["label"].eq(1), "count"].sum()
    observations.append(f"CO: {co} estación-días superan la guía OMS de 4 mg/m³ en todo el período; con una sola "
                        "clase no hay problema de clasificación y CO se analiza sólo con la regresión de la entrega.")
    report["observations"] = observations
    notes = [
        "Motivación: con los cortes EPA casi todas las jornadas son Buena (0 Desfavorable en prueba) y la accuracy "
        "es alta sin detectar episodios. Las guías OMS 2021 son el estándar sanitario internacional vigente y "
        "más exigente; producen clases evaluables en NO2 y PM10.",
        "Las guías OMS se definen sobre promedios de 24 h. Se usa el promedio de los registros publicados del día "
        "de medición (al menos 18 horas válidas y 24 filas meteorológicas, igual que la clasificación diaria "
        "original). NO2 se publica horario; PM10 ya es un móvil de 24 h, por lo que su promedio diario es una "
        "aproximación. No es un dictamen de cumplimiento normativo.",
        "Cortes: NO2 guía 25 µg/m³ y objetivo intermedio 2 de 50 µg/m³ (el objetivo intermedio 1, 120 µg/m³, "
        "tiene muy pocos días y se agrupa); PM10 guía 45 µg/m³. Se reutilizan los factores de conversión del "
        "experimento de índices alternativos.",
        "Se mantienen los cortes temporales (entrenamiento antes de 2024, validación 2024, prueba desde 2025, "
        "purga de 2 días), las entradas diarias del modelado original y la exclusión de objetivos interpolados.",
        "Se compara un conjunto adicional de entradas con el promedio del día anterior del mismo contaminante "
        "(pendiente de la segunda entrega). Es información disponible al estimar el día siguiente.",
        "Grilla explícita por validación: Random Forest (max_depth 8/12/sin límite × min_samples_leaf 5/20) y "
        "XGBoost (max_depth 3/4/6 × 300/600 árboles), cada uno con pesos de clase 0, 0.5 y 1. Se elige por F1 "
        "macro de validación; la selección se guarda antes de calcular prueba.",
        "Las métricas de prueba se informan para el elegido y para los finalistas de cada familia y conjunto de "
        "entradas, sin cambiar la elección. Accuracy se lee junto a F1 macro, recall por clase y la matriz.",
        "La interpretación usa TreeSHAP nativo de XGBoost sobre la prueba: describe cómo usa el modelo cada "
        "variable, no relaciones causales. ERA5 es meteorología histórica; un pronóstico operativo agregaría "
        "el error de la previsión meteorológica.",
    ]
    trial_cols = ["pollutant", "model", "feature_set", "weight_power", "max_depth", "min_samples_leaf",
                  "n_estimators", "validation_accuracy", "validation_f1_macro", "validation_balanced_accuracy"]
    finalists = finalists.assign(
        max_depth=finalists["max_depth"].map(lambda v: "sin límite" if pd.isna(v) else str(int(v))),
        min_samples_leaf=finalists["min_samples_leaf"].map(lambda v: "—" if pd.isna(v) else str(int(v))))
    selected_detail = details.loc[details["split"].eq("test") & details["station"].eq("todas")
                                  & details["selected_by_validation"]]
    stations = metrics.loc[metrics["split"].eq("test") & metrics["station"].ne("todas")
                           & metrics["selected_by_validation"]]
    shap_rows = pd.concat({p: (s["importance"] / s["importance"].sum()).head(10) for p, s in shaps.items()},
                          names=["pollutant", "variable"]).rename("share").reset_index()
    dist = distribution.loc[distribution["station"].eq("todas")].pivot_table(
        index=["pollutant", "category", "label"], columns="split", values="count", fill_value=0).reset_index()
    dist = dist.sort_values(["pollutant", "label"]).drop(columns="label")
    tables = [
        ("Distribución de clases OMS (estación-días)", dist),
        ("Finalistas por validación", finalists[trial_cols]),
        ("Prueba: elegido, finalistas y referencia", test[columns]),
        ("Detalle por clase del elegido — prueba", selected_detail[
            ["pollutant", "model", "category", "support", "precision", "recall", "f1", "false_positive_rate",
             "tp", "fp", "fn", "tn"]]),
        ("Elegido por estación — prueba", stations[["pollutant", "model", "station", "n", "accuracy", "f1_macro",
                                                    "balanced_accuracy", "false_positive_rate_macro"]]),
        ("Importancia SHAP agrupada (top 10)", shap_rows),
    ]
    figures = [("distribucion_oms.png", "Distribución de clases"),
               ("seleccion_validacion.png", "Selección por validación"),
               ("matrices_oms.png", "Matrices de confusión"),
               ("shap_importancia.png", "Importancia SHAP"),
               ("shap_dependencia.png", "Dependencia SHAP de la meteorología"),
               ("patrones_excedencias.png", "Patrones de excedencia")]
    md = ["# Clasificación diaria con las guías OMS 2021", "Estación-día, NO₂ y PM10. CO se describe."]
    md.extend("- " + item for item in observations)
    md.append("\n## Método y alcance\n")
    md.extend("- " + note for note in notes)
    html = ["<!doctype html><html lang='es'><meta charset='utf-8'><title>Clasificación OMS</title>",
            "<style>body{font:16px system-ui;max-width:1450px;margin:35px auto;padding:0 20px;color:#193049}"
            "table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border:1px solid #ccd5df}"
            "th{background:#edf3f8}section{overflow-x:auto}img{max-width:100%}li{margin:10px 0}</style>",
            "<h1>Clasificación diaria con las guías OMS 2021</h1><ul>"]
    html.extend(f"<li>{escape(item)}</li>" for item in observations)
    html.append("</ul><h2>Método y alcance</h2><ul>")
    html.extend(f"<li>{escape(note)}</li>" for note in notes)
    html.append("</ul>")
    for title, frame in tables:
        md.extend(["\n## " + title + "\n", md_table(frame)])
        html.extend([f"<h2>{escape(title)}</h2><section>", frame.to_html(index=False, na_rep="no evaluable",
                     float_format=lambda v: f"{v:.4f}"), "</section>"])
    for file, label in figures:
        md.append(f"![{label}](figuras/{file})")
        html.append(f"<h2>{label}</h2><img alt='{label}' src='figuras/{file}'>")
    md.append("\n## Fuentes\n")
    html.append("<h2>Fuentes</h2><ul>")
    for name, url in report["scales"]["sources"].items():
        md.append(f"- [{name}]({url})")
        html.append(f"<li><a href='{escape(url, quote=True)}'>{escape(name)}</a></li>")
    html.append("</ul></html>")
    (output / "resumen_oms.md").write_text("\n\n".join(md) + "\n", encoding="utf-8")
    (output / "resumen_oms.html").write_text("\n".join(html), encoding="utf-8")


def build_notebook(output, execute=True):
    import nbformat
    from jupyter_client import AsyncKernelManager
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter

    notebook_path = PROJECT / "notebooks/tercera_entrega_oms.ipynb"
    relative_output = output.resolve().relative_to(PROJECT).as_posix()
    cells = []
    markdown = lambda text: cells.append(nbformat.v4.new_markdown_cell(text))
    code = lambda text: cells.append(nbformat.v4.new_code_cell(text))
    markdown("# Tercera entrega — Clasificación con las guías OMS 2021\n"
             "Calidad del aire en CABA · UTN FRC · Grupo 12\n\n"
             "Continúa `tercera_entrega_modelado.ipynb`. Con los cortes EPA casi todos los días son *Buena* y el "
             "clasificador alcanza una accuracy alta sin detectar episodios. Aquí se reemplaza la etiqueta por las "
             "guías OMS 2021 (promedios de 24 h), se justifican los parámetros con una grilla en validación y se "
             "interpretan los modelos. Las celdas leen resultados persistidos por `.\\run.ps1 classify-oms`.")
    code('from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image, Markdown\n'
         'ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()\n'
         f'OUT = ROOT / "{relative_output}"\n'
         'report = json.loads((OUT / "reporte.json").read_text(encoding="utf-8"))\n'
         'metrics = pd.read_csv(OUT / "metricas_clasificacion.csv")\n'
         'detail = pd.read_csv(OUT / "metricas_por_clase.csv")\n'
         'trials = pd.read_csv(OUT / "candidatos_validacion.csv")\n'
         'pd.set_option("display.max_columns", 20)\n'
         'print("Cobertura diaria:", json.dumps(report["coverage"], indent=1))')
    markdown("## 1. Por qué cambiar de estándar\n\n"
             "| Contaminante | Guía OMS 2021 (24 h) | Objetivo intermedio usado | EPA (alto) |\n"
             "|---|---|---|---|\n"
             "| NO₂ | 25 µg/m³ (≈13 ppb) | 50 µg/m³ (OI-2) | ≥101 ppb |\n"
             "| PM10 | 45 µg/m³ | — | ≥155 µg/m³ |\n"
             "| CO | 4 mg/m³ (≈3,5 ppm) | — | ≥9,5 ppm |\n\n"
             "Los cortes EPA están pensados para episodios severos; en CABA casi no ocurren con estos contaminantes. "
             "La OMS fija valores basados en efectos en la salud a exposiciones más bajas. "
             "La etiqueta pasa a responder una pregunta útil: *¿el aire de esta estación cumple hoy la guía sanitaria?*")
    code('dist = pd.read_csv(OUT / "distribucion_categorias.csv")\n'
         'display(dist.query("station == \'todas\'").pivot_table(index=["pollutant","category"], columns="split", values="count", fill_value=0))\n'
         'display(Image(filename=str(OUT / "figuras/distribucion_oms.png")))')
    markdown("NO₂ queda con tres clases de tamaño comparable y PM10 con una excedencia minoritaria pero evaluable "
             "(decenas de días en prueba, cientos en entrenamiento). CO no supera la guía ningún día: no se clasifica.")
    markdown("## 2. Algoritmos, parámetros y selección\n\n"
             "Se mantienen la partición temporal, las entradas diarias y las métricas del modelado original. "
             "Se comparan una referencia (clase mayoritaria de entrenamiento), Random Forest y XGBoost. "
             "Justificación de la grilla:\n\n"
             "- **Random Forest**: 300 árboles (el error se estabiliza; más árboles sólo agregan tiempo), "
             "`max_features=0.8` como en el modelado original; se prueba profundidad 8, 12 o sin límite y "
             "`min_samples_leaf` 5 o 20 para controlar el sobreajuste con ~11.000 estación-días.\n"
             "- **XGBoost**: tasa de aprendizaje 0,05 con 300 o 600 árboles (aprendizaje lento, más estable); "
             "profundidad 3, 4 o 6 para variar las interacciones permitidas; `subsample` y `colsample_bytree` 0,8 "
             "y `reg_lambda=5` como regularización, igual que el modelado original.\n"
             "- **Ponderación de clases**: exponente 0 (sin pesos), 0,5 (suave) o 1 (balanceado), calculado sólo "
             "con entrenamiento. No se remuestrea ni se balancean validación y prueba.\n"
             "- **Entradas**: `meteo` (meteorología y calendario) o `rezago` (además, el promedio del día anterior).\n\n"
             "Se elige por F1 macro de validación (2024) y la elección se guarda antes de evaluar la prueba.")
    code('display(Image(filename=str(OUT / "figuras/seleccion_validacion.png")))\n'
         'display(pd.DataFrame({p: {"elegido": s["selected"], "F1 macro validación": round(s["validation_score"], 4)} '
         'for p, s in report["selections"].items()}).T)\n'
         'cols = ["model","feature_set","weight_power","max_depth","min_samples_leaf","n_estimators","validation_f1_macro","validation_balanced_accuracy"]\n'
         'for p in report["config"]["pollutants"]:\n'
         '    print(p, "— 10 mejores ensayos de validación")\n'
         '    display(trials.query("pollutant == @p").sort_values("validation_f1_macro", ascending=False)[cols].head(10))')
    markdown("## 3. Evaluación en prueba (2025–2026)\n\n"
             "Matriz de confusión, accuracy, precision, recall, F1 y tasa de falsos positivos "
             "(FPR = FP / (FP + TN) por clase frente al resto). La referencia mayoritaria muestra cuánto "
             "de la accuracy se explica sólo por el desbalance.")
    code('cols = ["pollutant","model","n","accuracy","precision_macro","recall_macro","f1_macro","balanced_accuracy","false_positive_rate_macro"]\n'
         'display(metrics.query("split == \'test\' and station == \'todas\'")[cols])\n'
         'display(Image(filename=str(OUT / "figuras/matrices_oms.png")))\n'
         'display(detail.query("split == \'test\' and station == \'todas\' and selected_by_validation")'
         '[["pollutant","model","category","support","precision","recall","f1","false_positive_rate","tp","fp","fn","tn"]])\n'
         'display(metrics.query("split == \'test\' and station != \'todas\' and selected_by_validation")'
         '[["pollutant","model","station","n","accuracy","f1_macro","balanced_accuracy"]])')
    markdown("## 4. Patrones e interpretación\n\n"
             "TreeSHAP descompone cada predicción en contribuciones de las variables (en log-odds). "
             "Se agrupan las columnas one-hot y los resúmenes diarios de una misma variable. "
             "Son asociaciones aprendidas por el modelo, no causas.")
    code('display(Image(filename=str(OUT / "figuras/shap_importancia.png")))\n'
         'display(Image(filename=str(OUT / "figuras/shap_dependencia.png")))\n'
         'display(Image(filename=str(OUT / "figuras/patrones_excedencias.png")))')
    code('display(Markdown((OUT / "resumen_oms.md").read_text(encoding="utf-8").split("## Método")[0]))')
    markdown("## 5. Reproducibilidad\n\n"
             "`.\\run.ps1 classify-oms` reentrena la grilla y regenera métricas, figuras, modelos y este notebook. "
             "Configuración: `config/clasificacion_oms.json` (grilla) y `config/indices_oms.json` (cortes OMS). "
             "No modifica los CSV de la segunda entrega ni los resultados de los experimentos anteriores.")
    nb = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python (.venv)", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    if execute:
        class ProjectKernelManager(AsyncKernelManager):
            def format_kernel_cmd(self, extra_arguments=None):
                self.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
                return super().format_kernel_cmd(extra_arguments)
        client = NotebookClient(nb, kernel_manager_class=ProjectKernelManager, timeout=180,
                                resources={"metadata": {"path": str(PROJECT)}})
        client.execute()
    nbformat.write(nb, notebook_path)
    html, _ = HTMLExporter().from_notebook_node(nb)
    notebook_path.with_suffix(".html").write_text(html, encoding="utf-8")
    return notebook_path


def run_oms_classifier(data_dir, config_path=None, output_dir=None):
    started = time.perf_counter()
    config_path = Path(config_path or PROJECT / "config/clasificacion_oms.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = PROJECT / "outputs/tercera_entrega"
    output = Path(output_dir or PROJECT / "outputs/clasificacion_oms")
    protected = [source, *(PROJECT / "outputs" / name for name in (
        "clasificacion_no2", "clasificacion_suave", "comparacion_clasificadores", "comparacion_indices"))]
    if output.resolve() in {path.resolve() for path in protected}:
        raise ValueError("Elegir una carpeta distinta de los experimentos anteriores")
    original = json.loads((source / "reporte.json").read_text(encoding="utf-8"))
    temporal = original["config"]
    data, _, audit = load_data(data_dir, temporal)
    if any(audit[key] != original[key] for key in ("input_sha256", "quality_sha256")):
        raise ValueError("Los CSV no corresponden al experimento original")
    scales = load_scales(SCALES)
    output.mkdir(parents=True, exist_ok=True)
    (output / "modelos").mkdir(exist_ok=True)
    write_json(output / "auditoria.json", audit)
    summaries, details, matrices, trials, finalists, distribution = [], [], [], [], [], []
    selections, models, shaps, daily_frames, coverage = {}, {}, {}, {}, {}
    for pollutant in TARGETS:
        spec = scales["pollutants"][pollutant]
        daily, base_features, coverage[pollutant] = daily_frame(data, pollutant, temporal)
        labels = classify_concentration(daily["value"], spec)
        distribution.extend(distribution_rows(daily, pollutant, spec["scheme"], "level", spec["classes"], labels))
        if pollutant not in config["pollutants"]:
            continue
        (output / pollutant).mkdir(exist_ok=True)
        best, pollutant_trials, selections[pollutant], models[pollutant], shaps[pollutant], daily_frames[pollutant] = \
            fit_pollutant(daily, base_features, pollutant, spec, config, output, (summaries, details, matrices))
        trials.extend(pollutant_trials)
        finalists.append(best)
        print(f"{pollutant}: elegido {selections[pollutant]['selected']} por validación", flush=True)
        gc.collect()
    metrics, detail_frame = pd.DataFrame(summaries), pd.DataFrame(details)
    trial_frame, distribution = pd.DataFrame(trials), pd.DataFrame(distribution)
    metrics.to_csv(output / "metricas_clasificacion.csv", index=False, encoding="utf-8-sig")
    detail_frame.to_csv(output / "metricas_por_clase.csv", index=False, encoding="utf-8-sig")
    trial_frame.to_csv(output / "candidatos_validacion.csv", index=False, encoding="utf-8-sig")
    distribution.to_csv(output / "distribucion_categorias.csv", index=False, encoding="utf-8-sig")
    write_json(output / "matrices_confusion.json", matrices)
    pd.concat({p: s["importance"] for p, s in shaps.items()}, names=["pollutant", "variable"]).rename(
        "mean_abs_shap").reset_index().to_csv(output / "importancia_shap.csv", index=False, encoding="utf-8-sig")
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "config": config, "scales": scales,
              "temporal_config": temporal, "coverage": coverage, "selections": selections, "models": models,
              "input_sha256": audit["input_sha256"], "quality_sha256": audit["quality_sha256"],
              "config_sha256": sha256(config_path), "scales_sha256": sha256(SCALES),
              "code_sha256": sha256(Path(__file__)),
              "versions": {name: importlib.metadata.version(name) for name in
                           ("numpy", "pandas", "scikit-learn", "xgboost", "matplotlib")},
              "elapsed_seconds": time.perf_counter() - started}
    make_figures(output, report, daily_frames, distribution, matrices, trial_frame, shaps)
    write_report(output, report, metrics, detail_frame, pd.concat(finalists), shaps, distribution)
    write_json(output / "reporte.json", report)
    for item in report["observations"]:
        print(item, flush=True)
    build_notebook(output)
    return output / "resumen_oms.html"
