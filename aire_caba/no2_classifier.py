"""Piloto de clasificación horaria directa de NO2 con niveles británicos DAQI."""

from datetime import datetime, timezone
import gc
from html import escape
import importlib.metadata
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from .air_quality_scales import classify_concentration, load_scales
from .aqi import classification_metrics
from .compare_indices import distribution_rows, evaluate
from .model_dataset import sha256
from .modeling import PROJECT, load_data, write_json


REGRESSION_NAME = "Regresion_XGBoost"
MAJORITY_NAME = "Mayoria_train"
DIRECT_NAMES = ("RandomForest", "XGBoost")


def encode_targets(labels):
    """Codifica sólo clases presentes, conservando su correspondencia con DAQI."""
    labels = np.asarray(labels, dtype=int)
    classes = np.unique(labels)
    if len(classes) < 2 or np.any(classes < 0):
        raise ValueError("Se requieren al menos dos clases de entrenamiento válidas")
    return classes, np.searchsorted(classes, labels)


def training_weights(labels, power):
    """Pesos de frecuencia inversa, suavizados y normalizados con entrenamiento."""
    if not 0 <= power <= 1:
        raise ValueError("El exponente de pesos debe estar entre cero y uno")
    labels = np.asarray(labels, dtype=int)
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    weights = (len(labels) / (len(counts) * counts[inverse])) ** power
    return weights / weights.mean()


def aligned_probabilities(model, X, original_classes, declared_count=10):
    """Expande probabilidades a DAQI; las clases no aprendidas conservan cero."""
    encoded_classes = np.asarray(model.classes_, dtype=int)
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    if np.any(encoded_classes < 0) or np.any(encoded_classes >= len(original_classes)):
        raise ValueError("El modelo no corresponde a la codificación de entrenamiento")
    output = np.zeros((len(X), declared_count), dtype=float)
    output[:, np.asarray(original_classes)[encoded_classes]] = probabilities
    if not np.isfinite(output).all() or np.any(output < 0) or not np.allclose(output.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Probabilidades inválidas")
    return output


def reconcile_regression(frame, regression):
    """Une los mismos objetivos evaluados por el regresor; rechaza datos obsoletos."""
    saved = regression.copy()
    saved["fecha_hora"] = pd.to_datetime(saved["fecha_hora"])
    expected = frame[["estacion", "fecha_hora", "split", "NO2"]]
    joined = expected.merge(saved, on=["estacion", "fecha_hora", "split"], how="outer",
                            validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all() or not np.allclose(joined["NO2"], joined["actual"], atol=1e-10, rtol=0):
        raise ValueError("Las predicciones del regresor no coinciden con las filas de evaluación")
    return frame[["estacion", "fecha_hora", "split"]].merge(
        saved[["estacion", "fecha_hora", "split", "prediction_XGBoost"]],
        on=["estacion", "fecha_hora", "split"], validate="one_to_one", how="left")["prediction_XGBoost"].to_numpy()


def fit_candidates(train, validation, features, config, spec, output, pollutant="NO2", cached_models=None):
    original_classes, encoded = encode_targets(train["label"].to_numpy())
    xtrain, xval = train[features], validation[features]
    yval = validation["label"].to_numpy()
    trials, parameters, scores = [], {}, {}
    cached_models = cached_models or {}
    for family in DIRECT_NAMES:
        best_model, best_score, best_parameters = None, -1, None
        for power in config["weight_powers"]:
            started = time.perf_counter()
            reused = (family, power) in cached_models
            if family == "RandomForest":
                model = RandomForestClassifier(**config["random_forest"], random_state=config["seed"],
                                               n_jobs=config["threads"])
                hyperparameters = config["random_forest"]
            else:
                model = XGBClassifier(**config["xgboost"], num_class=len(original_classes),
                                      random_state=config["seed"], n_jobs=config["threads"])
                hyperparameters = config["xgboost"] | {"num_class": len(original_classes)}
            weights = training_weights(train["label"], power)
            if reused:
                model = cached_models[(family, power)]
                print(f"{pollutant} / {family}: reutilizando el ajuste de pesos {power:g}", flush=True)
            else:
                print(f"{pollutant} / {family}: ajuste con exponente de pesos {power:g}", flush=True)
                model.fit(xtrain, encoded, sample_weight=weights)
            probabilities = aligned_probabilities(model, xval, original_classes, len(spec["classes"]))
            result = classification_metrics(yval, probabilities.argmax(axis=1), spec["classes"])
            score = result["summary"][config["selection_metric"]]
            params = hyperparameters | {"weight_power": power}
            trials.append({"model": family, "weight_power": power, "parameters": params, "reused": reused,
                           "fit_seconds": time.perf_counter() - started,
                           **{f"validation_{key}": value for key, value in result["summary"].items()}})
            print(f"{pollutant} / {family}: F1 macro validación={score:.4f}, "
                  f"recall clases presentes={result['summary']['balanced_accuracy']:.2%}", flush=True)
            if score > best_score:
                best_model, best_score, best_parameters = model, score, params
            else:
                del model
            gc.collect()
        bundle = {"model": best_model, "features": features, "target": f"{pollutant}_{spec['scheme']}_hourly",
                  "original_classes": original_classes.tolist(), "class_names": spec["classes"],
                  "scale": spec, "parameters": best_parameters,
                  "validation_f1_macro": best_score, "probabilities_calibrated": False}
        joblib.dump(bundle, output / "models" / f"{pollutant}_{family}.joblib", compress=3)
        parameters[family], scores[family] = best_parameters, best_score
        del best_model, bundle
        gc.collect()
    selection = {"selected_classifier": max(scores, key=scores.get), "selection_metric": config["selection_metric"],
                 "validation_scores": scores, "parameters": parameters,
                 "trained_classes": original_classes.tolist(),
                 "unlearned_classes": sorted(set(range(len(spec["classes"]))) - set(original_classes.tolist()))}
    # Se persiste la selección antes de calcular cualquier métrica de test.
    write_json(output / "seleccion_validacion.json", selection)
    pd.DataFrame(trials).drop(columns="parameters").to_csv(output / "candidatos_validacion.csv", index=False, encoding="utf-8-sig")
    return selection, trials, original_classes


def make_figures(output, metrics, details, matrices, selected):
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "tmp/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = output / "figuras"
    folder.mkdir(exist_ok=True)
    names = (REGRESSION_NAME, *DIRECT_NAMES)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for ax, name in zip(axes, names):
        row = next(item for item in matrices if item["model"] == name and item["split"] == "test"
                   and item["station"] == "todas" and item["resolution"] == "level")
        matrix = np.asarray(row["matrix"])
        support = matrix.sum(axis=1, keepdims=True)
        fractions = np.divide(matrix, support, out=np.zeros_like(matrix, dtype=float), where=support != 0)
        ax.imshow(fractions, vmin=0, vmax=1, cmap="Blues")
        for (i, j), count in np.ndenumerate(matrix):
            if count:
                ax.text(j, i, f"{count}\n{fractions[i, j]:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if fractions[i, j] > .5 else "#1b3048")
        ax.set(xticks=range(10), yticks=range(10), xticklabels=range(1, 11), yticklabels=range(1, 11),
               xlabel="Nivel DAQI predicho", ylabel="Nivel DAQI real", title=name + (" · elegido" if name == selected else ""))
    fig.suptitle("NO₂: regresión convertida y clasificación directa — prueba", fontweight="bold")
    fig.savefig(folder / "matrices_NO2.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    selected_rows = details.loc[details["resolution"].eq("level") & details["split"].eq("test")
                                & details["station"].eq("todas") & details["support"].gt(0)]
    labels = sorted(selected_rows["class_id"].unique())
    positions = np.arange(len(labels))
    for offset, name, color in zip((-.3, -.1, .1, .3), (MAJORITY_NAME, REGRESSION_NAME, *DIRECT_NAMES),
                                   ("#8193a6", "#596379", "#147d92", "#e28a35")):
        part = selected_rows.loc[selected_rows["model"].eq(name)].set_index("class_id").reindex(labels)
        ax.bar(positions + offset, part["recall"] * 100, width=.2, label=name, color=color)
    ax.set(xticks=positions, xticklabels=[f"Nivel {i+1}" for i in labels], ylabel="Recall (%)", ylim=(0, 108),
           title="Detección por nivel presente en prueba; todos pertenecen a contaminación baja")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(axis="y", alpha=.2)
    fig.savefig(folder / "recall_NO2.png", dpi=160)
    plt.close(fig)


def make_report(output, report, metrics, details):
    test = metrics.loc[metrics["split"].eq("test") & metrics["station"].eq("todas")
                       & metrics["resolution"].eq("level")]
    chosen = report["selection"]["selected_classifier"]
    row = test.loc[test["model"].eq(chosen)].iloc[0]
    previous = test.loc[test["model"].eq(REGRESSION_NAME)].iloc[0]
    difference = row["f1_macro_supported"] - previous["f1_macro_supported"]
    outcome = (f"El clasificador elegido por validación fue {chosen}. En prueba, su F1 macro sobre clases "
               f"presentes es {row['f1_macro_supported']:.4f}, frente a {previous['f1_macro_supported']:.4f} "
               f"del método de regresión convertido (diferencia {difference:+.4f}). "
               f"Accuracy {row['accuracy']:.2%} y balanced accuracy {row['balanced_accuracy']:.2%}.")
    report["outcome"] = outcome
    notes = [
        "Un clasificador de NO2 compartido entre estaciones; la estación es una entrada. "
        "La salida es el nivel británico, no una concentración ni una categoría diaria conjunta.",
        "Se reutilizan las mismas 59 entradas, objetivos medidos/censurados y filtros de plausibilidad. "
        "NO2 y los otros contaminantes no son predictores. Las concentraciones originales se conservan.",
        "Entrenamiento anterior a 2024; validación en 2024; prueba desde 2025, con la misma purga de dos días. "
        "Pesos y codificación de clases se calculan sólo con entrenamiento. Los modelos y el exponente de "
        "ponderación se eligen únicamente por F1 macro de validación; no se retocan después de ver el test.",
        "Se prueban pesos sin balancear, raíz de frecuencia inversa y frecuencia inversa completa. "
        "Todos se normalizan a media uno. No se crean ejemplos sintéticos ni se balancean validación y prueba.",
        "Las clases ausentes de entrenamiento no pueden aprenderse: conservan probabilidad cero y se indican "
        "en el manifiesto. Se codifican las presentes y se decodifican para no desplazar los niveles DAQI.",
        "Validación y prueba sólo contienen niveles 1, 2 y 3, todos de contaminación baja. "
        "No puede concluirse que el piloto detecte episodios moderados, altos o muy altos. "
        "El nivel 3 sólo tiene 19 ejemplos en test: sus métricas son inestables.",
        "F1/precision/recall macro promedian los diez niveles fijos y usan cero para clases no evaluables. "
        "Las variantes macro_supported consideran sólo niveles reales presentes. Se muestran ambas. "
        "FPR se calcula por clase contra el resto; cero FPR no garantiza detección.",
        "Se mantiene la conversión NO2 ppb × 1.9125 a µg/m³ y el redondeo único DAQI. "
        "Las cuatro bandas se evalúan aparte; en test todos los objetivos pertenecen a la banda baja.",
        "Los algoritmos siguen usando meteorología ERA5 histórica. No se evalúa aún el error adicional de "
        "pronósticos meteorológicos a 24/48 h. Las probabilidades del clasificador no están calibradas.",
        "El período de prueba ya se había inspeccionado al proponer este piloto: esta comparación es exploratoria. "
        "Una confirmación independiente requiere datos posteriores o un nuevo diseño de evaluación temporal.",
    ]
    cols = ["model", "n", "accuracy", "precision_macro", "recall_macro", "f1_macro",
            "f1_macro_supported", "balanced_accuracy", "false_positive_rate_macro"]
    chosen_detail = details.loc[details["model"].eq(chosen) & details["split"].eq("test")
                                & details["station"].eq("todas") & details["resolution"].eq("level")]
    tables = [("Comparación en prueba: diez niveles fijos", test[cols]),
              ("Detalle por nivel del clasificador elegido", chosen_detail[["category", "support", "predicted_count",
                "precision", "recall", "f1", "false_positive_rate", "tp", "fp", "fn", "tn"]]),
              ("Resultados por estación del clasificador elegido", metrics.loc[
                  metrics["model"].eq(chosen) & metrics["split"].eq("test")
                  & metrics["station"].ne("todas") & metrics["resolution"].eq("level"), ["station", *cols[1:]]]),
              ("Resultados de validación usados para comparar enfoques", metrics.loc[
                  metrics["split"].eq("validation") & metrics["station"].eq("todas") & metrics["resolution"].eq("level"), cols])]

    def md_table(frame):
        lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join("---" for _ in frame.columns) + " |"]
        for record in frame.itertuples(index=False, name=None):
            values = ["no evaluable" if pd.isna(v) else f"{v:.4f}" if isinstance(v, float) else str(v) for v in record]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    markdown = ["# Piloto: clasificación directa de NO₂", outcome, "## Método y límites\n"]
    markdown.extend("- " + note for note in notes)
    html = ["<!doctype html><html lang='es'><meta charset='utf-8'><title>Clasificación directa NO₂</title>",
            "<style>body{font:16px system-ui;max-width:1400px;margin:35px auto;padding:0 20px;color:#193049}"
            "table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border:1px solid #ccd5df}"
            "th{background:#edf3f8}section{overflow-x:auto}img{max-width:100%}li{margin:10px 0}</style>",
            "<h1>Piloto: clasificación directa de NO₂</h1>", f"<p>{escape(outcome)}</p><h2>Método y límites</h2><ul>"]
    html.extend(f"<li>{escape(note)}</li>" for note in notes)
    html.append("</ul>")
    for title, frame in tables:
        markdown.extend(["\n## " + title + "\n", md_table(frame)])
        html.extend([f"<h2>{escape(title)}</h2><section>", frame.to_html(index=False, na_rep="no evaluable",
                     float_format=lambda v: f"{v:.4f}"), "</section>"])
    for file, label in (("matrices_NO2.png", "Matrices de confusión"), ("recall_NO2.png", "Recall por nivel")):
        markdown.append(f"![{label}](figuras/{file})")
        html.append(f"<h2>{label}</h2><img alt='{label}' src='figuras/{file}'>")
    markdown.append("\n## Fuentes\n")
    html.append("<h2>Fuentes</h2><ul>")
    for name, url in report["scale_sources"].items():
        markdown.append(f"- [{name}]({url})")
        html.append(f"<li><a href='{escape(url, quote=True)}'>{escape(name)}</a></li>")
    html.append("</ul></html>")
    (output / "resumen_NO2.md").write_text("\n\n".join(markdown) + "\n", encoding="utf-8")
    (output / "clasificacion_NO2.html").write_text("\n".join(html), encoding="utf-8")


def run_no2_classifier(data_dir, config_path=None, output_dir=None):
    started = time.perf_counter()
    config_path = Path(config_path or PROJECT / "config/clasificacion_no2.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["selection_metric"] != "f1_macro" or not config["weight_powers"]:
        raise ValueError("El piloto selecciona con F1 macro y requiere al menos una ponderación")
    source = PROJECT / "outputs/tercera_entrega"
    output = Path(output_dir or PROJECT / "outputs/clasificacion_no2")
    if output.resolve() in (source.resolve(), (PROJECT / "outputs/comparacion_indices").resolve()):
        raise ValueError("Elegir una carpeta distinta para conservar los experimentos anteriores")
    original_report = json.loads((source / "reporte.json").read_text(encoding="utf-8"))
    scales = load_scales()
    spec = scales["pollutants"]["NO2"]
    data, features, audit = load_data(data_dir, original_report["config"])
    if any(audit[key] != original_report[key] for key in ("input_sha256", "quality_sha256")):
        raise ValueError("El dataset actual no coincide con el experimento de regresión")
    data = data.loc[data["NO2"].notna()].copy()
    data["label"] = classify_concentration(data["NO2"], spec)
    partitions = {name: data.loc[data["split"].eq(name)].copy() for name in ("train", "validation", "test")}
    if any(part.empty for part in partitions.values()) or any(name in features for name in ("CO", "NO2", "PM10", "label")):
        raise ValueError("Se requieren las tres particiones y entradas sin contaminantes ni etiquetas")
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    distribution = pd.DataFrame(distribution_rows(data, "NO2", spec["scheme"], "level", spec["classes"], data["label"]))
    distribution.to_csv(output / "distribucion_categorias.csv", index=False, encoding="utf-8-sig")
    write_json(output / "auditoria.json", audit)
    print(f"NO2: train={len(partitions['train']):,}, validation={len(partitions['validation']):,}, "
          f"test={len(partitions['test']):,}; {len(features)} entradas", flush=True)
    selection, trials, original_classes = fit_candidates(partitions["train"], partitions["validation"],
                                                        features, config, spec, output)
    validation_test = pd.concat([partitions["validation"], partitions["test"]], ignore_index=True)
    saved_regression = pd.read_csv(source / "predicciones_NO2.csv")
    previous = reconcile_regression(validation_test, saved_regression)
    predicted_labels = {REGRESSION_NAME: classify_concentration(previous, spec)}
    _, encoded = encode_targets(partitions["train"]["label"])
    dummy = DummyClassifier(strategy="most_frequent").fit(partitions["train"][features], encoded)
    probabilities = aligned_probabilities(dummy, validation_test[features], original_classes)
    predicted_labels[MAJORITY_NAME] = probabilities.argmax(axis=1)
    selected_probabilities = None
    importances = []
    for family in DIRECT_NAMES:
        bundle = joblib.load(output / "models" / f"NO2_{family}.joblib")
        probabilities = aligned_probabilities(bundle["model"], validation_test[features], original_classes)
        predicted_labels[family] = probabilities.argmax(axis=1)
        if family == selection["selected_classifier"]:
            selected_probabilities = probabilities
        importances.extend({"model": family, "feature": name, "importance": float(value)}
                           for name, value in zip(features, bundle["model"].feature_importances_))
        del bundle
        gc.collect()
    summaries, details, matrices = [], [], []
    actual = validation_test["label"].to_numpy()
    prediction_export = validation_test[["estacion", "fecha", "fecha_hora", "split", "NO2", "label"]].copy()
    for family, predicted in predicted_labels.items():
        prediction_export[f"prediction_{family}"] = predicted
        for resolution, names in (("level", spec["classes"]), ("band", spec["band_classes"])):
            current_actual, current_predicted = actual, predicted
            if resolution == "band":
                mapping = np.asarray(spec["level_to_band"])
                current_actual, current_predicted = mapping[actual], mapping[predicted]
            for split, part in partitions.items():
                if split == "train":
                    continue
                for station in ("todas", *sorted(part["estacion"].unique())):
                    mask = validation_test["split"].eq(split).to_numpy()
                    if station != "todas":
                        mask = mask & validation_test["estacion"].eq(station).to_numpy()
                    context = {"pollutant": "NO2", "scheme": spec["scheme"], "resolution": resolution,
                               "model": family, "split": split, "station": station,
                               "selected_classifier": family == selection["selected_classifier"]}
                    evaluate(current_actual[mask], current_predicted[mask], names, context, summaries, details, matrices)
    for index in range(len(spec["classes"])):
        prediction_export[f"selected_probability_level_{index+1}"] = selected_probabilities[:, index]
    prediction_export.to_csv(output / "predicciones_NO2.csv", index=False, encoding="utf-8-sig", float_format="%.9g")
    metrics, detail_frame = pd.DataFrame(summaries), pd.DataFrame(details)
    metrics.to_csv(output / "metricas_clasificacion.csv", index=False, encoding="utf-8-sig")
    detail_frame.to_csv(output / "metricas_por_clase.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(importances).to_csv(output / "importancia_variables.csv", index=False, encoding="utf-8-sig")
    write_json(output / "matrices_confusion.json", matrices)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "experiment": "direct_NO2_DAQI_classification",
              "config": config, "temporal_config": original_report["config"], "selection": selection,
              "features": features, "scale": spec, "scale_sources": {key: value for key, value in scales["sources"].items()
                                                                      if key != "india_thresholds"},
              "tuning_trials": trials, "sizes": {name: len(part) for name, part in partitions.items()},
              "input_sha256": audit["input_sha256"], "quality_sha256": audit["quality_sha256"],
              "source_predictions_sha256": sha256(source / "predicciones_NO2.csv"),
              "config_sha256": sha256(config_path), "scales_sha256": sha256(PROJECT / "config/indices_alternativos.json"),
              "code_sha256": sha256(Path(__file__)), "elapsed_seconds": time.perf_counter() - started,
              "independent_new_test": False, "probabilities_calibrated": False,
              "versions": {name: importlib.metadata.version(name) for name in
                           ("numpy", "pandas", "scikit-learn", "xgboost", "joblib")}}
    make_figures(output, metrics, detail_frame, matrices, selection["selected_classifier"])
    make_report(output, report, metrics, detail_frame)
    write_json(output / "reporte.json", report)
    print(report["outcome"], flush=True)
    return output / "clasificacion_NO2.html"
