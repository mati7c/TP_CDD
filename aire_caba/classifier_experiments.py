"""Comparación de técnicas y ponderaciones para CO y PM10; integra el piloto NO2."""

from datetime import datetime, timezone
import gc
from html import escape
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .air_quality_scales import classify_concentration, load_scales
from .compare_indices import make_figures
from .model_dataset import TARGETS, sha256
from .modeling import PROJECT, load_data, write_json
from .no2_classifier import aligned_probabilities, fit_candidates
from .soft_classifiers import SOFT, PREVIOUS, MAJORITY, matched_previous_predictions, score_predictions


def load_cached_soft(pollutant, features, spec, audit, config):
    folder = PROJECT / "outputs/clasificacion_suave/modelos"
    path = folder / f"{pollutant}_XGBoost_suave.ubj"
    metadata = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    if any(metadata[key] != audit[key] for key in ("input_sha256", "quality_sha256")):
        raise ValueError(f"{pollutant}: el modelo suave pertenece a otro dataset")
    if metadata["features"] != features or metadata["scale"] != spec:
        raise ValueError(f"{pollutant}: no coinciden entradas o escala del modelo suave")
    if metadata["parameters"]["weight_power"] != .5 or metadata["parameters"]["random_state"] != config["seed"]:
        raise ValueError(f"{pollutant}: semilla o ponderación distinta")
    if any(metadata["parameters"].get(key) != value for key, value in config["xgboost"].items()):
        raise ValueError(f"{pollutant}: hiperparámetros diferentes a los modelos suaves")
    if sha256(path) != metadata["model_sha256"]:
        raise ValueError(f"{pollutant}: el modelo suave cambió desde su exportación")
    model = XGBClassifier()
    model.load_model(path)
    model.set_params(n_jobs=config["threads"])
    return model, metadata


def write_report(output, report, metrics, details, trials, soft_summary):
    native = metrics.loc[metrics["split"].eq("test") & metrics["station"].eq("todas")
                         & metrics["resolution"].eq("level")]
    chosen = native.loc[native.apply(lambda row: row["model"] == report["selected_models"][row["pollutant"]], axis=1)].copy()
    chosen["weight_power"] = chosen["pollutant"].map(lambda p: report["selections"][p]["parameters"][report["selected_models"][p]]["weight_power"])
    chosen.to_csv(output / "metricas_elegidos_test.csv", index=False, encoding="utf-8-sig")
    columns = ["pollutant", "model", "n", "accuracy", "precision_macro", "recall_macro", "f1_macro",
               "f1_macro_supported", "balanced_accuracy", "false_positive_rate_macro", "f1_weighted", "ordinal_mae"]
    soft_short = soft_summary[["pollutant", "accuracy", "precision_macro", "recall_macro", "f1_macro", "false_positive_rate_macro"]].copy()
    soft_short.columns = ["Contaminante", "Accuracy", "Precision macro", "Recall macro", "F1 macro", "FPR macro"]
    supported = soft_summary[["pollutant", "classes_present", "classes_declared", "precision_macro_supported",
                               "recall_macro_supported", "f1_macro_supported", "balanced_accuracy", "f1_weighted", "ordinal_mae"]]
    notes = [
        "Se comparan Random Forest y XGBoost con pesos 0, 0.5 y 1 para CO y PM10. NO2 reutiliza su piloto previo. "
        "La configuración de árboles permanece fija; se elige técnica y ponderación por F1 macro de validación.",
        "Los XGBoost suaves ya entrenados se reutilizan después de comprobar hashes, datos, escala y parámetros. "
        "Los pesos se calculan sólo en entrenamiento. No se fabrican ejemplos ni se reequilibra test.",
        "Las selecciones por contaminante se guardan antes de calcular test. No se cambia una elección porque "
        "otra alternativa tenga mejores números de prueba. Las referencias previas y los XGBoost suaves se conservan.",
        "Macro considera todas las clases declaradas, con cero en métricas no evaluables. Macro_supported "
        "considera sólo categorías reales presentes. El FPR por clase es FP/(FP+TN), no uno menos precision.",
        "Los contaminantes usan escalas distintas: India para CO y DAQI británico para NO2/PM10. "
        "No hay un índice general de estación o CABA. En NO2 los tres niveles de test pertenecen a contaminación baja. "
        "En CO faltan las categorías superiores y en PM10 los casos elevados son escasos.",
        "Se conservan las mismas entradas, filas válidas, ventanas publicadas, conversiones y cortes temporales. "
        "ERA5 es meteorología histórica: esta experiencia no mide el error operativo del pronóstico a 24/48 h.",
        "Test ya había sido inspeccionado y motivó ampliar las pruebas: se trata de una comparación exploratoria. "
        "Una confirmación independiente requiere datos posteriores o un nuevo protocolo temporal.",
    ]
    observations = []
    for pollutant in TARGETS:
        row = chosen.loc[chosen["pollutant"].eq(pollutant)].iloc[0]
        before = soft_summary.loc[soft_summary["pollutant"].eq(pollutant)].iloc[0]
        observations.append(f"{pollutant}: elegido {row['model']}, ponderación {row['weight_power']:g}; "
                            f"F1 clases presentes test {row['f1_macro_supported']:.4f}, "
                            f"frente a {before['f1_macro_supported']:.4f} de XGBoost suave; "
                            f"accuracy {row['accuracy']:.2%}.")
    report["observations"] = observations
    selected_detail = details.loc[details["split"].eq("test") & details["station"].eq("todas")
                                  & details["resolution"].eq("level")]
    selected_detail = selected_detail.loc[selected_detail.apply(
        lambda row: row["model"] == report["selected_models"][row["pollutant"]], axis=1)]
    trial_cols = ["pollutant", "model", "weight_power", "validation_accuracy", "validation_precision_macro",
                  "validation_recall_macro", "validation_f1_macro", "validation_f1_macro_supported",
                  "validation_balanced_accuracy", "validation_false_positive_rate_macro"]
    tables = [("Resumen solicitado: XGBoost suave — test", soft_short),
              ("XGBoost suave: clases presentes y métricas adicionales", supported),
              ("Clasificadores elegidos por validación — test", chosen[["weight_power", *columns]]),
              ("Comparación de todos los enfoques finalistas — test", native[columns]),
              ("Las 18 variantes y sus métricas de validación", trials[trial_cols]),
              ("Detalle por categoría del clasificador elegido — test", selected_detail[
                  ["pollutant", "model", "category", "support", "precision", "recall", "f1",
                   "false_positive_rate", "tp", "fp", "fn", "tn"]])]

    def md_table(frame):
        lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join("---" for _ in frame.columns) + " |"]
        for row in frame.itertuples(index=False, name=None):
            values = ["no evaluable" if pd.isna(v) else f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    markdown = ["# Comparación ampliada de clasificadores", "Incluye el resumen de los tres XGBoost suaves y las pruebas adicionales."]
    markdown.extend("- " + item for item in observations)
    markdown.append("\n## Método y alcance\n")
    markdown.extend("- " + note for note in notes)
    html = ["<!doctype html><html lang='es'><meta charset='utf-8'><title>Comparación ampliada</title>",
            "<style>body{font:16px system-ui;max-width:1450px;margin:35px auto;padding:0 20px;color:#193049}"
            "table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border:1px solid #ccd5df}"
            "th{background:#edf3f8}section{overflow-x:auto}img{max-width:100%}li{margin:10px 0}</style>",
            "<h1>Comparación ampliada de clasificadores</h1><ul>"]
    html.extend(f"<li>{escape(item)}</li>" for item in observations)
    html.append("</ul><h2>Método y alcance</h2><ul>")
    html.extend(f"<li>{escape(note)}</li>" for note in notes)
    html.append("</ul>")
    for title, frame in tables:
        markdown.extend(["\n## " + title + "\n", md_table(frame)])
        html.extend([f"<h2>{escape(title)}</h2><section>", frame.to_html(index=False, na_rep="no evaluable",
                     float_format=lambda v: f"{v:.4f}"), "</section>"])
    for file, label in (("matrices_nuevas_escalas.png", "Matrices de los elegidos por validación"),
                        ("distribucion_nuevas_escalas.png", "Distribución por período")):
        markdown.append(f"![{label}](figuras/{file})")
        html.append(f"<h2>{label}</h2><img alt='{label}' src='figuras/{file}'>")
    markdown.append("\n## Fuentes\n")
    html.append("<h2>Fuentes</h2><ul>")
    for name, url in report["scales"]["sources"].items():
        markdown.append(f"- [{name}]({url})")
        html.append(f"<li><a href='{escape(url, quote=True)}'>{escape(name)}</a></li>")
    html.append("</ul></html>")
    (output / "resumen_comparacion.md").write_text("\n\n".join(markdown) + "\n", encoding="utf-8")
    (output / "resumen_comparacion.html").write_text("\n".join(html), encoding="utf-8")


def run_experiments(data_dir, config_path=None, output_dir=None):
    started = time.perf_counter()
    config_path = Path(config_path or PROJECT / "config/comparacion_clasificadores.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["pollutants_to_tune"] != ["CO", "PM10"] or config["weight_powers"] != [0, .5, 1]:
        raise ValueError("La comparación adicional utiliza CO/PM10 y las ponderaciones 0, 0.5 y 1")
    original_path = PROJECT / "outputs/tercera_entrega"
    soft_path = PROJECT / "outputs/clasificacion_suave"
    output = Path(output_dir or PROJECT / "outputs/comparacion_clasificadores")
    protected = [original_path, soft_path, PROJECT / "outputs/clasificacion_no2", PROJECT / "outputs/comparacion_indices"]
    if output.resolve() in {path.resolve() for path in protected}:
        raise ValueError("Usar una carpeta distinta de los experimentos anteriores")
    original = json.loads((original_path / "reporte.json").read_text(encoding="utf-8"))
    data, features, audit = load_data(data_dir, original["config"])
    if any(audit[key] != original[key] for key in ("input_sha256", "quality_sha256")):
        raise ValueError("Los datos difieren del experimento original")
    scales = load_scales()
    output.mkdir(parents=True, exist_ok=True)
    (output / "modelos").mkdir(exist_ok=True)
    write_json(output / "auditoria.json", audit)
    summaries, details, matrices, trials, selections, selected_models, model_metadata = [], [], [], [], {}, {}, {}
    for pollutant in config["pollutants_to_tune"]:
        spec = scales["pollutants"][pollutant]
        subset = data.loc[data[pollutant].notna()].copy()
        subset["label"] = classify_concentration(subset[pollutant], spec)
        parts = {name: subset.loc[subset["split"].eq(name)] for name in ("train", "validation", "test")}
        folder = output / pollutant
        (folder / "models").mkdir(parents=True, exist_ok=True)
        cached_model, cached_meta = load_cached_soft(pollutant, features, spec, audit, config)
        cache = {("XGBoost", .5): cached_model}
        selection, pollutant_trials, classes = fit_candidates(parts["train"], parts["validation"], features,
                                                              config, spec, folder, pollutant, cache)
        selections[pollutant], selected_models[pollutant] = selection, selection["selected_classifier"]
        trials.extend(row | {"pollutant": pollutant} for row in pollutant_trials)
        frame = pd.concat([parts["validation"], parts["test"]], ignore_index=True)
        soft_probabilities = aligned_probabilities(cached_model, frame[features], classes, len(spec["classes"]))
        previous = matched_previous_predictions(frame, pd.read_csv(original_path / f"predicciones_{pollutant}.csv"),
                                                pollutant, original["selected_by_validation"]["regression"][pollutant])
        predictions = {SOFT: soft_probabilities.argmax(axis=1), PREVIOUS: classify_concentration(previous, spec),
                       MAJORITY: np.full(len(frame), int(np.bincount(parts["train"]["label"]).argmax()))}
        exported = frame[["estacion", "fecha", "fecha_hora", "split", pollutant, "label"]].copy()
        for family in ("RandomForest", "XGBoost"):
            bundle = joblib.load(folder / "models" / f"{pollutant}_{family}.joblib")
            probabilities = aligned_probabilities(bundle["model"], frame[features], classes, len(spec["classes"]))
            predictions[family] = probabilities.argmax(axis=1)
            if family == selection["selected_classifier"]:
                target = output / "modelos" / f"{pollutant}_{family}"
                metadata = {key: value for key, value in bundle.items() if key != "model"}
                metadata |= {"input_sha256": audit["input_sha256"], "quality_sha256": audit["quality_sha256"]}
                if family == "XGBoost":
                    target = target.with_suffix(".ubj")
                    bundle["model"].save_model(target)
                else:
                    target = target.with_suffix(".joblib")
                    shutil.copyfile(folder / "models" / f"{pollutant}_{family}.joblib", target)
                metadata["model_sha256"] = sha256(target)
                write_json(target.with_suffix(".meta.json"), metadata)
                model_metadata[pollutant] = metadata | {"path": str(target.relative_to(output))}
                for index in range(len(spec["classes"])):
                    exported[f"chosen_probability_class_{index}"] = probabilities[:, index]
            del bundle
            gc.collect()
        for name, values in predictions.items():
            exported[f"prediction_{name}"] = values
        exported.to_csv(output / f"predicciones_{pollutant}.csv", index=False, encoding="utf-8-sig", float_format="%.9g")
        score_predictions(frame, spec, predictions, summaries, details, matrices, pollutant)
        print(f"{pollutant}: elegido {selection['selected_classifier']} por validación; evaluación de test guardada", flush=True)
        del cached_model, cache, soft_probabilities, subset, parts, frame, exported
        gc.collect()
    # NO2 ya fue comparado: integrar sus métricas y su selección sin volver a elegir con test.
    pilot = PROJECT / "outputs/clasificacion_no2"
    pilot_report = json.loads((pilot / "reporte.json").read_text(encoding="utf-8"))
    if any(pilot_report[key] != audit[key] for key in ("input_sha256", "quality_sha256")):
        raise ValueError("El piloto NO2 corresponde a otros datos")
    selections["NO2"] = pilot_report["selection"]
    selected_models["NO2"] = pilot_report["selection"]["selected_classifier"]
    if selected_models["NO2"] != "XGBoost" or selections["NO2"]["parameters"]["XGBoost"]["weight_power"] != .5:
        raise ValueError("El piloto NO2 cambió: revisar la exportación de su modelo elegido antes de integrar")
    trials.extend(row | {"pollutant": "NO2", "reused": True} for row in pilot_report["tuning_trials"])
    rename = {"Regresion_XGBoost": PREVIOUS}
    for file, destination in (("metricas_clasificacion.csv", summaries), ("metricas_por_clase.csv", details)):
        frame = pd.read_csv(pilot / file)
        frame["model"] = frame["model"].replace(rename)
        destination.extend(frame.to_dict("records"))
    for row in json.loads((pilot / "matrices_confusion.json").read_text(encoding="utf-8")):
        matrices.append(row | {"model": rename.get(row["model"], row["model"])})
    soft_metrics = pd.read_csv(soft_path / "metricas_clasificacion.csv")
    summaries.extend(soft_metrics.loc[soft_metrics["pollutant"].eq("NO2") & soft_metrics["model"].eq(SOFT)].to_dict("records"))
    soft_details = pd.read_csv(soft_path / "metricas_por_clase.csv")
    details.extend(soft_details.loc[soft_details["pollutant"].eq("NO2") & soft_details["model"].eq(SOFT)].to_dict("records"))
    matrices.extend(row for row in json.loads((soft_path / "matrices_confusion.json").read_text(encoding="utf-8"))
                    if row["pollutant"] == "NO2" and row["model"] == SOFT)
    for suffix in (".ubj", ".meta.json"):
        shutil.copyfile(soft_path / "modelos" / f"NO2_XGBoost_suave{suffix}", output / "modelos" / f"NO2_XGBoost{suffix}")
    model_metadata["NO2"] = json.loads((output / "modelos/NO2_XGBoost.meta.json").read_text(encoding="utf-8"))
    model_metadata["NO2"]["path"] = "modelos/NO2_XGBoost.ubj"
    metrics, detail_frame, trial_frame = pd.DataFrame(summaries), pd.DataFrame(details), pd.DataFrame(trials)
    metrics.to_csv(output / "metricas_clasificacion.csv", index=False, encoding="utf-8-sig")
    detail_frame.to_csv(output / "metricas_por_clase.csv", index=False, encoding="utf-8-sig")
    trial_frame.drop(columns="parameters").to_csv(output / "candidatos_validacion.csv", index=False, encoding="utf-8-sig")
    write_json(output / "matrices_confusion.json", matrices)
    write_json(output / "selecciones_validacion.json", selections)
    distribution = pd.read_csv(soft_path / "distribucion_categorias.csv")
    soft_summary = pd.read_csv(soft_path / "metricas_resumen_test.csv")
    soft_summary.to_csv(output / "metricas_xgboost_suave_test.csv", index=False, encoding="utf-8-sig")
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "config": config, "scales": scales,
              "selections": selections, "selected_models": selected_models, "models": model_metadata,
              "input_sha256": audit["input_sha256"], "quality_sha256": audit["quality_sha256"],
              "code_sha256": sha256(Path(__file__)), "config_sha256": sha256(config_path),
              "independent_new_test": False, "elapsed_seconds": time.perf_counter() - started}
    make_figures(output, report, distribution, matrices,
                 matrix_title="Clasificadores elegidos por validación — prueba con las mismas escalas")
    write_report(output, report, metrics, detail_frame, trial_frame, soft_summary)
    write_json(output / "reporte.json", report)
    for item in report["observations"]:
        print(item, flush=True)
    return output / "resumen_comparacion.html"
