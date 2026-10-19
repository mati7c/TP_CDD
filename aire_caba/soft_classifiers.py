"""Clasificación directa CO/NO2/PM10 con XGBoost y ponderación suave fija."""

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
from xgboost import XGBClassifier

from .air_quality_scales import classify_concentration, load_scales, views
from .compare_indices import distribution_rows, evaluate
from .model_dataset import TARGETS, sha256
from .modeling import PROJECT, load_data, write_json
from .no2_classifier import aligned_probabilities, encode_targets, training_weights


SOFT = "XGBoost_suave"
PREVIOUS = "Regresion_previa"
MAJORITY = "Mayoria_train"


def matched_previous_predictions(frame, saved, pollutant, selected_model):
    """Conserva orden, identidad y objetivos al comparar con la regresión previa."""
    keys = ["estacion", "fecha_hora", "split"]
    previous = saved.copy()
    previous["fecha_hora"] = pd.to_datetime(previous["fecha_hora"])
    joined = frame[keys + [pollutant]].merge(previous, on=keys, how="outer",
                                            validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all() or not np.allclose(joined[pollutant], joined["actual"], rtol=0, atol=1e-10):
        raise ValueError(f"{pollutant}: los objetivos no coinciden con la regresión previa")
    column = f"prediction_{selected_model}"
    return frame[keys].merge(previous[keys + [column]], on=keys, how="left", validate="one_to_one")[column].to_numpy()


def compatible_no2_bundle(config, spec, audit, features):
    path = PROJECT / "outputs/clasificacion_no2/models/NO2_XGBoost.joblib"
    report_path = PROJECT / "outputs/clasificacion_no2/reporte.json"
    if not config["reuse_no2_pilot_if_compatible"] or not path.exists() or not report_path.exists():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_params = config["xgboost"]
    if any(report[key] != audit[key] for key in ("input_sha256", "quality_sha256")):
        return None
    if report["config"]["seed"] != config["seed"]:
        return None
    bundle = joblib.load(path)
    if bundle["features"] != features or bundle["scale"] != spec or bundle["parameters"]["weight_power"] != .5:
        return None
    if any(bundle["parameters"].get(key) != value for key, value in expected_params.items()):
        return None
    return bundle, {"reused": True, "path": str(path.relative_to(PROJECT)), "sha256": sha256(path),
                    "pilot_report_sha256": sha256(report_path)}


def fit_or_reuse(pollutant, train, features, config, spec, audit):
    original_classes, encoded = encode_targets(train["label"])
    reuse = compatible_no2_bundle(config, spec, audit, features) if pollutant == "NO2" else None
    if reuse:
        bundle, provenance = reuse
        if bundle["original_classes"] != original_classes.tolist():
            raise ValueError("El piloto NO2 tiene una codificación de clases diferente")
        print("NO2: reutilizando XGBoost con ponderación suave del piloto validado", flush=True)
        return bundle["model"], original_classes, provenance
    print(f"{pollutant}: entrenando XGBoost con ponderación suave fija", flush=True)
    started = time.perf_counter()
    model = XGBClassifier(**config["xgboost"], num_class=len(original_classes),
                          random_state=config["seed"], n_jobs=config["threads"])
    model.fit(train[features], encoded, sample_weight=training_weights(train["label"], .5))
    return model, original_classes, {"reused": False, "fit_seconds": time.perf_counter() - started}


def score_predictions(frame, spec, predictions, summaries, details, matrices, pollutant):
    actual = frame["label"].to_numpy()
    for name, labels in predictions.items():
        for resolution, class_names in views(spec):
            y, pred = actual, labels
            if resolution == "band":
                mapping = np.asarray(spec["level_to_band"])
                y, pred = mapping[y], mapping[pred]
            for split in ("validation", "test"):
                for station in ("todas", *sorted(frame["estacion"].unique())):
                    mask = frame["split"].eq(split).to_numpy()
                    if station != "todas":
                        mask = mask & frame["estacion"].eq(station).to_numpy()
                    context = {"pollutant": pollutant, "scheme": spec["scheme"], "resolution": resolution,
                               "model": name, "split": split, "station": station}
                    evaluate(y[mask], pred[mask], class_names, context, summaries, details, matrices)


def report_figures(output, report, metrics, details, matrices):
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "tmp/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = output / "figuras"
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for ax, pollutant in zip(axes, TARGETS):
        row = next(item for item in matrices if item["pollutant"] == pollutant and item["model"] == SOFT
                   and item["split"] == "test" and item["resolution"] == "level" and item["station"] == "todas")
        matrix = np.asarray(row["matrix"])
        support = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, support, out=np.zeros_like(matrix, dtype=float), where=support != 0)
        ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        for (i, j), count in np.ndenumerate(matrix):
            if count:
                ax.text(j, i, f"{count}\n{normalized[i,j]:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if normalized[i,j] > .5 else "#172c44")
        names = report["scales"]["pollutants"][pollutant]["classes"]
        display_names = [name.split(" · ")[0] for name in names]
        ax.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=display_names,
               yticklabels=display_names, xlabel="Clase predicha", ylabel="Clase real", title=pollutant)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("XGBoost y ponderación suave — matrices de prueba", fontweight="bold")
    fig.savefig(folder / "matrices_tres_contaminantes.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    for ax, pollutant in zip(axes, TARGETS):
        frame = details.loc[details["pollutant"].eq(pollutant) & details["split"].eq("test")
                            & details["station"].eq("todas") & details["resolution"].eq("level")
                            & details["support"].gt(0)]
        labels = sorted(frame["class_id"].unique())
        positions = np.arange(len(labels))
        for offset, name, title, color in zip((-.25, 0, .25), (MAJORITY, PREVIOUS, SOFT),
                                             ("Mayoría", "Regresión previa", "XGBoost suave"),
                                             ("#8193a6", "#147d92", "#e28a35")):
            values = frame.loc[frame["model"].eq(name)].set_index("class_id").reindex(labels)["recall"]
            ax.bar(positions + offset, values * 100, width=.25, label=title, color=color)
        class_names = report["scales"]["pollutants"][pollutant]["classes"]
        ax.set(xticks=positions, xticklabels=[class_names[i].split(" · ")[0] for i in labels],
               ylabel="Recall (%)", ylim=(0, 105), title=pollutant)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Detección por categoría real presente — prueba", fontweight="bold")
    fig.savefig(folder / "recall_tres_contaminantes.png", dpi=160)
    plt.close(fig)


def write_report(output, report, metrics, details, summary):
    comparison = metrics.loc[metrics["split"].eq("test") & metrics["station"].eq("todas")
                             & metrics["resolution"].eq("level")]
    columns = ["pollutant", "model", "n", "accuracy", "precision_macro", "recall_macro", "f1_macro",
               "f1_macro_supported", "balanced_accuracy", "f1_weighted", "false_positive_rate_macro", "ordinal_mae"]
    observations = []
    for pollutant in TARGETS:
        chosen = comparison.loc[comparison["pollutant"].eq(pollutant) & comparison["model"].eq(SOFT)].iloc[0]
        previous = comparison.loc[comparison["pollutant"].eq(pollutant) & comparison["model"].eq(PREVIOUS)].iloc[0]
        observations.append(f"{pollutant}: accuracy {chosen['accuracy']:.2%}; F1 macro sobre clases presentes "
                            f"{chosen['f1_macro_supported']:.4f} frente a {previous['f1_macro_supported']:.4f} "
                            f"de la regresión previa; balanced accuracy {chosen['balanced_accuracy']:.2%}.")
    report["observations"] = observations
    notes = [
        "Tres clasificadores multiclase, uno por contaminante y compartidos entre estaciones. "
        "NO2/PM10 usan niveles británicos; CO usa categorías de India. No se combinan en un índice general.",
        "XGBoost usa la configuración elegida por validación en el piloto NO2 y exponente de pesos 0.5 fijo "
        "para los tres contaminantes. No se afirma que sea la configuración óptima para CO o PM10. "
        "Los pesos de cada contaminante se calculan sólo con sus ejemplos de entrenamiento.",
        "Se mantienen las 59 entradas, los cortes temporales y los descartes del modelado original. "
        "Entrenamiento anterior a 2024, validación en 2024 y prueba desde 2025; purga de dos días. "
        "Se excluyen objetivos interpolados y no se imputan faltantes del objetivo.",
        "NO2 reutiliza el modelo del piloto cuando se comprueba compatibilidad de datos, entradas, escala y "
        "parámetros; si no está disponible se entrena la misma configuración. Los modelos nuevos de CO y PM10 "
        "no se seleccionan ni ajustan usando test.",
        "Accuracy no demuestra detección de clases raras. Macro considera todas las categorías oficiales "
        "de la escala y usa cero cuando una métrica no puede calcularse; macro_supported sólo considera "
        "categorías reales presentes. Se informan ambas y el soporte por categoría.",
        "FPR se calcula por categoría contra el resto, FP/(FP+TN), y se promedia para el resumen macro. "
        "No equivale a la proporción de predicciones de esa categoría que fueron equivocadas; revisar también "
        "precision, recall y las matrices.",
        "En CO hay sólo tres de seis categorías en prueba. En NO2 sólo aparecen niveles 1–3, todos de "
        "contaminación baja. PM10 tiene los diez niveles en test, pero los elevados poseen pocos casos. "
        "Una clase sin ejemplos no puede darse por detectada ni evaluada.",
        "Las conversiones y promedios publicados se mantienen. La conversión de CO y su redondeo conservan "
        "las convenciones declaradas en el experimento de índices; no se certifica equivalencia instrumental "
        "ni cumplimiento normativo local.",
        "Se usan entradas ERA5 históricas y probabilidades sin calibración. Estos resultados no miden todavía "
        "el error operativo con meteorología pronosticada a 24/48 horas. El test ya había sido inspeccionado: "
        "la comparación es exploratoria y requiere confirmación independiente.",
        "Los tres modelos entrenados se exportan en formato nativo UBJ de XGBoost con metadatos JSON de "
        "entradas, codificación y escala. Las clases ausentes de entrenamiento conservan probabilidad cero. "
        "Los códigos exportados comienzan en cero: en DAQI el código 0 representa el nivel 1.",
    ]
    tables = [("Resumen solicitado: XGBoost suave, prueba", summary),
              ("Comparación con referencias en prueba", comparison[columns]),
              ("Validación: configuración fija sin nueva selección", metrics.loc[
                  metrics["split"].eq("validation") & metrics["station"].eq("todas")
                  & metrics["resolution"].eq("level"), columns]),
              ("Resultados por categoría de XGBoost suave — prueba", details.loc[
                  details["model"].eq(SOFT) & details["station"].eq("todas")
                  & details["split"].eq("test") & details["resolution"].eq("level"),
                  ["pollutant", "category", "support", "predicted_count", "precision", "recall", "f1",
                   "false_positive_rate", "tp", "fp", "fn", "tn"]]),
              ("XGBoost suave por estación — prueba", metrics.loc[
                  metrics["model"].eq(SOFT) & metrics["station"].ne("todas")
                  & metrics["split"].eq("test") & metrics["resolution"].eq("level"), ["station", *columns]])]

    def md_table(frame):
        lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join("---" for _ in frame.columns) + " |"]
        for row in frame.itertuples(index=False, name=None):
            text = ["no evaluable" if pd.isna(v) else f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
            lines.append("| " + " | ".join(text) + " |")
        return "\n".join(lines)

    md = ["# XGBoost con ponderación suave: CO, NO₂ y PM10", "Resultados horarios por estación."]
    md.extend("- " + item for item in observations)
    md.append("\n## Método y alcance\n")
    md.extend("- " + note for note in notes)
    html = ["<!doctype html><html lang='es'><meta charset='utf-8'><title>XGBoost suave: tres contaminantes</title>",
            "<style>body{font:16px system-ui;max-width:1500px;margin:35px auto;padding:0 20px;color:#193049}"
            "table{border-collapse:collapse;font-size:13px}th,td{padding:8px;border:1px solid #ccd5df}"
            "th{background:#edf3f8}section{overflow-x:auto}img{max-width:100%}li{margin:10px 0}</style>",
            "<h1>XGBoost con ponderación suave: CO, NO₂ y PM10</h1><ul>"]
    html.extend(f"<li>{escape(item)}</li>" for item in observations)
    html.append("</ul><h2>Método y alcance</h2><ul>")
    html.extend(f"<li>{escape(note)}</li>" for note in notes)
    html.append("</ul>")
    for title, frame in tables:
        md.extend(["\n## " + title + "\n", md_table(frame)])
        html.extend([f"<h2>{escape(title)}</h2><section>", frame.to_html(index=False, na_rep="no evaluable",
                     float_format=lambda value: f"{value:.4f}"), "</section>"])
    for file, label in (("matrices_tres_contaminantes.png", "Matrices de confusión"),
                        ("recall_tres_contaminantes.png", "Recall por categoría")):
        md.append(f"![{label}](figuras/{file})")
        html.append(f"<h2>{label}</h2><img alt='{label}' src='figuras/{file}'>")
    md.append("\n## Fuentes\n")
    html.append("<h2>Fuentes</h2><ul>")
    for name, url in report["scales"]["sources"].items():
        md.append(f"- [{name}]({url})")
        html.append(f"<li><a href='{escape(url, quote=True)}'>{escape(name)}</a></li>")
    html.append("</ul></html>")
    (output / "resumen_metricas.md").write_text("\n\n".join(md) + "\n", encoding="utf-8")
    (output / "resumen_metricas.html").write_text("\n".join(html), encoding="utf-8")


def run_soft_classifiers(data_dir, config_path=None, output_dir=None):
    started = time.perf_counter()
    config_path = Path(config_path or PROJECT / "config/clasificacion_suave.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["weight_power"] != .5:
        raise ValueError("Este experimento fija ponderación suave con exponente 0.5")
    source = PROJECT / "outputs/tercera_entrega"
    output = Path(output_dir or PROJECT / "outputs/clasificacion_suave")
    if output.resolve() in {source.resolve(), (PROJECT / "outputs/clasificacion_no2").resolve(),
                            (PROJECT / "outputs/comparacion_indices").resolve()}:
        raise ValueError("Elegir una carpeta distinta de los experimentos anteriores")
    original = json.loads((source / "reporte.json").read_text(encoding="utf-8"))
    scales = load_scales()
    data, features, audit = load_data(data_dir, original["config"])
    if any(audit[key] != original[key] for key in ("input_sha256", "quality_sha256")):
        raise ValueError("Los CSV no corresponden al experimento original")
    output.mkdir(parents=True, exist_ok=True)
    (output / "modelos").mkdir(exist_ok=True)
    write_json(output / "auditoria.json", audit)
    summaries, details, matrices, distribution, models = [], [], [], [], {}
    for pollutant in TARGETS:
        spec = scales["pollutants"][pollutant]
        subset = data.loc[data[pollutant].notna()].copy()
        subset["label"] = classify_concentration(subset[pollutant], spec)
        train = subset.loc[subset["split"].eq("train")]
        frame = subset.loc[subset["split"].isin(("validation", "test"))].copy()
        if len(train) == 0 or frame["split"].nunique() != 2:
            raise ValueError(f"{pollutant}: falta alguna partición temporal")
        for resolution, names in views(spec):
            labels = subset["label"] if resolution == "level" else np.asarray(spec["level_to_band"])[subset["label"]]
            distribution.extend(distribution_rows(subset, pollutant, spec["scheme"], resolution, names, labels))
        model, original_classes, provenance = fit_or_reuse(pollutant, train, features, config, spec, audit)
        probabilities = aligned_probabilities(model, frame[features], original_classes, len(spec["classes"]))
        chosen_regressor = original["selected_by_validation"]["regression"][pollutant]
        source_predictions = source / f"predicciones_{pollutant}.csv"
        previous = matched_previous_predictions(frame, pd.read_csv(source_predictions), pollutant, chosen_regressor)
        majority = int(np.bincount(train["label"], minlength=len(spec["classes"])).argmax())
        predictions = {SOFT: probabilities.argmax(axis=1), PREVIOUS: classify_concentration(previous, spec),
                       MAJORITY: np.full(len(frame), majority, dtype=int)}
        if pollutant == "NO2" and provenance["reused"]:
            pilot = pd.read_csv(PROJECT / "outputs/clasificacion_no2/predicciones_NO2.csv")
            pilot["fecha_hora"] = pd.to_datetime(pilot["fecha_hora"])
            matched = frame[["estacion", "fecha_hora", "split"]].merge(pilot, on=["estacion", "fecha_hora", "split"],
                                                                        how="left", validate="one_to_one")
            if not np.array_equal(predictions[SOFT], matched["prediction_XGBoost"]):
                raise ValueError("La reutilización de NO2 no reproduce el piloto")
            provenance["pilot_predictions_reproduced"] = True
        score_predictions(frame, spec, predictions, summaries, details, matrices, pollutant)
        exported = frame[["estacion", "fecha", "fecha_hora", "split", pollutant, "label"]].copy()
        for name, labels in predictions.items():
            exported[f"prediction_{name}"] = labels
        for index in range(len(spec["classes"])):
            exported[f"probability_class_{index}"] = probabilities[:, index]
        exported.to_csv(output / f"predicciones_{pollutant}.csv", index=False, encoding="utf-8-sig", float_format="%.9g")
        model_path = output / "modelos" / f"{pollutant}_XGBoost_suave.ubj"
        model.save_model(model_path)
        metadata = {"pollutant": pollutant, "features": features, "scale": spec,
                    "original_classes": original_classes.tolist(), "class_names": spec["classes"],
                    "parameters": config["xgboost"] | {"num_class": len(original_classes), "weight_power": .5,
                                                        "random_state": config["seed"], "n_jobs": config["threads"]},
                    "input_sha256": audit["input_sha256"], "quality_sha256": audit["quality_sha256"],
                    "provenance": provenance, "source_regressor": chosen_regressor,
                    "source_predictions_sha256": sha256(source_predictions),
                    "train_size": len(train), "validation_size": int(frame["split"].eq("validation").sum()),
                    "test_size": int(frame["split"].eq("test").sum()), "probabilities_calibrated": False,
                    "unlearned_classes": sorted(set(range(len(spec["classes"]))) - set(original_classes.tolist())),
                    "model_sha256": sha256(model_path)}
        write_json(model_path.with_suffix(".meta.json"), metadata)
        models[pollutant] = metadata
        latest = next(row for row in reversed(summaries) if row["pollutant"] == pollutant and row["model"] == SOFT
                      and row["split"] == "test" and row["resolution"] == "level" and row["station"] == "todas")
        print(f"{pollutant}: accuracy test={latest['accuracy']:.2%}, F1 clases presentes={latest['f1_macro_supported']:.4f}", flush=True)
        del model, probabilities, subset, frame, train
        gc.collect()
    metrics, detail_frame = pd.DataFrame(summaries), pd.DataFrame(details)
    summary = metrics.loc[metrics["model"].eq(SOFT) & metrics["split"].eq("test")
                          & metrics["station"].eq("todas") & metrics["resolution"].eq("level")].copy()
    summary["classes_declared"] = summary["pollutant"].map(lambda p: len(scales["pollutants"][p]["classes"]))
    summary["classes_trained"] = summary["pollutant"].map(lambda p: len(models[p]["original_classes"]))
    metrics.to_csv(output / "metricas_clasificacion.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "metricas_resumen_test.csv", index=False, encoding="utf-8-sig")
    detail_frame.to_csv(output / "metricas_por_clase.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(distribution).to_csv(output / "distribucion_categorias.csv", index=False, encoding="utf-8-sig")
    write_json(output / "matrices_confusion.json", matrices)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "config": config, "models": models,
              "scales": scales, "temporal_config": original["config"], "features": features,
              "input_sha256": audit["input_sha256"], "quality_sha256": audit["quality_sha256"],
              "config_sha256": sha256(config_path), "code_sha256": sha256(Path(__file__)),
              "scales_sha256": sha256(PROJECT / "config/indices_alternativos.json"),
              "independent_new_test": False, "probabilities_calibrated": False,
              "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "xgboost", "scikit-learn")},
              "elapsed_seconds": time.perf_counter() - started}
    report_figures(output, report, metrics, detail_frame, matrices)
    write_report(output, report, metrics, detail_frame, summary)
    write_json(output / "reporte.json", report)
    for observation in report["observations"]:
        print(observation, flush=True)
    return output / "resumen_metricas.html"
