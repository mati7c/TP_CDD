"""Reevalúa predicciones guardadas con UK/India; no entrena ni redefine EPA."""

from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .air_quality_scales import classify_concentration, convert_concentration, load_scales, views
from .aqi import CLASS_NAMES, categorize, classification_metrics
from .model_dataset import TARGETS, sha256
from .modeling import MODEL_NAMES, PROJECT, load_data, write_json


def distribution_rows(data, pollutant, scheme, resolution, names, labels):
    frame = data[["estacion", "split"]].copy()
    frame["label"] = labels
    output = []
    for station in ("todas", *sorted(frame["estacion"].unique())):
        subset = frame if station == "todas" else frame.loc[frame["estacion"].eq(station)]
        for split, part in subset.groupby("split", observed=True):
            counts = np.bincount(part["label"].to_numpy(), minlength=len(names))
            for label, name in enumerate(names):
                output.append({"pollutant": pollutant, "scheme": scheme, "resolution": resolution,
                               "station": station, "split": split, "label": label, "category": name,
                               "count": int(counts[label]), "fraction": float(counts[label] / len(part))})
    return output


def evaluate(actual, predicted, names, context, summaries, details, matrices):
    result = classification_metrics(actual, predicted, names)
    summaries.append(context | result["summary"] | {
        "classes_declared": len(names), "predicted_classes_present": len(np.unique(predicted)),
    })
    details.extend(context | row for row in result["classes"])
    matrices.append(context | {"classes": list(names), "matrix": result["confusion"]})


def score_saved_predictions(frame, train_values, pollutant, spec, winner, summaries, details, matrices):
    scales = [(spec["scheme"], resolution, names, classify_concentration)
              for resolution, names in views(spec)]
    scales.append(("EPA_3", "category", CLASS_NAMES, None))
    exported = frame.copy()
    exported["actual_converted"] = convert_concentration(frame["actual"], spec)
    for scheme, resolution, names, classify in scales:
        def labels(values):
            return categorize(values, pollutant) if classify is None else classify(values, spec, resolution)

        actual = labels(frame["actual"])
        train_labels = labels(train_values)
        if len(train_labels) == 0 or np.any(train_labels < 0):
            raise ValueError(f"{pollutant}: faltan objetivos válidos de entrenamiento")
        # Se fija con entrenamiento; no se busca la categoría mayoritaria del test.
        majority = int(np.bincount(train_labels, minlength=len(names)).argmax())
        columns = {name: labels(frame[f"prediction_{name}"]) for name in MODEL_NAMES}
        columns["Mayoria_train"] = np.full(len(frame), majority, dtype=np.int8)
        prefix = f"{scheme}_{resolution}"
        exported[f"actual_{prefix}"] = actual
        for name, predictions in columns.items():
            exported[f"{name}_{prefix}"] = predictions
            if np.any(predictions < 0) or np.any(actual < 0):
                raise ValueError(f"{pollutant}: predicción u objetivo inválido en resultados guardados")
            for split in ("validation", "test"):
                for station in ("todas", *sorted(frame["estacion"].unique())):
                    mask = frame["split"].eq(split).to_numpy()
                    if station != "todas":
                        mask = mask & frame["estacion"].eq(station).to_numpy()
                    if not mask.any():
                        continue
                    context = {"pollutant": pollutant, "scheme": scheme, "resolution": resolution,
                               "model": name, "selected_by_original_validation": name == winner,
                               "split": split, "station": station}
                    evaluate(actual[mask], predictions[mask], names, context, summaries, details, matrices)
    return exported


def selected_results(metrics, winners, *, scheme=None, resolution="level", model=None):
    selected = metrics.loc[metrics["split"].eq("test") & metrics["station"].eq("todas")
                           & metrics["resolution"].eq(resolution)].copy()
    if scheme:
        selected = selected.loc[selected["scheme"].eq(scheme)]
    if model:
        return selected.loc[selected["model"].eq(model)]
    return selected.loc[selected.apply(lambda row: row["model"] == winners[row["pollutant"]], axis=1)]


def make_figures(output, report, distribution, matrices,
                 matrix_title="Prueba con nuevas escalas: mismos modelos y concentraciones"):
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "tmp/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = output / "figuras"
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for ax, pollutant in zip(axes, TARGETS):
        winner = report["selected_models"][pollutant]
        chosen = next(row for row in matrices if row["pollutant"] == pollutant
                      and row["model"] == winner and row["resolution"] == "level"
                      and row["split"] == "test" and row["station"] == "todas")
        matrix = np.asarray(chosen["matrix"])
        support = matrix.sum(axis=1, keepdims=True)
        fractions = np.divide(matrix, support, out=np.zeros_like(matrix, dtype=float), where=support != 0)
        ax.imshow(fractions, vmin=0, vmax=1, cmap="Blues")
        for (i, j), count in np.ndenumerate(matrix):
            if count:
                ax.text(j, i, f"{count}\n{fractions[i, j]:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if fractions[i, j] > .5 else "#14263d")
        ticks = range(len(chosen["classes"]))
        labels = [name.split(" · ")[0] for name in chosen["classes"]]
        ax.set(xticks=ticks, yticks=ticks, xticklabels=labels, yticklabels=labels,
               xlabel="Clase predicha", ylabel="Clase real", title=f"{pollutant} — {winner}\n{chosen['scheme']}")
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle(matrix_title, fontweight="bold")
    fig.savefig(folder / "matrices_nuevas_escalas.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    colors = ("#8193a6", "#147d92", "#e28a35")
    for ax, pollutant in zip(axes, TARGETS):
        part = distribution.loc[distribution["pollutant"].eq(pollutant)
                                & distribution["resolution"].eq("level") & distribution["station"].eq("todas")]
        labels = report["scales"]["pollutants"][pollutant]["classes"]
        positions = np.arange(len(labels))
        for offset, split, color, title in zip((-.25, 0, .25), ("train", "validation", "test"), colors,
                                               ("Entrenamiento", "Validación", "Prueba")):
            fractions = part.loc[part["split"].eq(split)].sort_values("label")["fraction"].to_numpy()
            ax.bar(positions + offset, fractions * 100, width=.25, color=color, label=title)
        ax.set(xticks=positions, xticklabels=[v.split(" · ")[0] for v in labels],
               title=pollutant, ylabel="Porcentaje de registros reales")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Distribución observada: las clases no se balancean artificialmente", fontweight="bold")
    fig.savefig(folder / "distribucion_nuevas_escalas.png", dpi=160)
    plt.close(fig)


def make_report(output, report, metrics, distribution, details):
    winners = report["selected_models"]
    selected = selected_results(metrics, winners)
    columns = ["pollutant", "model", "classes_present", "classes_declared", "accuracy",
               "precision_macro", "recall_macro", "f1_macro", "f1_macro_supported",
               "balanced_accuracy", "false_positive_rate_macro"]
    summary = selected[columns].sort_values("pollutant")
    majority = selected_results(metrics, winners, model="Mayoria_train")
    majority = majority[["pollutant", "accuracy", "f1_macro", "f1_macro_supported", "balanced_accuracy"]]
    bands = selected_results(metrics, winners, resolution="band")[columns]
    test_dist = distribution.loc[distribution["station"].eq("todas") & distribution["split"].eq("test")
                                 & distribution["resolution"].eq("level")]
    test_dist = test_dist[["pollutant", "category", "count", "fraction"]]
    epa = selected_results(metrics, winners, scheme="EPA_3", resolution="category")[columns]
    class_detail = details.loc[details["split"].eq("test") & details["station"].eq("todas")
                               & details["resolution"].eq("level") & details["support"].gt(0)]
    class_detail = class_detail.loc[class_detail.apply(lambda row: row["model"] == winners[row["pollutant"]], axis=1)]
    class_detail = class_detail[["pollutant", "category", "support", "predicted_count", "precision", "recall", "f1",
                                 "false_positive_rate"]]

    def md_table(frame):
        headers = list(frame.columns)
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        for row in frame.itertuples(index=False, name=None):
            values = ["no evaluable" if pd.isna(v) else f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    observations = []
    for row in selected.itertuples():
        baseline = majority.loc[majority["pollutant"].eq(row.pollutant)].iloc[0]
        observations.append(
            f"{row.pollutant}: accuracy {row.accuracy:.2%}, F1 macro de clases presentes "
            f"{row.f1_macro_supported:.4f}, balanced accuracy {row.balanced_accuracy:.2%}. "
            f"Mayoría de entrenamiento: accuracy {baseline.accuracy:.2%} y "
            f"F1 de clases presentes {baseline.f1_macro_supported:.4f}.")
    limitations = [
        "Se reetiquetan las concentraciones reales y predichas guardadas; no se entrena de nuevo. "
        "MAE, RMSE y R² no cambian. Los modelos continúan seleccionados con la validación original de regresión.",
        "Esta prueba compara objetivos diferentes: cambiar etiquetas no demuestra que haya mejorado el mismo modelo "
        "ni permite comparar directamente los F1 de escalas con distinto número de clases. Los resultados de test ya "
        "inspeccionados no serán una evaluación ciega de futuras decisiones de selección.",
        "Se conservan las particiones temporales, los filtros de plausibilidad y la exclusión de objetivos interpolados. "
        "La nueva referencia de categoría mayoritaria se calcula sólo con entrenamiento.",
        "NO2: DAQI de 1 hora, ppb × 1.9125. PM10: límites DAQI aplicados al móvil de 24 horas publicado, "
        "sin promediarlo otra vez. No se certifica equivalencia instrumental europea ni se llama AQI diario oficial "
        "al máximo de valores móviles durante la jornada.",
        "CO: límites CPCB de 8 horas, ppm × 1.15, adoptando como convención 25 °C y 1013 mb. "
        "No se verificó esa condición de referencia en los sensores CABA. El redondeo de CO a una decimal "
        "es una convención explícita de esta comparación, no una instrucción atribuida a CPCB.",
        "Se informa por separado cada contaminante. No se combinan niveles DAQI con categorías India ni se "
        "modifica el clasificador diario EPA. No existe una nueva etiqueta general de estación/CABA en este experimento.",
        "Macro usa todas las clases declaradas y cero en las no evaluables. Macro_supported usa sólo clases reales "
        "presentes. Se informa cobertura y detalle por clase; FPR es FP/(FP+TN) de cada clase contra el resto. "
        "FPR de categorías no predichas puede ser cero aunque recall también sea cero.",
        "Los límites se verificaron en fuentes oficiales el 18/10/2026. Las conversiones son convenciones "
        "a condiciones de referencia fijas, no ajustes con meteorología futura.",
    ]
    tables = [("Resultados por nivel o categoría nativa — test", summary),
              ("Referencia de clase mayoritaria del entrenamiento — test", majority),
              ("Detalle de clases reales presentes — test", class_detail),
              ("Las cuatro bandas británicas — test", bands),
              ("Distribución real de niveles — test", test_dist),
              ("Control EPA original con las mismas predicciones — test", epa)]
    sources = report["scales"]["sources"]
    markdown = ["# Comparación UK/India por contaminante", "\nExperimento de reetiquetado sobre predicciones existentes.\n"]
    markdown.extend("- " + item for item in observations)
    markdown.append("\n## Alcance y método\n")
    markdown.extend("- " + item for item in limitations)
    for title, frame in tables:
        markdown.extend(["\n## " + title + "\n", md_table(frame)])
    markdown.extend(["\n## Figuras\n", "![Matrices](figuras/matrices_nuevas_escalas.png)",
                     "![Distribución](figuras/distribucion_nuevas_escalas.png)", "\n## Fuentes\n"])
    markdown.extend(f"- [{name}]({url})" for name, url in sources.items())
    (output / "resumen_comparacion.md").write_text("\n\n".join(markdown) + "\n", encoding="utf-8")
    html = ["<!doctype html><html lang='es'><meta charset='utf-8'><title>Comparación UK/India</title>",
            "<style>body{font:16px system-ui;max-width:1400px;margin:35px auto;padding:0 20px;color:#193049}"
            "table{border-collapse:collapse;font-size:13px}th,td{padding:8px;border:1px solid #ccd5df;text-align:left}"
            "th{background:#edf3f8}section{overflow-x:auto}img{max-width:100%}li{margin:10px 0}</style>",
            "<h1>Comparación UK/India por contaminante</h1>",
            "<p>Reetiquetado de predicciones existentes; no hubo un nuevo entrenamiento.</p><ul>"]
    html.extend(f"<li>{escape(item)}</li>" for item in observations)
    html.append("</ul><h2>Alcance y método</h2><ul>")
    html.extend(f"<li>{escape(item)}</li>" for item in limitations)
    html.append("</ul>")
    for title, frame in tables:
        html.extend([f"<h2>{escape(title)}</h2><section>",
                     frame.to_html(index=False, float_format=lambda x: f"{x:.4f}", na_rep="no evaluable"), "</section>"])
    html.extend(["<h2>Figuras</h2><img alt='Matrices de confusión' src='figuras/matrices_nuevas_escalas.png'>",
                 "<img alt='Distribución' src='figuras/distribucion_nuevas_escalas.png'><h2>Fuentes</h2><ul>"])
    html.extend(f"<li><a href='{escape(url, quote=True)}'>{escape(name)}</a></li>" for name, url in sources.items())
    html.append("</ul></html>")
    (output / "comparacion_indices.html").write_text("\n".join(html), encoding="utf-8")
    return observations


def run_comparison(data_dir, source_dir=None, output_dir=None):
    source = Path(source_dir or PROJECT / "outputs/tercera_entrega")
    output = Path(output_dir or PROJECT / "outputs/comparacion_indices")
    if source.resolve() == output.resolve():
        raise ValueError("Usar una carpeta distinta para conservar el experimento original")
    original = json.loads((source / "reporte.json").read_text(encoding="utf-8"))
    scales = load_scales()
    data, _, audit = load_data(data_dir, original["config"])
    previous_audit = json.loads((source / "auditoria.json").read_text(encoding="utf-8"))
    for key in ("input_sha256", "quality_sha256"):
        if audit[key] != previous_audit[key]:
            raise ValueError("Los datos actuales no corresponden al modelado guardado")
    output.mkdir(parents=True, exist_ok=True)
    summaries, details, matrices, distribution, sources = [], [], [], [], []
    sensitivity = []
    for pollutant in TARGETS:
        spec = scales["pollutants"][pollutant]
        subset = data.loc[data[pollutant].notna()]
        for resolution, names in views(spec):
            distribution.extend(distribution_rows(subset, pollutant, spec["scheme"], resolution, names,
                                                   classify_concentration(subset[pollutant], spec, resolution)))
        distribution.extend(distribution_rows(subset, pollutant, "EPA_3", "category", CLASS_NAMES,
                                               categorize(subset[pollutant], pollutant)))
        if pollutant == "CO":
            # Cuantifica el efecto de la condición de referencia sin elegir la que puntúa mejor.
            alternative = spec | {"factor": 1.1642}
            for split, part in subset.groupby("split", observed=True):
                changed = classify_concentration(part[pollutant], spec) != classify_concentration(part[pollutant], alternative)
                sensitivity.append({"pollutant": "CO", "split": split, "factor_primary_25C": spec["factor"],
                                    "factor_sensitivity_20C": alternative["factor"], "n": len(part),
                                    "labels_changed": int(changed.sum())})
        path = source / f"predicciones_{pollutant}.csv"
        sources.append({"path": str(path.relative_to(PROJECT)) if path.is_relative_to(PROJECT) else str(path),
                        "sha256": sha256(path)})
        frame = pd.read_csv(path, encoding="utf-8-sig")
        # Reconciliación exacta de objetivos antes de comparar índices.
        expected = subset.loc[subset["split"].isin(("validation", "test")), ["estacion", "fecha_hora", "split", pollutant]].copy()
        expected["fecha_hora"] = expected["fecha_hora"].dt.strftime("%Y-%m-%d %H:%M:%S")
        checked = frame.merge(expected, on=["estacion", "fecha_hora", "split"], validate="one_to_one", how="outer", indicator=True)
        if not checked["_merge"].eq("both").all() or not np.allclose(checked["actual"], checked[pollutant], rtol=0, atol=1e-10):
            raise ValueError(f"{pollutant}: las predicciones no coinciden con sus datos auditados")
        train = subset.loc[subset["split"].eq("train"), pollutant]
        exported = score_saved_predictions(frame, train, pollutant, spec,
                                           original["selected_by_validation"]["regression"][pollutant],
                                           summaries, details, matrices)
        exported.to_csv(output / f"predicciones_indices_{pollutant}.csv", index=False, encoding="utf-8-sig")
        print(f"{pollutant}: reevaluadas {len(frame):,} predicciones con {spec['scheme']}", flush=True)
    metrics, detail_frame, dist_frame = pd.DataFrame(summaries), pd.DataFrame(details), pd.DataFrame(distribution)
    metrics.to_csv(output / "metricas_clasificacion.csv", index=False, encoding="utf-8-sig")
    detail_frame.to_csv(output / "metricas_por_clase.csv", index=False, encoding="utf-8-sig")
    dist_frame.to_csv(output / "distribucion_categorias.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sensitivity).to_csv(output / "sensibilidad_conversion_CO.csv", index=False, encoding="utf-8-sig")
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "experiment": "relabel_saved_regressions",
        "retrained": False, "combined_station_category": False, "scales": scales,
        "selected_models": original["selected_by_validation"]["regression"],
        "original_config": original["config"], "source_predictions": sources,
        "input_sha256": audit["input_sha256"], "quality_sha256": audit["quality_sha256"],
        "scales_config_sha256": sha256(PROJECT / "config/indices_alternativos.json"),
        "regression_metrics_unchanged": True, "co_conversion_sensitivity": sensitivity,
    }
    # Control de regresión: las métricas EPA deben reproducir las persistidas, sin tolerar cambios de etiquetas.
    original_metrics = pd.read_csv(source / "metricas_clasificacion.csv")
    epa = metrics.loc[metrics["scheme"].eq("EPA_3") & metrics["model"].isin(MODEL_NAMES)]
    for row in epa.itertuples():
        before = original_metrics.loc[original_metrics["task"].eq(row.pollutant)
                                      & original_metrics["model"].eq(row.model)
                                      & original_metrics["split"].eq(row.split)
                                      & original_metrics["station"].eq(row.station)]
        for metric in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "false_positive_rate_macro"):
            if len(before) != 1 or not np.isclose(getattr(row, metric), before.iloc[0][metric], rtol=0, atol=1e-12):
                raise ValueError("El control EPA no reproduce las métricas originales")
    report["original_epa_metrics_reproduced"] = True
    write_json(output / "matrices_confusion.json", matrices)
    make_figures(output, report, dist_frame, matrices)
    observations = make_report(output, report, metrics, dist_frame, detail_frame)
    report["observations"] = observations
    write_json(output / "reporte.json", report)
    for observation in observations:
        print(observation, flush=True)
    return output / "comparacion_indices.html"
