"""Informe y notebook ejecutado a partir de resultados guardados del modelado."""

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / "tmp/matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .aqi import CLASS_NAMES
from .modeling import MODEL_NAMES, PROJECT, WEATHER

COLORS = {"Referencia": "#8193A6", "RandomForest": "#147D92", "XGBoost": "#E28A35"}
CLASS_COLORS = ["#2B8C6A", "#E8AF32", "#BF5263"]


def feature_group(feature):
    for prefix in ("estacion_", "dia_semana_", "mes_"):
        if feature.startswith(prefix):
            return prefix[:-1]
    for variable in WEATHER:
        if feature.startswith(variable + "_"):
            return variable
    return feature


def md_table(frame):
    headers = list(frame.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = ["no evaluable" if pd.isna(v) else f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def make_figures(output, report):
    figures = output / "figuras"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "white", "savefig.facecolor": "white"})
    regression = pd.read_csv(output / "metricas_regresion.csv")
    distribution = pd.read_csv(output / "distribucion_categorias.csv")
    matrices = json.loads((output / "matrices_confusion.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    units = {"CO": "ppm", "NO2": "ppb", "PM10": "µg/m³"}
    for ax, pollutant in zip(axes, ("CO", "NO2", "PM10")):
        subset = regression.query("task == @pollutant and split == 'test' and station == 'todas'").set_index("model").reindex(MODEL_NAMES)
        ax.bar(MODEL_NAMES, subset["mae"], color=[COLORS[name] for name in MODEL_NAMES])
        ax.set(title=pollutant, ylabel=f"MAE ({units[pollutant]})")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Error en prueba: concentraciones por hora", fontweight="bold")
    fig.savefig(figures / "mae_regresion.png", dpi=150)
    plt.close(fig)

    daily_dist = distribution.query("level == 'daily' and split != 'purga'").pivot_table(
        index="split", columns="category", values="count", aggfunc="sum", fill_value=0).reindex(
            index=["train", "validation", "test"], columns=CLASS_NAMES, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    bottom = np.zeros(3)
    for name, color in zip(CLASS_NAMES, CLASS_COLORS):
        values = daily_dist[name].to_numpy()
        ax.bar(["Entrenamiento", "Validación", "Prueba"], values, bottom=bottom, label=name, color=color)
        bottom += values
    for index, total in enumerate(bottom):
        ax.text(index, total + max(bottom) * .015, f"{int(total):,}", ha="center")
    ax.set(title="Distribución de jornadas por categoría", ylabel="Cantidad de estación-días")
    ax.legend(ncol=3)
    ax.set_ylim(0, max(bottom) * 1.18)
    fig.savefig(figures / "distribucion_diaria.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), layout="constrained")
    for ax, name in zip(axes, MODEL_NAMES):
        selected = next(row for row in matrices if row["task"] == "daily" and row["split"] == "test"
                        and row["station"] == "todas" and row["model"] == name)
        matrix = np.asarray(selected["matrix"])
        sums = matrix.sum(axis=1, keepdims=True)
        fractions = np.divide(matrix, sums, out=np.zeros_like(matrix, dtype=float), where=sums != 0)
        ax.imshow(fractions, cmap="Blues", vmin=0, vmax=1)
        for row in range(3):
            for col in range(3):
                text = f"{matrix[row, col]}\n{fractions[row, col]:.0%}" if sums[row, 0] else "0\nsin casos"
                ax.text(col, row, text, ha="center", va="center",
                        color="white" if fractions[row, col] > .55 else "#172B42", fontsize=9)
        ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
               xlabel="Categoría predicha", ylabel="Categoría real", title=name)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Clasificación diaria — matrices de confusión de prueba", fontweight="bold")
    fig.savefig(figures / "matrices_diarias.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, pollutant in zip(axes, ("CO", "NO2", "PM10")):
        frame = pd.read_csv(output / f"predicciones_{pollutant}.csv").query("split == 'test'")
        winner = report["selected_by_validation"]["regression"][pollutant]
        if len(frame) > 4000:
            frame = frame.sample(4000, random_state=42)
        prediction = frame[f"prediction_{winner}"]
        ax.scatter(frame["actual"], prediction, s=5, alpha=.25, color=COLORS[winner], rasterized=True)
        limit = max(frame["actual"].max(), prediction.max())
        ax.plot([0, limit], [0, limit], "--", color="#4D6178", linewidth=1)
        ax.set(title=f"{pollutant} — {winner}", xlabel=f"Valor real ({units[pollutant]})",
               ylabel=f"Predicción ({units[pollutant]})")
    fig.suptitle("Valores reales y predichos — muestra de prueba", fontweight="bold")
    fig.savefig(figures / "reales_predichos.png", dpi=150)
    plt.close(fig)

    importance = pd.read_csv(output / "importancia_variables.csv")
    winner = report["selected_by_validation"]["daily"]
    imp = importance.query("task == 'daily' and model == @winner").copy()
    if not imp.empty:
        imp["group"] = imp["feature"].map(feature_group)
        grouped = imp.groupby("group")["importance"].sum().nlargest(12).sort_values()
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        ax.barh(grouped.index, grouped, color=COLORS[winner])
        ax.set(title=f"Importancia de entradas — clasificador {winner}", xlabel="Importancia agrupada")
        fig.savefig(figures / "importancia_diaria.png", dpi=150)
        plt.close(fig)


def make_summary(output, report):
    reg = pd.read_csv(output / "metricas_regresion.csv")
    cls = pd.read_csv(output / "metricas_clasificacion.csv")
    detail = pd.read_csv(output / "metricas_por_clase.csv")
    winner = report["selected_by_validation"]["daily"]
    daily = cls.query("task == 'daily' and split == 'test' and station == 'todas'")
    daily_table = daily[["model", "n", "accuracy", "precision_macro", "recall_macro", "f1_macro",
                         "f1_macro_supported", "false_positive_rate_macro"]]
    reg_table = reg.query("split == 'test' and station == 'todas'")[["task", "model", "mae", "rmse", "r2", "mae_high"]]
    selected = daily.loc[daily.model.eq(winner)].iloc[0]
    baseline = daily.loc[daily.model.eq("Referencia")].iloc[0]
    minority = detail.query("task == 'daily' and split == 'test' and station == 'todas' and model == @winner")
    severe_support = int(minority.loc[minority.class_id.eq(2), "support"].iloc[0])
    lines = [
        "# Tercera entrega: resultados e interpretación",
        "",
        "Se compararon una referencia simple, Random Forest y XGBoost. Los modelos se eligieron por validación, antes de evaluar la prueba.",
        "",
        f"Se identificaron {report['outliers_original']} valores originales por encima de los máximos de plausibilidad, todos de CO. Ya estaban invalidados en el ETL. No se reintrodujeron valores interpolados como objetivos.",
        "",
        "## Regresión de concentraciones",
        "", md_table(reg_table), "",
    ]
    for pollutant, name in report["selected_by_validation"]["regression"].items():
        result = reg.query("task == @pollutant and model == @name and split == 'test' and station == 'todas'").iloc[0]
        reference = reg.query("task == @pollutant and model == 'Referencia' and split == 'test' and station == 'todas'").iloc[0]
        improvement = (1 - result.mae / reference.mae) * 100
        lines.append(f"- {pollutant}: validación seleccionó {name}. MAE de prueba {result.mae:.4f}; diferencia respecto de la referencia: {improvement:.1f}% de reducción del MAE.")
        stations = reg.query("task == @pollutant and model == @name and split == 'test' and station != 'todas'").sort_values("mae")
        best, worst = stations.iloc[0], stations.iloc[-1]
        lines.append(f"  El MAE varía entre {best.mae:.4f} en {best.station} y {worst.mae:.4f} en {worst.station}.")
        if result.r2 < 0:
            lines.append(f"  El R² de {pollutant} es negativo: en error cuadrático no mejora la media del conjunto de prueba. No se considera un pronóstico fiable por tener bajo error absoluto solamente.")
    lines += [
        "", "## Clasificación diaria", "", md_table(daily_table), "",
        f"Validación seleccionó {winner}; accuracy de prueba {selected.accuracy:.3f} y F1 macro fijo de tres clases {selected.f1_macro:.3f}. La referencia obtuvo F1 macro {baseline.f1_macro:.3f}.",
        "",
        "F1 macro_supported y balanced_accuracy promedian sólo las clases con casos reales; el F1 macro principal mantiene las tres etiquetas y asigna cero a las clases no evaluables. El detalle deja recall/precisión no definidos como vacíos, no como éxitos.",
        "",
        f"La clase Desfavorable tiene {severe_support} casos reales en prueba. " + (
            "No se puede estimar su sensibilidad con este período; los falsos positivos sí se pueden contar. Una accuracy alta no demuestra detección de episodios desfavorables."
            if severe_support == 0 else "Su recall debe interpretarse junto con este soporte y la matriz de confusión."),
        "",
        md_table(minority[["category", "support", "precision", "recall", "f1", "false_positive_rate", "fp", "fn"]]),
        "",
        "## Conocimiento obtenido y alcance",
        "",
        "- La clasificación está muy desbalanceada; la referencia mayoritaria puede tener accuracy alta sin detectar días moderados.",
        "- Los errores de concentraciones altas se analizan por separado con un umbral percentil 90 calculado exclusivamente en entrenamiento.",
        "- Las importancias describen asociaciones útiles para los modelos, no causas de contaminación. La estación y las variables meteorológicas pueden compartir información.",
        "- Las métricas por estación permiten identificar dónde empeora cada modelo; el modelo sigue siendo conjunto para las tres estaciones.",
        "- La evaluación es temporal con dos días de purga entre conjuntos. No se mezclaron estaciones de la misma fecha entre entrenamiento y prueba.",
        "- Las categorías se basan en cortes EPA, aplicados a los registros publicados y resumidos por la peor clase del día. Es una adaptación del proyecto, no el AQI diario oficial.",
        "- El conjunto sólo incluye CO, NO2 y PM10; una etiqueta Buena no describe contaminantes no medidos aquí.",
        "- ERA5 describe meteorología histórica. Estos resultados no validan todavía el error operativo a 24/48 horas con pronósticos meteorológicos.",
        "",
        "Fuentes: [EPA, AQI técnico](" + report["source"] + "); [definición de contaminantes de Buenos Aires Data](https://data.buenosaires.gob.ar/es_AR/dataset/calidad-aire/resource/juqdkmgo-294-resource).",
    ]
    moderate = minority.loc[minority.class_id.eq(1)].iloc[0]
    if moderate.support > 0:
        lines += [
            "", f"En días moderados, {winner} logra recall {moderate.recall:.3f} y precisión {moderate.precision:.3f}; "
            f"pierde {int(moderate.fn)} de {int(moderate.support)} días moderados y genera {int(moderate.fp)} falsas alarmas de esa categoría. "
            "Es una primera aproximación con limitaciones para detectar episodios, no un sistema de alerta validado.",
        ]
    importance = pd.read_csv(output / "importancia_variables.csv")
    importance = importance.query("task == 'daily' and model == @winner").copy()
    if not importance.empty:
        importance["group"] = importance.feature.map(feature_group)
        top = importance.groupby("group").importance.sum().nlargest(5)
        lines += ["", "En el clasificador elegido, las entradas de mayor importancia agrupada son: "
                  + ", ".join(f"{name} ({value:.1%})" for name, value in top.items())
                  + ". Estas importancias son internas al modelo y no identifican causas."]
    (output / "resumen_resultados.md").write_text("\n".join(lines), encoding="utf-8")


def build_notebook(output, execute=True):
    import nbformat
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    from jupyter_client import AsyncKernelManager

    notebook_path = PROJECT / "notebooks/tercera_entrega_modelado.ipynb"
    relative_output = output.resolve().relative_to(PROJECT).as_posix()
    cells = []
    markdown = lambda text: cells.append(nbformat.v4.new_markdown_cell(text))
    code = lambda text: cells.append(nbformat.v4.new_code_cell(text))
    markdown("# Tercera entrega — Modelado y evaluación\n"
             "Calidad del aire en CABA · UTN FRC · Grupo 12\n\n"
             "Tres regresiones (CO, NO₂ y PM10) y clasificación diaria por estación. "
             "Se comparan referencia, Random Forest y XGBoost. Este notebook presenta resultados calculados, con código reproducible.")
    code('from pathlib import Path\nimport json\nimport sys\nimport pandas as pd\nfrom IPython.display import display, Image, Markdown\n'
         'ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()\n'
         'sys.path.insert(0, str(ROOT))\n'
         f'OUT = ROOT / "{relative_output}"\n'
         'report = json.loads((OUT / "reporte.json").read_text(encoding="utf-8"))\n'
         'audit = json.loads((OUT / "auditoria.json").read_text(encoding="utf-8"))\n'
         'pd.set_option("display.max_columns", 20)\n'
         'print("Filas de entrada:", audit["input_rows"])\nprint("Cobertura diaria:", report["coverage"])')
    markdown("## 1. Dataset y decisiones\n\n"
             "Entrada: dataset_objetivo_listo_modelar.csv, con one-hot de estación, día de semana, mes y hora. "
             "Las fechas identifican y dividen períodos, pero no son predictores directos. "
             "Para cada regresión se excluyen los tres contaminantes de las entradas y se filtran los objetivos sin dato, "
             "interpolados o fuera de los límites del notebook de segunda entrega. Los censurados conservan LD/2.\n\n"
             "Los máximos de plausibilidad son reglas del proyecto: CO 10 ppm, NO₂ 400 ppb y PM10 600 µg/m³. "
             "Los valores altos dentro de esos límites se conservan.")
    code('display(pd.DataFrame(audit["original_outliers"]["by_station"]))\n'
         'display(pd.DataFrame(audit["targets"]).T)\n'
         'print("Columnas predictoras horarias:", len(report["features"]["hourly"]))')
    markdown("## 2. Etiquetas EPA adaptadas\n\n"
             "| Clase | CO (ppm) | NO₂ (ppb) | PM10 (µg/m³) |\n"
             "|---|---|---|---|\n| Buena | 0–4,4 | 0–53 | 0–54 |\n"
             "| Moderada | 4,5–9,4 | 54–100 | 55–154 |\n"
             "| Desfavorable, agrupada | ≥9,5 | ≥101 | ≥155 |\n\n"
             "Se aplica el truncamiento EPA: CO a un decimal; NO₂ y PM10 a enteros. "
             "Las categorías EPA desde 101 se agrupan como Desfavorable. "
             "Los datos publicados ya son CO móvil 8 h, NO₂ 1 h y PM10 móvil 24 h; no se recalculan esas ventanas. "
             "La etiqueta diaria es la peor categoría entre los registros del día y los tres contaminantes. "
             "Esto es una adaptación del proyecto y no reproduce el AQI diario oficial.\n\n"
             "Cada estación-día necesita al menos 18 horas válidas de cada contaminante y las 24 filas meteorológicas. "
             "Los conteos y categorías observadas no entran como predictores. "
             "Las variables meteorológicas diarias se resumen con media, mínimo y máximo; además se suma la precipitación.")
    code('distribution = pd.read_csv(OUT / "distribucion_categorias.csv")\n'
         'display(distribution.query("level == \'daily\'").pivot_table(index="split", columns="category", values="count", aggfunc="sum", fill_value=0))\n'
         'display(Image(filename=str(OUT / "figuras/distribucion_diaria.png")))')
    markdown("## 3. Partición temporal\n\n"
             "Entrenamiento anterior a 2024; validación en 2024; prueba desde 2025. "
             "Se purgan dos días al final de entrenamiento y validación para separar ventanas vecinas. "
             "Los cortes son comunes a todas las estaciones. Los parámetros se eligen con validación; "
             "la prueba no participa en la selección.")
    code('display(pd.DataFrame({"horario": report["splits_hourly"], "diario": report["splits_daily"]}).fillna(0).astype(int))\n'
         'display(pd.Series(report["config"], name="Configuración"))')
    markdown("## 4. Algoritmos y parámetros\n\n"
             "Regresión: mediana por estación, Random Forest Regressor y XGBoost Regressor. "
             "Clasificación diaria: clase mayoritaria, Random Forest Classifier y XGBoost Classifier. "
             "Las clases se balancean únicamente durante el ajuste de los clasificadores, con pesos calculados en entrenamiento. "
             "Las métricas conservan la distribución natural de validación y prueba. No se crean muestras sintéticas.\n\n"
             "Se prueban dos configuraciones XGBoost. Se elige por MAE para regresión y F1 macro para clasificación. "
             "Las predicciones de concentración negativas se limitan a cero; las observaciones no se rellenan.")
    code('display(pd.DataFrame(report["tuning_trials"]))\n'
         'display(report["parameters"])\n'
         'display(report["selected_by_validation"])')
    markdown("## 5. Evaluación de concentraciones\n\n"
             "MAE y RMSE se expresan en las unidades de cada contaminante; R² es complementario. "
             "MAE_high mide el error en concentraciones que superan el percentil 90 de entrenamiento. "
             "También se evalúan las categorías derivadas de cada predicción horaria.")
    code('reg = pd.read_csv(OUT / "metricas_regresion.csv")\n'
         'display(reg.query("split == \'test\' and station == \'todas\'")[["task","model","n","mae","rmse","r2","mae_high"]])\n'
         'display(Image(filename=str(OUT / "figuras/mae_regresion.png")))\n'
         'display(Image(filename=str(OUT / "figuras/reales_predichos.png")))')
    code('display(reg.query("split == \'test\' and station != \'todas\'")[["task","model","station","n","mae","rmse","r2"]])')
    markdown("## 6. Clasificación: métricas requeridas\n\n"
             "Se informan matriz de confusión, precision, recall, F1-score, accuracy y tasa de falsos positivos. "
             "Para cada clase: FPR = FP / (FP + TN), considerando esa clase frente al resto. "
             "Se agregan balanced accuracy y F1 ponderado como complementos.\n\n"
             "El F1 macro principal promedia las tres clases fijas. El F1 macro_supported sólo promedia clases "
             "con casos reales. Una clase sin soporte no tiene recall evaluable: en el detalle figura vacío. "
             "No se interpreta un FPR bajo ni una accuracy alta como evidencia de detectar eventos ausentes.")
    code('cls = pd.read_csv(OUT / "metricas_clasificacion.csv")\n'
         'display(cls.query("task == \'daily\' and station == \'todas\'")[["split","model","n","accuracy","precision_macro","recall_macro","f1_macro","f1_macro_supported","balanced_accuracy","false_positive_rate_macro"]])\n'
         'display(Image(filename=str(OUT / "figuras/matrices_diarias.png")))')
    code('detail = pd.read_csv(OUT / "metricas_por_clase.csv")\n'
         'winner = report["selected_by_validation"]["daily"]\n'
         'display(detail.query("task == \'daily\' and split == \'test\' and station == \'todas\' and model == @winner")[["category","support","precision","recall","f1","false_positive_rate","tp","fp","fn","tn"]])\n'
         'display(cls.query("task == \'daily\' and split == \'test\' and station != \'todas\'")[["model","station","n","accuracy","f1_macro","f1_macro_supported","false_positive_rate_macro"]])')
    markdown("## 7. Patrones e interpretación\n\n"
             "Las importancias se agrupan para reunir las columnas de una misma variable one-hot o de sus resúmenes "
             "meteorológicos. Describen la utilización de entradas por el modelo y no relaciones causales.")
    code('importance_plot = OUT / "figuras/importancia_diaria.png"\n'
         'if importance_plot.exists():\n    display(Image(filename=str(importance_plot)))\n'
         'display(Markdown((OUT / "resumen_resultados.md").read_text(encoding="utf-8")))')
    markdown("## 8. Reproducibilidad y aplicación futura\n\n"
             "Instalar requirements-modeling.txt en .venv y ejecutar desde PowerShell: .\\run.ps1 model. "
             "La configuración está en config/modeling.json. "
             "Los modelos, esquemas y parámetros se guardan en outputs/tercera_entrega/models; "
             "las métricas, predicciones y figuras quedan junto a ellos. "
             "Los CSV de segunda entrega no se modifican.\n\n"
             "ERA5 se acepta como aproximación inicial para aprender la relación meteorología–contaminación. "
             "El error de pronóstico meteorológico y los horizontes reales de 24/48 h deberán validarse en uso operativo. "
             "Este trabajo todavía no implementa frontend/backend.")
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


def build_report(output=None, execute=True):
    output = Path(output or PROJECT / "outputs/tercera_entrega")
    report = json.loads((output / "reporte.json").read_text(encoding="utf-8"))
    make_figures(output, report)
    make_summary(output, report)
    return build_notebook(output, execute)
