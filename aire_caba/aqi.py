"""Tres etiquetas del proyecto basadas en los cortes de concentración de EPA."""

import numpy as np

CLASS_NAMES = ("Buena", "Moderada", "Desfavorable")
BREAKPOINTS = {"CO": (4.5, 9.5), "NO2": (54.0, 101.0), "PM10": (55.0, 155.0)}
EPA_SOURCE = "https://document.airnow.gov/technical-assistance-document-for-the-reporting-of-daily-air-quailty.pdf"


def categorize(values, pollutant):
    """EPA trunca CO a un decimal y NO2/PM10 a enteros; -1 indica dato inválido."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values) & (values >= 0)
    labels = np.full(values.shape, -1, dtype=np.int8)
    factor = 10 if pollutant == "CO" else 1
    truncated = np.floor(values[valid] * factor + 1e-10) / factor
    labels[valid] = np.searchsorted(BREAKPOINTS[pollutant], truncated, side="right")
    return labels


def classification_metrics(actual, predicted, class_names=CLASS_NAMES):
    """Matriz real x predicha y métricas multiclase; FPR por clase contra el resto."""
    actual, predicted = np.asarray(actual, dtype=int), np.asarray(predicted, dtype=int)
    if actual.shape != predicted.shape or actual.ndim != 1 or len(actual) == 0:
        raise ValueError("Se requieren dos vectores de etiquetas del mismo tamaño, no vacíos")
    if len(class_names) < 2 or len(set(class_names)) != len(class_names):
        raise ValueError("Se requieren al menos dos nombres de clase distintos")
    labels = np.arange(len(class_names))
    if not (np.isin(actual, labels).all() and np.isin(predicted, labels).all()):
        raise ValueError("Las etiquetas deben corresponder a las clases declaradas")
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    np.add.at(matrix, (actual, predicted), 1)
    total = int(matrix.sum())
    rows = []
    for label, name in enumerate(class_names):
        tp = int(matrix[label, label])
        fn = int(matrix[label].sum() - tp)
        fp = int(matrix[:, label].sum() - tp)
        tn = total - tp - fn - fp
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        # cero para una clase presente que nunca fue predicha; no se inventa soporte.
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
        rows.append({
            "class_id": label, "category": name, "support": tp + fn,
            "predicted_count": tp + fp, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        })
    supported = [row for row in rows if row["support"] > 0]
    fprs = [row["false_positive_rate"] for row in rows if row["false_positive_rate"] is not None]
    summary = {
        "n": total, "accuracy": float(np.trace(matrix) / total),
        "precision_macro": sum(row["precision"] or 0 for row in rows) / len(labels),
        "recall_macro": sum(row["recall"] or 0 for row in rows) / len(labels),
        "f1_macro": sum(row["f1"] or 0 for row in rows) / len(labels),
        "precision_macro_supported": sum(row["precision"] or 0 for row in supported) / len(supported),
        "recall_macro_supported": sum(row["recall"] for row in supported) / len(supported),
        "f1_macro_supported": sum(row["f1"] or 0 for row in supported) / len(supported),
        "balanced_accuracy": sum(row["recall"] for row in supported) / len(supported),
        "f1_weighted": sum((row["f1"] or 0) * row["support"] for row in rows) / total,
        "false_positive_rate_macro": float(np.mean(fprs)) if fprs else None,
        "ordinal_mae": float(np.abs(actual - predicted).mean()),
        "classes_present": len(supported),
    }
    return {"summary": summary, "classes": rows, "confusion": matrix.tolist()}
