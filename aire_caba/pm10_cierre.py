"""Entrega OMS con NO2 conservado y PM10 con una sola variable histórica nueva.

Ejecutar: python -m aire_caba.pm10_cierre
La notebook muestra resultados persistidos, como la entrega OMS original.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from .air_quality_scales import classify_concentration
from .compare_indices import evaluate
from .model_dataset import sha256
from .modeling import PROJECT, load_data, write_json
from .no2_classifier import training_weights
from .oms_classifier import LAG, daily_frame, feature_group, make_figures, md_table, shap_importance

SOURCE = PROJECT / 'outputs/clasificacion_oms'
DEFAULT_OUTPUT = PROJECT / 'outputs/clasificacion_oms_pm10_cierre'
CONFIG = PROJECT / 'config/pm10_cierre.json'
NOTEBOOK = PROJECT / 'notebooks/tercera_entrega_oms_pm10_cierre.ipynb'
CLOSE = 'pm_close_lag1'


def add_previous_close(hourly, daily):
    """Cierre de d-1 asociado exactamente a d; sin saltar faltantes ni backfill.

    load_data ya aplicó los filtros de estado, plausibilidad e interpolación.
    La convención heredada representa HORA=24 como fecha + un día.
    """
    closing = hourly.loc[hourly.fecha_hora.eq(hourly.fecha + pd.Timedelta(days=1)),
                         ['estacion', 'fecha', 'PM10']].copy()
    if closing.duplicated(['estacion', 'fecha']).any():
        raise ValueError('Más de un cierre para una misma estación y fecha')
    closing['fecha'] += pd.Timedelta(days=1)
    closing = closing.rename(columns={'PM10': CLOSE})
    return daily.merge(closing, on=['estacion', 'fecha'], how='left', validate='one_to_one')


def select_threshold(actual, scores, fp_budget):
    """Mayor TP en calibración con FP acotados; empates: menos FP y mayor corte."""
    actual = np.asarray(actual, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if set(np.unique(actual)) != {0, 1} or actual.shape != scores.shape:
        raise ValueError('La calibración requiere ambas clases y vectores del mismo tamaño')
    if not np.isfinite(scores).all() or fp_budget < 0:
        raise ValueError('Probabilidades o presupuesto de falsas alarmas inválidos')
    _, recall, thresholds = precision_recall_curve(actual, scores)
    tp = np.rint(recall[:-1] * actual.sum()).astype(int)
    predicted = len(scores) - np.searchsorted(np.sort(scores), thresholds, side='left')
    fp = predicted - tp
    allowed = np.flatnonzero(fp <= fp_budget)
    if not len(allowed):
        return float(np.nextafter(scores.max(), np.inf))
    best = max(allowed, key=lambda i: (tp[i], -fp[i], thresholds[i]))
    return float(thresholds[best])


def predict_daily(model, daily, metadata):
    """Devuelve score y clase usando el corte guardado; no usar model.predict()."""
    values = daily[metadata['features']].astype(np.float32)
    scores = model.predict_proba(values)[:, 1].astype(float)
    return scores, (scores >= metadata['decision_threshold']).astype(np.int8)


def fit_model(train, features, config):
    model = XGBClassifier(**config['xgboost'], objective='binary:logistic', eval_metric='logloss',
                          random_state=config['seed'], n_jobs=config['threads'])
    model.fit(train[features].astype(np.float32), train.label,
              sample_weight=training_weights(train.label, config['weight_power']))
    return model


def temporal_controls(frame, features, config, temporal):
    """Reproduce sólo los tres controles del modelo final, sin otras variantes."""
    from .aqi import classification_metrics
    rows, saved = [], []
    for control in config['temporal_controls']:
        year = control['calibration_year']
        start = pd.Timestamp(f'{year}-01-01')
        next_start = pd.Timestamp(f'{year + 1}-01-01')
        end = pd.Timestamp(f'{year + 2}-01-01')
        gap = pd.Timedelta(days=temporal['gap_days'])
        train = frame.loc[frame.fecha.lt(start - gap)]
        calibration = frame.loc[frame.fecha.ge(start) & frame.fecha.lt(next_start - gap)]
        evaluation = frame.loc[frame.fecha.ge(next_start) & frame.fecha.lt(end) & frame.split.ne('purga')]
        assert train.fecha.max() < calibration.fecha.min() < evaluation.fecha.min()
        model = fit_model(train, features, config)
        scores = model.predict_proba(calibration[features].astype(np.float32))[:, 1].astype(float)
        threshold = select_threshold(calibration.label, scores, control['calibration_false_positive_budget'])
        probabilities, predicted = predict_daily(model, evaluation, {'features': features, 'decision_threshold': threshold})
        result = classification_metrics(evaluation.label, predicted, ['Cumple guía OMS', 'Supera guía OMS'])
        positive = result['classes'][1]
        rows.append(control | {'threshold': threshold, **result['summary'],
                               **{k + '_positive': positive[k] for k in ('precision', 'recall', 'f1', 'false_positive_rate')},
                               **{k: positive[k] for k in ('tp', 'fp', 'fn', 'tn', 'support')}})
        part = evaluation[['estacion', 'fecha', 'label']].copy()
        part['evaluation_year'], part['score'], part['predicted'] = year + 1, probabilities, predicted
        saved.append(part)
    all_predictions = pd.concat(saved, ignore_index=True)
    combined = classification_metrics(all_predictions.label, all_predictions.predicted, ['Cumple guía OMS', 'Supera guía OMS'])
    return pd.DataFrame(rows), all_predictions, combined


def write_summary(output, report, metrics, detail, config):
    selected = metrics.loc[metrics.split.eq('test') & metrics.station.eq('todas') & metrics.selected_by_validation]
    positive = detail.loc[detail.pollutant.eq('PM10') & detail.split.eq('test') & detail.station.eq('todas') & detail.class_id.eq(1)].iloc[0]
    pm = selected.loc[selected.pollutant.eq('PM10')].iloc[0]
    no2_observation = next(s for s in report['original_observations'] if s.startswith('NO2:'))
    co_observation = next(s for s in report['original_observations'] if s.startswith('CO:'))
    observations = [no2_observation,
                    f"PM10: XGBoost con 56 entradas, pesos suaves y una variable nueva: el cierre previo. "
                    f"En prueba detecta {int(positive.tp)} de {int(positive.support)} excedencias, con {int(positive.fp)} falsas alarmas. "
                    f"Precision de excedencias {positive.precision:.2%}, recall {positive.recall:.2%}, F1 {positive.f1:.2%}; "
                    f"accuracy {pm.accuracy:.2%} y F1 macro {pm.f1_macro:.4f}.", co_observation]
    report['observations'] = observations
    notes = [
        'NO2 conserva el modelo, los resultados, los parámetros y las predicciones originales. No se reentrena.',
        'PM10 mantiene las 55 entradas originales (incluido el promedio anterior) y agrega únicamente pm_close_lag1: el último registro publicado del día anterior, que ya es un móvil de 24 h. No es una medición instantánea.',
        'El cierre agrega información de cómo terminó la jornada frente al promedio de todos sus registros. Se seleccionó una versión pequeña mediante validación y controles temporales, sin exigir que toda variable con poca importancia fuera inútil.',
        'Parámetros de PM10: 200 árboles, profundidad 2, min_child_weight 10, learning_rate 0.05, reg_lambda 20, subsample y colsample_bytree 0.8, método hist, semilla 42. Se restringe la complejidad de los árboles y se conservan los parámetros del experimental elegido.',
        'Ponderación suave: frecuencia inversa elevada a 0.5 y normalizada a media 1, calculada sólo con entrenamiento. No se duplican filas ni se balancean los datos de validación o prueba.',
        f"Umbral de PM10: {report['models']['PM10']['decision_threshold']:.12f}. Se maximizaron detecciones en validación con hasta {config['validation_false_positive_budget']} falsas alarmas. Este presupuesto no garantiza la misma tasa futura y el score no está calibrado como probabilidad sanitaria.",
        'Entrenamiento anterior a 2024, validación 2024 y prueba desde 2025, con purga de dos días. Se conservan las filas, los filtros de calidad, la cobertura mínima de 18 registros y el redondeo de la entrega original.',
        'Los controles temporales entrenan antes de 2020/2021/2022, calibran en ese año y evalúan el siguiente. Sólo se reproducen los controles de este modelo final.',
        'La selección de entradas y umbral no se recalcula con test. El test ya fue consultado en experimentos del proyecto: estos resultados siguen siendo exploratorios y requieren una evaluación futura independiente.',
        'Para pronosticar el día d se requieren el promedio y el cierre del día d-1 ya publicados. La disponibilidad y latencia del proveedor aún deben verificarse. No permite usar el cierre completo de hoy antes de que hoy termine.',
        'La meteorología del día objetivo es ERA5 histórico. El desempeño con pronósticos meteorológicos emitidos no se midió. Las predicciones son diarias por estación, no para cualquier punto de CABA.',
        'PM10 se etiqueta con el promedio diario de sus registros móviles de 24 h: es una aproximación heredada y no equivale exactamente al promedio del día calendario ni a un dictamen de cumplimiento de la OMS.',
        'TreeSHAP describe asociaciones del modelo y no demuestra causalidad. La interpretación de NO2 mantiene el modelo original; la de PM10 corresponde al modelo reducido.',
    ]
    cols = ['pollutant', 'model', 'n', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'balanced_accuracy', 'false_positive_rate_macro']
    parts = ['# Clasificación diaria con las guías OMS 2021', '\n'.join('- ' + s for s in observations),
             '\n## Método y alcance\n', '\n'.join('- ' + s for s in notes),
             '\n## Modelos finales en prueba\n', md_table(selected[cols]),
             '\n## Métricas por clase\n', md_table(detail.loc[detail.split.eq('test') & detail.station.eq('todas') & detail.selected_by_validation,
                 ['pollutant', 'category', 'support', 'precision', 'recall', 'f1', 'false_positive_rate', 'tp', 'fp', 'fn', 'tn']])]
    (output / 'resumen_oms.md').write_text('\n\n'.join(parts) + '\n', encoding='utf-8')


def build_notebook(output):
    import nbformat
    from jupyter_client import AsyncKernelManager
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter

    template = PROJECT / 'notebooks/tercera_entrega_oms_modificado.ipynb'
    nb = nbformat.read(template, as_version=4)
    original_cells = deepcopy(nb.cells)
    relative = output.resolve().relative_to(PROJECT).as_posix()
    nb.cells[0].source = nb.cells[0].source.replace('`.\\run.ps1 classify-oms`', '`python -m aire_caba.pm10_cierre`')
    nb.cells[0].source += '\n\n**Versión final de PM10:** una sola variable nueva (cierre anterior), 56 entradas y ponderación suave. Se conserva íntegro el NO₂ original.'
    nb.cells[1].source = nb.cells[1].source.replace('outputs/clasificacion_oms"', relative + '"')
    nb.cells[5].source = nb.cells[5].source.replace('Se mantienen la partición temporal', 'Para **NO₂** se mantienen la partición temporal')
    nb.cells[5].source += '''

### PM10: modelo final con cierre previo

Se mantienen las 55 entradas originales y se agrega sólo `pm_close_lag1`: el último PM10 publicado de ayer. Ese registro ya resume las 24 horas anteriores. El promedio de ayer (`valor_dia_anterior`) resume todos los registros móviles publicados ese día; ambos aportan información distinta sobre el nivel general y el cierre de la jornada.

XGBoost usa **200 árboles, profundidad 2, `min_child_weight=10`, `reg_lambda=20`, tasa 0,05, submuestreo de filas y columnas 0,8 y semilla 42**. La ponderación es suave (exponente 0,5). Son los parámetros del experimental elegido; aquí no se repite la grilla anterior de PM10.

El corte se calibra en 2024 maximizando las detecciones con un presupuesto de hasta 14 falsas alarmas. Se eligió una versión reducida mediante validación y controles temporales anteriores. El test ya había sido visto en el proyecto: la comparación es exploratoria y no es una nueva evaluación ciega.

La mejora frente al original corresponde al conjunto de cambios de entradas, parámetros, ponderación y umbral; no se atribuye sólo a la variable nueva.
'''
    nb.cells[6].source += '''

print("PM10 — configuración final y umbral")
pm10_meta = report["models"]["PM10"]
display(pd.Series(pm10_meta["parameters"], name="valor").to_frame())
print("Umbral de decisión:", pm10_meta["decision_threshold"])
print("Variables:", len(pm10_meta["features"]), "· única nueva: pm_close_lag1")
print("PM10 — controles temporales del modelo final")
display(pd.read_csv(OUT / "control_temporal_PM10.csv")[["evaluation_year","n","precision_positive","recall_positive","f1_positive","tp","fp","fn"]])
'''
    nb.cells[7].source += '\n\nPM10 muestra únicamente el modelo final con cierre previo. NO₂ conserva sus resultados anteriores. La precision y el recall de la clase que supera la guía se leen en la tabla por clase, además de los promedios macro.'
    nb.cells[9].source += '\n\nPara PM10 se recalculan SHAP y sus gráficos con el modelo reducido. Los gráficos por año, mes y día de la semana describen las mismas observaciones que antes; no dependen de cambiar el modelo.'
    nb.cells[12].source = '''## 5. Reproducibilidad

Desde la raíz del proyecto, con el entorno de `requirements-modeling.txt` instalado:

```powershell
python -m aire_caba.pm10_cierre
```

Reentrena sólo el PM10 final con su única variable nueva, calibra el umbral en validación, reproduce sus tres controles temporales y genera resultados, figuras, modelo y esta notebook. **NO₂ se carga del modelo original y se verifica que sus datos y resultados no cambien.** Configuración: `config/pm10_cierre.json`; etiquetas: `config/indices_oms.json`.

Para regenerar únicamente la notebook desde los resultados guardados: `python -m aire_caba.pm10_cierre --report-only`. Para predecir con PM10, cargar su modelo y metadatos y utilizar `predict_daily`: `model.predict()` aplica otro corte.

El cierre previo debe estar publicado al emitir el pronóstico. La meteorología es ERA5 histórica y la etiqueta de PM10 promedia registros móviles de 24 h; faltan validar la disponibilidad operativa, el uso de pronósticos meteorológicos y una evaluación futura independiente. Los resultados corresponden a estación-día.
'''
    class ProjectKernelManager(AsyncKernelManager):
        def format_kernel_cmd(self, extra_arguments=None):
            self.kernel_spec.argv = [sys.executable, '-B', '-m', 'ipykernel_launcher', '-f', '{connection_file}']
            return super().format_kernel_cmd(extra_arguments)
    client = NotebookClient(nb, kernel_manager_class=ProjectKernelManager, timeout=180,
                            resources={'metadata': {'path': str(PROJECT)}})
    client.execute()
    nbformat.validate(nb)
    assert len(nb.cells) == len(original_cells)
    assert [c.id for c in nb.cells] == [c.id for c in original_cells]
    for i in (2, 3, 4, 8, 10, 11):
        assert nb.cells[i].source == original_cells[i].source
    nbformat.write(nb, NOTEBOOK)
    html, _ = HTMLExporter().from_notebook_node(nb)
    NOTEBOOK.with_suffix('.html').write_text(html, encoding='utf-8')
    return NOTEBOOK


def run(data_dir, output, config_path=CONFIG, execute_notebook=True):
    started = perf_counter()
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    if output.resolve() in {SOURCE.resolve(), (PROJECT / 'outputs/tercera_entrega').resolve()}:
        raise ValueError('Elegir una carpeta distinta de los resultados originales')
    original = json.loads((SOURCE / 'reporte.json').read_text(encoding='utf-8'))
    source_paths = [SOURCE / 'modelos/NO2_XGBoost_rezago.ubj', SOURCE / 'modelos/PM10_XGBoost_rezago.ubj',
                    SOURCE / 'predicciones_NO2.csv', SOURCE / 'metricas_clasificacion.csv', SOURCE / 'metricas_por_clase.csv']
    before = {p.relative_to(PROJECT).as_posix(): sha256(p) for p in source_paths}
    data, _, audit = load_data(data_dir, original['temporal_config'])
    for k in ('input_sha256', 'quality_sha256'):
        if audit[k] != original[k]:
            raise ValueError('Los datos no coinciden con los usados en la entrega OMS')
    (output / 'modelos').mkdir(parents=True, exist_ok=True)
    report = deepcopy(original)
    report['created_at_utc'] = datetime.now(timezone.utc).isoformat()
    report['original_observations'] = original['observations']
    report['pm10_final_config'] = config
    report['code_sha256'] = sha256(Path(__file__))
    report['config_sha256'] = sha256(Path(config_path))
    daily, base, coverage = daily_frame(data, 'PM10', original['temporal_config'])
    daily = add_previous_close(data, daily)
    features = [*base, LAG, CLOSE]
    assert len(features) == 56
    spec = original['scales']['pollutants']['PM10']
    daily['label'] = classify_concentration(daily.value, spec)
    train, validation, test = [daily.loc[daily.split.eq(s)] for s in ('train', 'validation', 'test')]
    assert train.fecha.max() < validation.fecha.min() < test.fecha.min()
    model = fit_model(train, features, config)
    values = model.predict_proba(validation[features].astype(np.float32))[:, 1].astype(float)
    threshold = select_threshold(validation.label, values, config['validation_false_positive_budget'])
    metadata = {'pollutant': 'PM10', 'model': config['model_name'], 'features': features, 'extra_feature': CLOSE,
                'scale': spec, 'class_names': spec['classes'], 'original_classes': [0, 1],
                'parameters': config['xgboost'] | {'weight_power': config['weight_power'], 'random_state': config['seed']},
                'decision_threshold': threshold, 'train_size': len(train), 'validation_size': len(validation), 'test_size': len(test),
                'probabilities_calibrated': False, 'requires_previous_day_close_published': True,
                'target_averages_published_rolling_24h_values': True}
    selection = {'selected': config['model_name'], 'features': features, 'parameters': metadata['parameters'],
                 'decision_threshold': threshold, 'validation_false_positive_budget': config['validation_false_positive_budget'],
                 'test_used_to_select_threshold': False, 'test_already_seen_in_project': True}
    write_json(output / 'seleccion_PM10_antes_de_test.json', selection)
    # Sólo se ajusta el umbral en validación. NO2 se carga, no se reentrena.
    controls, control_predictions, control_summary = temporal_controls(daily, features, config, original['temporal_config'])
    controls.to_csv(output / 'control_temporal_PM10.csv', index=False, encoding='utf-8-sig')
    control_predictions.to_csv(output / 'predicciones_control_temporal_PM10.csv', index=False, encoding='utf-8-sig')
    write_json(output / 'resumen_control_temporal_PM10.json', control_summary)
    print('PM10: modelo ajustado, umbral calibrado y controles temporales reproducidos.', flush=True)
    records = ([], [], [])
    export = pd.concat([validation, test], ignore_index=True)
    probabilities, predicted = predict_daily(model, export, metadata)
    summary_exceedance = []
    for split in ('validation', 'test'):
        for station in ('todas', *sorted(export.estacion.unique())):
            mask = export.split.eq(split) & (True if station == 'todas' else export.estacion.eq(station))
            context = {'pollutant': 'PM10', 'scheme': spec['scheme'], 'resolution': 'level', 'model': config['model_name'],
                       'selected_by_validation': True, 'split': split, 'station': station}
            evaluate(export.loc[mask, 'label'].to_numpy(), predicted[mask], spec['classes'], context, *records)
            from .aqi import classification_metrics
            result = classification_metrics(export.loc[mask, 'label'], predicted[mask], spec['classes'])
            positive = result['classes'][1]
            summary_exceedance.append(context | result['summary'] | {
                k + '_positive': positive[k] for k in ('precision', 'recall', 'f1', 'false_positive_rate')} | {
                k: positive[k] for k in ('tp', 'fp', 'fn', 'tn', 'support')} | {
                'average_precision': float(average_precision_score(export.loc[mask, 'label'], probabilities[mask])),
                'roc_auc': float(roc_auc_score(export.loc[mask, 'label'], probabilities[mask])) if export.loc[mask, 'label'].nunique() == 2 else None})
    pm_metrics, pm_detail, pm_matrices = records
    source_metrics = pd.read_csv(SOURCE / 'metricas_clasificacion.csv')
    source_detail = pd.read_csv(SOURCE / 'metricas_por_clase.csv')
    metrics = pd.concat([source_metrics.loc[source_metrics.pollutant.eq('NO2')], pd.DataFrame(pm_metrics)], ignore_index=True)
    detail = pd.concat([source_detail.loc[source_detail.pollutant.eq('NO2')], pd.DataFrame(pm_detail)], ignore_index=True)
    pd.testing.assert_frame_equal(metrics.loc[metrics.pollutant.eq('NO2')].reset_index(drop=True), source_metrics.loc[source_metrics.pollutant.eq('NO2')].reset_index(drop=True))
    pd.testing.assert_frame_equal(detail.loc[detail.pollutant.eq('NO2')].reset_index(drop=True), source_detail.loc[source_detail.pollutant.eq('NO2')].reset_index(drop=True))
    matrices = [m for m in json.loads((SOURCE / 'matrices_confusion.json').read_text(encoding='utf-8')) if m['pollutant'] == 'NO2'] + pm_matrices
    metrics.to_csv(output / 'metricas_clasificacion.csv', index=False, encoding='utf-8-sig')
    detail.to_csv(output / 'metricas_por_clase.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(summary_exceedance).to_csv(output / 'metricas_excedencias_PM10.csv', index=False, encoding='utf-8-sig')
    write_json(output / 'matrices_confusion.json', matrices)
    predictions = export[['estacion', 'fecha', 'split', 'value', LAG, CLOSE, 'label']].rename(columns={'value': 'PM10_promedio_diario'}).copy()
    predictions['prediction_' + config['model_name']] = predicted
    predictions['selected_probability_class_0'] = 1 - probabilities
    predictions['selected_probability_class_1'] = probabilities
    predictions.to_csv(output / 'predicciones_PM10.csv', index=False, encoding='utf-8-sig')
    for filename in ('predicciones_NO2.csv', 'distribucion_categorias.csv'):
        shutil.copyfile(SOURCE / filename, output / filename)
    for extension in ('ubj', 'meta.json'):
        filename = 'NO2_XGBoost_rezago.' + extension
        shutil.copyfile(SOURCE / 'modelos' / filename, output / 'modelos' / filename)
    target = output / 'modelos' / ('PM10_' + config['model_name'] + '.ubj')
    model.save_model(target)
    metadata['model_sha256'] = sha256(target)
    write_json(target.with_suffix('.meta.json'), metadata)
    reloaded = XGBClassifier()
    reloaded.load_model(target)
    reloaded_scores, reloaded_predicted = predict_daily(reloaded, export, metadata)
    np.testing.assert_array_equal(probabilities, reloaded_scores)
    np.testing.assert_array_equal(predicted, reloaded_predicted)
    pm_validation = next(r for r in pm_metrics if r['split'] == 'validation' and r['station'] == 'todas')
    selection['validation_score'] = pm_validation['f1_macro']
    selection['validation_positive_metrics'] = next(r for r in summary_exceedance if r['split'] == 'validation' and r['station'] == 'todas')
    report['selections']['PM10'] = selection
    report['models']['PM10'] = metadata
    report['coverage']['PM10'] = coverage
    report['pm10_previous_close_missing'] = int(daily[CLOSE].isna().sum())
    trials = pd.read_csv(SOURCE / 'candidatos_validacion.csv')
    pm_trial = {'pollutant': 'PM10', 'model': 'XGBoost', 'feature_set': 'rezago_cierre', 'weight_power': config['weight_power'],
                **config['xgboost'], **{'validation_' + k: v for k, v in pm_validation.items() if k not in ('pollutant', 'scheme', 'resolution', 'model', 'selected_by_validation', 'split', 'station')}}
    trials = pd.concat([trials.loc[trials.pollutant.eq('NO2')], pd.DataFrame([pm_trial])], ignore_index=True)
    trials.to_csv(output / 'candidatos_validacion.csv', index=False, encoding='utf-8-sig')
    no2_daily, _, _ = daily_frame(data, 'NO2', original['temporal_config'])
    no2_daily['label'] = classify_concentration(no2_daily.value, original['scales']['pollutants']['NO2'])
    no2_meta = original['models']['NO2']
    no2 = XGBClassifier()
    no2.load_model(output / 'modelos/NO2_XGBoost_rezago.ubj')
    no2_test = no2_daily.loc[no2_daily.split.eq('test')]
    shap_results = {}
    for pollutant, fitted, frame, cols, classes, name in [
        ('NO2', no2, no2_test, no2_meta['features'], 3, no2_meta['model']),
        ('PM10', model, test, features, 2, config['model_name'])]:
        importance, positive = shap_importance(fitted, frame[cols].astype(np.float32), cols, classes - 1)
        shap_results[pollutant] = {'model': name, 'positive_class': original['scales']['pollutants'][pollutant]['classes'][-1],
                                   'importance': importance.groupby(feature_group).sum().sort_values(ascending=False),
                                   'positive': positive, 'values': frame[cols]}
    pd.concat({p: s['importance'] for p, s in shap_results.items()}, names=['pollutant', 'variable']).rename('mean_abs_shap').reset_index().to_csv(
        output / 'importancia_shap.csv', index=False, encoding='utf-8-sig')
    distribution = pd.read_csv(output / 'distribucion_categorias.csv')
    (output / 'figuras').mkdir(exist_ok=True)
    make_figures(output, report, {'NO2': no2_daily, 'PM10': daily}, distribution, matrices, trials, shap_results)
    # Estas figuras describen las mismas observaciones: conservarlas exactamente.
    for filename in ('distribucion_oms.png', 'patrones_excedencias.png', 'patrones_dia_semana.png'):
        shutil.copyfile(SOURCE / 'figuras' / filename, output / 'figuras' / filename)
    write_summary(output, report, metrics, detail, config)
    report['elapsed_seconds'] = perf_counter() - started
    write_json(output / 'reporte.json', report)
    assert all(sha256(PROJECT / p) == h for p, h in before.items())
    write_json(output / 'auditoria.json', audit | {
        'source_files_sha256': before,
        'original_files_unchanged': all(sha256(PROJECT / p) == h for p, h in before.items()),
        'no2_model_identical': sha256(output / 'modelos/NO2_XGBoost_rezago.ubj') == before['outputs/clasificacion_oms/modelos/NO2_XGBoost_rezago.ubj'],
        'no2_predictions_identical': sha256(output / 'predicciones_NO2.csv') == before['outputs/clasificacion_oms/predicciones_NO2.csv'],
        'no2_metrics_unchanged': True, 'pm10_reload_reproduces_predictions': True,
        'notebook_template': 'notebooks/tercera_entrega_oms_modificado.ipynb'})
    print(pd.DataFrame(summary_exceedance).query("split == 'test' and station == 'todas'")[
        ['n', 'accuracy', 'precision_positive', 'recall_positive', 'f1_positive', 'tp', 'fp', 'fn']].to_string(index=False), flush=True)
    if execute_notebook:
        build_notebook(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=PROJECT / 'data')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--skip-notebook', action='store_true')
    args = parser.parse_args()
    if args.report_only:
        print(build_notebook(args.output_dir))
    else:
        with threadpool_limits(limits=4):
            print(run(args.data_dir, args.output_dir, args.config, not args.skip_notebook))


if __name__ == '__main__':
    main()
