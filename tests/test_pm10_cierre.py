import unittest

import numpy as np
import pandas as pd

from aire_caba.pm10_cierre import CLOSE, add_previous_close, predict_daily, select_threshold


def hourly():
    rows = []
    for day in pd.date_range('2023-01-01', periods=4):
        for hour in range(1, 25):
            rows.append({'estacion': 'centenario', 'fecha': day,
                         'fecha_hora': day + pd.Timedelta(hours=hour), 'PM10': day.day * 10. + hour})
    return pd.DataFrame(rows)


class Pm10CierreTests(unittest.TestCase):
    def test_previous_close_is_calendar_day_and_ignores_future_values(self):
        data = hourly()
        days = data[['estacion', 'fecha']].drop_duplicates()
        result = add_previous_close(data, days)
        self.assertTrue(np.isnan(result.iloc[0][CLOSE]))
        self.assertEqual(result.iloc[1][CLOSE], 34.)
        changed = data.copy()
        changed.loc[changed.fecha.ge(pd.Timestamp('2023-01-03')), 'PM10'] = 999.
        later = add_previous_close(changed, days)
        pd.testing.assert_frame_equal(result.loc[result.fecha.le('2023-01-03')], later.loc[later.fecha.le('2023-01-03')])

    def test_missing_close_is_not_replaced_by_earlier_day_or_hour(self):
        data = hourly()
        days = data[['estacion', 'fecha']].drop_duplicates()
        removed = data.loc[~(data.fecha.eq('2023-01-02') & data.fecha_hora.eq('2023-01-03'))]
        result = add_previous_close(removed, days)
        self.assertTrue(np.isnan(result.loc[result.fecha.eq('2023-01-03'), CLOSE].iloc[0]))
        self.assertEqual(result.loc[result.fecha.eq('2023-01-04'), CLOSE].iloc[0], 54.)

    def test_duplicate_closing_measurement_rejected(self):
        data = hourly()
        with self.assertRaises(ValueError):
            add_previous_close(pd.concat([data, data.iloc[[23]]]), data[['estacion', 'fecha']].drop_duplicates())

    def test_threshold_maximizes_detections_inside_budget(self):
        actual, scores = np.array([0, 0, 1, 0, 1]), np.array([.1, .6, .5, .4, .8])
        for budget in range(4):
            cut = select_threshold(actual, scores, budget)
            yp = scores >= cut
            self.assertLessEqual(int(np.sum(yp & (actual == 0))), budget)
            feasible = [int(np.sum((scores >= c) & (actual == 1))) for c in np.unique(scores)
                        if int(np.sum((scores >= c) & (actual == 0))) <= budget]
            if feasible:
                self.assertEqual(int(np.sum(yp & (actual == 1))), max(feasible))

    def test_threshold_can_emit_no_alerts(self):
        self.assertGreater(select_threshold([0, 1], [1., 0.], 0), 1.)
        with self.assertRaises(ValueError):
            select_threshold([0, 0], [.1, .3], 0)

    def test_prediction_uses_saved_threshold_and_feature_order(self):
        class StubModel:
            def predict_proba(self, values):
                self.columns = list(values.columns)
                return np.array([[.55, .45], [.8, .2]])
        model = StubModel()
        frame = pd.DataFrame({'b': [3, 4], 'a': [1, 2], 'unused': [9, 9]})
        scores, labels = predict_daily(model, frame, {'features': ['a', 'b'], 'decision_threshold': .4})
        np.testing.assert_array_equal(scores, [.45, .2])
        np.testing.assert_array_equal(labels, [1, 0])
        self.assertEqual(model.columns, ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
