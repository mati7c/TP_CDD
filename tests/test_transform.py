import unittest

try:
    import pandas as pd
    from aire_caba.transform import CALENDAR, METEO_RANGE, POLLUTANTS, STATIONS, build_target, fill_short_gaps, flatline
except ImportError:
    pd = None


def entrada(hours, values):
    """Filas del 01/01/2021 con HORA dada; `values` asigna texto a columnas de contaminantes."""
    rows = []
    for i, h in enumerate(hours):
        row = {"FECHA": "01JAN2021:00:00:00", "HORA": str(h), "CAL_dia_semana": "5"}
        row.update({f"CAL_{c}": "0" for c in CALENDAR})
        for s in STATIONS + ("PALERMO",):
            row.update({f"{p}_{s}": "s/d" for p in POLLUTANTS})
            row.update({f"METEO_{s}_{v}": str((lo + hi) / 2) for v, (lo, hi) in METEO_RANGE.items()})
        row.update({k: v[i] for k, v in values.items()})
        rows.append(row)
    return pd.DataFrame(rows)


@unittest.skipIf(pd is None, "requiere pandas (requirements-etl.txt)")
class TransformTests(unittest.TestCase):
    def test_cleaning_rules_and_hour_convention(self):
        hours = list(range(1, 25)) + [0, 30]
        co = ["0.5", "<0.05", "s/d", "0.7", "40", "#REF!"] + ["0.5"] * 18 + ["", ""]
        pm10 = ["0"] + ["20"] * 23 + ["", ""]
        no2 = ["15"] * 24 + ["", ""]
        data, report = build_target(entrada(hours, {"CO_CENTENARIO": co, "PM10_CENTENARIO": pm10, "NO2_CENTENARIO": no2}))
        cen = data[data.estacion == "centenario"].set_index("fecha_hora")
        self.assertEqual(report["filas_hora_invalida_descartadas"], 2)
        self.assertEqual(cen.index[-1], "2021-01-02 00:00")
        self.assertAlmostEqual(cen.loc["2021-01-01 02:00", "CO"], 0.025)
        self.assertEqual(cen.loc["2021-01-01 02:00", "CO_estado"], "censurado")
        self.assertEqual(cen.loc["2021-01-01 03:00", "CO_estado"], "interpolado")
        self.assertEqual(cen.loc["2021-01-01 05:00", "CO_estado"], "interpolado")
        self.assertEqual(cen.loc["2021-01-01 01:00", "PM10_estado"], "fuera_rango")
        self.assertEqual(cen.loc["2021-01-01 02:00", "PM10"], 20)
        self.assertTrue(cen["NO2"].isna().all())
        self.assertEqual(set(cen["NO2_estado"]), {"congelado"})
        self.assertNotIn("palermo", set(data.estacion))

    def test_invalid_hour_with_data_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "HORA fuera"):
            build_target(entrada([0], {"CO_CENTENARIO": ["0.4"]}))

    def test_only_short_internal_gaps_are_filled(self):
        s = pd.Series([1.0, None, None, 4.0, None, None, None, None, 9.0, None])
        filled, mask = fill_short_gaps(s, 3)
        self.assertEqual(mask.tolist(), [False, True, True, False, False, False, False, False, False, False])
        self.assertEqual(filled[1], 2.0)

    def test_flatline_ignores_censored(self):
        value = pd.Series([1.0] * 4 + [2.0])
        self.assertEqual(flatline(value, pd.Series(["medido"] * 5), 4).tolist(), [True] * 4 + [False])
        self.assertFalse(flatline(value, pd.Series(["censurado"] * 5), 4).any())
