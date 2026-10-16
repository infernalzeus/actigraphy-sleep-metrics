import unittest

import numpy as np
import pandas as pd

from nonparametric import compute_mxlx, intradailyVariability


class NonparametricRefactorTests(unittest.TestCase):
    def test_exact_minute_windows_and_onsets(self):
        times = pd.date_range('2026-01-01', periods=2 * 1440, freq='min')
        minute = np.arange(len(times)) % 1440
        activity = np.ones(len(times), dtype=float)
        activity[(minute >= 8 * 60) & (minute < 18 * 60)] = 10.0
        activity[(minute < 5 * 60)] = 0.0
        result = compute_mxlx(pd.DataFrame({'Time': times, 'SVM': activity}))
        self.assertEqual(result['M10_onset_time'], '08:00')
        self.assertEqual(result['L5_onset_time'], '00:00')
        self.assertEqual(result['M10_mean_activity'], 10.0)
        self.assertEqual(result['L5_mean_activity'], 0.0)

    def test_resolution_and_missing_window_policy(self):
        times = pd.date_range('2026-01-01', periods=1440, freq='min')
        data = pd.DataFrame({'Time': times, 'SVM': np.arange(1440, dtype=float)})
        data = data.drop(index=100)
        result = compute_mxlx(data, m_window_hours=1, l_window_hours=1,
                              resolution_minutes=10)
        self.assertTrue(result['M10_onset_time'].endswith('0'))
        self.assertFalse(result['mxlx_complete_profile'])

    def test_iv_does_not_join_across_missing_gap(self):
        values = np.array([[0.0, np.nan, 10.0, 5.0]])
        self.assertTrue(np.isnan(intradailyVariability(values, LOCF=False)))
        self.assertFalse(np.isnan(intradailyVariability(values, LOCF=True)))

if __name__ == '__main__':
    unittest.main()
