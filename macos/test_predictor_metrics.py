import unittest

import numpy as np

from predictor_metrics import PredictorMetrics, top_mask


class PredictorMetricsTests(unittest.TestCase):
    def fixture(self):
        metrics = PredictorMetrics([2] * 42, [np.array([0, 1])] * 42, [np.array([0])] * 42)
        ids = np.tile(np.arange(8), (42, 1))
        weights = np.ones((42, 8))
        block = np.zeros((42, 288))
        block[:, :8] = 1 / 8
        wanted = np.zeros_like(block, dtype=bool)
        wanted[:, :2] = True
        return metrics, ids, weights, block, wanted

    def test_delivery_and_cached_hits_are_separate(self):
        metrics, ids, weights, block, wanted = self.fixture()
        ready = np.zeros_like(wanted)
        ready[:, [0, 2]] = True
        metrics.observe(wanted, ready, ids, weights, block, block)
        report = metrics.report()
        self.assertEqual(report['counts']['wanted'], 84)
        self.assertEqual(report['counts']['ready'], 84)
        self.assertEqual(report['counts']['late'], 42)
        self.assertEqual(report['counts']['cached_outside_wanted'], 42)
        self.assertEqual(report['wanted_delivery_fraction'], .5)
        self.assertEqual(report['mean_salience_coverage']['ready'], .25)
        self.assertEqual(report['allocated_slots'], 84)
        self.assertEqual(report['per_layer'][0]['counts']['wanted_ready'], 1)

    def test_calibration_waits_for_full_causal_horizon(self):
        metrics, ids, weights, block, wanted = self.fixture()
        for _ in range(15):
            metrics.observe(wanted, wanted, ids, weights, block, block)
        self.assertEqual(metrics.report()['block_calibration']['full_windows'], 0)
        metrics.observe(wanted, wanted, ids, weights, block, block)
        calibration = metrics.report()['block_calibration']
        self.assertEqual(calibration['full_windows'], 1)
        self.assertEqual(calibration['mean_total_variation'], 0)
        self.assertEqual(calibration['soft_share_bin_error'], 0)
        self.assertEqual(calibration['top_budget_observed_salience_share'], .25)
        self.assertAlmostEqual(calibration['mean_cross_entropy_nats'], np.log(8))
        for _ in range(40):
            metrics.observe(wanted, wanted, ids, weights, block, block)
        self.assertEqual(metrics.report()['block_calibration']['full_windows'], 41)
        self.assertEqual(len(metrics.predictions), 16)
        self.assertEqual(len(metrics.shares), 16)

    def test_uses_forecast_at_window_start_and_per_token_shares(self):
        metrics, ids, weights, block, wanted = self.fixture()
        forecast = block.copy()
        forecast[:, :8] = 0
        forecast[:, 8] = 1
        metrics.observe(wanted, wanted, ids, weights, forecast, block)
        forecast[:] = block
        for _ in range(15):
            metrics.observe(wanted, wanted, ids, weights * 100, block, block)
        calibration = metrics.report()['block_calibration']
        self.assertEqual(calibration['mean_total_variation'], 1)
        self.assertEqual(calibration['top_budget_predicted_salience_share'], 1)
        self.assertEqual(calibration['top_budget_observed_salience_share'], .125)

    def test_invalid_probabilities_fail_and_ties_preserve_budget(self):
        metrics, ids, weights, block, wanted = self.fixture()
        with self.assertRaises(ValueError):
            metrics.observe(wanted, wanted, ids, weights, block * 2, block)
        self.assertTrue((top_mask(np.ones((42, 288)), [3] * 42).sum(-1) == 3).all())


if __name__ == '__main__':
    unittest.main()
