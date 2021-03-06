import unittest

import numpy as np

from threshold_optimizer.threshold_optimizer import ThresholdOptimizer

y_score = [0.1, 0.6, 0.8, 0.01, 0.55, 0.93, 0.3, 0.82, 0.22, 0.46]
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]

threshold_optimizer = ThresholdOptimizer(y_score,
                                         y_true,
                                         10)


class ThresholdOptimizerTestCases(unittest.TestCase):
    def test_convert_classes(self):
        test_classes = threshold_optimizer.convert_classes(0.3)
        expected_classes = np.array([0, 1, 1, 0, 1, 1, 1, 1, 0, 1])
        assert (test_classes == expected_classes).all()

    def test_get_best_metrics(self):
        metric_type = 'test_accuracy'
        # test for maximizing metrics
        test_best_score, test_best_threshold = threshold_optimizer._get_best_metrics(metric_type,
                                                                                     y_score,
                                                                                     greater_is_better=True)
        best_score, best_threshold = 0.93, 0.5211111111111112
        self.assertEqual(best_score, test_best_score)
        self.assertEqual(best_threshold, test_best_threshold)

        # test for minimizing metrics
        test_best_score, test_best_threshold = threshold_optimizer._get_best_metrics(metric_type,
                                                                                     y_score,
                                                                                     greater_is_better=False)
        best_score, best_threshold = 0.01, 0.3166666666666667
        self.assertEqual(best_score, test_best_score)
        self.assertEqual(best_threshold, test_best_threshold)


if __name__ == '__main__':
    unittest.main()
