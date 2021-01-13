import unittest

from budgies.threshold_optimizer import ThresholdOptimizer

y_score = [0.1, 0.6, 0.8, 0.01, 0.55, 0.93, 0.3, 0.82, 0.22, 0.46]
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]

threshold_optimizer = ThresholdOptimizer(y_score,
                                         y_true)


class MyTestCase(unittest.TestCase):
    def test_convert_classes(self):
        test_classes = threshold_optimizer.convert_classes(0.3)
        expected_classes = [0, 1, 1, 0, 1, 1, 1, 1, 0, 1]
        self.assertEqual(test_classes, expected_classes)


if __name__ == '__main__':
    unittest.main()
