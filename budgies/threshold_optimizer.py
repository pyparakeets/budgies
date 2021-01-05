from typing import Union, Tuple

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score


class ThresholdOptimizer:
    def __init__(self,
                 predicted_probabilities: Union[np.ndarray, pd.Series, list],
                 y_test: Union[np.ndarray, pd.Series, list],
                 search_space_size: int = 100):
        """

        Args:
            predicted_probabilities: output from the application of test/validation data from model/estimator.
                This should be a list, numpy array or pandas series containing probabilities
                that are to be converted into class predictions. If multidimensional input is given,
                it defaults to use predictions for class 1 during optimization.
            y_test: The true class values from the test/validation set passed into the model/estimator for predictions.
            search_space_size: The number of possible probability threshold values to optimze for
        """
        self.predicted_probabilities = predicted_probabilities
        if len(self.predicted_probabilities.shape) == 2:
            self.predicted_probabilities = self.predicted_probabilities[:, 1]
        self.search_space = np.linspace(0, 1, search_space_size)
        self.y_test = np.array(y_test)
        self.optimized_metrics = dict()
        self._supported_metrics = [
            'f1', 'accuracy', 'sensitivity', 'specificity',
            'precision', 'recall',
        ]

    def set_search_space(self,
                         search_space_size: int):
        """set the number of possible probability threshold values to optimze for

        This function is useful to reset the size of the search space after initializing the ThresholdOptimizer object.

        Args:
            search_space_size: The number of possible probability threshold values to optimze for
        """
        self.search_space = np.linspace(0, 1, search_space_size)

    def convert_classes(self,
                        threshold: int) -> np.ndarray:
        """Convert predicted probabilities into binary classes based on a threshold/cutoff value

        Args:
            threshold: The probability threshold value to determine predicted classes.
                        This follows a greater than or equal to format for determining class 1

        Returns: 1 dimensional numpy array of classes

        """
        classes = np.where(self.predicted_probabilities >= threshold, 1, 0)
        return classes

    def _get_best_metrics(self,
                          metric_type: str,
                          scores: list,
                          optimization: str = 'max') -> Tuple[int, int]:
        """computes optimized metrics based which supported metric was specified

        Args:
            optimization:
            metric_type:
            scores:

        Returns: best score and best threshold for a specified metric

        """
        if optimization.lower() == 'max':
            best_score = max(scores)
        elif optimization.lower() == 'min':
            best_score = min(scores)
        else:
            raise ValueError('Wrong value passed into optimization parameter. Should be max or min')
        best_index = scores.index(best_score)
        best_threshold = self.search_space[best_index]
        self.optimized_metrics.update(
            {
                metric_type: {
                    'best_score': best_score,
                    'best_threshold': best_threshold,
                    'all_scores': scores,
                },
            },
        )
        print(f'best {metric_type}: {best_score} occurs at threshold {best_threshold}')
        return best_score, best_threshold

    def get_best_f1_metrics(self) -> Tuple[int, int]:
        """Optimizes threshold for F1 score

        Returns: best F1 score and threshold at which best F1 score occurs

        """
        f1_scores = list()
        for i in self.search_space:
            classes = self.convert_classes(threshold=i)
            f1_scores.append(f1_score(classes, self.y_test))
        best_f1_score, best_f1_threshold = self._get_best_metrics(
            metric_type='f1_score',
            scores=f1_scores,
            optimization='max'
        )
        return best_f1_score, best_f1_threshold

    def get_best_sensitivity_metrics(self) -> Tuple[int, int]:
        """Optimizes threshold for sensitivity score

        Returns: best sensitivity score and threshold at which best sensitivity score occurs

        """
        sensitivity_scores = list()
        for i in self.search_space:
            classes = self.convert_classes(threshold=i)
            tn, fp, fn, tp = confusion_matrix(self.y_test, classes).ravel()
            sensitivity = tp / (tp + fn)
            sensitivity_scores.append(sensitivity)
        best_sensitivity_score, best_sensitivity_threshold = self._get_best_metrics(
            metric_type='sensitivity_score',
            scores=sensitivity_scores,
            optimization='max'
        )
        return best_sensitivity_score, best_sensitivity_threshold

    def get_best_specificity_metrics(self) -> Tuple[int, int]:
        """Optimizes threshold for specificity

        Returns: best specificity score and threshold at which best specificity score occurs

        """
        specificity_scores = list()
        for i in self.search_space:
            classes = self.convert_classes(threshold=i)
            tn, fp, fn, tp = confusion_matrix(self.y_test, classes).ravel()
            specificity = tn / (tn + fp)
            specificity_scores.append(specificity)
        best_specificity_score, best_specificity_threshold = self._get_best_metrics(
            metric_type='specificity_score',
            scores=specificity_scores,
            optimization='max'
        )
        return best_specificity_score, best_specificity_threshold

    def get_best_accuracy_metrics(self) -> Tuple[int, int]:
        """Optimizes threshold for accuracy

        Returns: best accuracy score and threshold at which best accuracy score occurs

        """
        accuracy_scores = list()
        for i in self.search_space:
            classes = self.convert_classes(threshold=i)
            accuracy_scores.append(accuracy_score(classes, self.y_test))
        best_accuracy_score, best_accuracy_threshold = self._get_best_metrics(
            metric_type='accuracy_score',
            scores=accuracy_scores,
            optimization='max'
        )
        return best_accuracy_score, best_accuracy_threshold

    def get_best_precision_metrics(self) -> Tuple[int, int]:
        """Optimizes threshold for precision

        Returns: best precision score and threshold at which best precision score occurs

        """
        precision_scores = list()
        for i in self.search_space:
            classes = self.convert_classes(threshold=i)
            precision_scores.append(precision_score(classes, self.y_test))
        best_precision_score, best_precision_threshold = self._get_best_metrics(
            metric_type='precision_score',
            scores=precision_scores,
            optimization='max'
        )
        return best_precision_score, best_precision_threshold

    def get_best_recall_metrics(self) -> Tuple[int, int]:
        """Optimizes threshold for recall

        Returns: best recall score and threshold at which best recall score occurs

        """
        recall_scores = list()
        for i in self.search_space:
            classes = self.convert_classes(threshold=i)
            recall_scores.append(recall_score(classes, self.y_test))
        best_recall_score, best_recall_threshold = self._get_best_metrics(
            metric_type='precision_score',
            scores=recall_scores,
            optimization='max'
        )
        return best_recall_score, best_recall_threshold

    def optimize_metrics(self,
                         metrics: list = None):
        """Function to optimize for supported metrics in a batch format

        Args:
            metrics: Optional. Should be specified if only specific supported metrics are
                    to be optimized. input must be a subset one of the supported metrics.
                    If no metrics are applied, all metrics will be optimized for.

        """

        if metrics is None:
            metrics = self._supported_metrics
        else:
            metrics = [metric.lower() for metric in metrics]
            assert all(metric in self._supported_metrics for metric in metrics)
        for i in metrics:
            super(ThresholdOptimizer, self).__getattribute__(f'get_best_{i}_metrics')()
