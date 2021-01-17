from typing import Union, Tuple

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score


class ThresholdOptimizer:
    def __init__(self,
                 y_score: Union[np.ndarray, pd.Series, list],
                 y_true: Union[np.ndarray, pd.Series, list],
                 search_space_size: int = 100):
        """

        Args:
            y_score: output from the application of test/validation data from model/estimator.
                This should be a list, numpy array or pandas series containing probabilities
                that are to be converted into class predictions. If multidimensional input is given,
                it defaults to use predictions for class 1 during optimization.
            y_true: The true class values from the test/validation set passed into the model/estimator for predictions.
            search_space_size: The number of possible probability threshold values to optimze for
        """
        self.y_score = np.array(y_score)
        if len(self.y_score.shape) == 2:
            self.y_score = self.y_score[:, 1]
        min_threshold, max_threshold = min(self.y_score), max(self.y_score)
        self.search_space = np.linspace(min_threshold, max_threshold, search_space_size)
        self.y_true = np.array(y_true)
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
                        threshold: float) -> np.ndarray:
        """Convert predicted probabilities into binary classes based on a threshold/cutoff value

        Args:
            threshold: The probability threshold value to determine predicted classes.
                        This follows a greater than or equal to format for determining class 1

        Returns: 1 dimensional numpy array of classes

        """
        classes = np.where(self.y_score >= threshold, 1, 0)
        return classes

    def _get_best_metrics(self,
                          metric_type: str,
                          scores: list,
                          greater_is_better: bool = True,
                          verbose: bool = True) -> Tuple[int, int]:
        """computes optimized metrics based which supported metric was specified

        Args:
            metric_type: The name of the mertic to optimize for. It should be one of the supported metrics
            scores: Computed metrics for all threshold values in the search space
            greater_is_better: Optional. Indicator of whether to optimize by finding the maximum metric value
                            or the minimum metric value
            verbose: Optional. Option of whether to output results of optimization. Defaults to true

        Returns: Best score and best threshold for a specified metric

        """
        if greater_is_better:
            best_score = max(scores)
        else:
            best_score = min(scores)

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
        if verbose:
            print(f'best {metric_type}: {best_score} occurs at threshold {best_threshold}')
        return best_score, best_threshold

    def get_best_f1_metrics(self,
                            verbose: bool = True) -> Tuple[int, int]:
        """Optimizes threshold for F1 score

        Returns: best F1 score and threshold at which best F1 score occurs

        Args:
            verbose: Optional. Option of whether to output results of optimization

        """
        f1_scores = list()
        for i in self.search_space:
            classes = self.convert_classes(threshold=i)
            f1_scores.append(f1_score(classes, self.y_true))
        best_f1_score, best_f1_threshold = self._get_best_metrics(
            metric_type='f1_score',
            scores=f1_scores,
            greater_is_better=True,
            verbose=verbose
        )
        return best_f1_score, best_f1_threshold

    def get_best_sensitivity_metrics(self,
                                     verbose: bool = True) -> Tuple[int, int]:
        """Optimizes threshold for sensitivity score

        Returns: best sensitivity score and threshold at which best sensitivity score occurs

        Args:
            verbose: Optional. Option of whether to output results of optimization

        """
        sensitivity_scores = list()
        for i in self.search_space:
            classes = self.convert_classes(threshold=i)
            tn, fp, fn, tp = confusion_matrix(self.y_true, classes).ravel()
            sensitivity = tp / (tp + fn)
            sensitivity_scores.append(sensitivity)
        best_sensitivity_score, best_sensitivity_threshold = self._get_best_metrics(
            metric_type='sensitivity_score',
            scores=sensitivity_scores,
            greater_is_better=True,
            verbose=verbose
        )
        return best_sensitivity_score, best_sensitivity_threshold

    def get_best_specificity_metrics(self,
                                     verbose: bool = True) -> Tuple[int, int]:
        """Optimizes threshold for specificity

        Returns: best specificity score and threshold at which best specificity score occurs

        Args:
            verbose: Optional. Option of whether to output results of optimization

        """
        specificity_scores = list()
        for i in self.search_space:
            classes = self.convert_classes(threshold=i)
            tn, fp, fn, tp = confusion_matrix(self.y_true, classes).ravel()
            specificity = tn / (tn + fp)
            specificity_scores.append(specificity)
        best_specificity_score, best_specificity_threshold = self._get_best_metrics(
            metric_type='specificity_score',
            scores=specificity_scores,
            greater_is_better=True,
            verbose=verbose
        )
        return best_specificity_score, best_specificity_threshold

    def get_best_accuracy_metrics(self,
                                  verbose: bool = True) -> Tuple[int, int]:
        """Optimizes threshold for accuracy

        Returns: best accuracy score and threshold at which best accuracy score occurs

        Args:
            verbose: Optional. Option of whether to output results of optimization

        """
        accuracy_scores = list()
        for i in self.search_space:
            classes = self.convert_classes(threshold=i)
            accuracy_scores.append(accuracy_score(classes, self.y_true))
        best_accuracy_score, best_accuracy_threshold = self._get_best_metrics(
            metric_type='accuracy_score',
            scores=accuracy_scores,
            greater_is_better=True
        )
        return best_accuracy_score, best_accuracy_threshold

    def get_best_precision_metrics(self,
                                   verbose: bool = True) -> Tuple[int, int]:
        """Optimizes threshold for precision

        Returns: best precision score and threshold at which best precision score occurs

        Args:
            verbose: Optional. Option of whether to output results of optimization

        """
        precision_scores = list()
        for i in self.search_space:
            classes = self.convert_classes(threshold=i)
            precision_scores.append(precision_score(classes, self.y_true))
        best_precision_score, best_precision_threshold = self._get_best_metrics(
            metric_type='precision_score',
            scores=precision_scores,
            greater_is_better=True,
            verbose=verbose
        )
        return best_precision_score, best_precision_threshold

    def get_best_recall_metrics(self,
                                verbose: bool = True) -> Tuple[int, int]:
        """Optimizes threshold for recall

        Returns: best recall score and threshold at which best recall score occurs

        Args:
            verbose: Optional. Option of whether to output results of optimization

        """
        recall_scores = list()
        for i in self.search_space:
            classes = self.convert_classes(threshold=i)
            recall_scores.append(recall_score(classes, self.y_true))
        best_recall_score, best_recall_threshold = self._get_best_metrics(
            metric_type='precision_score',
            scores=recall_scores,
            greater_is_better=True,
            verbose=verbose
        )
        return best_recall_score, best_recall_threshold

    def optimize_metrics(self,
                         metrics: list = None,
                         verbose: int = 1):
        """Function to optimize for supported metrics in a batch format

        Args:
            verbose: Optional. Option of whether to output results of optimization
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
            super(ThresholdOptimizer, self).__getattribute__(f'get_best_{i}_metrics')(verbose=verbose)
