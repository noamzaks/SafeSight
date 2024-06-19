from dataclasses import dataclass
from typing import List


@dataclass
class TestResults:
    """
    Whereas:
        * TP - # true positives
        * TN - # true negatives
        * FP - # false positives
        * TP - # false negatives

    Definitions:
        * Accuracy = (TP + TN) / (TP + TN + FP + FN)
        * Precision = (TP) / (TP + FP)
        * False Positive Rate = (FP) / (FP + TN)
        * False Negative Rate = (FN) / (FN + TP)
    """

    accuracy: float
    precision: float
    false_positive_rate: float
    false_negative_rate: float

    def __str__(self):
        return (
            f"TestResults("
            f"accuracy={self.accuracy:.3f}, "
            f"precision={self.precision:.3f}, "
            f"false_positive_rate={self.false_positive_rate:.3f}, "
            f"false_negative_rate={self.false_negative_rate:.3f})"
        )

    def __init__(self, test_labels: List[bool], test_results: List[bool]) -> None:
        assert len(test_results) == len(test_labels)

        tp = sum(
            [int(label and result) for label, result in zip(test_labels, test_results)]
        )
        tn = sum(
            [
                int(not label and not result)
                for label, result in zip(test_labels, test_results)
            ]
        )
        fp = sum(
            [
                int((not label) and result)
                for label, result in zip(test_labels, test_results)
            ]
        )
        fn = sum(
            [
                int(label and (not result))
                for label, result in zip(test_labels, test_results)
            ]
        )

        try:
            self.accuracy = (tp + tn) / (tp + tn + fp + fn)
        except ZeroDivisionError:
            self.accuracy = 0
        try:
            self.precision = tp / (tp + fp)
        except ZeroDivisionError:
            self.precision = 0
        try:
            self.false_positive_rate = fp / (fp + tn)
        except ZeroDivisionError:
            self.false_positive_rate = 0
        try:
            self.false_negative_rate = fn / (fn + tp)
        except ZeroDivisionError:
            self.false_negative_rate = 0
