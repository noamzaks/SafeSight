import torch
from dataclasses import dataclass


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

    def __init__(self, test_labels: list[bool], test_results: list[bool]) -> None:
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
                int(not label and result)
                for label, result in zip(test_labels, test_results)
            ]
        )
        fn = sum(
            [
                int(label and not result)
                for label, result in zip(test_labels, test_results)
            ]
        )

        self.accuracy = (tp + tn) / (tp + tn + fp + fn)
        self.precision = tp / (tp + fp)
        self.false_positive_rate = fp / (fp + tn)
        self.false_negative_rate = fn / (fn + tp)


class Model:
    def train(self, dataset: torch.Dataset):
        pass

    def test(self, dataset: torch.Dataset) -> TestResults:
        pass
