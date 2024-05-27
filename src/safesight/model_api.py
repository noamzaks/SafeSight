import torch
from dataclasses import dataclass


@dataclass
class TestResults:
    """
    See Typst doc for definitions of values.
    """

    accuracy: float
    precision: float
    false_positive_rate: float
    false_negative_rate: float

    def __init__(self, test_labels: list[bool], test_results: list[bool]) -> None:
        assert len(test_results) == len(test_labels)
        self.accuracy = len(
            [i for i in range(len(test_labels)) if test_labels[i] == test_results[i]]
        )


class Model:
    def train(self, dataset: torch.Dataset):
        pass

    def test(self, dataset: torch.Dataset) -> TestResults:
        pass
