from typing import Optional
from PIL.Image import Image
from dataclasses import dataclass


@dataclass
class Evaluation:
    result: bool


class Pipeline:
    def process_image(self, image: Image) -> Optional[Evaluation]:
        """
        Processes an image, potentially modifying internal state, potentially
        returning an evaluation.
        """
        pass

    def evaluate(self) -> Evaluation:
        """
        Evaluates the current internal state.
        """
        pass
