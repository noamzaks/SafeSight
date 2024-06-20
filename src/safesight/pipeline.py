from sys import stderr
from typing import Optional
from PIL.Image import Image
import PIL.Image
from dataclasses import dataclass
from multiprocessing import Queue
from datetime import datetime
import numpy as np


@dataclass
class Evaluation:
    result: bool
    raw_answer: str = ""
    timestamp: datetime = datetime(2000, 1, 1)


class Pipeline:
    def process_image(self, image: Image) -> Evaluation:
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

    def run_pipeline(self, image_queue: Queue, evaluation_queue: Queue):
        print("Pipeline started", file=stderr)
        while True:
            image_array = image_queue.get()
            print(f"Got {image_array} in pipeline", file=stderr)
            image = PIL.Image.fromarray(image_array)
            print(f"Got {image} in pipeline", file=stderr)
            evaluation = self.process_image(image)
            evaluation_queue.put(evaluation)
            print(f"Returned {evaluation} from pipeline", file=stderr)
