from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Queue
from sys import stderr

import PIL.Image
from PIL.Image import Image


@dataclass
class Evaluation:
    result: bool
    # Raw answer, when available from LLM.
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
            # print(f"Got {image_array} in pipeline", file=stderr)
            image = PIL.Image.fromarray(image_array)
            # print(f"Got {image} in pipeline", file=stderr)
            evaluation = self.process_image(image)
            evaluation_queue.put(evaluation)
            # print(f"Returned {evaluation} from pipeline", file=stderr)
