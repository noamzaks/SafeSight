from datetime import datetime
from pathlib import Path

import torch
from PIL.Image import Image

from safesight.our_model import ModelSettings, Net
from safesight.pipeline import Evaluation, Pipeline


class OurModelPipeline(Pipeline):
    """
    A pipeline that runs a model we trained, stored on disk as a `.pth` file.
    """

    def prepare(self):
        pass

    def cleanup(self):
        pass

    def __init__(self, model_path: Path, model_settings: ModelSettings) -> None:
        self.net = Net(model_settings)
        self.net.load_state_dict(torch.load(str(model_path)))
        self.net.train(False)

    def process_image(self, image: Image) -> Evaluation:
        # print("Got image in process_image", file=stderr)
        label = self.net.evaluate_image(image)
        if label == "accident":
            evaluation = Evaluation(True, timestamp=datetime.now())
        else:
            evaluation = Evaluation(False, timestamp=datetime.now())

        return evaluation
