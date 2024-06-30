from datetime import datetime
from pathlib import Path
from sys import stderr

import torch
from PIL.Image import Image

from safesight.custom_model import Net
from safesight.pipeline import Evaluation, Pipeline


class CustomModelPipeline(Pipeline):
    """
    A pipeline that runs a model we trained, stored on disk as a `.pth` file.
    """

    def prepare(self):
        pass

    def cleanup(self):
        pass

    def __init__(self, model_path: Path) -> None:
        self.net = None
        self.model_path = model_path
        # load_state_dict also loads the ModelSettings, which are saved to disk.
        # self.net.load_state_dict(torch.load(str(model_path)))
        # self.net.train(False)

    def process_image(self, image: Image) -> Evaluation:
        if self.net is None:
            self.net = Net()
            self.net.load_state_dict(torch.load(str(self.model_path)))
            self.net.train(False)
        label = self.net.evaluate_image(image.convert("RGB"))
        if label == "accident":
            evaluation = Evaluation(True, timestamp=datetime.now())
        else:
            evaluation = Evaluation(False, timestamp=datetime.now())
        return evaluation
