from pathlib import Path
from typing import Optional
from safesight.pipeline import Pipeline, Evaluation
from safesight.our_model import Net, ModelSettings
from PIL.Image import Image
import torch


class OurModelPipeline(Pipeline):
    """
    A pipeline that runs a model we trained, stored on disk as a `.pth` file.
    """

    def __init__(self, model_path: Path, model_settings: ModelSettings) -> None:
        self.net = Net(model_settings)
        self.net.load_state_dict(torch.load(str(model_path)))
        self.net.train(False)

    def process_image(self, image: Image) -> Optional[Evaluation]:
        label = self.net.evaluate_image(image)
        if label == "accident":
            self.last_evaluation = Evaluation(True)
        else:
            self.last_evaluation = Evaluation(False)

        return self.last_evaluation

    def evaluate(self) -> Evaluation:
        return self.last_evaluation
