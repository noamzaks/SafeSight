from pathlib import Path
from typing import Optional
from safesight.pipeline import Pipeline, Evaluation
from safesight.our_model import Net, ModelSettings, evaluate_image
from PIL.Image import Image
import torch


class OurModelPipeline(Pipeline):
    def __init__(
        self, model_path: Path, model_settings: ModelSettings, transform
    ) -> None:
        self.net = Net(model_settings)
        self.net.load_state_dict(torch.load(str(model_path)))
        self.net.train(False)
        self.transform = transform

    def process_image(self, image: Image) -> Optional[Evaluation]:
        label = evaluate_image(self.net, image, self.transform)
        if label == "accident":
            self.last_evaluation = Evaluation(True)
        else:
            self.last_evaluation = Evaluation(False)

        return self.last_evaluation

    def evaluate(self) -> Evaluation:
        return self.last_evaluation
