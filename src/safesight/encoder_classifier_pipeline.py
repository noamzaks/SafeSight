from pathlib import Path
from datetime import datetime
from typing import Callable, List
import torch
from PIL.Image import Image
from safesight.pipeline import Pipeline, Evaluation
from safesight.encoder import EncoderClassifier, CLIPEncoder
import PIL.Image

class EncoderClassifierPipeline(Pipeline):
    def __init__(self,  encoder: Callable[[Image], torch.Tensor], layer_sizes: List[int], model_path: Path) -> None:
        super().__init__()
        self.model = EncoderClassifier(encoder, layer_sizes)
        self.model.load_head(model_path)
    
    def prepare(self):
        pass

    def cleanup(self):
        pass

    def process_image(self, image: Image) -> Evaluation:
        result = self.model.evaluate_image(image)
        evaluation = Evaluation(result=result=="accident", timestamp=datetime.now())
        return evaluation

if __name__ == "__main__":
    MODEL_HEAD_PATH = Path("encoder_models/Hidden200-200-B32-lr0.01-mom0.9-epochs20-zaksaset.pth")
    MODEL_LAYERS = [200, 200]
    ENCODER = CLIPEncoder("ViT-L/14@336px")
    pipeline = EncoderClassifierPipeline(ENCODER.encode_image, MODEL_LAYERS, MODEL_HEAD_PATH)
    print(pipeline.process_image(PIL.Image.open("zaksaset/test/accident/1_3.png")))
    print(pipeline.process_image(PIL.Image.open("zaksaset/test/nonaccident/5_21.png")))
    print(pipeline.process_image(PIL.Image.open("zaksaset/test/nonaccident/1_6.png")))