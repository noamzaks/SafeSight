from PIL.Image import Image
from ultralytics import YOLO

from safesight.pipeline import Pipeline, Evaluation

class YOLOPipeline(Pipeline):
    def __init__(self, model_path: str, threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.threshold = threshold

    def process_image(self, image: Image) -> Evaluation:
        results = self.model(image, verbose=False)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > self.threshold and results.names[int(class_id)].lower() == "accident":
                return Evaluation(True)
        return Evaluation(False)

    def prepare(self):
        pass

    def cleanup(self):
        pass