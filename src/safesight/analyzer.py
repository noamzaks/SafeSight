from pathlib import Path
from typing import Callable

from PIL.Image import Image

from safesight.blip_pipeline import BlipPipeline
from safesight.camera import Camera
from safesight.pipeline import Evaluation


class Analyzer:
    def __init__(self, question="Is there an accident in this photo?") -> None:
        self.blip_pipeline = BlipPipeline(question)

    def run_analyzer(
        self,
        camera: Camera,
        evaluation_callback: Callable[[Image, Evaluation], None],
        sampling_step: int = 1,
    ):
        frame = camera.get_image()
        while frame:
            self.blip_pipeline.process_image(frame)
            evaluation_callback(frame, self.blip_pipeline.evaluate())
            for _ in range(sampling_step - 1):
                if not camera.get_image():
                    return
            frame = camera.get_image()
