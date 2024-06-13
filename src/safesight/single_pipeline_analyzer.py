from typing import Callable, Type

from PIL.Image import Image

from safesight.pipeline import Pipeline
from safesight.analyzer import Analyzer
from safesight.camera import Camera
from safesight.pipeline import Evaluation

SAVE_TEMP_FRAMES = True


class SinglePipelineAnalyzer(Analyzer):
    def __init__(
        self, pipeline: Type[Pipeline], *pipeline_args, **pipeline_kwargs
    ) -> None:
        self.pipeline = pipeline(*pipeline_args, **pipeline_kwargs)
        self.stopped = False

    def run_analyzer(
        self,
        camera: Camera,
        evaluation_callback: Callable[[Image, Evaluation], None],
        sampling_step: int = 1,
    ):
        frame = camera.get_image()
        while frame:
            self.pipeline.process_image(frame)
            evaluation_callback(frame, self.pipeline.evaluate())
            if self.stopped:
                return
            for _ in range(sampling_step - 1):
                if not camera.get_image():
                    return
            frame = camera.get_image()

    def stop_analysis(self):
        self.stopped = True
