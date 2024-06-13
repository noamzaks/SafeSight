from typing import Callable
from safesight.camera import Camera
from safesight.pipeline import Evaluation
from PIL.Image import Image


class Analyzer:
    def __init__(self) -> None:
        pass

    def run_analyzer(
        self,
        camera: Camera,
        evaluation_callback: Callable[[Image, Evaluation], None],
        sampling_step: int = 1,
    ):
        """
        Runs the analyzer on a given Camera. Calls `evaluation_callback` when an
        evaluation is reached.

        To run on a file, see safesight.file_camera
        """
        pass

    def stop_analysis(self) -> None:
        pass
