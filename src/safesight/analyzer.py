from typing import Callable
from blip_pipeline import BlipPipeline
from camera import Camera
from pathlib import Path


class Analyzer:
    def __init__(self) -> None:
        self.blip_pipeline = BlipPipeline()

    def run_analyzer(self, camera: Camera, evaluation_callback: Callable):
        pass

    def run_analyzer_video(self, video_file: Path, evaluation_callback: Callable):
        pass
