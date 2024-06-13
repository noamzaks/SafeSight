from typing import Callable
from safesight.blip_pipeline import BlipPipeline
from safesight.camera import Camera
from pathlib import Path


class Analyzer:
    def __init__(self) -> None:
        pass

    def run_analyzer(self, camera: Camera, evaluation_callback: Callable):
        pass

    def run_analyzer_video(self, video_file: Path, evaluation_callback: Callable):
        pass
