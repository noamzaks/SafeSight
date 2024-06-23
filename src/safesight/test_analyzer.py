import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

import click
import cv2
from PIL.Image import Image

from safesight.analyzer import Analyzer
from safesight.cli import cli
from safesight.file_camera import FileCamera
from safesight.net_tester import TRANSFORM_OF_SIZE
from safesight.our_model import ModelSettings
from safesight.our_model_pipeline import OurModelPipeline
from safesight.pipeline import Evaluation
from safesight.single_pipeline_analyzer import SinglePipelineAnalyzer


@cli.group()
def analyzer():
    """Commands to run the finished Analyzers. (currently python==3.8.*)"""
    pass


def test_model_analyzer(
    video_path: Path, model_path: Path, model_settings: ModelSettings
):
    analyzer = Analyzer()
    camera = FileCamera(video_path)
    pipeline = OurModelPipeline(model_path, model_settings)
    analyzer.add_pipeline(pipeline)
    analyzer.run_analyzer(camera, 30)


if __name__ == "__main__":
    video_path = Path("youtube_dataset/test.mp4")
    model_path = Path("models/model0.pth")
    settings = ModelSettings(
        internal_layer_size=6,
        epochs=3,
        learning_rate=0.001,
        momentum=0.9,
        transform=TRANSFORM_OF_SIZE(500),
    )

    test_model_analyzer(video_path, model_path, settings)
