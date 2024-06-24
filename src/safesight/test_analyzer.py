import signal
from pathlib import Path

from safesight.analyzer import Analyzer
from safesight.cli import cli
from safesight.file_camera import FileCamera
from safesight.net_tester import TRANSFORM_OF_SIZE
from safesight.our_model import ModelSettings
from safesight.our_model_pipeline import OurModelPipeline


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
    analyzer.start_analyzer(camera, 30, memory_size=1 << 30)  # 1 GB of shared memory
    signal.signal(signal.SIGINT, lambda _, __: analyzer.stop_analysis())
    signal.pause()


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
