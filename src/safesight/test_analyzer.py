import signal
from pathlib import Path

from safesight.analyzer import Analyzer
from safesight.cli import cli
from safesight.custom_model_pipeline import CustomModelPipeline
from safesight.gemini_pipeline import GeminiPipeline


@cli.group()
def analyzer():
    """Commands to run the finished Analyzers. (currently python==3.8.*)"""
    pass


# @analyzer.command()
# @click.option(
#     "--video-path", type=click.Path(exists=True, dir_okay=False, file_okay=True)
# )
# @click.option(
#     "--model-path", type=click.Path(exists=True, dir_okay=False, file_okay=True)
# )
def run_analyzer(model_path: Path):
    analyzer = Analyzer()
    model_pipeline = CustomModelPipeline(model_path)
    gemini_pipeline = GeminiPipeline()
    analyzer.add_pipeline("custom_model", model_pipeline, True)
    analyzer.add_pipeline("gemini", gemini_pipeline, False)
    analyzer.start_analyzer(30, memory_size=1 << 30)  # 1 GB of shared memory
    signal.signal(signal.SIGINT, lambda _, __: analyzer.stop_analysis())
    signal.pause()


if __name__ == "__main__":
    video_path = Path("long_videos/10_secs.mp4")
    model_path = Path("models/model2.pth")
    run_analyzer(model_path)
