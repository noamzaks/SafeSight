from dataclasses import dataclass
import csv
from pathlib import Path
import pathlib
import sys
import time
from typing import Any, Dict, List, Tuple, Type

from PIL.Image import Image
import click
import cv2

from safesight.analyzer import Analyzer
from safesight.blip_pipeline import BlipPipeline
from safesight.cli import cli
from safesight.net_tester import TRANSFORM_OF_SIZE
from safesight.our_model import ModelSettings
from safesight.single_pipeline_analyzer import SinglePipelineAnalyzer
from safesight.file_camera import FileCamera
from safesight.pipeline import Evaluation
from safesight.our_model_pipeline import OurModelPipeline

DEBUG = True


@cli.group()
def analyzer():
    """Commands to run the finished Analyzers. (currently python==3.8.*)"""
    pass


@dataclass
class AnalyzerTestResults:
    positives: int
    negatives: int
    average_time_taken: float


# @analyzer.command()
# @click.argument("file", type=click.Path(exists=True, dir_okay=False, file_okay=True))
# @click.option("-s", "--sampling_step", default=100)
# @click.option("-l", "--frame_limit", default=100)
# @click.option("-q", "--quiet", default=False)
def test_model_analyzer(
    file: str,
    sampling_step: int = 100,
    frame_limit: int = 100,
    quiet: bool = False,
    analyzer_type: Type[Analyzer] = SinglePipelineAnalyzer,
    analyzer_args: List[Any] = [
        BlipPipeline,
        "Is there a road accident in this video?",
    ],
) -> AnalyzerTestResults:
    """
    Test the SinglePipelineAnalyzer.
    """
    analyzer = analyzer_type(*analyzer_args)
    file_camera = FileCamera(Path(file))
    if not quiet:
        print(f"Video name: {file}")
        print(f"Sampling with step of {sampling_step}")
        print(f"#Frames in video: {file_camera.video.get(cv2.CAP_PROP_FRAME_COUNT)}")
        print(
            f"<Width, Height> of video: <{file_camera.video.get(cv2.CAP_PROP_FRAME_WIDTH)},"
            f"{file_camera.video.get(cv2.CAP_PROP_FRAME_HEIGHT)}>"
        )

    frame_number = 0
    results: list[Evaluation] = []
    times: list[float] = []

    last_time = time.time()

    temp_frames_dir = Path("./temp_frames/")
    temp_frames_dir.mkdir(parents=True, exist_ok=True)

    def process_result(image: Image, evaluation: Evaluation):
        nonlocal frame_number, last_time
        current_time = time.time()
        time_taken = current_time - last_time

        output_line = f"Frame #{frame_number}, evaluation: {evaluation}, time taken: {time_taken:.3f}."
        if not quiet:
            print(output_line)

        if DEBUG:
            print(output_line, file=sys.stderr)  # Feedback for running into file.
            image.save(str(temp_frames_dir / f"{frame_number}.jpg"))

        frame_number += 1
        results.append(evaluation)
        times.append(time_taken)
        last_time = time.time()
        if frame_number > frame_limit:
            analyzer.stop_analysis()

    analyzer.run_analyzer(file_camera, process_result, sampling_step)

    positives = int(len([e for e in results if e.result]))
    negatives = int(frame_number - positives)
    if not quiet:
        print(
            f"""
                Overall results:
                Model answered True {positives}/{frame_number} times, {100.0 * (positives / frame_number):.2f}%
                Model answered False {negatives}/{frame_number} times, {100.0 * (negatives / frame_number):.2f}%
                Average time taken per frame: {sum(times) / len(times):.3f}
                """
        )
    return AnalyzerTestResults(positives, negatives, sum(times) / len(times))


analyzer_dict: Dict[str, Tuple[Type[Analyzer], List]] = {
    "Blip": (
        SinglePipelineAnalyzer,
        [BlipPipeline, "Is there a road accident in this video?"],
    )
}


@analyzer.command()
@click.argument(
    "test_file", type=click.Path(exists=True, dir_okay=False, file_okay=True)
)
@click.option("-q", "--quiet", default=False)
def run_tests(test_file: str, quiet: bool = False):
    """
    Run tests on the Analyzers, according to the `test_file`, which should be a JSON file in the
    following format (each line constitues a test):

    Analyzer Profile,File ,Sampling Step,Frame Limit

    Example:
    Blip            ,a.mp4,100          ,100

    ...

    Analyzer Type should be one of the following:
        * Blip

    Outputs its results in CSV format to standard output.
    """

    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "Analyzer Profile",
            "File",
            "Sampling Step",
            "Frame Limit",
            "Positives",
            "Negatives",
            "Frames",
            "Positive %",
            "Negative %",
            "Avg. Time Taken",
        ]
    )
    with open(test_file, newline="") as csvfile:
        reader = csv.reader(csvfile)
        for test in reader:
            analyzer_profile, file, sampling_step, frame_limit = map(str.strip, test)
            analyzer_type, analyzer_args = analyzer_dict[analyzer_profile]
            if not quiet:
                print(f"Running {analyzer_profile} on {file}...", file=sys.stderr)
            result = test_model_analyzer(
                file,
                int(sampling_step),
                int(frame_limit),
                True,
                analyzer_type,
                analyzer_args,
            )

            frame_number = result.positives + result.negatives
            writer.writerow(
                [
                    analyzer_profile,
                    file,
                    sampling_step,
                    frame_limit,
                    result.positives,
                    result.negatives,
                    f"{100.0 * (result.positives / frame_number):.2f}",
                    f"{100.0 * (result.negatives / frame_number):.2f}",
                    f"{result.average_time_taken:.3f}",
                ]
            )


def test_model_analyzer(
    video_path: Path, model_path: Path, model_settings: ModelSettings
):
    analyzer = Analyzer()
    camera = FileCamera(video_path)
    pipeline = OurModelPipeline(model_path, model_settings)
    analyzer.add_pipeline(pipeline)
    analyzer.run_analyzer(camera, 30)


if __name__ == "__main__":
    video_path = Path("data/videos/5pGjkv5lZXE.mp4")
    model_path = Path("models/model0.pth")
    settings = ModelSettings(
        internal_layer_size=6,
        epochs=3,
        learning_rate=0.001,
        momentum=0.9,
        transform=TRANSFORM_OF_SIZE(500),
    )

    test_model_analyzer(video_path, model_path, settings)
