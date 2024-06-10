import sys
from itertools import count
from pathlib import Path
from PIL.Image import Image
import cv2
from safesight.analyzer import Analyzer
from safesight.file_camera import FileCamera
from safesight.pipeline import Evaluation

DEBUG = True


def test_analyzer(file: str, sampling_step: int = 100):
    analyzer = Analyzer()
    file_camera = FileCamera(Path(file))
    print(f"Video name: {file}")
    print(f"#Frames in video: {file_camera.video.get(cv2.CAP_PROP_FRAME_COUNT)}")
    print(
        f"<Width, Height> of video: <{file_camera.video.get(cv2.CAP_PROP_FRAME_WIDTH)},"
        f"{file_camera.video.get(cv2.CAP_PROP_FRAME_HEIGHT)}>"
    )
    print(f"Sampling with step of {sampling_step}")

    frame_number = 0
    results: list[Evaluation] = []

    def process_result(image: Image, evaluation: Evaluation):
        nonlocal frame_number
        print(f"Frame #{frame_number}, evaluation: {evaluation.result}.")
        if DEBUG:
            print(
                f"Frame #{frame_number}, evaluation: {evaluation.result}.",
                file=sys.stderr,
            )  # Feedback for running into file.
        frame_number += 1
        results.append(evaluation)

    analyzer.run_analyzer(file_camera, process_result, sampling_step)

    positives = int(len([e for e in results if e.result]))
    negatives = int(frame_number - positives)
    print(
        f"""
            Overall results:
            Model answered True {positives}/{frame_number} times, {100.0 * (positives / frame_number)}%
            Model answered False {negatives}/{frame_number} times, {100.0 * (negatives / frame_number)}%
            """
    )


if __name__ == "__main__":
    test_analyzer("data/videos/5pGjkv5lZXE.mp4", 200)
