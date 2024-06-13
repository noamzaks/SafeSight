from pathlib import Path
import sys
import time

from PIL.Image import Image
import cv2

from safesight.blip_pipeline import BlipPipeline
from safesight.single_pipeline_analyzer import SinglePipelineAnalyzer
from safesight.file_camera import FileCamera
from safesight.pipeline import Evaluation

DEBUG = True


def test_analyzer(file: str, sampling_step: int = 100, frame_limit: int = 100):
    analyzer = SinglePipelineAnalyzer(
        BlipPipeline, "Is there a road accident in this video?"
    )
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
    times: list[float] = []

    last_time = time.time()

    def process_result(image: Image, evaluation: Evaluation):
        nonlocal frame_number, last_time
        current_time = time.time()
        time_taken = current_time - last_time

        output_line = f"Frame #{frame_number}, evaluation: {evaluation}, time taken: {time_taken:.3f}."
        print(output_line)

        if DEBUG:
            print(output_line, file=sys.stderr)  # Feedback for running into file.
            image.save(f"./temp_images/{frame_number}.jpg")

        frame_number += 1
        results.append(evaluation)
        times.append(time_taken)
        last_time = time.time()
        if frame_number > frame_limit:
            analyzer.stop_analysis()

    analyzer.run_analyzer(file_camera, process_result, sampling_step)

    positives = int(len([e for e in results if e.result]))
    negatives = int(frame_number - positives)
    print(
        f"""
            Overall results:
            Model answered True {positives}/{frame_number} times, {100.0 * (positives / frame_number):.2f}%
            Model answered False {negatives}/{frame_number} times, {100.0 * (negatives / frame_number):.2f}%
            Average time taken per frame: {sum(times) / len(times):.3f}
            """
    )


if __name__ == "__main__":
    test_analyzer("data/videos/9DRFJxKHc6g.mp4", 100)
