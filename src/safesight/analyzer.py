import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from typing import List

import numpy as np

from safesight.camera import Camera
from safesight.pipeline import Evaluation, Pipeline


@dataclass
class PipelineProcess:
    pipeline: Pipeline
    image_queue: mp.Queue
    evaluation_queue: mp.Queue
    process: mp.Process


class Analyzer:
    """
    The final product, that runs on a Camera and checks for accidents.
    """

    def __init__(self) -> None:
        self.pipelines: List[PipelineProcess] = []

    def run_analyzer(self, camera: Camera, evaluations_per_second: int):
        """
        Runs the analyzer on a given Camera. Calls `evaluation_callback` when an
        evaluation is reached.

        To run on a file, see safesight.file_camera
        """
        frame_interval = 1 / float(evaluations_per_second)
        frames_so_far = 0
        while True:
            time_started = time.time()
            image = camera.get_image()
            image_array = np.array(image)

            for pipeline in self.pipelines:
                # print(f"Put {image} in pipelines")
                pipeline.image_queue.put(image_array)

            for pipeline in self.pipelines:
                while True:
                    try:
                        evaluation = pipeline.evaluation_queue.get(timeout=0.01)
                    except queue.Empty:
                        break

                    assert type(evaluation) == Evaluation

                    if not evaluation.result:
                        continue
                    """
                    TODO: we want a view like they have - https://github.com/dolongbien/HumanBehaviorBKU
                    For this, I think it will be useful to be able to dump the analyzer output in a CSV fashion (we don't need a fancy viewer)
                    """
                    print(f"From {pipeline.pipeline}: {evaluation}")

            frames_so_far += 1
            # TODO: change to evaluations_per_second or something like that
            if frames_so_far % 100 == 0:
                print(f"Frame #{frames_so_far}")
            time_now = time.time()
            time_to_sleep = frame_interval - (time_now - time_started)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

    def add_pipeline(self, pipeline: Pipeline):
        """
        Add a pipeline to the Analyzer and start running it.
        """
        image_queue = mp.Queue()
        evaluation_queue = mp.Queue()
        process = mp.Process(
            target=pipeline.run_pipeline, args=(image_queue, evaluation_queue)
        )
        self.pipelines.append(
            PipelineProcess(pipeline, image_queue, evaluation_queue, process)
        )
        process.start()

    def stop_analysis(self) -> None:
        # TODO: implement this, although it's not very important
        pass
