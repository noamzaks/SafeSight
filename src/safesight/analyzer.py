import multiprocessing as mp
import queue
import time
from sys import stderr
from typing import List

import numpy

from safesight.camera import Camera
from safesight.pipeline import Evaluation, Pipeline


class Analyzer:
    """
    The final product, that runs on a Camera and checks for accidents.
    """
    pipelines: List[Pipeline]

    # image_queue: mp.Queue
    # evaluation_queue: mp.Queue

    def __init__(self) -> None:
        self.pipelines = []

    def run_analyzer(self, camera: Camera, evaluations_per_second: int, *, num_processes: int = mp.cpu_count()):
        """
        Runs the analyzer on a given Camera.
        Analyses in all pipelines, uses up to *num_processes* processes.

        To run on a file, see safesight.file_camera
        """
        print(f"Running analyzer with {num_processes} processes and {len(self.pipelines)} pipelines", file=stderr)
        with mp.Pool(num_processes) as pool:
            frame_interval = 1 / float(evaluations_per_second)
            frames_so_far = 0
            while True:
                time_started = time.time()
                image = camera.get_image()
                image_array = numpy.array(image)

                # for pipeline in self.pipelines:
                #     # print(f"Put {image} in pipelines")
                #     pipeline.image_queue.put(image_array)

                # for pipeline in self.pipelines:
                #     while True:
                #         try:
                #             evaluation = pipeline.evaluation_queue.get(timeout=0.01)
                #         except queue.Empty:
                #             # print("Nothing in queue")
                #             break
                #
                #         assert type(evaluation) == Evaluation
                #
                #         if evaluation.result:
                #             pass
                #         print(f"From {pipeline.pipeline}: {evaluation}")

                jobs = []
                for pipeline in self.pipelines:
                    jobs.append(pool.apply_async(pipeline.process_image, args=(image_array,),
                                                 callback=lambda evaluation: print(f"From {pipeline}: {evaluation}"), error_callback=lambda e: print(e, file=f"Error: {stderr}")))

                [job.wait() for job in jobs] # todo: uncomment if want for all pipelines to finish before next frame
                frames_so_far += 1
                if frames_so_far % 100 == 0:
                    print(f"Frame #{frames_so_far}")
                time_now = time.time()
                time_to_sleep = frame_interval - (time_now - time_started)
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)

    def add_pipeline(self, pipeline: Pipeline):
        """
        Add a pipeline to the Analyzer.
        """
        self.pipelines.append(pipeline)

    def stop_analysis(self) -> None:
        pass
