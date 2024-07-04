import multiprocessing as mp
import multiprocessing.connection
import random
import struct
import time
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from sys import stderr
from typing import List, Optional, Dict

from safesight.file_camera import FileCamera
from safesight.pipeline import Pipeline

UINT_BITMASK = 0xffffffff


class MemoryControl(Enum):
    """
    STOP = 0
    CONTINUE = 1
    Control bytes in the shared memory.
    """

    # Frame terminators:
    FRAME_END = 0xff & -1
    FRAME_NOT_READY = 0xff & -2

    # Control bytes:
    WAIT = UINT_BITMASK & -1
    RESET_INDEX = UINT_BITMASK & -2
    CLOSE = UINT_BITMASK & -3


class Analyzer:
    """
    The final product, that runs on a Camera and checks for accidents.
    """

    @dataclass
    class _PipelineData:
        name: str
        pipeline: Pipeline
        autorun: bool
        process: Optional[Process]

    pipelines: List[_PipelineData]
    running: bool
    stopping: bool
    memory: Optional[SharedMemory]
    camera_proc: Optional[Process]
    autorun_proc: Optional[Process]
    results_proc: Optional[Process]

    def __init__(self) -> None:
        self.pipelines = []
        self.running = False
        self.stopping = False
        self.memory = None
        self.camera_proc = None
        self.results_proc = None

    def add_pipeline(self, name: str, pipeline: Pipeline, autorun: bool = True) -> None:
        """
        Add a pipeline to the Analyzer.
        """
        self.pipelines.append(Analyzer._PipelineData(name, pipeline, autorun, None))

    def start_analyzer(self, frames_pes_second: int, memory_size: int) -> bool:
        """
        Starts each pipeline in a separate process and then the camera. To stop, call stop_analysis.

        @param frames_pes_second Frames per second to evaluate
        @param memory_size Size of the shared memory in bytes (has to fit at least 2 frames)
        """

        if self.running:
            print("Tried to start the analyzer while it was already running.", file=stderr)
            return False

        init_success = False
        try:
            self.running = True

            random_salt = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=8))
            shared_memory_name = f"safesight_analyzer-{random_salt}"

            self.memory = SharedMemory(create=True, size=memory_size, name=shared_memory_name)
            self.memory.buf[0:4] = struct.pack(">I", MemoryControl.WAIT.value)

            evaluation_queues = {pipeline.name: Queue() for pipeline in self.pipelines}
            camera_to_autorun_queue = Queue()
            camera_to_eval_queue = Queue()
            process_index_queues = {p.name: Queue() for p in self.pipelines}

            self.results_proc = Process(name='eval', target=self._eval_process,
                                        args=(evaluation_queues, camera_to_eval_queue, process_index_queues))
            self.results_proc.start()

            for pipeline in self.pipelines:
                p = Process(name=f'pipeline-{pipeline.name}', target=pipeline.pipeline.run_pipeline,
                            kwargs={"shared_memory_name": shared_memory_name,
                                    "evaluation_queue": evaluation_queues[pipeline.name],
                                    "index_queue": process_index_queues[pipeline.name]})
                pipeline.process = p
                p.start()

            self.autorun_proc = Process(name='autorun', target=self._adaptive_rate_process,
                                        args=(camera_to_autorun_queue, process_index_queues))
            self.autorun_proc.start()

            self.camera_proc = Process(name='camera', target=self._camera_process,
                                       args=(frames_pes_second,),
                                       kwargs={"shared_memory_name": shared_memory_name,
                                               "index_queues": [camera_to_eval_queue, camera_to_autorun_queue]})
            self.camera_proc.start()

            init_success = True
        finally:
            if not init_success:
                self.stop_analysis()

                return init_success

    @staticmethod
    def _camera_process(frames_per_second: int, *, shared_memory_name: str,
                        index_queues: List[Queue]):
        camera = FileCamera(Path("../../data/videos/test.mp4"))
        print(f"[{mp.current_process().pid}] Starting camera process.", file=stderr)
        mem = None
        index = 0
        print_step = round(frames_per_second * 5 / 100) * 100
        try:
            mem = SharedMemory(name=shared_memory_name)
            buff = mem.buf

            frame_num = 0
            buff[index:index + 4] = struct.pack(">I", MemoryControl.WAIT.value)

            last_time = 0

            while True:
                t = time.time()
                if t - last_time < 1 / frames_per_second:
                    time.sleep(max(0.0, 1 / frames_per_second - (t - last_time)))
                    continue

                img = camera.get_image()
                if img is None:
                    break

                last_time = t
                frame_num += 1
                if frame_num % print_step == 0:
                    print(f"[{mp.current_process().pid}] CAMERA: Frame {frame_num}.", file=stderr)

                size = img.size
                frame_len = size[0] * size[1] * 4

                if frame_len + 8 > len(buff) // 2:
                    print(f"Dropping frame {frame_num}, not enough memory", file=stderr)
                    continue
                if index + frame_len + 8 + 2 >= len(buff):
                    buff[0:4] = struct.pack(">I", MemoryControl.WAIT.value)
                    buff[index:index + 4] = struct.pack(">I", MemoryControl.RESET_INDEX.value)
                    index = 0

                for q in index_queues:
                    q.put((frame_num, index))

                buff[index + 8 + frame_len] = MemoryControl.FRAME_NOT_READY.value
                buff[index + 4:index + 8] = struct.pack(">HH", *size)
                buff[index:index + 4] = struct.pack(">I", frame_num)

                buff[index + 8:index + 8 + frame_len] = img.convert("RGBA").tobytes()
                buff[index + 8 + frame_len + 1:index + 8 + frame_len + 1 + 4] = struct.pack(">I",
                                                                                            MemoryControl.WAIT.value)
                buff[index + 8 + frame_len] = MemoryControl.FRAME_END.value

                index += frame_len + 8 + 1

        finally:
            print("Camera process closing", file=stderr)
            if mem:
                mem.buf[index:index + 4] = struct.pack(">I", MemoryControl.CLOSE.value)
                # mem.close()
            for q in index_queues:
                q.put((None, index))

    @staticmethod
    def _adaptive_rate_process(index_queue: Queue, pipeline_index_queues: Dict[str, Queue]):
        while True:
            frame, index = index_queue.get()
            if frame is None:
                break

            # TODO: Adaptive rate
            for _, q in pipeline_index_queues.items():
                q.put(index)

    @staticmethod
    def _eval_process(evaluation_queues: Dict[str, Queue], index_queue: Queue,
                      pipeline_index_queues: Dict[str, Queue]):
        frames = dict()
        while len(evaluation_queues) > 0:
            if not index_queue.empty():
                frame, index = index_queue.get()
                if frame is None:
                    break
                frames[frame] = index

            for pipeline, q in evaluation_queues.copy().items():
                item = q.get()
                if item is None:
                    # q.close()
                    evaluation_queues.pop(pipeline)
                    continue
                frame_num, evaluation = item
                # print(f"{pipeline},{frame_num},{evaluation.result}")
                print(f"{pipeline} evaluated frame {frame_num}, result: {evaluation.result} ({evaluation.raw_answer})",
                      file=stderr)

    def stop_analysis(self) -> None:
        """
        Tries to stop everything gracefully, but if it doesn't, just terminates all the processes.
        """
        if not self.running:
            print("Tried to stop the analyzer while it was not running.", file=stderr)
            return
        if self.stopping:
            print("Tried to stop the analyzer while it was already stopping.", file=stderr)
            return

        self.stopping = True

        if self.camera_proc:
            self.camera_proc.kill()

        if self.autorun_proc:
            self.autorun_proc.kill()

        if self.results_proc:
            self.results_proc.kill()

        # Give the system time to stop gracefully
        multiprocessing.connection.wait([p.sentinel for p in mp.active_children()], timeout=5)

        for p in [self.camera_proc, self.results_proc] + [p.process for p in self.pipelines]:
            if p.is_alive():
                p.terminate()

        if self.memory:
            self.memory.close()
            self.memory.unlink()

        self.camera_proc = None
        self.autorun_proc = None
        self.results_proc = None
        for p in self.pipelines:
            p.process = None
        self.memory = None
        self.running = False
        self.stopping = False
