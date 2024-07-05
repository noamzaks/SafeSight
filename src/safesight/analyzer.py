import multiprocessing as mp
import multiprocessing.connection
import queue
import random
import signal
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

KILL_TIMEOUT = 5


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


class _ProcessSigIntIgnore(mp.Process):
    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        super().run()


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
            print(f"[{mp.current_process().pid}] Starting analyzer.", file=stderr)

            random_salt = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=8))
            shared_memory_name = f"safesight_analyzer-{random_salt}"

            self.memory = SharedMemory(create=True, size=memory_size, name=shared_memory_name)
            self.memory.buf[0:4] = struct.pack(">I", MemoryControl.WAIT.value)

            evaluation_queues = {pipeline.name: Queue() for pipeline in self.pipelines}
            camera_to_autorun_queue = Queue()
            camera_to_eval_queue = Queue()
            process_index_queues = {p.name: Queue() for p in self.pipelines}

            self.results_proc = _ProcessSigIntIgnore(name='eval', target=self._eval_process,
                                                     args=(
                                                         evaluation_queues, camera_to_eval_queue, process_index_queues))
            self.results_proc.start()

            for pipeline in self.pipelines:
                p = _ProcessSigIntIgnore(name=f'pipeline-{pipeline.name}', target=pipeline.pipeline.run_pipeline,
                                         kwargs={"shared_memory_name": shared_memory_name,
                                                 "evaluation_queue": evaluation_queues[pipeline.name],
                                                 "index_queue": process_index_queues[pipeline.name]})
                pipeline.process = p
                p.start()

            self.autorun_proc = _ProcessSigIntIgnore(name='autorun', target=self._adaptive_rate_process,
                                                     args=(camera_to_autorun_queue,
                                                           {p.name: process_index_queues[p.name] for p in
                                                            self.pipelines if p.autorun}))
            self.autorun_proc.start()

            self.camera_proc = _ProcessSigIntIgnore(name='camera', target=self._camera_process,
                                                    args=(frames_pes_second,),
                                                    kwargs={"shared_memory_name": shared_memory_name,
                                                            "index_queues": [camera_to_eval_queue,
                                                                             camera_to_autorun_queue]})
            self.camera_proc.start()

            init_success = True
        finally:
            if not init_success:
                self.stop_analysis()

                return init_success

    @staticmethod
    def _camera_process(frames_per_second: int, *, shared_memory_name: str,
                        index_queues: List[Queue]):

        def terminate():
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, lambda _, __: terminate())

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
            print(f"[{mp.current_process().pid}] Camera process exiting.", file=stderr)
            if mem:
                mem.buf[index:index + 4] = struct.pack(">I", MemoryControl.CLOSE.value)
                # mem.close()
            for i_queue in index_queues:
                i_queue.put((None, index))
                i_queue.close()
            return

    @staticmethod
    def _adaptive_rate_process(index_queue: Queue, pipeline_index_queues: Dict[str, Queue]):
        print(f"[{mp.current_process().pid}] Starting autorun process.", file=stderr)
        while True:
            frame, index = index_queue.get()
            if frame is None:
                break

            # TODO: Adaptive rate
            for _, q in pipeline_index_queues.items():
                q.put(index)

        for q in pipeline_index_queues.values():
            q.close()
        print(f"[{mp.current_process().pid}] Autorun process exiting.", file=stderr)

    @staticmethod
    def _eval_process(evaluation_queues: Dict[str, Queue], index_queue: Queue,
                      pipeline_index_queues: Dict[str, Queue]):
        print(f"[{mp.current_process().pid}] Starting evaluation process.", file=stderr)

        frames = dict()
        last_positive = 0

        while len(evaluation_queues) > 0:
            if not index_queue.empty():
                frame, index = index_queue.get()
                if frame is None:
                    for q in pipeline_index_queues.values():
                        q.put(index)
                    continue
                frames[frame] = index
            for pipeline, q in evaluation_queues.copy().items():
                try:
                    item = q.get(False)
                except queue.Empty:
                    continue
                if item is None:
                    # q.close()
                    evaluation_queues.pop(pipeline)
                    continue
                frame_num, evaluation = item
                if frame_num not in frames:
                    frame, index = index_queue.get()
                    if frame is None:
                        for p_queue in pipeline_index_queues.values():
                            p_queue.put(index)
                        break
                    frames[frame] = index

                # print(f"{pipeline},{frame_num},{evaluation.result}")
                print(f"{pipeline} evaluated frame {frame_num}, result: {evaluation.result} ({evaluation.raw_answer})",
                      file=stderr)

                if pipeline == "custom_model" and evaluation.result:
                    t = time.time()
                    if t - last_positive > 5:
                        last_positive = t
                        pipeline_index_queues["gemini"].put(frames[frame_num])

        for q in evaluation_queues.values():
            q.close()
        index_queue.close()
        for q in pipeline_index_queues.values():
            q.close()
        print(f"[{mp.current_process().pid}] Evaluation process exiting.", file=stderr)

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
            self.camera_proc.terminate()

        # if self.autorun_proc:
        #     self.autorun_proc.kill()
        #
        # if self.results_proc:
        #     self.results_proc.kill()

        # Give the system time to stop gracefully
        t = time.time()
        while len(mp.active_children()) > 0 and time.time() - t < KILL_TIMEOUT:
            mp.connection.wait([p.sentinel for p in mp.active_children()], timeout=0.1)

        gracefully = True
        for p in [self.camera_proc, self.autorun_proc, self.results_proc] + [p.process for p in self.pipelines]:
            if p.is_alive():
                gracefully = False
                print(f"Forcefully terminating process {p.pid}.", file=stderr)
                p.kill()

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

        if gracefully:
            print("Stopped the analyzer gracefully.", file=stderr)
        self.stopping = False
