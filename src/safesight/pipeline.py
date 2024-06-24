import multiprocessing as mp
import struct
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from multiprocessing.shared_memory import SharedMemory
from sys import stderr
from typing import Tuple

import PIL.Image
from PIL.Image import Image


@dataclass
class Evaluation:
    result: bool
    # Raw answer, when available from LLM.
    raw_answer: str = ""
    timestamp: datetime = datetime(2000, 1, 1)


class Pipeline(ABC):
    @abstractmethod
    def process_image(self, image: Image) -> Evaluation:
        """
        Processes an image and returns an evaluation.

        @param image: The image to process. Should not be modified, a copy should be made is changes are necessary.
        """
        pass

    @abstractmethod
    def prepare(self):
        """
        Called before the pipeline is started.
        """
        pass

    @abstractmethod
    def cleanup(self):
        """
        Called after the pipeline is stopped.
        """
        pass

    def run_pipeline(self, *, shared_memory_name: str, evaluation_queue: mp.SimpleQueue):
        from safesight.analyzer import MemoryControl as MemCtrl
        print(f"[{mp.current_process().pid}] Starting pipeline {self}.", file=stderr)

        self.prepare()

        mem = None
        try:
            mem = SharedMemory(name=shared_memory_name)
            buff = mem.buf.toreadonly()

            index = 0
            reset_countdown = 10
            while True:
                frame_num = struct.unpack(">I", buff[index:index + 4])[0]

                if frame_num == MemCtrl.WAIT.value:
                    continue
                if frame_num == MemCtrl.CLOSE.value:
                    break
                if frame_num == MemCtrl.RESET_INDEX.value:
                    index = 0
                    continue

                # noinspection PyTypeChecker
                size: Tuple[int, int] = struct.unpack(">HH", buff[index + 4:index + 8])
                if size[0] * size[1] == 0:
                    print(f"[{mp.current_process().pid}] Unexpected size: {size}.", file=stderr)
                    if reset_countdown == 0:
                        print(f"[{mp.current_process().pid}] Forcing index reset.", file=stderr)
                        index = 0
                    else:
                        reset_countdown -= 1
                        time.sleep(0.01)
                    continue
                reset_countdown = 10

                frame_len = size[0] * size[1] * 4  # RGBA
                terminator = buff[index + 8 + frame_len]

                if terminator == MemCtrl.FRAME_NOT_READY.value:
                    continue
                if terminator != MemCtrl.FRAME_END.value:
                    print(f"[{mp.current_process().pid}] Unexpected terminator: {terminator}.", file=stderr)
                    continue

                frame = buff[index + 8:index + 8 + frame_len]
                image = PIL.Image.frombuffer("RGBA", size, frame)
                evaluation = self.process_image(image)
                print(f"[{mp.current_process().pid}] Evaluated frame #{frame_num}, result: {evaluation.result}",
                      file=stderr)
                evaluation_queue.put((frame_num, evaluation))

                index = index + 8 + frame_len + 1

        finally:
            if mem is not None:
                mem.close()
            evaluation_queue.put(None)
            evaluation_queue.close()
            self.cleanup()
            print(f"[{mp.current_process().pid}] Closing pipeline {self}.", file=stderr)
