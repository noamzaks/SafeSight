from pathlib import Path
import sys
import time
from typing import Optional

from PIL import Image as Im
from PIL.Image import Image
from safesight.camera import Camera

import cv2


class FileCamera(Camera):
    """
    Imitates a real-time camera. When get_image is called, returns a frame according to how much
    time has passed.
    """

    def __init__(self, file: Path) -> None:
        self.video = cv2.VideoCapture(str(file))
        self.start_time: float = 0

    def start(self):
        self.start_time = time.time()

    def get_image(self) -> Optional[Image]:
        """
        Returns a frame from the video according to how much time has passed since start() was called.
        If start() was not called, calls start().
        If there was an error in handling the video (couldn't move video position, couldn't
        read frame) returns None.
        """

        success, image = self.video.read()
        if not success:
            return None
        color_converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        PIL_image = Im.fromarray(color_converted)
        return PIL_image

        if self.start_time == 0:
            self.start()

        current_time = time.time()
        time_since_start = current_time - self.start_time

        # Set the current position of the video in milliseconds.
        if not self.video.set(cv2.CAP_PROP_POS_MSEC, time_since_start * 1000.0):
            return None

        success, image = self.video.read()
        if not success:
            return None

        color_converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        PIL_image = Im.fromarray(color_converted)
        return PIL_image
