from pathlib import Path
from typing import Optional, Tuple

from PIL import Image as Im
from PIL.Image import Image
from safesight.camera import Camera

import cv2


class FileCamera(Camera):
    def __init__(self, file: Path) -> None:
        self.video = cv2.VideoCapture(str(file))

    def get_image(self, resize: Optional[Tuple[int, int]] = None) -> Optional[Image]:
        success, image = self.video.read()
        if not success:
            print("Failed to read image from video file")
            return None
        color_converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        PIL_image = Im.fromarray(color_converted)
        if resize:
            PIL_image.resize(resize)
        return PIL_image
