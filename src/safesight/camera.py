from PIL.Image import Image
import cv2
from typing import Optional


class Camera:
    """
    A camera to interface with the analyzer.

    See also: FileCamera, a camera sub-class to simulate a camera from a video file.
    """

    def __init__(self, index: int) -> None:
        self.video = cv2.VideoCapture(index)

    def get_image(self) -> Optional[Image]:
        read, image = self.video.read()
        if not read:
            return None


if __name__ == "__main__":
    cam = Camera(-1)
    cam.get_image()
