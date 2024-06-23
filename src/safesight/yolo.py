from typing import Union, List

from ultralytics import YOLO
from pathlib import Path

from safesight.cli import cli


def get_all_files(directory: Union[Path, str], extention: str) -> List[Path]:
    return list(Path(directory).glob(f"**/*.{extention}"))


@cli.group()
def yolo():
    """Commands for YOLO model"""
    pass


@yolo.command()
def main():  # TODO: refactor as command
    # Load a model
    model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model

    # Run batched inference on a list of images
    results = model(get_all_files("data", "jpg"))  # return a list of Results objects

    # Process results list
    for result in results:
        orig_path = Path(result.path).relative_to("data")
        prediction_path = Path("yolo_predictions") / orig_path
        result.save(filename=str(prediction_path))  # save to disk
