import os
import time
import click
import PIL.Image

from safesight.cli import pipeline
from safesight.yolo_pipeline import YOLOPipeline

@pipeline.command()
@click.option(
    "--image-directory-path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.option(
    "--output-path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.option(
    "--sleep-time",
    type=float,
    default=0
)
def run_pipeline(image_directory_path: str, output_path: str, sleep_time: float):
    pipeline = YOLOPipeline("best.pt")

    for image_name in os.listdir(image_directory_path):
        if not image_name.endswith(".png"):
            continue

        image = PIL.Image.open(os.path.join(image_directory_path, image_name))
        evaluation = pipeline.process_image(image)
        with open(os.path.join(output_path, f"{image_name}_{evaluation.result}.png"), "wb") as f:
            image.save(f, "PNG")
        print(f"{image_name},{evaluation.result}")
        time.sleep(sleep_time)
