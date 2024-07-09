import os
import time
import click
import PIL.Image
from tqdm import tqdm

from safesight.cli import pipeline
from safesight.custom_model_pipeline import CustomModelPipeline
from safesight.pipeline import Pipeline
from safesight.yolo_pipeline import YOLOPipeline
from safesight.gemini_pipeline import GeminiPipeline

PIPELINES = ["yolo", "gemini", "custom_model"]

def get_pipeline(pipeline_name: str) -> Pipeline:
    if pipeline_name == "yolo":
        return YOLOPipeline("best.pt")
    elif pipeline_name == "gemini":
        return GeminiPipeline()
    elif pipeline_name == "custom_model":
        return CustomModelPipeline("models/model2.pth")
    return None

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
@click.option(
    "--pipeline-name",
    type=click.Choice(PIPELINES)
)
@click.option(
    "--save-images",
    is_flag=True,
    default=False
)
def run_pipeline(image_directory_path: str, output_path: str, sleep_time: float, pipeline_name: str, save_images: bool = False):
    """ Runs a pipeline on all images in the directory recursively. Saves the results in the output directory.
    If save_images then saves the images with a name according to the result of the pipeline."""
    pipeline = get_pipeline(pipeline_name)

    images = []
    for path, dirs, files in os.walk(image_directory_path):
        for image_name in files:
            if image_name.endswith(".png"):
                images.append(os.path.join(path, image_name))

    with open(os.path.join(output_path, "results.csv"), "w") as results:
        with tqdm(images) as images_tqdm:
            for image_path in images_tqdm:
                images_tqdm.set_description(f"Processing {image_path}")
                image = PIL.Image.open(image_path)
                evaluation = pipeline.process_image(image)
                if save_images:
                    with open(os.path.join(output_path, f"{os.path.basename(image_path[:-4])}_{evaluation.result}.png"), "wb") as f:
                        image.save(f, "PNG")
                results.write(f"{image_path[len(image_directory_path)+1:]},{evaluation.result}\n")
                time.sleep(sleep_time)
