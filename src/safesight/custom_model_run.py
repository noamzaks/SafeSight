import csv
import matplotlib.pyplot as plt
from safesight.cli import cli
import click
from safesight.custom_model_pipeline import CustomModelPipeline
from safesight.file_camera import FileCamera
from pathlib import Path


@cli.group()
def run():
    """
    Commands to train and run custom models.
    """


# @run.command()
# @click.option(
#     "--video-path", type=click.Path(exists=True, dir_okay=False, file_okay=True)
# )
# @click.option(
#     "--model-path", type=click.Path(exists=True, dir_okay=False, file_okay=True)
# )
def run_model(video_path: Path, model_path: Path):
    pipeline = CustomModelPipeline(model_path)
    file_camera = FileCamera(video_path)

    image = file_camera.get_image()
    frame_num = 0
    while image:
        evaluation = pipeline.process_image(image)
        print(f"{frame_num},{evaluation.result}")
        image = file_camera.get_image()
        frame_num += 1

    print("Stopped evaluation")


def plot_results(result_file: Path):
    results = []
    with open(result_file, "r") as file:
        results = [1 if line.split(",")[1].strip() == "True" else 0 for line in file]
    plt.plot(range(1, len(results) + 1), results)
    plt.savefig("results.png")


if __name__ == "__main__":
    # run_model(Path("data/videos/9DRFJxKHc6g.mp4"), Path("models/model0.pth"))
    plot_results(Path("9D_video_results.csv"))
