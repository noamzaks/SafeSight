from pathlib import Path
import torchvision.transforms.v2 as transforms
import torch
from sys import stderr
from typing import List
from safesight.our_model import Net, evaluate_image, run_net_test, ModelSettings
import PIL.Image

settings: List[ModelSettings] = [
    ModelSettings(
        image_size=250,
        internal_layer_size=6,
        epochs=3,
        learning_rate=0.001,
        momentum=0.9,
    ),
    ModelSettings(
        image_size=500,
        internal_layer_size=16,
        epochs=10,
        learning_rate=0.001,
        momentum=0.9,
    ),
    ModelSettings(
        image_size=500,
        internal_layer_size=6,
        epochs=10,
        learning_rate=0.001,
        momentum=0.9,
    ),
    ModelSettings(
        image_size=250,
        internal_layer_size=16,
        epochs=10,
        learning_rate=0.001,
        momentum=0.9,
    ),
]


def net_tester(
    settings: List[ModelSettings],
    train_dir: Path,
    test_dir: Path,
    model_dir: Path = Path("./models"),
):
    model_dir.mkdir(parents=True, exist_ok=True)

    for i, setting in enumerate(settings):
        print(f"Running test {i}, with settings {setting}...", file=stderr)
        results = run_net_test(
            setting, train_dir, test_dir, model_dir / Path(f"model{i}.pth")
        )
        print(f"{i}: Settings: {setting}; Results: {results}")


if __name__ == "__main__":
    # net = Net(settings[0])
    # net.load_state_dict(torch.load("./models/model0.pth"))
    # image_size = settings[0].image_size
    # transform = transforms.Compose(
    #     [
    #         transforms.ToImage(),
    #         transforms.ToDtype(torch.float32, scale=True),
    #         transforms.Resize((image_size, image_size)),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ]
    # )
    # evaluate_image(net, PIL.Image.open("data/test/accident/test15_15.jpg"), transform)
    net_tester(settings, Path("./zaksaset"), Path("data/test"))
