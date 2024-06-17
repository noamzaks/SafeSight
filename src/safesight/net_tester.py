from pathlib import Path
import torchvision.transforms.v2 as v2
import torch
from sys import stderr
from typing import List
from safesight.model_api import TestResults
from safesight.our_model import Net, evaluate_image, run_net_test, ModelSettings, test


def TRANSFORM_OF_SIZE(size):
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((size, size)),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


settings: List[ModelSettings] = [
    ModelSettings(
        internal_layer_size=6,
        epochs=3,
        learning_rate=0.001,
        momentum=0.9,
        transform=TRANSFORM_OF_SIZE(500),
    ),
    ModelSettings(
        internal_layer_size=32,
        epochs=15,
        learning_rate=0.001,
        momentum=0.9,
        transform=TRANSFORM_OF_SIZE(500),
    ),
    ModelSettings(
        internal_layer_size=16,
        epochs=10,
        learning_rate=0.001,
        momentum=0.9,
        transform=TRANSFORM_OF_SIZE(500),
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
    net_tester(settings, Path("data/train"), Path("data/test"))
