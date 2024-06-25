import torchvision.transforms.v2 as v2
import torch
from typing import List, Callable
from dataclasses import dataclass


@dataclass
class ModelSettings:
    internal_layer_size: int
    epochs: int
    learning_rate: float
    momentum: float
    transform: Callable


def TRANSFORM_OF_SIZE(size: int):
    """
    Returns a pytorch transform function that scales images to `size`x`size`.
    """
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
        epochs=10,
        learning_rate=0.001,
        momentum=0.9,
        transform=TRANSFORM_OF_SIZE(500),
    ),
    ModelSettings(
        internal_layer_size=6,
        epochs=3,
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
