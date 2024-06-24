import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import PIL.Image
import click
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
from PIL.Image import Image

from safesight.test_results import TestResults
from safesight.cli import cli
from safesight.model_settings import settings as custom_model_settings
from safesight.model_settings import ModelSettings


class Net(nn.Module):
    """
    A Neural Network to process images.
    To use, run something like:
        `
        net = Net(settings)
        train(net, traindir)
        outputs = net(transformed_inputs)
        `

    Where `transformed_inputs` are a tensor of inputs that went through the
    transform function the network was trained with.

    To load a pre-trained model:
        `
        net = Net()
        net.load_state_dict(torch.load(model_path))
        `
    Which also loads the ModelSettings.
    """

    def __init__(self, settings: Optional[ModelSettings] = None):
        super(Net, self).__init__()
        if settings:
            self.apply_settings(settings)

    def _get_image_size(self) -> Tuple[int, int]:
        dummy_image = PIL.Image.new("RGB", (1, 1))
        transformed_image = self.settings.transform(dummy_image)
        return transformed_image.size()[1:3]

    def _get_conv_output(self, input_size: Tuple[int, int]):
        # Create a dummy tensor with the input size and pass it through the conv and pool layers
        with torch.no_grad():
            x = torch.zeros(1, 3, input_size[0], input_size[1])
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            return x.numel()  # Total number of elements in the tensor

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def evaluate_image(self, image: Image) -> Union[str, int]:
        """
        Evaluate an image with the network.
        Returns the label it predicts for the image, or the numerical label
        if it can't find the matching string.
        """
        # print("in evaluate_image")
        # print(self.settings.transform)
        # print(image)
        transformed = self.settings.transform(image).unsqueeze(0)
        # print(transformed, file=sys.stderr)
        output = self(transformed)
        _, label = torch.max(output.data, 1)
        label = int(label)

        try:
            return self.idx_to_class[label]
        except KeyError:
            return label

    def apply_settings(self, settings: ModelSettings):
        self.settings = settings

        self.idx_to_class: Dict[int, str] = {}
        self.class_to_idx: Dict[str, int] = {}

        self.conv1 = nn.Conv2d(3, settings.internal_layer_size, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(settings.internal_layer_size, 16, 5)

        # Calculate the size of the feature map after the convolutional and pooling layers
        conv_output_size = self._get_conv_output(self._get_image_size())

        self.fc1 = nn.Linear(conv_output_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def get_extra_state(self) -> ModelSettings:
        return self.settings

    def set_extra_state(self, state: ModelSettings) -> None:
        self.apply_settings(state)


def train(traindir: Path, net: Net):
    """
    Train a net on the dataset at `traindir`, which is set up like so:
        traindir/accident/image1.jpg
                         /image2.jpg
                         /...
        traindir/nonaccident/image1.jpg
                            /image2.jpg
                            /...
    """

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(), lr=net.settings.learning_rate, momentum=net.settings.momentum
    )

    trainset = torchvision.datasets.ImageFolder(
        str(traindir), transform=net.settings.transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=2
    )

    net.class_to_idx = trainset.class_to_idx
    net.idx_to_class = {v: k for k, v in net.class_to_idx.items()}

    for epoch in range(net.settings.epochs):  # loop over the dataset multiple times
        print(f"Running epoch {epoch}...", file=sys.stderr)

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # if i % 10 == 9:
            #     print(
            #         f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}",
            #         file=sys.stderr,
            #     )
            #     running_loss = 0.0
        print(f"Finished epoch {epoch}, Loss: {running_loss}", file=sys.stderr)

    print("Finished Training", file=sys.stderr)


def test(testdir: Path, net: Net) -> Tuple[List[int], List[int]]:
    """
    Runs the model on the testdir with the net's transform.
    Returns a tuple of the form (labels, predictions).
    `testdir` should be organized like training directories are.
    """
    testset = torchvision.datasets.ImageFolder(
        str(testdir), transform=net.settings.transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )
    dataiter = iter(testloader)
    images, batch_labels = next(dataiter)

    labels = []
    predictions = []

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, batch_labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, batch_predictions = torch.max(outputs.data, 1)
            batch_predictions = list(map(int, batch_predictions))
            batch_labels = list(map(int, batch_labels))

            labels += batch_labels
            predictions += batch_predictions

    return (labels, predictions)


def run_with_settings_list(
    settings: List[ModelSettings],
    train_dir: Path,
    test_dir: Path,
    model_dir: Path = Path("./models"),
):
    """
    Train and test models with different parameters.
    `traindir` and `testdir` should be organized like so:
        traindir/accident/image1.jpg
                         /image2.jpg
                         /...
        traindir/nonaccident/image1.jpg
                            /image2.jpg
                            /...
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    for i, setting in enumerate(settings):
        print(f"Running test {i}, with settings {setting}...", file=sys.stderr)
        results = run_with_settings(
            setting, train_dir, test_dir, model_dir / Path(f"model{i}.pth")
        )
        print(f"{i}: Settings: {setting}; Results: {results}")


def run_with_settings(
    settings: ModelSettings,
    traindir: Path,
    testdir: Path,
    save_path: Optional[Path],
) -> TestResults:
    net = Net(settings)
    train(traindir, net)
    if save_path:
        torch.save(net.state_dict(), str(save_path))
    labels, predictions = test(testdir, net)
    labels = list(map(bool, labels))
    predictions = list(map(bool, predictions))
    return TestResults(labels, predictions)


@cli.group()
def custom_model():
    """
    Commands to train and run custom models.
    """


@custom_model.command()
@click.option(
    "--train-path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default=Path("data/train"),
    show_default=True,
)
@click.option(
    "--test-path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default=Path("data/test"),
    show_default=True,
)
def train_and_run_image(train_path: Path, test_path: Path):
    """
    Train and test image classification models with different parameters, taken from
    safesight.model_settings.
    `test-path` and `test-path` should be directories organized like so:

        train-path/accident/image1.jpg

        train-path/nonaccident/image2.jpg

        train-path/accident/...

        test-path/accident/image1.jpg

        test-path/nonaccident/image2.jpg

        test-path/accident/...

    Outputs results into STDOUT, and saves the models in ./models.

    WARNING - Overwrites files at ./models/model0.pth, ./models/model1.pth, ...
    """
    model_dir = Path("./models")
    model_dir.mkdir(parents=True, exist_ok=True)

    for i, setting in enumerate(custom_model_settings):
        print(f"Running test {i}, with settings {setting}...", file=sys.stderr)
        results = run_with_settings(
            setting, train_path, test_path, model_dir / Path(f"model{i}.pth")
        )
        print(f"{i}: Settings: {setting}; Results: {results}")
