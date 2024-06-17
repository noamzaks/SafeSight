from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, List
from PIL.Image import Image
import PIL.Image
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.utils.data
import torchvision
import torchvision.transforms.v2 as transforms
import torch.optim as optim


import torch
import torch.nn as nn
import torch.nn.functional as F

from safesight.model_api import TestResults


@dataclass
class ModelSettings:
    internal_layer_size: int
    epochs: int
    learning_rate: float
    momentum: float
    transform: Callable


class Net(nn.Module):
    def __init__(self, settings: ModelSettings):
        super(Net, self).__init__()
        self.settings = settings

        self.idx_to_class: Dict[str, int] = {}
        self.class_to_idx: Dict[str, int] = {}

        self.conv1 = nn.Conv2d(3, settings.internal_layer_size, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(settings.internal_layer_size, 16, 5)

        # Calculate the size of the feature map after the convolutional and pooling layers
        conv_output_size = self._get_conv_output(self.get_image_size())

        self.fc1 = nn.Linear(conv_output_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def get_image_size(self) -> Tuple[int, int]:
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


def train(traindir: Path, net: Net, settings: ModelSettings):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(), lr=settings.learning_rate, momentum=settings.momentum
    )

    trainset = torchvision.datasets.ImageFolder(
        str(traindir), transform=settings.transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=2
    )

    net.class_to_idx = trainset.class_to_idx
    net.idx_to_class = {v: k for k, v in net.class_to_idx.items()}

    for epoch in range(settings.epochs):  # loop over the dataset multiple times
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


def test(
    testdir: Path, net: Net, settings: ModelSettings
) -> Tuple[List[int], List[int]]:
    """
    Runs the model on the testdir with the transform.
    Returns a tuple of the form (labels, predictions).
    """
    testset = torchvision.datasets.ImageFolder(
        str(testdir), transform=settings.transform
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


def evaluate_image(net: Net, image: Image, settings: ModelSettings) -> int:
    transformed = settings.transform(image).unsqueeze(0)
    output = net(transformed)
    print(output.data)
    _, label = torch.max(output.data, 1)
    return int(label)


def run_net_test(
    settings: ModelSettings,
    traindir: Path,
    testdir: Path,
    save_path: Optional[Path],
) -> TestResults:
    net = Net(settings)
    train(traindir, net, settings)
    if save_path:
        torch.save(net.state_dict(), str(save_path))
    labels, predictions = test(testdir, net, settings)
    labels = list(map(bool, labels))
    predictions = list(map(bool, predictions))
    return TestResults(labels, predictions)
