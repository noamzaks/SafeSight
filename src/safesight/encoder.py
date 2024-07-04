import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import PIL.Image
import clip
import torch
import torch.utils.data
import torchvision
from PIL.Image import Image
from torch import nn
from torch import optim

from safesight.test_results import TestResults


@dataclass
class TrainingSettings:
    learning_rate: float
    momentum: float
    epochs: int


class CLIPEncoder:
    def __init__(
        self,
        model: str = "ViT-B/32",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.model, self.preprocess = clip.load(model, device=self.device)

    def encode_image(self, image: Image) -> torch.Tensor:
        # unsqueeze to wrap it in a tensor, the model expects batches.
        prepared_image = self.preprocess(image)
        assert type(prepared_image) == torch.Tensor
        prepared_image = prepared_image.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output: torch.Tensor = self.model.encode_image(prepared_image)
        return output.squeeze()


class EncoderHead(nn.Module):
    def __init__(self, in_features):
        self.layer1 = nn.Linear(in_features, 20)
        self.layer2 = nn.Linear(20, 2)

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class EncoderClassifier:
    """
    A binary classifier (accident/nonaccident) that first passes images through an
    encoder, then a neural network (the head).
    """

    def __init__(
        self,
        encoder: Callable[[Image], torch.Tensor] = CLIPEncoder("ViT-B/32").encode_image,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = encoder
        features = self._get_num_features()
        self.head = nn.Sequential(nn.Linear(in_features=features, out_features=2))
        self.head.to(self.device)

    def _get_num_features(self):
        """
        Get the number of features that the encoder returns (size of the vector).
        """
        image = PIL.Image.new("RGB", (1, 1))
        encoding = self.encoder(image)
        return encoding.size(0)

    def evaluate_image(self, image: Image) -> int:
        """
        Evaluate an image by passing it through the encoder and then the head.
        """
        encoding = self.encoder(image)
        result = self.head(encoding)
        _, label = torch.max(result.data, 1)
        return int(label)

    def train_head(
        self, dataset_path: Path, settings: TrainingSettings
    ) -> Dict[int, str]:
        """
        Train the head according to a labeled dataset. The dataset should be a directory of
        the form:
            traindir/accident/image1.jpg

                             /image2.jpg

                             /...

            traindir/nonaccident/image1.jpg

                                /image2.jpg

                                /...

        Returns a dictionary from the numeric dataset labels (returned by evaluate_image) to the
        string labels (names of the subdirectories of dataset_path)
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            self.head.parameters(),
            lr=settings.learning_rate,
            momentum=settings.momentum,
        )

        trainset = torchvision.datasets.ImageFolder(
            str(dataset_path), transform=self.encoder
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=4, shuffle=True, num_workers=2
        )

        for epoch in range(settings.epochs):  # loop over the dataset multiple times
            print(f"Running epoch {epoch}...", file=sys.stderr)

            running_loss = 0.0
            for _, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.head(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
            print(f"Finished epoch {epoch}, Loss: {running_loss}", file=sys.stderr)

        print("Finished Training", file=sys.stderr)
        self.idx_to_class = {k: v for v, k in trainset.class_to_idx.items()}
        return self.idx_to_class

    def test(self, dataset_path: Path) -> TestResults:
        """
        Test the model on a dataset, which is a dir organized like in `train_head`.
        """
        dataset = torchvision.datasets.ImageFolder(
            str(dataset_path), transform=self.encoder
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=4, shuffle=True, num_workers=2
        )

        labels = []
        predictions = []
        with torch.no_grad():
            for data in dataloader:
                images, batch_labels = data
                # calculate outputs by running images through the network
                outputs = self.head(images)
                # the class with the highest energy is what we choose as prediction
                _, batch_predictions = torch.max(outputs.data, 1)
                batch_predictions = list(map(int, batch_predictions))
                batch_labels = list(map(int, batch_labels))

                labels += batch_labels
                predictions += batch_predictions

        labels = list(map(bool, labels))
        predictions = list(map(bool, predictions))
        return TestResults(labels, predictions)

    def save_head(self, path: Path):
        torch.save(self.head, path)

    def load_head(self, path: Path):
        """
        Load a head model previously saved by `save_head`.
        """
        self.head.load_state_dict(torch.load(path))

    def run_on_video(self, video_path: Path) -> List[bool]:
        """
        Evaluate each frame in the video at video_path. Outputs to
        """
        # file_camera = FileCamera(video_path)


if __name__ == "__main__":
    encoder = CLIPEncoder(
        "ViT-L/14@336px", device="cuda" if torch.cuda.is_available() else "cpu"
    )
    model = EncoderClassifier(encoder.encode_image)
    model.train_head(
        Path("zaksaset/train"),
        TrainingSettings(learning_rate=0.01, momentum=0.9, epochs=10),
    )
    print(model.test(Path("zaksaset/test")))
    model.save_head(Path("modelhead2.pth"))
