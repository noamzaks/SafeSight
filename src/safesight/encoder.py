import sys
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable, Dict, List, Optional

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
        super(EncoderHead, self).__init__()
        self.layer1 = nn.Linear(in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.sigmoid(x)
        return x


class EncoderClassifier:
    """
    A binary classifier (accident/nonaccident) that first passes images through an
    encoder, then a neural network (the head).
    """

    def __init__(
        self, encoder: Callable[[Image], torch.Tensor], layer_sizes: List[int]
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = encoder
        features = self._get_num_features()
        if layer_sizes:
            hidden_layers = []
            for i in range(len(layer_sizes) - 1):
                hidden_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                hidden_layers.append(nn.ReLU())

            self.head = nn.Sequential(
                nn.Linear(features, layer_sizes[0]),
                nn.ReLU(),
                *hidden_layers,
                nn.Linear(layer_sizes[-1], 1),
                nn.Sigmoid(),
            )
        else:
            self.head = nn.Sequential(nn.Linear(features, 1), nn.Sigmoid())
        print(self.head)
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
        criterion = nn.BCELoss()
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
                labels = labels.type(torch.float)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.head(inputs).squeeze()
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
                outputs = self.head(images).squeeze()
                # round to 1 or 0 to get the predictions
                batch_predictions = outputs.round()
                batch_predictions = list(map(int, batch_predictions))
                batch_labels = list(map(int, batch_labels))

                labels += batch_labels
                predictions += batch_predictions

        labels = list(map(bool, labels))
        predictions = list(map(bool, predictions))
        return TestResults(labels, predictions)

    def save_head(self, path: Path):

        torch.save(self.head.state_dict(), path)

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


def train_and_test_encoder_classifier(
    encoder: Callable[[Image], torch.Tensor],
    training_settings: TrainingSettings,
    layers: List[int],
    train_dataset: Path,
    test_dataset: Path,
    save_model_path: Optional[Path],
) -> TestResults:
    model = EncoderClassifier(encoder, layers)
    model.train_head(train_dataset, training_settings)
    if save_model_path:
        model.save_head(save_model_path)
    return model.test(test_dataset)


if __name__ == "__main__":
    print(
        0,
        train_and_test_encoder_classifier(
            CLIPEncoder("ViT-B/32").encode_image,
            TrainingSettings(learning_rate=0.01, momentum=0.9, epochs=7),
            layers=[],
            train_dataset=Path("zaksaset/train"),
            test_dataset=Path("zaksaset/test"),
            save_model_path=Path(
                "encoder_models/Linear-B32-lr0.01-mom0.9-epochs7-zaksaset.pth"
            ),
        ),
    )
    print(
        1,
        train_and_test_encoder_classifier(
            CLIPEncoder("ViT-B/32").encode_image,
            TrainingSettings(learning_rate=0.01, momentum=0.9, epochs=10),
            layers=[50],
            train_dataset=Path("zaksaset/train"),
            test_dataset=Path("zaksaset/test"),
            save_model_path=Path(
                "encoder_models/Hidden50-B32-lr0.01-mom0.9-epochs7-zaksaset.pth"
            ),
        ),
    )
    print(
        2,
        train_and_test_encoder_classifier(
            CLIPEncoder("ViT-B/32").encode_image,
            TrainingSettings(learning_rate=0.01, momentum=0.9, epochs=10),
            layers=[50, 50],
            train_dataset=Path("zaksaset/train"),
            test_dataset=Path("zaksaset/test"),
            save_model_path=Path(
                "encoder_models/Hidden50-50-B32-lr0.01-mom0.9-epochs10-zaksaset.pth"
            ),
        ),
    )
    print(
        3,
        train_and_test_encoder_classifier(
            CLIPEncoder("ViT-B/32").encode_image,
            TrainingSettings(learning_rate=0.05, momentum=0.9, epochs=10),
            layers=[50],
            train_dataset=Path("zaksaset/train"),
            test_dataset=Path("zaksaset/test"),
            save_model_path=Path(
                "encoder_models/Hidden50-B32-lr0.05-mom0.9-epoch7-zaksaset.pth"
            ),
        ),
    )
    print(
        4,
        train_and_test_encoder_classifier(
            CLIPEncoder("ViT-B/32").encode_image,
            TrainingSettings(learning_rate=0.05, momentum=0.9, epochs=20),
            layers=[200, 200],
            train_dataset=Path("zaksaset/train"),
            test_dataset=Path("zaksaset/test"),
            save_model_path=Path(
                "encoder_models/Hidden200-200-B32-lr0.01-mom0.9-epochs20-zaksaset.pth"
            ),
        ),
    )
