import os

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from hw3_2 import get_statistic, get_fashion_mnist_model


PROJECT_NAME = "hw3_fashion_mnist"
CHECKPOINT_PATH = "checkpoints"
LABEL = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boat",
}


class SampleTester:
    def __init__(self, project_name, model, test_images, test_data_loader, test_transforms, checkpoint_path):
        self.project_name = project_name
        self.model = model
        self.test_images = test_images
        self.test_data_loader = test_data_loader
        self.transforms = test_transforms
        self.latest_file_path = os.path.join(
            checkpoint_path, f"{project_name}_checkpoint_latest.pt"
        )
        print("MODEL FILE: {0}".format(self.latest_file_path))
        self.model.load_state_dict(torch.load(self.latest_file_path, map_location=torch.device('cpu')))

    def test(self):
        self.model.eval()
        sample_id = 0

        with torch.no_grad():
            for input_tensor, target_tensor in self.test_data_loader:
                input_tensor = self.transforms(input_tensor)
                output_tensor = self.model(input_tensor)
                predictions = torch.argmax(output_tensor, dim=1)
                images = []  # array of images
                plot = False

                for i in range(len(input_tensor)):
                    image = self.test_images[sample_id][0]
                    prd = predictions[i].item()
                    target = target_tensor[i].item()
                    images.append(image)
                    sample_id += 1

                    # if the answer is wrong, plot this batch
                    if prd != target:
                        plot = True

                if plot:
                    plot_predictions(images, predictions, target_tensor)
                    break


def plot_predictions(images, predictions, targets):
    fig, rows = plt.subplots(nrows=2, ncols=5)
    i = 0

    for r in range(len(rows)):
        ax = rows[r]

        for c in range(len(ax)):
            fig.set_size_inches(10, 5)
            predict = predictions[i].item()
            target = targets[i].item()
            ax[c].text(0, 32, f"Predict: {LABEL[predict]}")
            ax[c].text(0, 36, f"Target: {LABEL[target]}")
            ax[c].imshow(images[i], cmap='gray')
            ax[c].axis('off')
            print(f"[{i}] predict: {predict} / target: {target}")
            i += 1

    fig.show()


def plot_wears(*label_ids):
    """
    Plot the matching sample images.
    It will draw 3 images per each clothes variant.
    :param label_ids: label identifiers to match
    """
    data_path = os.path.join("_00_data", "j_fashion_mnist")
    images = datasets.FashionMNIST(data_path, train=False, download=True)
    fig, rows = plt.subplots(nrows=len(label_ids), ncols=3)
    i = 0

    for r in range(len(rows)):
        label = label_ids[r]
        ax = rows[r]

        for c in range(len(ax)):
            target = images[i][1]

            while target != label:
                i += 1
                target = images[i][1]

            fig.set_size_inches(10, 5)
            ax[c].text(0, 32, LABEL[target])
            ax[c].imshow(images[i][0], cmap='gray')
            ax[c].axis('off')
            i += 1

    fig.show()


def get_fashion_mnist_test_data():
    data_path = os.path.join("_00_data", "j_fashion_mnist")
    f_mnist_test_images = datasets.FashionMNIST(data_path, train=False, download=True)
    f_mnist_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transforms.ToTensor())
    f_test_mean, f_test_std = get_statistic(f_mnist_test, dim=3)
    test_data_loader = DataLoader(dataset=f_mnist_test, shuffle=False, batch_size=10)

    f_mnist_transforms = nn.Sequential(
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=f_test_mean, std=f_test_std),
    )

    return f_mnist_test_images, test_data_loader, f_mnist_transforms


def main():
    test_images, test_data_loader, test_transforms = get_fashion_mnist_test_data()
    test_model = get_fashion_mnist_model()
    tester = SampleTester(
        PROJECT_NAME, test_model, test_images,
        test_data_loader, test_transforms, CHECKPOINT_PATH
    )
    tester.test()
    plot_wears(2, 4)


if __name__ == "__main__":
    main()
