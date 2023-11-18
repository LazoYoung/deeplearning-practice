import os
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

BASE_PATH = '.'
# BASE_PATH = '/content/drive/MyDrive/Colab Notebooks'

# Appending BASE_PATH to system path
sys.path.append(BASE_PATH)

CHECKPOINT_PATH = os.path.join(BASE_PATH, "checkpoints")
if not os.path.isdir(CHECKPOINT_PATH):
    os.makedirs(os.path.join(BASE_PATH, "checkpoints"))


from hw3_2 import get_statistic, get_fashion_mnist_model, PROJECT_NAME


class Tester:
    def __init__(self, project_name, model, test_data_loader, transforms, checkpoint_path):
        self.project_name = project_name
        self.model = model
        self.test_data_loader = test_data_loader
        self.transforms = transforms

        # file path to latest checkpoint
        self.latest_file_path = os.path.join(
            checkpoint_path, f"{project_name}_checkpoint_latest.pt"
        )

        print("MODEL FILE: {0}".format(self.latest_file_path))

        # Load model states from the checkpoint
        self.model.load_state_dict(torch.load(self.latest_file_path, map_location=torch.device('cpu')))

    def test(self):
        self.model.eval()  # disable dropout layers

        correct_num = 0
        tested_sample_num = 0

        # disable model optimization
        with torch.no_grad():
            for test_batch in self.test_data_loader:
                input_test, target_test = test_batch

                # transform/normalize input
                input_test = self.transforms(input_test)

                # feed forward
                output_test = self.model(input_test)

                # decode one-hot vectors
                predicted_test = torch.argmax(output_test, dim=1)
                correct_num += torch.sum(torch.eq(predicted_test, target_test))
                tested_sample_num += len(input_test)

            test_accuracy = 100.0 * correct_num / tested_sample_num

        print(f"TEST RESULTS: {test_accuracy:6.3f}%")

    def test_single(self, input_test):
        self.model.eval()  # disable dropout layers

        # disable model optimizations
        with torch.no_grad():
            # transform/normalize input
            input_test = self.transforms(input_test)

            # feed forward
            output_test = self.model(input_test)

            # decode one-hot vectors
            predicted_test = torch.argmax(output_test, dim=1)

        return predicted_test.item()


def get_fashion_mnist_test_data():
    data_path = os.path.join(BASE_PATH, "_00_data", "j_fashion_mnist")
    f_mnist_test_images = datasets.FashionMNIST(data_path, train=False, download=True)
    f_mnist_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transforms.ToTensor())
    f_test_mean, f_test_std = get_statistic(f_mnist_test, dim=3)

    print("Num Test Samples: ", len(f_mnist_test))
    print("Sample Shape: ", f_mnist_test[0][0].shape)  # torch.Size([1, 28, 28])
    print("Mean: ", f_test_mean, ", Std: ", f_test_std)

    test_data_loader = DataLoader(dataset=f_mnist_test, batch_size=len(f_mnist_test))

    f_mnist_transforms = nn.Sequential(
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=f_test_mean, std=f_test_std),
    )

    return f_mnist_test_images, test_data_loader, f_mnist_transforms


def main():
    images, test_data_loader, test_transforms = get_fashion_mnist_test_data()
    test_model = get_fashion_mnist_model()
    tester = Tester(
        PROJECT_NAME, test_model, test_data_loader,
        test_transforms, CHECKPOINT_PATH
    )
    tester.test()
    print()

    img, label = images[0]
    print("     LABEL:", label)
    plt.imshow(img)
    plt.show()

    output = tester.test_single(
        torch.tensor(np.array(images[0][0])).unsqueeze(dim=0).unsqueeze(dim=0)
    )
    print("PREDICTION:", output)


if __name__ == "__main__":
    main()
