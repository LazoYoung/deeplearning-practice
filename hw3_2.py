import os
import sys
from datetime import datetime

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import datasets
from torchvision.transforms import transforms

import wandb

# Global variables
PROJECT_NAME = "hw3_fashion_mnist"
BASE_PATH = '.'
# BASE_PATH = '/content/drive/MyDrive/Colab Notebooks'

# Appending BASE_PATH to system path
sys.path.append(BASE_PATH)

# Importing external code...
# must be done after appending BASE_PATH to system path
from _01_code._06_fcn_best_practice.c_trainer import EarlyStopping
from _01_code._99_common_utils.utils import strfdelta

CHECKPOINT_PATH = os.path.join(BASE_PATH, "checkpoints")
if not os.path.isdir(CHECKPOINT_PATH):
    os.makedirs(os.path.join(BASE_PATH, "checkpoints"))


class Trainer:
    def __init__(
            self, project_name, model, optimizer, train_data_loader, validation_data_loader,
            transforms, run_time_str, wandb, device, checkpoint_file_path
    ):
        self.project_name = project_name
        self.model = model
        self.optimizer = optimizer
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.transforms = transforms
        self.run_time_str = run_time_str
        self.wandb = wandb
        self.device = device
        self.checkpoint_file_path = checkpoint_file_path
        # reduction='mean' ensures loss_fn to produce scalar output (although not required)
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')

    def run(self):
        early_stopping = EarlyStopping(
            patience=self.wandb.config.early_stop_patience,
            delta=self.wandb.config.early_stop_delta,
            project_name=self.project_name,
            checkpoint_file_path=self.checkpoint_file_path,
            run_time_str=self.run_time_str
        )
        n_epochs = self.wandb.config.epochs
        training_start_time = datetime.now()

        for epoch in range(1, n_epochs + 1):
            train_loss, train_accuracy = self.train_model()

            if epoch == 1 or epoch % self.wandb.config.validation_intervals == 0:
                validation_loss, validation_accuracy = self.validate_model()

                elapsed_time = datetime.now() - training_start_time
                epoch_per_second = 0 if elapsed_time.seconds == 0 else epoch / elapsed_time.seconds

                message, early_stop = early_stopping.check_and_save(validation_loss, self.model)

                print(
                    f"[Epoch {epoch:>3}] "
                    f"T_loss: {train_loss:7.5f}, "
                    f"T_accuracy: {train_accuracy:6.4f} | "
                    f"V_loss: {validation_loss:7.5f}, "
                    f"V_accuracy: {validation_accuracy:6.4f} | "
                    f"{message} | "
                    f"T_time: {strfdelta(elapsed_time, '%H:%M:%S')}, "
                    f"T_speed: {epoch_per_second:4.3f}"
                )

                self.wandb.log({
                    "Epoch": epoch,
                    "Training loss": train_loss,
                    "Training accuracy (%)": train_accuracy,
                    "Validation loss": validation_loss,
                    "Validation accuracy (%)": validation_accuracy,
                    "Training speed (epochs/sec.)": epoch_per_second,
                })

                if early_stop:  # Discontinue training if we run out of patience...
                    break

        elapsed_time = datetime.now() - training_start_time
        print(f"Final training time: {strfdelta(elapsed_time, '%H:%M:%S')}")

    def train_model(self):
        self.model.train()
        loss_sum = 0.0
        correct_num = 0
        trained_sample_num = 0
        train_num = 0

        for batch in self.train_data_loader:
            input_tensor, target_tensor = batch
            input_tensor = input_tensor.to(self.device)
            target_tensor = target_tensor.to(self.device)
            input_tensor = self.transforms(input_tensor)
            output_tensor = self.model(input_tensor)
            loss = self.loss_fn(output_tensor, target_tensor)
            loss_sum += loss.item()  # aggregate loss (assert it's scalar)

            # decode one-hot vectors
            predicted_tensor = torch.argmax(output_tensor, dim=1)
            correct_num += torch.sum(torch.eq(predicted_tensor, target_tensor)).item()
            trained_sample_num += len(input_tensor)
            train_num += 1

            self.optimizer.zero_grad()
            loss.backward()  # compute gradients
            self.optimizer.step()  # update parameters

        train_loss = loss_sum / train_num
        train_accuracy = 100.0 * correct_num / trained_sample_num

        return train_loss, train_accuracy

    def validate_model(self):
        self.model.eval()
        loss_sum = 0.0
        correct_num = 0
        validated_sample_num = 0
        validate_num = 0

        # optimization is not required
        with torch.no_grad():
            for validation_batch in self.validation_data_loader:
                input_tensor, target_tensor = validation_batch
                input_tensor = input_tensor.to(device=self.device)
                target_tensor = target_tensor.to(device=self.device)
                input_tensor = self.transforms(input_tensor)
                output_tensor = self.model(input_tensor)
                loss_sum += self.loss_fn(output_tensor, target_tensor).item()

                predicted_tensor = torch.argmax(output_tensor, dim=1)
                correct_num += torch.sum(torch.eq(predicted_tensor, target_tensor)).item()

                validated_sample_num += len(input_tensor)
                validate_num += 1

        validation_loss = loss_sum / validate_num
        validation_accuracy = 100.0 * correct_num / validated_sample_num

        return validation_loss, validation_accuracy


def get_fashion_mnist_data(augmented: bool, num_workers=1):
    data_path = os.path.join(BASE_PATH, "_00_data", "j_fashion_mnist")
    f_mnist_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=transforms.ToTensor())
    f_mnist_train, f_mnist_validation = random_split(f_mnist_train, [55_000, 5_000])

    if augmented:
        train_transforms = nn.Sequential(
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAdjustSharpness(sharpness_factor=0.1, p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        )
        transformed = []
        for image, label in f_mnist_train:
            transformed_image = train_transforms(image)
            transformed.append((transformed_image, label))
        f_mnist_train = ConcatDataset([f_mnist_train, transformed])

    # get mean, standard deviation of training dataset
    f_train_mean, f_train_std = get_statistic(f_mnist_train, dim=3)

    # A transformer that normalizes the dataset
    f_mnist_transforms = nn.Sequential(
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=f_train_mean, std=f_train_std),
    )

    print("Num Train Samples: ", len(f_mnist_train))
    print("Num Validation Samples: ", len(f_mnist_validation))
    print("Sample Shape: ", f_mnist_train[0][0].shape)  # torch.Size([1, 28, 28])
    print("Label Shape: ", type(f_mnist_train[0][1]))
    print("Mean: ", f_train_mean, ", Std: ", f_train_std)
    print("Number of Data Loading Workers:", num_workers)

    train_data_loader = DataLoader(
        dataset=f_mnist_train, batch_size=wandb.config.batch_size, shuffle=True,
        pin_memory=True, num_workers=num_workers
    )

    validation_data_loader = DataLoader(
        dataset=f_mnist_validation, batch_size=wandb.config.batch_size,
        pin_memory=True, num_workers=num_workers
    )

    return train_data_loader, validation_data_loader, f_mnist_transforms


def get_statistic(dataset, dim):
    dataset = torch.stack([t for t, _ in dataset], dim=dim)
    mean = dataset.view(-1).mean().item()
    std = dataset.view(-1).std().item()

    return mean, std


def get_fashion_mnist_model():
    class MyModel(nn.Module):
        def __init__(self, in_channel, n_output):
            super().__init__()

            self.model = nn.Sequential(
                # B x 1 x 28 x 28 --> B x 6 x (28 - 5 + 1) x (28 - 5 + 1) = B x 6 x 24 x 24
                nn.Conv2d(in_channels=in_channel, out_channels=6, kernel_size=(5, 5), stride=(1, 1)),
                # B x 6 x 24 x 24 --> B x 6 x 12 x 12
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(num_features=6),
                nn.ReLU(),
                # B x 6 x 12 x 12 --> B x 16 x (12 - 5 + 1) x (12 - 5 + 1) = B x 16 x 8 x 8
                nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1)),
                # B x 16 x 8 x 8 --> B x 16 x 4 x 4
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(num_features=16),
                nn.ReLU(),
                nn.Flatten(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.BatchNorm1d(num_features=128),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(128, n_output),
            )

        def forward(self, x):
            return self.model(x)

    return MyModel(in_channel=1, n_output=10)


def main(online=False):
    config = {
        'epochs': 500,
        'batch_size': 2048,
        'validation_intervals': 10,
        'learning_rate': 1e-3,
        'early_stop_patience': 10,
        'early_stop_delta': 1e-5,
        'weight_decay': 0.0,
        'dropout': True,
        'augment': True,
    }

    if config['augment']:
        augment_name = "image_augment"
    else:
        augment_name = "no_image_augment"

    run_time_str = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')
    name = "{0}_{1}".format(augment_name, run_time_str)

    wandb.init(
        mode="online" if online else "offline",
        project=PROJECT_NAME,
        notes="fashion mnist dataset",
        tags=["cnn", "fashion_mnist", "image_augment"],
        name=name,
        config=config
    )
    print(wandb.config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device {device}.")

    train_data_loader, validation_data_loader, train_transforms = get_fashion_mnist_data(wandb.config.augment)
    train_model = get_fashion_mnist_model()
    train_model.to(device)
    wandb.watch(train_model)

    from torchinfo import summary
    summary(train_model, input_size=(1, 1, 28, 28))

    optimizer = optim.Adam(train_model.parameters(), lr=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)
    trainer = Trainer(
        PROJECT_NAME, train_model, optimizer,
        train_data_loader, validation_data_loader, train_transforms,
        run_time_str, wandb, device, CHECKPOINT_PATH
    )
    trainer.run()
    wandb.finish()


if __name__ == "__main__":
    main()
