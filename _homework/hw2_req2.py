import os
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime
from argparse import ArgumentParser, BooleanOptionalAction
import wandb


class TitanicDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        feature = self.X[idx]
        target = self.y[idx]
        return {'input': feature, 'target': target}

    def __str__(self):
        return "Data Size: {0}, Input Shape: {1}, Target Shape: {2}".format(
            len(self.X), self.X.shape, self.y.shape
        )


class TitanicTestDataset(Dataset):
    def __init__(self, X):
        self.X = torch.FloatTensor(X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        feature = self.X[idx]
        return {'input': feature}

    def __str__(self):
        return "Data Size: {0}, Input Shape: {1}".format(
            len(self.X), self.X.shape
        )


class TitanicModel(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(n_input, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, n_output),
        )

    def forward(self, x):
        x = self.model(x)
        return x


def get_model_and_optimizer():
    model = TitanicModel(n_input=11, n_output=2)
    optimizer = optim.SGD(model.parameters(), lr=wandb.config.learning_rate)

    return model, optimizer


def get_data():
    PATH = os.path.join(os.path.pardir, "_00_data", "0_titanic")
    train_data_path = os.path.join(PATH, "train.csv")
    test_data_path = os.path.join(PATH, "test.csv")
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    all_df = pd.concat([train_df, test_df], sort=False)
    all_df = preprocess(all_df)

    train_X = all_df[~all_df["Survived"].isnull()].drop("Survived", axis=1).reset_index(drop=True)
    train_y = train_df["Survived"]

    test_X = all_df[all_df["Survived"].isnull()].drop("Survived", axis=1).reset_index(drop=True)

    dataset = TitanicDataset(train_X.values, train_y.values)
    train_dataset, validation_dataset = random_split(dataset, [0.8, 0.2])
    test_dataset = TitanicTestDataset(test_X.values)

    train_data_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
    validation_data_loader = DataLoader(validation_dataset, batch_size=len(validation_dataset))
    test_data_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    return train_data_loader, validation_data_loader, test_data_loader


def preprocess(dataset):
    print("Preprocessing dataset...")
    # adjust fare
    Fare_mean = dataset[["Pclass", "Fare"]].groupby("Pclass").mean().reset_index()
    Fare_mean.columns = ["Pclass", "Fare_mean"]
    dataset = pd.merge(dataset, Fare_mean, on="Pclass", how="left")
    dataset.loc[(dataset["Fare"].isnull()), "Fare"] = dataset["Fare_mean"]

    # adjust name
    name_df = dataset["Name"].str.split("[,.]", n=2, expand=True)
    name_df.columns = ["family_name", "honorific", "name"]
    name_df["family_name"] = name_df["family_name"].str.strip()
    name_df["honorific"] = name_df["honorific"].str.strip()
    name_df["name"] = name_df["name"].str.strip()
    dataset = pd.concat([dataset, name_df], axis=1)

    # adjust age
    honorific_age_mean = dataset[["honorific", "Age"]].groupby("honorific").median().round().reset_index()
    honorific_age_mean.columns = ["honorific", "honorific_age_mean", ]
    dataset = pd.merge(dataset, honorific_age_mean, on="honorific", how="left")
    dataset.loc[(dataset["Age"].isnull()), "Age"] = dataset["honorific_age_mean"]
    dataset = dataset.drop(["honorific_age_mean"], axis=1)

    # derive columns
    dataset["family_num"] = dataset["Parch"] + dataset["SibSp"]
    dataset.loc[dataset["family_num"] == 0, "alone"] = 1
    dataset["alone"].fillna(0, inplace=True)

    # drop redundant columns
    dataset = dataset.drop(["PassengerId", "Name", "family_name", "name", "Ticket", "Cabin"], axis=1)

    # reduce honorific variants
    dataset.loc[
        ~(
                (dataset["honorific"] == "Mr") |
                (dataset["honorific"] == "Miss") |
                (dataset["honorific"] == "Mrs") |
                (dataset["honorific"] == "Master")
        ),
        "honorific"
    ] = "other"
    dataset["Embarked"].fillna("missing", inplace=True)

    # encode category samples
    category_features = dataset.columns[dataset.dtypes == "object"]
    from sklearn.preprocessing import LabelEncoder
    for category_feature in category_features:
        le = LabelEncoder()
        if dataset[category_feature].dtypes == "object":
            le = le.fit(dataset[category_feature])
            dataset[category_feature] = le.transform(dataset[category_feature])

    return dataset


def train(model, optimizer, train_data_loader, validation_data_loader):
    print("Training model...")

    n_epochs = wandb.config.epochs
    loss_fn = nn.MSELoss()  # Use a built-in loss function
    next_print_epoch = 100

    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        num_trains = 0
        for train_batch in train_data_loader:
            output_train = model(train_batch['input'])
            loss = loss_fn(output_train, train_batch['target'])
            loss_train += loss.item()
            num_trains += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_validation = 0.0
        num_validations = 0
        with torch.no_grad():
            for validation_batch in validation_data_loader:
                output_validation = model(validation_batch['input'])
                loss = loss_fn(output_validation, validation_batch['target'])
                loss_validation += loss.item()
                num_validations += 1

        wandb.log({
            "Epoch": epoch,
            "Training loss": loss_train / num_trains,
            "Validation loss": loss_validation / num_validations
        })

        if epoch >= next_print_epoch:
            print(
                f"Epoch {epoch}, "
                f"Training loss {loss_train / num_trains:.4f}, "
                f"Validation loss {loss_validation / num_validations:.4f}"
            )
            next_print_epoch += 100


def test(test_data_loader):
    print("Testing model...")
    batch = next(iter(test_data_loader))
    print("{0}".format(batch['input'].shape))
    my_model = TitanicModel(n_input=11, n_output=2)
    output_batch = my_model(batch['input'])
    prediction_batch = torch.argmax(output_batch, dim=1)
    for idx, prediction in enumerate(prediction_batch, start=892):
        print(idx, prediction.item())


def main(args):
    current_time_str = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')
    config = {
        'epochs': args.epoch,
        'batch_size': args.batch_size,
        'learning_rate': 1e-3,
        'n_hidden_unit_list': [20, 20],
    }

    wandb.init(
        mode="online" if args.wandb else "disabled",
        project="my_model_training",
        notes="My first wandb experiment",
        tags=["my_model", "california_housing"],
        name=current_time_str,
        config=config
    )
    print(args)
    print(wandb.config)

    train_data_loader, validation_data_loader, test_data_loader = get_data()
    model, optimizer = get_model_and_optimizer()

    wandb.watch(model)
    train(model, optimizer, train_data_loader, validation_data_loader)
    wandb.finish()
    test(test_data_loader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--wandb", action=BooleanOptionalAction, default=False, help="True or False"
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=512, help="Batch size (default: 512)"
    )
    parser.add_argument(
        "-e", "--epoch", type=int, default=1_000, help="Number of training epochs (default: 1000)"
    )
    args = parser.parse_args()

    main(args)
