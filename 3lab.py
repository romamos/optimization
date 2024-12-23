import os

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

from train import train # Assuming 'train.py' exists in the same directory

DEVICE = torch.device("cpu")
BATCHSIZE = 128
CLASSES = 10
DIR = os.getcwd()
EPOCHS = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10


def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 3) # Suggesting the number of layers

    conv_layers = []
    conv_layers.append(nn.Conv2d(3, 16, kernel_size=3, padding=1))
    conv_layers.append(nn.ReLU())
    conv_layers.append(nn.MaxPool2d(2, 2))

    model_layers = []
    model_layers.extend(conv_layers)
    model_layers.append(nn.Flatten())


    in_features = 16 * 16 * 16
    for i in range(n_layers): # Using n_layers here
        out_features = trial.suggest_int("n_units_l{}".format(i), 32, 512)
        model_layers.append(nn.Linear(in_features, out_features))
        model_layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5) # Dropout for each layer
        model_layers.append(nn.Dropout(p))

        in_features = out_features

    model_layers.append(nn.Linear(in_features, CLASSES))
    model_layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*model_layers)



def get_cifar10():
    # Load CIFAR10 dataset.
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(DIR, train=True, download=True, transform=transform),
        batch_size=BATCHSIZE,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(DIR, train=False, transform=transform),
        batch_size=BATCHSIZE,
        shuffle=True,
    )

    return train_loader, valid_loader


def objective(trial):
    # Generate the model.
    model = define_model(trial).to(DEVICE)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]) # Changed here
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the CIFAR10 dataset.
    train_loader, valid_loader = get_cifar10()

    # Training of the model.
    for epoch in range(EPOCHS):
        # ... (rest of the training and validation code remains the same)

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100) # or any other number of trials

