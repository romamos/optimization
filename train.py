import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
# PyTorch TensorBoard support
from torch.utils.tensorboard import (SummaryWriter)

from torch.utils.data import DataLoader
# functions to show an image
import matplotlib.pyplot as plt
import numpy as np
from model import build_model, classes



batch_size = 4
model_path = 'models/cifar_model.pth'


def show_images(trainloader):
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

        # get some random training images

    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))  # Исправлено: добавлен batch_size


def plot_loss(losses, optimizer_n):
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, marker='o', label=f'Loss ({optimizer_n})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss per Epoch ({optimizer_n})')
    plt.legend()
    plt.grid()
    plt.savefig(f'loss_plot_{optimizer_n}.png')
    plt.close()
    print(f"Loss plot saved as 'loss_plot_{optimizer_n}.png'")


def train(optimizer_n):
    print('Load data')
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    # show_images(trainloader)
    net = build_model()
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    if optimizer_n == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    elif optimizer_n == 'Rprop':
        optimizer = optim.Rprop(net.parameters(), lr=0.001)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_n}")  # Добавлена обработка неизвестного оптимизатора

    # Удалена лишняя строка optimizer = optim.SGD(...)

    losses = []
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_loss = 0.0
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
            epoch_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        avg_loss = epoch_loss / len(trainloader)
        losses.append(avg_loss)
    plot_loss(losses, optimizer_n)
    print('Finished Training')
    os.makedirs('models', exist_ok=True)
    torch.save(net.state_dict(), f'models/{optimizer_n}_model.pth')


# writer = SummaryWriter('SGD')

if __name__ == '__main__':  # Исправлено: name
    train(optimizer_n='SGD')
    train(optimizer_n='Rprop')