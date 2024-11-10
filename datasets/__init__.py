import os

from torchvision import transforms
from torchvision.datasets import CIFAR10

def rescale(x):
    return x * 2 - 1

def get_dataset(args, config):
    transform = []
    transform.append(transforms.Resize(config.data.image_size))
    if config.data.random_flip:
        transform.append(transforms.RandomHorizontalFlip())
    transform.append(transforms.ToTensor())
    if config.data.rescaled:
        transform.append(transforms.Lambda(rescale))
    transform = transforms.Compose(transform)

    dataset_path = os.path.join(args.exp, 'datasets', 'cifar10')
    dataset = CIFAR10(dataset_path, train=True, download=True, transform=transform)

    return dataset
