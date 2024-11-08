import os

from torchvision import transforms
from torchvision.datasets import CIFAR10

def get_dataset(args, config):
    if config.data.random_flip:
        train_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.ToTensor()
        ])
    test_transform = transforms.Compose([
        transforms.Resize(config.data.image_size),
        transforms.ToTensor()
    ])

    train_dataset_path = os.path.join(args.exp, 'datasets', 'cifar10_train')
    train_dataset = CIFAR10(train_dataset_path, train=True, download=True, transform=train_transform)
    test_dataset_path = os.path.join(args.exp, 'datasets', 'cifar10_test')
    test_dataset = CIFAR10(test_dataset_path, train=False, download=True, transform=test_transform)

    return train_dataset, test_dataset
