import torch
from torchvision import datasets, transforms

from PIL import Image
import numpy as np


def normalize_img(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    return transforms.Normalize(mean=mean, std=std)


def image_tranformer(resize=255, center_crop=224):
    return transforms.Compose([transforms.Resize(resize),
                               transforms.CenterCrop(center_crop),
                               transforms.ToTensor(),
                               normalize_img()])


def load_dir(data_dir='flowers'):

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        'train': transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     normalize_img()]),
        'test': image_tranformer(),
    }

    # Done: Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'test': datasets.ImageFolder(train_dir, transform=data_transforms['test']),
        'valid': datasets.ImageFolder(train_dir, transform=data_transforms['test'])
    }

    # Done: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64),
    }

    return dataloaders, image_datasets
