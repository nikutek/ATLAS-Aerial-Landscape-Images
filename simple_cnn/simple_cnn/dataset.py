import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import yaml
from pathlib import Path


def get_data_loaders(config: dict):
    """
    Parameters
    -----------
    config : dict (loaded config.yaml)
    
    Return
    -------
    dataloaders : dict {'train': ..., 'val': ..., 'test' ...}
    dataset_sizes : dict  {'train': int, 'val': int, 'test': int}
    class_names   : list[str]
    """
    data_dir = Path(config['data']['processed_dir'])
    img_size = config['data']['img_size']
    batch = config['data']['batch_size']
    workers = config['data']['num_workers']
    mean = config['data']['mean']
    std = config['data']['std']
    
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.2),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    base_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    data_transforms = {
        'train': train_transforms,
        'val':   base_transforms,
        'test':  base_transforms,
    }
    
    image_datasets = {
        split: datasets.ImageFolder(data_dir / split, data_transforms[split])
        for split in ['train', 'val', 'test']
    }
    
    dataloaders = {
        split: DataLoader(
            image_datasets[split],
            batch_size  = batch,
            shuffle     = (split == 'train'),
            num_workers = workers,
            pin_memory  = True,
        )
        for split in ['train', 'val', 'test']
    }
    
    dataset_sizes = {split: len(image_datasets[split]) for split in ['train', 'val', 'test']}
    class_names   = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names
    