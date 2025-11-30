from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

CIFAR10_NORMALIZATION = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

def get_data_loader(config: dict, train: bool, batch_size: int = None, shuffle: bool = None) -> DataLoader:
    """
    Get DataLoader for CIFAR dataset.

    Parameters
    ----------
    config : dict
        Configuration dictionary with DataLoader parameters.
    train : bool
        Whether to get the training or test DataLoader.
    batch_size : int, optional
        The batch size for the DataLoader.
    shuffle : bool, optional
        Whether to shuffle the dataset.
    
    Returns
    -------
    data_loader : DataLoader
        The DataLoader for the CIFAR dataset.
    """

    if batch_size is None:
        batch_size = config["batch_size"]

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*CIFAR10_NORMALIZATION)
    ])

    train_transform = transforms.Compose([
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(*CIFAR10_NORMALIZATION)
    ])

    full_train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                                  download=True, transform=train_transform)
    full_val_dataset   = datasets.CIFAR10(root='./data', train=True, 
                                                    download=True, transform=test_transform)
    test_dataset       = datasets.CIFAR10(root='./data', train=False, 
                                                    download=True, transform=test_transform)
    
    if not train:
        dataset = test_dataset
    elif train and config["use_val"]:
        val_proportion = 0.15
        dataset_size = len(full_train_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(val_proportion * dataset_size))
        
        # Shuffle indices manually with a fixed seed
        np.random.seed(42)
        np.random.shuffle(indices)
        
        train_indices, val_indices = indices[split:], indices[:split]

        # Create Subsets using the specific transforms
        dataset = Subset(full_train_dataset, train_indices)
        val_ds  = Subset(full_val_dataset, val_indices)
    else:
        dataset = full_train_dataset

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if shuffle is not None else train,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        persistent_workers=config["persistent_workers"]
    )

    if train and config["use_val"]:
        val_dataloader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
            persistent_workers=config["persistent_workers"]
        )

        return data_loader, val_dataloader

    return data_loader