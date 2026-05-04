"""
CIFAR-10 and CIFAR-100 data loading utilities.

Provides dataset classes and dataloader builders for image classification.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from typing import Dict, Optional, Tuple

# CIFAR-10 normalization stats
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# CIFAR-100 normalization stats
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def get_cifar_transforms(
    dataset: str = "cifar10",
    split: str = "train",
    image_size: int = 32,
) -> transforms.Compose:
    """Get preprocessing transforms for CIFAR datasets.
    
    Args:
        dataset: 'cifar10' or 'cifar100'
        split: 'train' or 'test'
        image_size: Target image size (CIFAR default is 32)
    Returns:
        torchvision transforms composition
    """
    mean = CIFAR10_MEAN if dataset == "cifar10" else CIFAR100_MEAN
    std = CIFAR10_STD if dataset == "cifar10" else CIFAR100_STD

    if split == "train":
        return transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


class CIFARDataset(Dataset):
    """Wrapper around torchvision CIFAR datasets with unified interface."""
    
    def __init__(
        self,
        root: str,
        dataset: str = "cifar10",
        split: str = "train",
        image_size: int = 32,
        download: bool = True,
    ) -> None:
        """
        Args:
            root: Data directory (e.g., './data')
            dataset: 'cifar10' or 'cifar100'
            split: 'train' or 'test'
            image_size: Target image size
            download: Whether to download if not present
        """
        super().__init__()
        self.dataset_name = dataset
        self.split = split
        self.image_size = image_size

        transform = get_cifar_transforms(dataset, split, image_size)

        if dataset == "cifar10":
            self.dataset = datasets.CIFAR10(
                root=root,
                train=(split == "train"),
                transform=transform,
                download=download,
            )
        elif dataset == "cifar100":
            self.dataset = datasets.CIFAR100(
                root=root,
                train=(split == "train"),
                transform=transform,
                download=download,
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset}. Use 'cifar10' or 'cifar100'")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Returns a dict with 'pixel_values' and 'labels'."""
        image, label = self.dataset[idx]
        return {
            "pixel_values": image,   # (3, H, W) tensor
            "labels": torch.tensor(label, dtype=torch.long),
        }


def collate_fn_cifar(batch: list) -> Dict[str, torch.Tensor]:
    """Collate function for CIFAR dataloaders.
    
    Stacks pixel_values and labels into batches.
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)
    labels = torch.stack([item["labels"] for item in batch], dim=0)
    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }


def build_dataloaders(
    data_root: str,
    dataset: str = "cifar10",
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = 32,
    download: bool = True,
    shuffle: bool = True,
    drop_last: bool = False,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, Optional[DataLoader], Dataset, Optional[Dataset]]:
    """Build train and (optionally) validation dataloders.
    
    Args:
        data_root: Root directory for dataset storage
        dataset: 'cifar10' or 'cifar100'
        batch_size: Batch size per GPU
        num_workers: DataLoader workers
        image_size: Target image size
        download: Download dataset if not present
        shuffle: Whether to shuffle training data
        drop_last: Drop incomplete last batch
        distributed: Use distributed sampler
        rank: Process rank for distributed training
        world_size: Total number of processes
    Returns:
        train_loader, val_loader, train_dataset, val_dataset
    """
    # Training dataset
    train_dataset = CIFARDataset(
        root=data_root,
        dataset=dataset,
        split="train",
        image_size=image_size,
        download=download,
    )

    train_sampler = None
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None) and shuffle,
        sampler=train_sampler,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=collate_fn_cifar,
        pin_memory=True,
    )

    # Validation/test dataset
    val_dataset = CIFARDataset(
        root=data_root,
        dataset=dataset,
        split="test",
        image_size=image_size,
        download=download,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collate_fn_cifar,
        pin_memory=True,
    )

    return train_loader, val_loader, train_dataset, val_dataset
