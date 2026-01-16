import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def get_flowers102_dataloaders(root='./data', batch_size=32, num_workers=2, download=True):
    """
    Returns DataLoaders for train, val, test with an 80-10-10 split.
    Since Flowers102 official split is unbalanced (small train, large test),
    we merge all and re-split to meet the challenge's 80% training requirement.
    """
    
    # Define transforms
    # Level 1: Basic scaling and normalization (ImageNet stats)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224), # Basic crop for baseline
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    # Load ALL data partitions
    # We download 'train' just to ensure data is present, but we will combine everything.
    # Flowers102 has splits: 'train', 'val', 'test'.
    
    print("Loading Flowers102 dataset...")
    # Using 'test' split to init because it's the largest, then we'll verify file paths 
    # Actually, torchvision datasets allow downloading. We need to construct a unified dataset.
    
    # Helper to get all samples
    all_samples = []
    all_labels = []
    
    # We load each split to get the file paths and labels
    for split in ['train', 'val', 'test']:
        ds = datasets.Flowers102(root=root, split=split, download=download)
        all_samples.extend(ds._image_files)
        all_labels.extend(ds._labels)
        
    print(f"Total images found: {len(all_samples)}")
    
    # Create a custom class to hold the merged dataset so we can apply different transforms
    class MergedFlowersDataset(torch.utils.data.Dataset):
        def __init__(self, image_files, labels, transform=None):
            self.image_files = image_files
            self.labels = labels
            self.transform = transform
            from PIL import Image
            self.loader = datasets.folder.default_loader

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            path = self.image_files[idx]
            target = self.labels[idx]
            sample = self.loader(path)
            if self.transform:
                sample = self.transform(sample)
            return sample, target

    # Split indices: 80% Train, 10% Val, 10% Test
    indices = np.arange(len(all_samples))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, shuffle=True, random_state=42, stratify=all_labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, shuffle=True, random_state=42, stratify=np.array(all_labels)[temp_idx])
    
    print(f"Split sizes -> Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Create Subsets/Datasets with appropriate transforms
    # Note: We pass the lists, but we need to create separate dataset instances 
    # to assign different transforms (e.g., if we add augmentation later for train only)
    
    # For efficiency, we just create 3 datasets with the subsets of files 
    train_ds = MergedFlowersDataset(
        [all_samples[i] for i in train_idx], 
        [all_labels[i] for i in train_idx], 
        transform=data_transforms['train']
    )
    
    val_ds = MergedFlowersDataset(
        [all_samples[i] for i in val_idx], 
        [all_labels[i] for i in val_idx], 
        transform=data_transforms['val']
    )
    
    test_ds = MergedFlowersDataset(
        [all_samples[i] for i in test_idx], 
        [all_labels[i] for i in test_idx], 
        transform=data_transforms['test']
    )

    dataloaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }
    
    dataset_sizes = {'train': len(train_ds), 'val': len(val_ds), 'test': len(test_ds)}
    class_names = [str(i) for i in range(102)] # Flowers102 doesn't give text labels easily in basic load, usually just IDs 0-101
    
    return dataloaders, dataset_sizes, class_names
