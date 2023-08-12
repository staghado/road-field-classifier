import os
import glob
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# Ignore warnings
warnings.filterwarnings("ignore")

# Map class names to numeric labels
CLASS_MAP = {
    'field': 0,
    'road': 1
}

class ClassificationDataset(Dataset):
    """Custom PyTorch dataset for image classification."""

    def __init__(self, image_paths, transform=None):
        """
        Initializes the dataset.

        Args:
            image_paths (list): List of paths to image files.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.transform = transform
        self.image_paths = image_paths

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing 'image' and 'label'.
        """
        img_name = self.image_paths[idx]
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        # Determine label based on directory name
        label = CLASS_MAP['field'] if img_name.split('/')[-2] == 'fields' else CLASS_MAP['road']
        sample = {'image': image, 'label': label}

        return sample
