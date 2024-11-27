import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import SimpleITK as sitk
import numpy as np


class CTMulticlassDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        PyTorch Dataset for 3D CT images and their corresponding multiclass masks.
        Args:
            image_dir (str): Directory containing CT images.
            mask_dir (str): Directory containing multiclass masks.
            transform (callable, optional): Transformations to apply to both images and masks.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nrrd')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nrrd')])
        self.transform = transform

        print(f"Found {len(self.image_files)} images and {len(self.mask_files)} masks.")
        assert len(self.image_files) == len(self.mask_files), \
            "Number of images and masks must match!"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load paths
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        # Read image and mask
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)

        # Convert to NumPy arrays
        image_array = sitk.GetArrayFromImage(image).astype(np.float32)
        mask_array = sitk.GetArrayFromImage(mask).astype(np.int64)

        # Normalize the image (scale to [0, 1])
        image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))

        # Apply transformations
        if self.transform:
            image_array, mask_array = self.transform(image_array, mask_array)

        # Convert to PyTorch tensors
        image_tensor = torch.tensor(image_array).unsqueeze(0)  # Add channel dimension (C, D, H, W)
        mask_tensor = torch.tensor(mask_array)

        return image_tensor, mask_tensor


def random_augmentation(image, mask):
    """
    Apply random augmentations to the image and mask.
    """
    # Random horizontal flip
    if np.random.rand() > 0.5:
        image = np.flip(image, axis=2).copy()
        mask = np.flip(mask, axis=2).copy()

    # Random vertical flip
    if np.random.rand() > 0.5:
        image = np.flip(image, axis=1).copy()
        mask = np.flip(mask, axis=1).copy()

    return image, mask


def prepare_data_loaders(image_dir, mask_dir, batch_size=2, train_ratio=0.7, val_ratio=0.2):
    """
    Prepare PyTorch DataLoaders for training, validation, and testing.
    Args:
        image_dir (str): Directory containing CT images.
        mask_dir (str): Directory containing multiclass masks.
        batch_size (int): Batch size for DataLoaders.
        train_ratio (float): Proportion of training data.
        val_ratio (float): Proportion of validation data.
    Returns:
        train_loader, val_loader, test_loader: PyTorch DataLoaders.
    """
    dataset = CTMulticlassDataset(image_dir=image_dir, mask_dir=mask_dir, transform=random_augmentation)

    # Split dataset
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# Set directories
image_dir = '/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/images'
mask_dir = '/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/multiclass_masks'

# Prepare DataLoaders
train_loader, val_loader, test_loader = prepare_data_loaders(image_dir, mask_dir, batch_size=2)

# Example usage
for images, masks in train_loader:
    print(f"Image batch shape: {images.shape}")  # Should be (batch_size, 1, D, H, W)
    print(f"Mask batch shape: {masks.shape}")    # Should be (batch_size, D, H, W)
    break