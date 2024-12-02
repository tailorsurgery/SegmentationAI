import os
import time
import torch
import gc
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
import random

# Extract patches from the volume
def extract_patches(volume, patch_size=(128, 128, 128), stride=(64, 64, 64)):
    _, D, H, W = volume.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    patches = []
    for z in range(0, D - pd + 1, sd):
        for y in range(0, H - ph + 1, sh):
            for x in range(0, W - pw + 1, sw):
                patches.append((z, z+pd, y, y+ph, x, x+pw))
    return patches

# Evaluate model performance
def evaluate_model(model, test_loader, device, num_classes):
    print("Starting evaluation...")
    start_time = time.time()
    model.eval()
    total_dice_scores = np.zeros(num_classes)
    num_batches = 0

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            # Compute Dice scores for each class
            for cls in range(1, num_classes):  # Skip background (class 0)
                intersection = ((predictions == cls) & (masks == cls)).sum().item()
                union = ((predictions == cls) | (masks == cls)).sum().item()
                dice_score = 2.0 * intersection / (union + 1e-6)
                total_dice_scores[cls] += dice_score

            num_batches += 1

    avg_dice_scores = total_dice_scores / num_batches
    elapsed_time = time.time() - start_time
    print(f"Evaluation completed in {elapsed_time:.2f} seconds.")
    print(f"Average Dice Scores (per class): {avg_dice_scores}")
    return avg_dice_scores

# Visualize predictions
def visualize_predictions(model, test_loader, device, num_classes, num_samples=5):
    print("Starting visualization...")
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            # Visualize predictions
            for i in range(min(num_samples, images.size(0))):
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                plt.imshow(images[i, 0].cpu().numpy(), cmap="gray")
                plt.title("Input Image")
                plt.subplot(1, 3, 2)
                plt.imshow(masks[i].cpu().numpy(), cmap="jet", vmin=0, vmax=num_classes-1)
                plt.title("Ground Truth Mask")
                plt.subplot(1, 3, 3)
                plt.imshow(predictions[i].cpu().numpy(), cmap="jet", vmin=0, vmax=num_classes-1)
                plt.title("Predicted Mask")
                plt.show()

            if idx + 1 >= num_samples:
                break

    elapsed_time = time.time() - start_time
    print(f"Visualization completed in {elapsed_time:.2f} seconds.")

# Predict single volume
def predict(model, image, device):
    print("Starting prediction...")
    start_time = time.time()
    model.eval()
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    elapsed_time = time.time() - start_time
    print(f"Prediction completed in {elapsed_time:.2f} seconds.")
    return prediction

# Dataset class
class PatchBasedDataset(Dataset):
    def __init__(self, image_dir, mask_dir, patch_size=(128, 128, 128), stride=(64, 64, 64), num_cases=2):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nrrd')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nrrd')])
        self.patch_size = patch_size
        self.stride = stride

        print(f"Found {len(self.image_files)} images and {len(self.mask_files)} masks.")
        assert len(self.image_files) == len(self.mask_files), \
            "Number of images and masks must match!"

        if num_cases is not None:
            selected_indices = random.sample(range(len(self.image_files)), min(num_cases, len(self.image_files)))
            self.image_files = [self.image_files[i] for i in selected_indices]
            self.mask_files = [self.mask_files[i] for i in selected_indices]

        # Precompute patch indices for each image
        self.patch_indices = []
        for img_file in self.image_files:
            image_path = os.path.join(self.image_dir, img_file)
            image = sitk.ReadImage(image_path)
            image_array = sitk.GetArrayFromImage(image).astype(np.float32)
            image_array = image_array[np.newaxis, ...]  # Add channel dimension
            patches = extract_patches(image_array, self.patch_size, self.stride)
            self.patch_indices.append(patches)

    def __len__(self):
        return sum(len(patches) for patches in self.patch_indices)

    def __getitem__(self, idx):
        cumulative_patches = 0
        for img_idx, patches in enumerate(self.patch_indices):
            if idx < cumulative_patches + len(patches):
                patch_idx = idx - cumulative_patches
                patch_coords = patches[patch_idx]
                break
            cumulative_patches += len(patches)

        image_path = os.path.join(self.image_dir, self.image_files[img_idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[img_idx])

        print(f"Processing case: {os.path.basename(image_path)}")

        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
        image_array = sitk.GetArrayFromImage(image).astype(np.float32)
        mask_array = sitk.GetArrayFromImage(mask).astype(np.int64)

        image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
        image_array = image_array[np.newaxis, ...]

        mask_array = np.clip(mask_array, 0, 5)

        z_start, z_end, y_start, y_end, x_start, x_end = patch_coords
        image_patch = image_array[:, z_start:z_end, y_start:y_end, x_start:x_end]
        mask_patch = mask_array[z_start:z_end, y_start:y_end, x_start:x_end]

        image_tensor = torch.tensor(image_patch, dtype=torch.float32)
        mask_tensor = torch.tensor(mask_patch, dtype=torch.int64)

        return image_tensor, mask_tensor

# 3D U-Net model
class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(kernel_size=2)
        )
        self.middle = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, out_channels, kernel_size=1)
        )

    def forward(self, x):
        enc = self.encoder(x)
        mid = self.middle(enc)
        dec = self.decoder(mid)
        return dec

# Training function
def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-3):
    print("Starting training...")
    start_time = time.time()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc="Training"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            del images, masks, outputs
            gc.collect()
            torch.cuda.empty_cache()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        print(f"Training Loss: {train_loss:.4f}")

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                running_val_loss += loss.item()

                del images, masks, outputs, loss
                gc.collect()
                torch.cuda.empty_cache()

        val_loss = running_val_loss / len(val_loader)
        val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Epoch {epoch + 1} completed in {time.time() - epoch_start:.2f} seconds.")
        gc.collect()
        torch.cuda.empty_cache()


    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds.")
    return train_losses, val_losses

# Main
if __name__ == "__main__":
    # Paths
    image_dir = '/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/images'
    mask_dir = '/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/multiclass_masks'
    dataset_dir = '/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/processed/processed_dataset.pth'
    model_save_path = '/Users/samyakarzazielbachiri/Documents/SegmentationAI/models/3d_unet/3d_unet_model.pth'

    if os.path.exists(dataset_dir):
        print("Loading preprocessed dataset...")
        full_dataset = torch.load(dataset_dir)
    else:
        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            raise FileNotFoundError("Image or mask directory not found!")
            # TODO: Download the dataset from gcloud
            # https://console.cloud.google.com/storage/browser/segmentai_dataset
        full_dataset = PatchBasedDataset(image_dir, mask_dir)
        torch.save(full_dataset, dataset_dir)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True, num_workers=10//2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False, num_workers=10//2, pin_memory=True)

    device = torch.device("cpu")
    model = UNet3D(1, 6).to(device)

    train_losses, val_losses = train_model(model, train_loader, val_loader, device, epochs=1, lr=1e-3)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    test_loader = val_loader  # For simplicity
    evaluate_model(model, test_loader, device, 6)
    visualize_predictions(model, test_loader, device, 6)