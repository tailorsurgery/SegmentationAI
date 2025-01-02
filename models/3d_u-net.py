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

# Visualization function
def visualize_patch(image_patch, mask_patch):
    """
    Visualize the middle slice of a 3D image and mask patch.
    """
    # Calculate the middle slice in the z-dimension
    z_middle = (image_patch.shape[1] // 2)+20

    # Extract the middle slice for visualization
    image_slice = image_patch[0, z_middle, :, :]  # First channel, middle z-slice
    mask_slice = mask_patch[z_middle, :, :]  # Middle z-slice of the mask

    # Plot the image and mask side-by-side
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_slice, cmap="gray")
    plt.title("Image Patch (Middle Slice)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask_slice, cmap="jet", vmin=0, vmax=5)
    plt.title("Mask Patch (Middle Slice)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# Extract patches from the volume
def extract_patches(volume, patch_size=(128, 128, 128), stride=(64, 64, 64)):
    _, d, h, w = volume.shape

    pd, ph, pw = patch_size
    sd, sh, sw = stride

    patches = []
    for z in range(0, d - pd + 1, sd):
        for y in range(0, h - ph + 1, sh):
            for x in range(0, w - pw + 1, sw):
                patches.append((z, z+pd, y, y+ph, x, x+pw))
    print(f"*Volume Shape {volume.shape}")
    print(f"*Patch Size {patch_size}")
    print(f"*Stride {stride}")
    print(f"*Number of Patches {len(patches)}")
    return patches

# Evaluate model performance
def evaluate_model(model, test_loader, device, num_classes):
    print("*********************Starting evaluation...*********************")
    start_time = time.time()
    model.eval()
    total_dice_scores = np.zeros(num_classes)
    num_batches = 5

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
def visualize_predictions(model, test_loader, device, num_classes, num_samples=20):
    print("*********************Starting visualization...*********************")
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


def evaluate_and_visualize_model(
    model, test_loader, device, num_classes, visualize=False, num_samples=5
):
    """
    Evaluate the model and optionally visualize predictions.

    Args:
        model (torch.nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run the evaluation (CPU or GPU).
        num_classes (int): Number of segmentation classes.
        visualize (bool): Whether to visualize predictions. Default is False.
        num_samples (int): Number of samples to visualize. Default is 5.

    Returns:
        np.ndarray: Average Dice scores for each class.
    """
    print("*********************Starting evaluation...*********************")
    start_time = time.time()
    model.eval()
    total_dice_scores = np.zeros(num_classes)
    num_batches = 0

    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(test_loader, desc="Evaluating")):
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

            # Visualize predictions if enabled
            if visualize and idx < num_samples:
                for i in range(min(num_samples, images.size(0))):
                    plt.figure(figsize=(12, 4))
                    plt.subplot(1, 3, 1)
                    plt.imshow(images[i, 0].cpu().numpy(), cmap="gray")
                    plt.title("Input Image")
                    plt.axis("off")

                    plt.subplot(1, 3, 2)
                    plt.imshow(masks[i].cpu().numpy(), cmap="jet", vmin=0, vmax=num_classes - 1)
                    plt.title("Ground Truth Mask")
                    plt.axis("off")

                    plt.subplot(1, 3, 3)
                    plt.imshow(predictions[i].cpu().numpy(), cmap="jet", vmin=0, vmax=num_classes - 1)
                    plt.title("Predicted Mask")
                    plt.axis("off")

                    plt.tight_layout()
                    plt.show()

    avg_dice_scores = total_dice_scores / num_batches
    elapsed_time = time.time() - start_time

    print(f"Evaluation completed in {elapsed_time:.2f} seconds.")
    print(f"Average Dice Scores (per class): {avg_dice_scores}")

    return avg_dice_scores

# Predict single volume
def predict(model, image, device):
    print("*********************Starting prediction...*********************")
    start_time = time.time()
    model.eval()
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    print(f"Image tensor: {image_tensor[0].item()}")
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    elapsed_time = time.time() - start_time
    print(f"Prediction completed in {elapsed_time:.2f} seconds.")
    return prediction

# Dataset class
class PatchBasedDataset(Dataset):
    def __init__(self, image_dir, mask_dir, patch_size=(128, 128, 128), stride=(64, 64, 64), num_cases=None, visualize=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nrrd')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nrrd')])
        self.patch_size = patch_size
        self.stride = stride
        self.visualize = visualize

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
        print(f"    With masks: {os.path.basename(mask_path)}")
        print(f"Accessing patch {idx} from image {os.path.basename(self.image_files[img_idx])}")

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

        if self.visualize:
            visualize_patch(image_patch, mask_patch)

        image_tensor = torch.tensor(image_patch, dtype=torch.float32)
        mask_tensor = torch.tensor(mask_patch, dtype=torch.int64)
        # print(f"Image tensor: {image_tensor}")
        # print(f"Mask tensor: {mask_tensor[0].item()}")

        return image_tensor, mask_tensor


# 3D U-Net model
class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()

        # Encoder blocks
        self.encoder1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool3d(2)

        self.encoder2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.encoder3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Middle block
        self.middle = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Up-convolution blocks
        self.upconv3 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(
            nn.Conv3d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Final output layer
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path with skip connections
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))

        # Middle block
        mid = self.middle(self.pool(enc3))

        # Decoder path
        dec3 = self.upconv3(mid)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)

# Training function
def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-3):
    print("*********************Starting training...*********************")
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
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pth")
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
    region = 'arms'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.dirname(script_dir)
    # go back one folder to get to the root folder
    print(script_dir)
    image_dir = script_dir + '/data/segmentai_dataset/images/' + region
    mask_dir = script_dir + '/data/segmentai_dataset/multiclass_masks/' + region
    dataset_dir = script_dir + '/data/segmentai_dataset/processed/' + region + 'sss_processed_dataset.pth'
    model_save_path = script_dir + '/models/unet/' + region + '_3d_unet_model'

    if os.path.exists(dataset_dir):
        print("Loading preprocessed dataset...")
        full_dataset = torch.load(dataset_dir)
    else:
        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            print("Image dir: ", image_dir)
            print("Mask dir: ", mask_dir)
            raise FileNotFoundError("Image or mask directory not found!")
            # TODO: Download the dataset from gcloud storage
            # https://console.cloud.google.com/storage/browser/segmentai_dataset
        full_dataset = PatchBasedDataset(image_dir, mask_dir)
        torch.save(full_dataset, dataset_dir)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(1, 6).to(device)

    training = False
    if training:
        # TODO: Change number of epochs more than 2 (20??)
        train_losses, val_losses = train_model(model, train_loader, val_loader, device, epochs=20, lr=1e-3)
        torch.save(model.state_dict(), f"{model_save_path}_training.pth")
        print(f"Model saved to {model_save_path}_training.pth")

    evaluate = True
    if evaluate:
        '''evaluate_model(model, val_loader, device, 6)
        visualize_predictions(model, val_loader, device, 6)'''
        evaluate_and_visualize_model(model, val_loader, device, 6, visualize=False, num_samples=5)
        torch.save(model.state_dict(), f"{model_save_path}_evaluation.pth")
        print(f"Model saved to {model_save_path}_training.pth")
