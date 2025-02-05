import os
import time
import torch
import gc
import numpy as np
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import random
import signal
import sys


# Patch extraction function with validation
def extract_patches(volume, patch_size=(128, 128, 128), stride=(128, 128, 128)):
    _, d, h, w = volume.shape

    pd, ph, pw = patch_size
    sd, sh, sw = stride

    patches = []
    for z in range(0, d - pd + 1, sd):
        for y in range(0, h - ph + 1, sh):
            for x in range(0, w - pw + 1, sw):
                patch = volume[:, z:z + pd, y:y + ph, x:x + pw]
                if patch.shape == (1, pd, ph, pw):  # Ensure valid shape
                    patches.append((z, z + pd, y, y + ph, x, x + pw))
                else:
                    print(f"Skipping invalid patch with shape {patch.shape}")
    print(f"*Volume Shape {volume.shape}")
    print(f"*Patch Size {patch_size}")
    print(f"*Stride {stride}")
    print(f"*Number of Patches {len(patches)}")
    return patches


# Custom collate function for DataLoader
def custom_collate(batch):
    images, masks = zip(*batch)

    # Find maximum dimensions in the batch
    max_depth = max(img.shape[1] for img in images)
    max_height = max(img.shape[2] for img in images)
    max_width = max(img.shape[3] for img in images)

    # Create padded tensors
    padded_images = torch.zeros((len(images), 1, max_depth, max_height, max_width), dtype=torch.float32)
    padded_masks = torch.zeros((len(masks), max_depth, max_height, max_width), dtype=torch.long)

    for i, (img, mask) in enumerate(zip(images, masks)):
        padded_images[i, :, :img.shape[1], :img.shape[2], :img.shape[3]] = img
        padded_masks[i, :mask.shape[0], :mask.shape[1], :mask.shape[2]] = mask

    return padded_images, padded_masks


# Dataset class with error handling and debug statements
class PatchBasedDataset(Dataset):
    def __init__(self, image_dir, mask_dir, patch_size=(128, 128, 128), stride=(128, 128, 128), num_cases=None,
                 visualize=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nrrd')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nrrd')])
        self.patch_size = patch_size
        self.stride = stride
        self.visualize = visualize

        print(f"Found {len(self.image_files)} images and {len(self.mask_files)} masks.")
        assert len(self.image_files) == len(self.mask_files), "Number of images and masks must match!"

        if num_cases is not None:
            selected_indices = random.sample(range(len(self.image_files)), min(num_cases, len(self.image_files)))
            self.image_files = [self.image_files[i] for i in selected_indices]
            self.mask_files = [self.mask_files[i] for i in selected_indices]

        # Precompute patch indices for each image
        self.patch_indices = []
        for img_file, msk_file in zip(self.image_files, self.mask_files):
            image_path = os.path.join(self.image_dir, img_file)
            mask_path = os.path.join(self.mask_dir, msk_file)

            # Read them with SimpleITK
            image_sitk = sitk.ReadImage(image_path)
            mask_sitk = sitk.ReadImage(mask_path)

            image_array = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
            mask_array = sitk.GetArrayFromImage(mask_sitk).astype(np.int64)

            # Normalization for the image
            min_val, max_val = image_array.min(), image_array.max()
            if max_val > min_val:
                image_array = (image_array - min_val) / (max_val - min_val)

            # Add channel dimension: (1, D, H, W)
            image_array = image_array[np.newaxis, ...]

            # Extract patch coords for the image
            coords_list = extract_patches(
                volume=image_array,
                patch_size=self.patch_size,
                stride=self.stride
            )

            valid_coords = []
            for (z_start, z_end, y_start, y_end, x_start, x_end) in coords_list:
                # Slice the mask (which is shape (D,H,W)) to see if it's all zeros
                mask_patch = mask_array[z_start:z_end, y_start:y_end, x_start:x_end]

                if mask_patch.sum() == 0:
                    # skip if mask patch has no foreground
                    continue

                # keep this coordinate
                valid_coords.append((z_start, z_end, y_start, y_end, x_start, x_end))

            # Now we store only the coords that have some foreground
            self.patch_indices.append(valid_coords)

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

        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)

        image_array = sitk.GetArrayFromImage(image).astype(np.float32)
        mask_array = sitk.GetArrayFromImage(mask).astype(np.int64)

        image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
        image_array = image_array[np.newaxis, ...]

        z_start, z_end, y_start, y_end, x_start, x_end = patch_coords
        image_patch = image_array[:, z_start:z_end, y_start:y_end, x_start:x_end]
        mask_patch = mask_array[z_start:z_end, y_start:y_end, x_start:x_end]

        image_tensor = torch.tensor(image_patch, dtype=torch.float32)
        mask_tensor = torch.tensor(mask_patch, dtype=torch.long)

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


class AttentionBlock3D(nn.Module):
    """
    A 3D Attention Gate as described in 'Attention U-Net: Learning Where to Look for the Pancreas'
    (https://arxiv.org/abs/1804.03999).

    Typically:
      - g: gating signal (from decoder)
      - x: features from encoder (skip connection)
    """
    def __init__(self, in_channels_g, in_channels_x, inter_channels):
        super(AttentionBlock3D, self).__init__()

        # W_g: transform gating signal
        self.W_g = nn.Sequential(
            nn.Conv3d(in_channels_g, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(inter_channels)
        )

        # W_x: transform skip connection
        self.W_x = nn.Sequential(
            nn.Conv3d(in_channels_x, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(inter_channels)
        )

        # psi: combines the two transforms and outputs alpha mask
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        # ReLU for gating
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating (decoder)
        # x: skip connection (encoder)
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # resample gating if needed (e.g., if shapes differ)
        # But typically you ensure shapes match by design (e.g., with MaxPool / UpConv).
        # For example, if x is bigger, you might do a resize or interpolation here.
        # g1 = F.interpolate(g1, size=x1.shape[2:], mode='trilinear', align_corners=False)

        # combine
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)      # shape: (N,1,D,H,W)

        # Multiply skip connection by attention map
        out = x * psi
        return out

class UNet3D_Attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D_Attention, self).__init__()

        # --- [Encoder blocks] ---
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

        # --- [Middle block] ---
        self.middle = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # --- [Attention Gates for skip connections] ---
        # We create an AttentionBlock for each skip (encoder1->decoder1, encoder2->decoder2, encoder3->decoder3)
        # gating channel ~ channel of decoder feature
        # x channel ~ channel of skip connection
        # inter_channels can be ~ half of x's channels or something similar

        self.att3 = AttentionBlock3D(in_channels_g=256, in_channels_x=256, inter_channels=256)
        self.att2 = AttentionBlock3D(in_channels_g=128, in_channels_x=128, inter_channels=64)
        self.att1 = AttentionBlock3D(in_channels_g=64, in_channels_x=64, inter_channels=32)

        # --- [Decoder blocks with up-convs] ---
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

        # --- [Final output layer] ---
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # --- Encoder path ---
        enc1 = self.encoder1(x)              # shape: (64, D, H, W)
        enc2 = self.encoder2(self.pool(enc1)) # shape: (128, D/2, H/2, W/2)
        enc3 = self.encoder3(self.pool(enc2)) # shape: (256, D/4, H/4, W/4)

        # --- Middle ---
        mid = self.middle(self.pool(enc3))    # shape: (512, D/8, H/8, W/8)

        # --- Decoder path with Attention ---

        # 1) Decoder block at level 3
        dec3 = self.upconv3(mid)  # upsample from 512 -> 256
        # apply attention gate to enc3
        enc3_att = self.att3(g=dec3, x=enc3)  # returns refined skip
        # concat
        dec3 = torch.cat([dec3, enc3_att], dim=1)
        dec3 = self.decoder3(dec3)

        # 2) Decoder block at level 2
        dec2 = self.upconv2(dec3) # upsample from 256 -> 128
        enc2_att = self.att2(g=dec2, x=enc2)
        dec2 = torch.cat([dec2, enc2_att], dim=1)
        dec2 = self.decoder2(dec2)

        # 3) Decoder block at level 1
        dec1 = self.upconv1(dec2) # upsample from 128 -> 64
        enc1_att = self.att1(g=dec1, x=enc1)
        dec1 = torch.cat([dec1, enc1_att], dim=1)
        dec1 = self.decoder1(dec1)

        # --- Final ---
        output = self.final_conv(dec1)
        return output

# GPU usage monitoring function
def log_gpu_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 2
        reserved = torch.cuda.memory_reserved() / 1024 ** 2
        print(f"GPU Memory: Allocated={allocated:.2f} MB, Reserved={reserved:.2f} MB")


def compute_iou(pred, target, num_classes=2, present_classes=None):
    if present_classes is None:
        present_classes = list(range(1, num_classes))

    ious = []
    for cls in present_classes:
        pred_mask = (pred == cls)
        tgt_mask = (target == cls)
        intersection = (pred_mask & tgt_mask).sum().item()
        union = (pred_mask | tgt_mask).sum().item()
        if union > 0:
            ious.append(intersection / union)

    return np.mean(ious) if ious else 0.0


def compute_dice(pred, target, num_classes=2, present_classes=None):
    if present_classes is None:
        # By default, include all classes [0..(num_classes-1)]
        present_classes = list(range(1, num_classes))

    dices = []

    for cls in present_classes:
        pred_mask = (pred == cls)
        target_mask = (target == cls)

        intersection = (pred_mask & target_mask).sum().item()
        denom = pred_mask.sum().item() + target_mask.sum().item()

        if denom == 0:
            # If this class does not appear in either pred or target, skip
            continue

        dice_score = 2.0 * intersection / denom
        dices.append(dice_score)

    if len(dices) == 0:
        return 0.0

    return float(np.mean(dices))


def compute_classwise_iou_and_dice(pred, target, class_list):
    """
    Compute per-class IoU and Dice for each class in 'class_list',
    then return dictionaries of values + mean IoU + mean Dice.

    pred, target: 3D or 4D tensors (e.g., shape (N,D,H,W))
    class_list: e.g. [0,1,2,...] or [0,9,10,11,12,13,14,15,16]
    """
    # Flatten if shape is (N,D,H,W) -> (all_voxels, ) for easier summations
    if pred.dim() == 4:
        pred = pred.view(-1)
        target = target.view(-1)

    class_ious = {}
    class_dices = {}

    for cls in class_list:
        pred_mask = (pred == cls)
        tgt_mask = (target == cls)

        intersection = (pred_mask & tgt_mask).sum().item()
        union = (pred_mask | tgt_mask).sum().item()
        pred_sum = pred_mask.sum().item()
        tgt_sum = tgt_mask.sum().item()

        # IoU
        if union > 0:
            iou_val = intersection / union
        else:
            iou_val = 0.0

        # Dice
        denom = pred_sum + tgt_sum
        if denom > 0:
            dice_val = 2.0 * intersection / denom
        else:
            dice_val = 0.0

        class_ious[cls] = iou_val
        class_dices[cls] = dice_val

    # compute the means
    mean_iou = np.mean(list(class_ious.values()))
    mean_dice = np.mean(list(class_dices.values()))
    return class_ious, class_dices, mean_iou, mean_dice


# Training function with IoU and Dice
def train_model(model, train_loader, val_loader, device, optimizer, epochs=20, lr=1e-3,
                model_save_path="with_attention"):
    print("********************* Starting training *********************")
    start_time = time.time()

    # Example: Weighted cross-entropy
    class_weights = torch.tensor([0.1, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = GradScaler()

    CLASS_LIST = list(range(2))

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        running_loss = 0.0

        # -----------------------------
        # TRAINING LOOP
        # -----------------------------
        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            # Log GPU usage periodically
            if (batch_idx + 1) % 10 == 0:
                log_gpu_usage()

            del images, masks, outputs, loss
            gc.collect()
            torch.cuda.empty_cache()

        train_loss = running_loss / len(train_loader)
        print(f"Training Loss: {train_loss:.4f}")

        # -----------------------------
        # VALIDATION LOOP
        # -----------------------------
        if epoch % 2 == 0:
            model.eval()
            val_loss = 0.0

            # Accumulators for classwise IoU & Dice across entire val set
            class_intersections = {cls: 0 for cls in CLASS_LIST}
            class_unions = {cls: 0 for cls in CLASS_LIST}
            class_predsums = {cls: 0 for cls in CLASS_LIST}
            class_targetsums = {cls: 0 for cls in CLASS_LIST}
            slice_saved = False
            with torch.no_grad():
                for images, masks in tqdm(val_loader, desc="Validation"):
                    images, masks = images.to(device), masks.to(device)
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                    val_loss += loss.item()

                    # Predicted labels
                    preds = torch.argmax(outputs, dim=1)
                    SAVE_DIR = "./saved_validation_slices"
                    os.makedirs(SAVE_DIR, exist_ok=True)

                    # Save a slice
                    if not slice_saved:
                        for i in range(images.size(0)):  # Iterate over the batch
                            slice_idx = images.size(2) + 12 // 2  # Select the middle slice
                            if slice_idx < images.size(2):  # Ensure slice index is valid
                                # Extract slices
                                input_slice = images[i, 0, slice_idx, :, :].cpu().numpy()
                                gt_slice = masks[i, slice_idx, :, :].cpu().numpy()
                                pred_slice = preds[i, slice_idx, :, :].cpu().numpy()

                                # Normalize slices to [0, 255]
                                def normalize_to_uint8(slice_data):
                                    return ((slice_data - np.min(slice_data)) / (
                                            np.max(slice_data) - np.min(slice_data)) * 255).astype(np.uint8)

                                input_slice = normalize_to_uint8(input_slice)
                                gt_slice = normalize_to_uint8(gt_slice)
                                pred_slice = normalize_to_uint8(pred_slice)

                                # Save slices
                                Image.fromarray(input_slice).save(
                                    os.path.join(SAVE_DIR, f"epoch_{epoch + 1}_input_slice.png"))
                                Image.fromarray(gt_slice).save(
                                    os.path.join(SAVE_DIR, f"epoch_{epoch + 1}_ground_truth_slice.png"))
                                Image.fromarray(pred_slice).save(
                                    os.path.join(SAVE_DIR, f"epoch_{epoch + 1}_prediction_slice.png"))

                                print(f"Saved validation slices for epoch {epoch + 1}.")
                                slice_saved = True  # Save only once per validation phase
                                break

                    # Accumulate stats for each class
                    # Flatten so we can sum easily (N, D,H,W) -> all voxels
                    preds_flat = preds.view(-1)
                    masks_flat = masks.view(-1)

                    for cls in CLASS_LIST:
                        pred_mask = (preds_flat == cls)
                        tgt_mask = (masks_flat == cls)

                        inter = (pred_mask & tgt_mask).sum().item()
                        uni = (pred_mask | tgt_mask).sum().item()

                        class_intersections[cls] += inter
                        class_unions[cls] += uni
                        class_predsums[cls] += pred_mask.sum().item()
                        class_targetsums[cls] += tgt_mask.sum().item()

                    del images, masks, outputs, preds
                    gc.collect()
                    torch.cuda.empty_cache()

            val_loss /= len(val_loader)

            # -----------------------------
            # COMPUTE CLASS-WISE METRICS
            # -----------------------------
            class_ious = {}
            class_dices = {}
            for cls in CLASS_LIST:
                intersection = class_intersections[cls]
                union = class_unions[cls]
                pred_sum = class_predsums[cls]
                tgt_sum = class_targetsums[cls]

                # IoU
                iou_val = intersection / union if union > 0 else 0.0
                # Dice
                dice_val = (2.0 * intersection) / (pred_sum + tgt_sum) if (pred_sum + tgt_sum) > 0 else 0.0

                class_ious[cls] = iou_val
                class_dices[cls] = dice_val

            # Mean IoU & Dice across classes
            mean_iou = np.mean(list(class_ious.values()))
            mean_dice = np.mean(list(class_dices.values()))

            # Print results
            print(f"Validation Loss: {val_loss:.4f}\n")

            print("Class-wise IoU:")
            for cls in CLASS_LIST:
                print(f"  - Class {cls}: {class_ious[cls]:.4f}")
            print(f"Mean IoU: {mean_iou:.4f}\n")

            print("Class-wise Dice:")
            for cls in CLASS_LIST:
                print(f"  - Class {cls}: {class_dices[cls]:.4f}")
            print(f"Mean Dice: {mean_dice:.4f}\n")

        # Save model each epoch
        torch.save(model.state_dict(), f"{model_save_path}_epoch{epoch + 1}.pth")
        print(f"Model saved to {model_save_path}_epoch{epoch + 1}.pth")

    print(f"Training completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    region = 'binary'
    script_dir = "/export/fhome/skarzazi/tfg/SegmentationAI"
    image_dir = f"{script_dir}/data/segmentai_dataset/images/{region}"
    mask_dir = f"{script_dir}/data/segmentai_dataset/multiclass_masks/{region}"
    dataset_dir = f"{script_dir}/data/segmentai_dataset/processed/this_one_dataset.pth"
    model_save_path = f"{script_dir}/models/unet/with_attention"

    # Load dataset
    if os.path.exists(dataset_dir):
        print("Loading preprocessed dataset...")
        full_dataset = torch.load(dataset_dir)
    else:
        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            print("Image dir: ", image_dir)
            print("Mask dir: ", mask_dir)
            raise FileNotFoundError("Image or mask directory not found!")
        full_dataset = PatchBasedDataset(image_dir, mask_dir, visualize=False)
        torch.save(full_dataset, dataset_dir)

    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1, pin_memory=True,
                              collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=1, pin_memory=True,
                            collate_fn=custom_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNet3D_Attention(1, 2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training
    train_model(model, train_loader, val_loader, device, optimizer, epochs=50, lr=1e-3)
    torch.save(model.state_dict(), f"{model_save_path}_final.pth")
    print(f"Model saved to {model_save_path}_final.pth")
