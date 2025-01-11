import gc
import os
import time
import torch
from torch.cuda.amp import autocast
import SimpleITK as sitk
import numpy as np
from models.unet3d import UNet3D

script_dir = '/Users/samyakarzazielbachiri/Documents/SegmentationAI'

case = "240018-2"

# Patch parameters
PATCH_SIZE = (128, 128, 128)  # Size of each patch (depth, height, width)
STRIDE = (64, 64, 64)  # Stride for patch extraction

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the trained model
model = UNet3D(1, 9).to(device)  # Output channels = number of classes
model_path = f"{script_dir}/models/unet/knee_3d_unet_model_training.pth"
print(f"Loading model weights from: {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device))
print("Model weights loaded successfully.")
model.eval()

# Preprocess input data
image_path = script_dir + f"/data/segmentai_dataset/images/knee/{case}_images.nrrd"
print(f"Loading input image from: {image_path}")
image = sitk.ReadImage(image_path)
print("Image loaded successfully.")

# Normalize (preserve original shape)
image_array = sitk.GetArrayFromImage(image).astype(np.float32)
print(f"Original image shape: {image_array.shape}")
image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))

# Patch creation
def create_patches(volume, patch_size, stride):
    pd, ph, pw = patch_size
    sd, sh, sw = stride
    d, h, w = volume.shape

    patches = []
    patch_indices = []
    for z in range(0, d - pd + 1, sd):
        for y in range(0, h - ph + 1, sh):
            for x in range(0, w - pw + 1, sw):
                patch = volume[z:z+pd, y:y+ph, x:x+pw]
                patches.append(patch)
                patch_indices.append((z, z+pd, y, y+ph, x, x+pw))

    return np.array(patches), patch_indices

patches, patch_indices = create_patches(image_array, PATCH_SIZE, STRIDE)
print(f"Number of patches created: {len(patches)}")

# Perform inference on patches
torch.cuda.empty_cache()
gc.collect()

print("Starting inference on patches...")
start_time = time.time()

# Prepare output array for binary masks
num_classes = 9  # Number of classes
binary_masks = [np.zeros(image_array.shape, dtype=np.uint8) for _ in range(num_classes)]

with torch.no_grad():
    for patch, (z_start, z_end, y_start, y_end, x_start, x_end) in zip(patches, patch_indices):
        patch_tensor = torch.tensor(patch).unsqueeze(0).unsqueeze(0).to(device)
        with autocast():
            output = model(patch_tensor)  # Output shape: [1, num_classes, d, h, w]
        patch_predictions = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # Predicted class for each voxel

        # Update binary masks for each class
        for cls in range(num_classes):
            binary_masks[cls][z_start:z_end, y_start:y_end, x_start:x_end] += (patch_predictions == cls).astype(np.uint8)

print(f"Inference on patches completed in {time.time() - start_time:.2f} seconds.")

# Save the binary masks as separate NRRD files
for cls, binary_mask in enumerate(binary_masks):
    binary_mask_image = sitk.GetImageFromArray(binary_mask)
    binary_mask_image.CopyInformation(image)  # Ensure spatial metadata (origin, spacing) matches the input
    output_path = f"{script_dir}/binary_mask_class_{cls}_{case}.nrrd"
    sitk.WriteImage(binary_mask_image, output_path)
    print(f"Class {cls} binary mask saved to: {output_path}")