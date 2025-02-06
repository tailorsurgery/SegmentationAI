import gc
import os
import time
import torch
import SimpleITK as sitk
import numpy as np
from torch.cuda.amp import autocast
from models.unet3d_mejorado import UNet3D  # Update import based on your project structure

script_dir = '/path/to/SegmentationAI'
case = "240018-2"

# Patch parameters
PATCH_SIZE = (128, 128, 128)  # (depth, height, width)
STRIDE = (64, 64, 64)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the trained model
model = UNet3D(in_channels=1, out_channels=17).to(device)
model_path = f"{script_dir}/models/unet/knee_3d_unet_model_training.pth"
print(f"Loading model weights from: {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device))
print("Model weights loaded successfully.")
model.eval()

# Preprocess input data
image_path = f"{script_dir}/data/segmentai_dataset/images/knee/{case}_images.nrrd"
print(f"Loading input image from: {image_path}")
image = sitk.ReadImage(image_path)
image_array = sitk.GetArrayFromImage(image).astype(np.float32)
print(f"Original image shape: {image_array.shape} (D, H, W)")

# Normalize
#image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array) + 1e-8)

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
    return patches, patch_indices

patches, patch_indices = create_patches(image_array, PATCH_SIZE, STRIDE)
print(f"Number of patches created: {len(patches)}")

torch.cuda.empty_cache()
gc.collect()

print("Starting inference on patches...")
start_time = time.time()

# ---------------------------
# Single multi-class mask
# ---------------------------
multi_class_mask = np.zeros_like(image_array, dtype=np.uint8)

with torch.no_grad():
    for patch, (z_start, z_end, y_start, y_end, x_start, x_end) in zip(patches, patch_indices):
        # (D, H, W) => (1,1,D,H,W)
        patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with autocast():
            output = model(patch_tensor)  # shape: (1, num_classes, d, h, w)

        # Predicted class per voxel: shape (d, h, w)
        patch_predictions = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        # Place these class IDs in the appropriate location in the multi_class_mask
        multi_class_mask[z_start:z_end, y_start:y_end, x_start:x_end] = patch_predictions

        del patch_tensor, output, patch_predictions
        torch.cuda.empty_cache()
        gc.collect()

print(f"Inference on patches completed in {time.time() - start_time:.2f} seconds.")

# Convert to SimpleITK image
mask_image = sitk.GetImageFromArray(multi_class_mask)
mask_image.CopyInformation(image)

# Save as a single multi-class NRRD
output_path = f"{script_dir}/data/segmentai_dataset/multiclass_masks/knee/{case}_multiclass_mask.nrrd"
sitk.WriteImage(mask_image, output_path)
print(f"Multi-class mask saved to: {output_path}")