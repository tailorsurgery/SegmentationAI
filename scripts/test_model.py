import gc
import os
import time
from torch.cuda.amp import autocast
script_dir = '/Users/samyakarzazielbachiri/Documents/SegmentationAI'

from models.unet3d import UNet3D
import torch
import SimpleITK as sitk
import numpy as np

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the trained model
model = UNet3D(1, 6).to(device)
model_path = f"{script_dir}/models/arms-3dunet.pth"
print(f"Loading model weights from: {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device))
print("Model weights loaded successfully.")
model.eval()

# Preprocess input data
image_path = script_dir + "/data/segmentai_dataset/images/arms/240023-2_images.nrrd"
print(f"Loading input image from: {image_path}")
image = sitk.ReadImage(image_path)
print("Image loaded successfully.")

# Normalize and downsample
image_array = sitk.GetArrayFromImage(image).astype(np.float32)
print(f"Original image shape: {image_array.shape}")
image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))

# Optional: Downsample
downsample_factor = 4
image_array = image_array[::downsample_factor, ::downsample_factor, ::downsample_factor]

# Ensure the shape is divisible by the required factor
required_divisibility = 8
new_shape = [
    (dim // required_divisibility) * required_divisibility for dim in image_array.shape
]
image_array = image_array[:new_shape[0], :new_shape[1], :new_shape[2]]
print(f"Downsampled and adjusted shape: {image_array.shape}")

input_tensor = torch.tensor(image_array).unsqueeze(0).unsqueeze(0).to(device)
print(f"Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")

# Perform inference
torch.cuda.empty_cache()
gc.collect()

print("Starting inference...")
start_time = time.time()

with torch.no_grad():
    output = model(input_tensor)

print(f"Inference completed in {time.time() - start_time:.2f} seconds.")
predictions = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()


# Save the predictions
predicted_image = sitk.GetImageFromArray(predictions)

# Save the predictions
output_path = f"{script_dir}/scripts/output/fist_prediction_240023-2.nrrd"
print("Resampling prediction to match original image size...")

# Resample predicted image
resampler = sitk.ResampleImageFilter()
resampler.SetSize(image.GetSize())
resampler.SetOutputSpacing(image.GetSpacing())
resampler.SetOutputOrigin(image.GetOrigin())
resampler.SetOutputDirection(image.GetDirection())
resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # For segmentation masks
resampled_prediction = resampler.Execute(predicted_image)

# Save the resampled predicted image
sitk.WriteImage(resampled_prediction, output_path)
print(f"Inference complete. Resampled predictions saved to: {output_path}")

