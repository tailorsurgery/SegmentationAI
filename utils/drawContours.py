import nrrd
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

# Step 1: Load the NRRD file
file_path = "/Users/samyakarzazielbachiri/Documents/SegmentationAI/scripts/output/images/240023-2_images.nrrd"
image_data, header = nrrd.read(file_path)

# Step 2: Select a slice from the 3D image (e.g., middle slice)
slice_index = (image_data.shape[0] // 2) + 30  # Select the middle slice
slice_data = image_data[slice_index, :, :]

# Step 3: Generate the contour
# Apply a threshold to segment the image
threshold_value = 226  # Adjust based on your image intensity range

binary_slice = slice_data > threshold_value


# Find contours
contours = measure.find_contours(binary_slice, level=0.5)

# Step 4: Visualize the contour
plt.figure(figsize=(8, 8))
plt.imshow(slice_data, cmap="gray")
for contour in contours:
    plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color="blue")

plt.title("Contour Overlay on CT Slice")
plt.axis("off")
plt.show()