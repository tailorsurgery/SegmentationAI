'''import numpy as np
import nibabel as nib
from stl import mesh
import os
import time

case = 240056

def create_voxel_grid(stl_mesh, grid_size):
    # Create an empty 3D voxel grid
    voxel_grid = np.zeros(grid_size, dtype=np.uint8)

    # Deduplicate points and calculate bounds
    unique_points = np.unique(stl_mesh.points.reshape(-1, 3), axis=0)
    if unique_points.shape[0] < 3:
        raise ValueError("The mesh has fewer than 3 unique points; cannot compute bounds.")

    min_bounds = unique_points.min(axis=0)
    max_bounds = unique_points.max(axis=0)

    # Debugging output for bounds
    print(f"Unique Min bounds: {min_bounds}, Unique Max bounds: {max_bounds}")

    # Check dimensions of min_bounds and max_bounds
    if min_bounds.shape[0] != 3 or max_bounds.shape[0] != 3:
        raise ValueError("Invalid mesh bounds: min_bounds and max_bounds must have shape (3,)")

    # Calculate the scaling factor for voxelization
    scale = np.array(grid_size) / (max_bounds - min_bounds)

    # Convert STL vertices to voxel grid indices
    for face in stl_mesh.vectors:
        # Get the vertices of the face and scale them to voxel indices
        scaled_vertices = ((face - min_bounds) * scale).astype(int)

        # Clamp the indices to be within the voxel grid size
        scaled_vertices = np.clip(scaled_vertices, 0, np.array(grid_size) - 1)

        # Add filled voxels for the face
        fill_voxels(voxel_grid, scaled_vertices)

    return voxel_grid

def fill_voxels(voxel_grid, vertices):
    # Convert vertices to a list of indices
    x_coords = [vertices[i][0] for i in range(3)]
    y_coords = [vertices[i][1] for i in range(3)]
    z_coords = [vertices[i][2] for i in range(3)]

    # Use a simple approach to fill the voxel space (triangular prism filling)
    for x in range(min(x_coords), max(x_coords) + 1):
        for y in range(min(y_coords), max(y_coords) + 1):
            # Interpolate the z-coordinate within the triangle defined by the vertices
            for z in range(min(z_coords), max(z_coords) + 1):
                if point_in_triangle((x, y, z), vertices):
                    voxel_grid[x, y, z] = 1  # Mark as filled voxel

def point_in_triangle(pt, triangle):
    # This function checks if the point is inside the triangle
    v0 = triangle[1] - triangle[0]
    v1 = triangle[2] - triangle[0]
    v2 = pt - triangle[0]

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    # Check for degenerate triangle case (dot product is zero)
    denominator = dot00 * dot11 - dot01 * dot01
    if np.isclose(denominator, 0):
        return False  # The triangle is degenerate

    invDenom = 1 / denominator
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom

    return (u >= 0) and (v >= 0) and (u + v <= 1)

def stl_to_nifti(input_path, output_file, grid_size=(100, 100, 100)):
    # List to hold each mask
    masks = []

    # Process all STL files in the input directory
    for filename in os.listdir(input_path):
        if filename.endswith('.stl'):
            stl_file_path = os.path.join(input_path, filename)
            try:
                stl_mesh = mesh.Mesh.from_file(stl_file_path)
                # Check if the mesh is empty
                if len(stl_mesh.vectors) == 0:
                    raise ValueError(f"The mesh from {stl_file_path} has no triangles.")

                # Create the voxel grid from the STL mesh
                voxel_data = create_voxel_grid(stl_mesh, grid_size)
                
                # Append to masks list
                masks.append(voxel_data[np.newaxis, ...])  # Add a new axis for stacking

                print(f'Successfully processed {filename}.')
            except Exception as e:
                print(f'Error processing {filename}: {e}')
    
    if masks:
        # Stack masks along a new dimension (0th)
        masks_stacked = np.vstack(masks)
        # Save the stacked masks as a NIfTI file
        nifti_img = nib.Nifti1Image(masks_stacked, np.eye(4))
        nib.save(nifti_img, output_file)
        print(f'Successfully saved all masks to {output_file}.')
    else:
        print('No masks to save.')

for i in range(1,2):
    print(f"Processing case {case}...")
    try:
        # Directory paths
        input_path = f'C:\\Users\\samya\\Downloads\\RESEGMENTED\\binary_stl\\{case}'
        output_path = f'C:\\Users\\samya\\Downloads\\RESEGMENTED\\dataset\\masks\\{case}.nii'

        # Run the conversion
        time_start = time.time()
        stl_to_nifti(input_path, output_path, grid_size=(100, 100, 100))
        time_end = time.time()
        print(f"Total time taken: {time_end - time_start:.2f} seconds")
    except Exception as e:
        print(f"Error processing case {case}: {e}")
        continue
    case+=1'''

import os
import numpy as np
import nibabel as nib
from stl import mesh
import matplotlib.pyplot as plt

def load_stl_as_volume(stl_file_path, target_shape):
    """Load a binary STL file and return it as a 3D NumPy array."""
    # Load the STL file
    model = mesh.Mesh.from_file(stl_file_path)

    # Create a volume to store the model
    volume = np.zeros(target_shape, dtype=np.uint8)

    # Set up voxel size
    voxel_size = (1, 1, 1)  # You can adjust this to change the size of the volume
    
    # Iterate over the faces of the mesh and fill the volume
    for face in model.vectors:
        # Get the vertices of the face
        for vertex in face:
            # Calculate voxel indices based on vertex coordinates
            x, y, z = (vertex / voxel_size).astype(int)

            # Ensure we are within the bounds of the volume
            if 0 <= x < target_shape[0] and 0 <= y < target_shape[1] and 0 <= z < target_shape[2]:
                volume[x, y, z] = 1  # Mark the voxel as occupied

    return volume

def save_volume_as_nifti(volume, output_file_path):
    """Save the 3D volume as a NIfTI file."""
    nifti_img = nib.Nifti1Image(volume, np.eye(4))
    nib.save(nifti_img, output_file_path)
    print(f'Successfully saved NIfTI file to: {output_file_path}')

def main(case):
    # Define the directory containing STL files
    stl_directory = f'./../TS_DATASET/dataset/binary_stl/{case}'  # Update this path as needed
    output_nifti_file = f'./../TS_DATASET/dataset/masks/{case}_masks.nii'
    
    # Define the target shape of the volume (adjust according to your needs)
    target_shape = (100, 100, 100)  # Example shape, change as necessary

    # Create a combined volume
    combined_volume = np.zeros(target_shape, dtype=np.uint8)

    # Load each STL file from the directory
    for file_name in os.listdir(stl_directory):
        if file_name.endswith('.stl'):
            file_path = os.path.join(stl_directory, file_name)
            print(f'Processing {file_path}...')
            volume = load_stl_as_volume(file_path, target_shape)
            combined_volume += volume  # Combine volumes

    # Save the combined volume as a NIfTI file
    save_volume_as_nifti(combined_volume, output_nifti_file)

    # Visualize the middle slice
    middle_slice_index = combined_volume.shape[0] // 2
    plt.imshow(combined_volume[middle_slice_index, :, :], cmap='gray')
    plt.title(f'Middle Slice (Index {middle_slice_index})')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    case = 240056
    main(case)
