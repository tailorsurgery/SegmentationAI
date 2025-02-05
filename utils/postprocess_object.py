import nrrd
import numpy as np
from skimage import measure
import trimesh
import pyvista as pv


def load_nrrd(file_path):
    """
    Load a .nrrd segmentation file.

    Parameters:
        file_path (str): Path to the .nrrd file.

    Returns:
        data (numpy.ndarray): Segmentation mask array.
        header (dict): Metadata from the .nrrd file.
    """
    data, header = nrrd.read(file_path)
    print("Loaded NRRD file.")
    print("Data shape:", data.shape)
    print("Header info:", header)
    return data, header


def extract_label_names(header):
    """
    Extract label names from the .nrrd header.

    Parameters:
        header (dict): Metadata from the .nrrd file.

    Returns:
        label_names (dict): Mapping of label indices to their names.
    """
    label_names = {}
    for key, value in header.items():
        if key.startswith("label_"):
            label_index = int(key.split("_")[1])  # Extract label index
            label_names[label_index] = value.strip()  # Remove extra spaces or nulls
    return label_names


def create_3d_mesh(label_mask, spacing):
    """
    Create a 3D mesh from a binary mask using the Marching Cubes algorithm.

    Parameters:
        label_mask (numpy.ndarray): Binary mask for the label.
        spacing (tuple): Voxel spacing (dx, dy, dz).

    Returns:
        trimesh.Trimesh: 3D mesh object.
    """
    print("Generating 3D mesh...")
    verts, faces, normals, _ = measure.marching_cubes(label_mask, level=0, spacing=spacing)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    print("3D mesh generated.")
    return mesh


def smooth_and_wrap_mesh(mesh, iterations=10, wrap_resolution=100):
    """
    Smooth and wrap a 3D mesh.

    Parameters:
        mesh (trimesh.Trimesh): Input 3D mesh.
        iterations (int): Number of smoothing iterations.
        wrap_resolution (int): Resolution for wrapping.

    Returns:
        trimesh.Trimesh: Smoothed and wrapped 3D mesh.
    """
    print(f"Smoothing the mesh with {iterations} iterations...")
    smoothed_mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=iterations)

    print(f"Cleaning and wrapping the mesh...")
    # Convert to PyVista mesh
    pv_mesh = pv.wrap(smoothed_mesh)

    # Ensure the mesh is clean
    cleaned_mesh = pv_mesh.clean()

    # Optionally triangulate the mesh for further stability
    triangulated_mesh = cleaned_mesh.triangulate()

    # Convert back to Trimesh
    wrapped_trimesh = trimesh.Trimesh(
        vertices=np.array(triangulated_mesh.points),
        faces=np.array(triangulated_mesh.faces)
    )
    print("Wrapping completed.")
    return wrapped_trimesh


def save_mesh(mesh, output_path):
    """
    Save a 3D mesh to a file.

    Parameters:
        mesh (trimesh.Trimesh): 3D mesh to save.
        output_path (str): Path to save the file (e.g., .stl format).
    """
    print(f"Saving the mesh to {output_path}...")
    mesh.export(output_path)
    print("Mesh saved.")


def process_nrrd_file(file_path, output_dir):
    """
    Process a .nrrd file to extract 3D objects for all labels and save them as .stl files.

    Parameters:
        file_path (str): Path to the .nrrd file.
        output_dir (str): Directory to save the generated .stl files.
    """
    # Load the .nrrd file
    data, header = load_nrrd(file_path)

    # Extract voxel spacing from the header
    spacing = tuple(np.abs(np.array(header['space directions']).diagonal()))
    print("Voxel spacing:", spacing)

    # Extract label names from the header
    label_names = extract_label_names(header)
    print("Label names:", label_names)

    # Process each label
    for label, name in label_names.items():
        print(f"Processing label {label}: {name}...")

        # Generate binary mask for the label
        label_mask = (data == label).astype(np.uint8)

        # Create 3D mesh
        mesh = create_3d_mesh(label_mask, spacing)

        # Smooth and wrap the mesh
        smoothed_wrapped_mesh = smooth_and_wrap_mesh(mesh, iterations=50, wrap_resolution=200)

        # Save the mesh to an .stl file
        output_path = f"{output_dir}/{name}.stl"
        save_mesh(smoothed_wrapped_mesh, output_path)


if __name__ == "__main__":
    # Input NRRD file path
    out_p = "/Users/samyakarzazielbachiri/Documents/SegmentationAI/scripts/output"
    nrrd_file = out_p + "/multiclass_masks/240026-2_multiclass_mask.nrrd"  # Update with your file path

    # Output directory for saving .stl files
    output_directory = out_p  # Update with your output path

    # Process the NRRD file
    process_nrrd_file(nrrd_file, output_directory)