import os
import numpy as np
import SimpleITK as sitk
import time

def combine_binary_masks_with_alignment(image_path, mask_dir, output_path):
    """
    Combina máscaras binarias de un caso en una sola máscara multiclasificada, asegurando la alineación espacial.

    Args:
        image_path (str): Ruta a la imagen original (para referencia).
        mask_dir (str): Directorio que contiene las máscaras binarias.
        output_path (str): Ruta donde se guardará la máscara multiclasificada.
    """
    # Leer la imagen de referencia para obtener tamaño y metadatos
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)
    image_shape = image_array.shape
    print(f"Forma de la imagen: {image_shape}")

    # Crear una máscara vacía para la salida (misma forma que la imagen)
    multiclass_mask = np.zeros(image_shape, dtype=np.uint8)

    # Extraer el nombre del caso a partir del archivo de imagen
    case_name = os.path.basename(image_path).split("_images")[0]
    print(f"Procesando el caso: {case_name}")

    # Listar todas las máscaras binarias que coincidan con el caso
    mask_files = sorted(
        [f for f in os.listdir(mask_dir) if f.startswith(case_name) and f.endswith('.nrrd')]
    )
    print(f"Máscaras encontradas para el caso {case_name}: {mask_files}")

    # Combinar máscaras, asignando un valor único a cada una
    class_value = 1
    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)

        # Leer la máscara y sus metadatos
        mask = sitk.ReadImage(mask_path)
        print(f"Procesando máscara: {mask_file}")

        # Resamplear la máscara para que coincida en espacio y tamaño con la imagen
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(image)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampled_mask = resampler.Execute(mask)

        # Convertir la máscara resampleada en un array
        resampled_mask_array = sitk.GetArrayFromImage(resampled_mask)

        # Asignar el valor de la clase a las regiones correspondientes
        multiclass_mask[resampled_mask_array > 0] = class_value
        print(f"Asignando clase {class_value} a {mask_file}")
        class_value += 1

    # Guardar la máscara multiclasificada
    multiclass_sitk = sitk.GetImageFromArray(multiclass_mask)
    multiclass_sitk.SetSpacing(image.GetSpacing())
    multiclass_sitk.SetOrigin(image.GetOrigin())
    multiclass_sitk.SetDirection(image.GetDirection())
    sitk.WriteImage(multiclass_sitk, output_path)
    print(f"Máscara multiclasificada guardada en {output_path}")

# Directorios principales
image_dir = "/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/images"
mask_dir = "/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/masks"
output_dir = "/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/multiclass_masks"

# Asegurarse de que el directorio de salida exista
os.makedirs(output_dir, exist_ok=True)

# Iterar sobre todas las imágenes en el directorio de imágenes
#image_files = [f for f in os.listdir(image_dir) if f.endswith("_images.nrrd")]
#print(image_files)
image_files = ['240014-2_images.nrrd', '240014-3_images.nrrd']
time.sleep(5)
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    case_name = os.path.basename(image_path).split("_images")[0]
    output_path = os.path.join(output_dir, f"{case_name}_multiclass_mask.nrrd")

    print(f"\nProcesando el caso: {case_name}")
    combine_binary_masks_with_alignment(image_path, mask_dir, output_path)

print("Procesamiento completado para todas las imágenes.")