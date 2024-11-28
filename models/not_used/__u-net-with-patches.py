import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from torch import nn, optim
import nrrd
import numpy as np
from scipy.ndimage import zoom
import time
import matplotlib.pyplot as plt
import multiprocessing


multiprocessing.set_start_method('spawn', force=True)
torch.cuda.empty_cache()


# Función para extraer patches de volúmenes 3D
def extract_patches(volume, patch_size=(128, 128, 128), stride=(64, 64, 64)):
    d, h, w = volume.shape[-3:]  # Para manejar máscaras con dimensión extra
    patches = []
    indices = []
    for z in range(0, d - patch_size[0] + 1, stride[0]):
        for y in range(0, h - patch_size[1] + 1, stride[1]):
            for x in range(0, w - patch_size[2] + 1, stride[2]):
                patch = volume[..., z:z + patch_size[0], y:y + patch_size[1], x:x + patch_size[2]]
                patches.append(patch)
                indices.append((z, y, x))
    return patches, indices

class MultiplanarDataset3D(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, limit=None, patch_size=(128, 128, 128), stride=(64, 64, 64)):
        print(f"Iniciando la carga del dataset.")
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride

        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nrrd')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nrrd')])
        print(f"Archivos de imagen encontrados: {len(self.image_files)}")
        print(f"Archivos de máscara encontrados: {len(self.mask_files)}")
        
        # Emparejar cada imagen con sus máscaras correspondientes
        self.image_mask_pairs = []
        for image_file in self.image_files:
            base_name = image_file.split("_images")[0]
            masks = [f for f in self.mask_files if f.startswith(base_name)]
            if masks:
                self.image_mask_pairs.append((image_file, masks))
                print(f"Emparejado {image_file} con máscaras: {masks}")
        
        if limit:  # Limitar el dataset para pruebas rápidas
            self.image_mask_pairs = self.image_mask_pairs[:limit]
            print(f"Dataset limitado a las primeras {limit} parejas imagen-máscara.")
        print(f"Dataset inicializado con {len(self.image_mask_pairs)} parejas imagen-máscara.")

        # Generar patches de todas las imágenes y máscaras
        self.patches = []
        for image_file, mask_files in self.image_mask_pairs:
            # Cargar imagen
            img_path = os.path.join(self.image_dir, image_file)
            print(f"Cargando imagen desde: {img_path}")
            image_data, _ = nrrd.read(img_path)
            print(f"Imagen cargada con forma: {image_data.shape}")

            # Cargar todas las máscaras para esta imagen
            mask_data_list = []
            for mask_file in mask_files:
                mask_path = os.path.join(self.mask_dir, mask_file)
                print(f"Cargando máscara desde: {mask_path}")
                mask_data, _ = nrrd.read(mask_path)
                print(f"Máscara cargada con forma: {mask_data.shape}")

                # Redimensionar máscara para que coincida con las dimensiones de la imagen
                mask_resized = self.resize_mask(mask_data, image_data.shape)
                mask_data_list.append(mask_resized)
            print(f"Total de máscaras cargadas para esta imagen: {len(mask_data_list)}")
            combined_mask = np.stack(mask_data_list, axis=0)
            print(f"Máscaras combinadas con forma: {combined_mask.shape}")

            # Extraer patches de imagen y máscara
            image_patches, _ = extract_patches(image_data, self.patch_size, self.stride)
            mask_patches, _ = extract_patches(combined_mask, self.patch_size, self.stride)
            for img_patch, mask_patch in zip(image_patches, mask_patches):
                self.patches.append((img_patch, mask_patch))

        print(f"Total de patches generados: {len(self.patches)}")

    def __len__(self):
        return len(self.patches)  # Número de patches

    def __getitem__(self, idx):
        image_patch, mask_patch = self.patches[idx]

        if self.transform:
            image_patch = self.transform(image_patch)
            mask_patch = self.transform(mask_patch)
            return image_patch, mask_patch
        else:
            # Añadir dimensión de canal a la imagen
            image_tensor = torch.tensor(image_patch, dtype=torch.float32).unsqueeze(0)
            mask_tensor = torch.tensor(mask_patch, dtype=torch.float16)
            return image_tensor, mask_tensor

    def resize_mask(self, mask, target_shape):
        """
        Redimensionar una máscara 3D para que coincida con la forma objetivo utilizando interpolación por vecino más cercano.
        """
        factors = [t / s for s, t in zip(mask.shape, target_shape)]
        resized_mask = zoom(mask, factors, order=0)  # Interpolación por vecino más cercano
        return resized_mask.astype(np.uint8)

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, out_channels, kernel_size=1)
        )

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec
    
def train_model_3d(model, dataloader, optimizer, criterion, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

def main():
    # Device configuration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS backend")
    else:
        device = torch.device("cpu")
        print("MPS backend not available, using CPU")
    
  

    # Rutas al dataset
    image_dir = "/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/images"
    mask_dir = "/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/masks"

    # Hiperparámetros
    batch_size = 1  # Ajustar según la memoria GPU
    learning_rate = 0.001
    epochs = 2
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")

    # Tamaño de patches y stride
    patch_size = (128, 128, 128)
    stride = (64, 64, 64)

    # Crear dataset y dataloader
    dataset = MultiplanarDataset3D(image_dir, mask_dir, limit=5, patch_size=patch_size, stride=stride)
    print(f"Dataset creado con tamaño: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    print(f"Dataloader creado con tamaño: {len(dataloader)}")
    # After processing a batch
    del dataset
    torch.cuda.empty_cache()
    # Definir modelo, pérdida y optimizador
    num_masks = dataloader.patches[0][1].shape[0]  # Número de máscaras por patch
    model = UNet3D(in_channels=1, out_channels=num_masks).to(device)
    criterion = nn.BCEWithLogitsLoss()  # Pérdida para múltiples etiquetas binarias
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Entrenar el modelo
    train_model_3d(model, dataloader, optimizer, criterion, device, epochs)

if __name__ == "__main__":
    main()