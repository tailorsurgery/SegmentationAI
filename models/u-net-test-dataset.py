import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from torch import nn, optim
from PIL import Image
import nrrd
import numpy as np
from scipy.ndimage import zoom
import time
import matplotlib.pyplot as plt

class MultiplanarDataset3D(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, limit=None):
        print(f"Iniciando la carga del dataset.")
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nrrd')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nrrd')])
        print(f"Archivos de imagen encontrados: {len(self.image_files)}")
        print(f"Archivos de máscara encontrados: {len(self.mask_files)}")
        
        # Pair each image with its corresponding mask files
        self.image_mask_pairs = []
        for image_file in self.image_files:
            base_name = image_file.split("_images")[0]
            masks = [f for f in self.mask_files if f.startswith(base_name)]
            if masks:
                self.image_mask_pairs.append((image_file, masks))
                print(f"Emparejado {image_file} con máscaras: {masks}")
        
        if limit:  # Limit the dataset for quick testing
            self.image_mask_pairs = self.image_mask_pairs[:limit]
            print(f"Dataset limitado a las primeras {limit} parejas imagen-máscara.")
        self.transform = transform
        print(f"Dataset inicializado con {len(self.image_mask_pairs)} parejas imagen-máscara.")

    def __len__(self):
        print(f"Obteniendo la longitud del dataset: {len(self.image_mask_pairs)}")
        return len(self.image_mask_pairs)  # Number of image-mask pairs

    def __getitem__(self, idx):
        print(f"Obteniendo el ítem en el índice: {idx}")
        # Load image and its corresponding mask files
        image_file, mask_files = self.image_mask_pairs[idx]
        img_path = os.path.join(self.image_dir, image_file)
        print(f"Cargando imagen desde: {img_path}")
        image_data, _ = nrrd.read(img_path)
        print(f"Imagen cargada con forma: {image_data.shape}")

        # Load all masks for this image
        mask_data_list = []
        for mask_file in mask_files:
            mask_path = os.path.join(self.mask_dir, mask_file)
            print(f"Cargando máscara desde: {mask_path}")
            mask_data, _ = nrrd.read(mask_path)
            print(f"Máscara cargada con forma: {mask_data.shape}")

            # Resize mask to match image dimensions
            mask_resized = self.resize_mask(mask_data, image_data.shape)
            mask_data_list.append(mask_resized)
        
        print(f"Total de máscaras cargadas para esta imagen: {len(mask_data_list)}")
        combined_mask = np.stack(mask_data_list, axis=0)
        print(f"Máscaras combinadas con forma: {combined_mask.shape}")

        mask_tensor = torch.tensor(combined_mask, dtype=torch.float16)  # Shape: [num_masks, D, H, W]
        print(f"Tensor de máscara creado con dtype: {mask_tensor.dtype}")

        if self.transform:
            print("Aplicando transformaciones.")
            image = self.transform(image_data)
            mask = self.transform(combined_mask)
            return image, mask
        else:
            image_tensor = torch.tensor(image_data, dtype=torch.float32)
            print(f"Tensor de imagen creado con dtype: {image_tensor.dtype}")
            return image_tensor, mask_tensor

    def resize_mask(self, mask, target_shape):
        """
        Resize a 3D mask to match the target 3D shape using nearest-neighbor interpolation.
        """
        factors = [t / s for s, t in zip(mask.shape, target_shape)]
        resized_mask = zoom(mask, factors, order=0)  # Nearest-neighbor interpolation
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
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, out_channels, kernel_size=1)
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

            # Ensure masks are 5D (batch_size, num_masks, D, H, W)
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
    # Paths to dataset
    image_dir = "/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/images"
    mask_dir = "/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/masks"

    # Hyperparameters
    batch_size = 1  # Adjust based on GPU memory
    learning_rate = 0.001
    epochs = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset and dataloader
    dataset = MultiplanarDataset3D(image_dir, mask_dir, limit=5)
    print(f"Dataset creado con tamaño: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataloader creado con tamaño: {len(dataloader)}")

    # Define model, loss, and optimizer
    num_masks = len(dataset.image_mask_pairs[0][1])  # Number of masks per image
    model = UNet3D(in_channels=1, out_channels=num_masks).to(device)
    criterion = nn.BCEWithLogitsLoss()  # Multi-label binary cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model_3d(model, dataloader, optimizer, criterion, device, epochs)

if __name__ == "__main__":
    main()