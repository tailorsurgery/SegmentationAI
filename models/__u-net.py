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
from sklearn.metrics import jaccard_score
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Dataset Class for Multiple Images and Masks
class MultiplanarDataset(Dataset):
    def __init__(self, image_dir, mask_dir, plane='axial', transform=None, limit=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nrrd')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nrrd')])
        
        # Pair each image with its corresponding mask files
        self.image_mask_pairs = []
        for image_file in self.image_files:
            base_name = image_file.split("_images")[0]
            masks = [f for f in self.mask_files if f.startswith(base_name)]
            if masks:
                self.image_mask_pairs.append((image_file, masks))
        
        if limit:  # Limit the dataset for quick testing
            self.image_mask_pairs = self.image_mask_pairs[50:50+limit]
        #print(self.image_mask_pairs)
        self.plane = plane
        self.transform = transform

    def __len__(self):
        return len(self.image_mask_pairs)  # Number of image-mask pairs

    def __getitem__(self, idx):
        # Load image and its corresponding mask files
        image_file, mask_files = self.image_mask_pairs[idx]
        img_path = os.path.join(self.image_dir, image_file)
        image_data, _ = nrrd.read(img_path)

        # Load all masks for this image
        mask_data_list = []
        for mask_file in mask_files:
            mask_path = os.path.join(self.mask_dir, mask_file)
            mask_data, _ = nrrd.read(mask_path)
            mask_data_list.append(mask_data)

        # Normalize image
        image_data = image_data.astype(np.float32)
        image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))

        # Resize all masks to match image dimensions
        for i in range(len(mask_data_list)):
            mask_data_list[i] = self.resize_mask(mask_data_list[i], image_data.shape)

        # Extract the desired plane for the image and masks
        if self.plane == 'axial':
            image = image_data[:, :, image_data.shape[2] // 2]
            masks = [mask[:, :, mask.shape[2] // 2] for mask in mask_data_list]
        elif self.plane == 'sagittal':
            image = image_data[:, image_data.shape[1] // 2, :]
            masks = [mask[:, mask.shape[1] // 2, :] for mask in mask_data_list]
        elif self.plane == 'coronal':
            image = image_data[image_data.shape[0] // 2, :, :]
            masks = [mask[mask.shape[0] // 2, :, :] for mask in mask_data_list]
        else:
            raise ValueError("Invalid plane. Choose from 'axial', 'sagittal', or 'coronal'.")

        # Combine masks into a single multi-class mask
        '''combined_mask = np.zeros_like(image, dtype=np.uint8)
        for i, mask in enumerate(masks):
            combined_mask[mask > 0] = i + 1  # Assign a unique label for each bone (e.g., bone 1 -> 1, bone 2 -> 2)
'''
        # Convert to PIL format for transformations
        image = Image.fromarray((image * 255).astype(np.uint8))
        combined_mask = Image.fromarray(combined_mask)

        if self.transform:
            image = self.transform(image)
            combined_mask = self.transform(combined_mask)

        return image, combined_mask


    @staticmethod
    def resize_mask(mask, target_shape):
        """
        Resize mask to match target shape using nearest-neighbor interpolation.
        """
        factors = [t / s for s, t in zip(mask.shape, target_shape)]
        resized_mask = zoom(mask, factors, order=0)
        return resized_mask.astype(np.uint8)
    

# U-Net Model Definition
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec

# Training Function
def train_model(model, dataloader, optimizer, criterion, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            print("Images:", images)
            print("Masks:", masks)


            # Ensure masks are 3D (batch_size, height, width)
            masks = masks.squeeze(1)  # Remove channel dimension
            masks = masks.long()  # Cast to LongTensor

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")


def evaluate_model(model, dataloader, device):
    model.eval()
    dice_scores = []
    jaccard_scores = []
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            masks = masks.squeeze(1).long()  # Ensure masks are LongTensors

            outputs = model(images)
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1).cpu().numpy()
            masks = masks.cpu().numpy()

            for pred, mask in zip(preds, masks):
                if mask.sum() == 0 and pred.sum() == 0:
                    dice = 1.0
                    jaccard = 1.0
                else:
                    intersection = (pred * mask).sum()
                    dice = 2.0 * intersection / (pred.sum() + mask.sum() + 1e-8)
                    jaccard = jaccard_score(mask.flatten(), pred.flatten(), average='binary', zero_division=1)
                dice_scores.append(dice)
                jaccard_scores.append(jaccard)

    print(f"Average Dice Score: {np.mean(dice_scores):.4f}")
    print(f"Average Jaccard Index: {np.mean(jaccard_scores):.4f}")


def visualize_predictions(model, dataloader, device, num_samples=3):
    model.eval()
    samples = 0
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1).cpu().numpy()  # Predicted class indices
            images = images.cpu().numpy()
            masks = masks.cpu().numpy()

            for img, mask, pred in zip(images, masks, preds):
                if samples >= num_samples:
                    return
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                plt.title("Input Image")
                plt.imshow(img[0], cmap='gray')  # Single-channel input image
                plt.subplot(1, 3, 2)
                plt.title("Ground Truth Mask")
                plt.imshow(mask, cmap='gray')  # Ground truth multi-class mask
                plt.subplot(1, 3, 3)
                plt.title("Predicted Mask")
                plt.imshow(pred, cmap='gray')  # Predicted multi-class mask
                plt.show()
                samples += 1

def debug_dataset_loading(dataset, num_samples=3):
    for i in range(min(num_samples, len(dataset))):
        image, mask = dataset[i]  # Load image and mask
        print("******", mask)
        # Convert tensors to numpy arrays and squeeze extra dimensions
        image_np = image.numpy().squeeze()  # Remove channel dimension from image
        mask_np = mask.numpy().squeeze()  # Remove channel dimension from mask

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Input Image")
        plt.imshow(image_np, cmap='gray')  # Visualize single-channel image
        plt.subplot(1, 2, 2)
        plt.title("Ground Truth Mask")
        plt.imshow(mask_np, cmap='gray')  # Visualize single-channel mask
        plt.show()
                
# Main Function
def main():
    # Paths to dataset
    image_dir ="/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/images"
    mask_dir = "/Users/samyakarzazielbachiri/Documents/SegmentationAI/data/segmentai_dataset/masks"

    # Hyperparameters
    batch_size = 2
    learning_rate = 0.001
    epochs = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformations
    transform = Compose([Resize((512, 512)), ToTensor()])

    # Create dataset and dataloader
    dataset = MultiplanarDataset(image_dir, mask_dir, plane='axial', transform=transform, limit=50)
    print(dataset)


    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define model, loss, and optimizer
    num_classes = len(dataset.image_mask_pairs[0][1]) + 1  # Number of masks + 1 for background
    model = UNet(in_channels=1, out_channels=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Debugging dataset loading...")
    debug_dataset_loading(dataset, num_samples=3)
    # Train the model
    print("Training on Multiple Images and Masks:")
    train_time = time.time()
    train_model(model, dataloader, optimizer, criterion, device, epochs)
    print(f"Training time: {time.time() - train_time:.4f} seconds")

    eval_time = time.time()
    evaluate_model(model, dataloader, device)
    print(f"Evaluation time: {time.time() - eval_time:.4f} seconds")

    vis_time = time.time()
    visualize_predictions(model, dataloader, device)
    print(f"Visualization time: {time.time() - vis_time:.4f} seconds")
    torch.save(model.state_dict(), "/Users/samyakarzazielbachiri/Documents/SegmentationAI/models/unet/u_net_model.pth")

if __name__ == "__main__":
    main()
    
