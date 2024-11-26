import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import nrrd
from torchvision.transforms import Compose, Resize, ToTensor
import sys
#sys.path.append("/Users/samyakarzazielbachiri/Documents/SegmentationAI/models/sam2/")
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra


from hydra.core.config_store import ConfigStore

print(f"Hydra search path: {ConfigStore.instance().repo}")

# Dataset class for loading images and masks
class BoneSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nrrd')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nrrd')])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        image, _ = nrrd.read(img_path)
        mask, _ = nrrd.read(mask_path)

        # Normalize image and convert mask to integer
        image = (image - image.min()) / (image.max() - image.min())
        mask = mask.astype('int64')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

def load_sam2_model(config_path, checkpoint_path):
    """
    Load SAM 2 model using the provided configuration and checkpoint.
    """
    config_dir = os.path.dirname(config_path)
    config_name = os.path.basename(config_path)

    # Clear any previous Hydra initialization
    from hydra.core.global_hydra import GlobalHydra
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # Initialize Hydra with the config directory
    from hydra import initialize_config_dir, compose
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)

    print(f"Configuration Loaded: {cfg}")
    # Pass the cfg dictionary directly to build_sam2
    model = build_sam2(cfg, checkpoint_path)
    return SAM2ImagePredictor(model)

# Training Loop for Fine-Tuning
def finetune_model(model, dataloader, optimizer, criterion, device, epochs=10):
    model.model.train()  # Set the model to training mode
    for epoch in range(epochs):
        total_loss = 0
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model.model(images)
            loss = criterion(outputs, masks)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

    # Save the fine-tuned model
    torch.save(model.model.state_dict(), "finetuned_sam2.pth")
    print("Fine-tuned model saved as 'finetuned_sam2.pth'")

# Main Function
def main():
    path = "/Users/samyakarzazielbachiri/Documents/SegmentationAI/"
    print(f"Current working directory: {path}")
    image_dir = os.path.join(path, "data/segmentai_dataset/images")
    mask_dir = os.path.join(path, "data/segmentai_dataset/masks")

    config_path = os.path.join(path, "models/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml")
    print(f"Config path: {config_path}")
    checkpoint_path = os.path.join(path, "models/sam2/checkpoints/sam2.1_hiera_large.pt")

    batch_size = 4
    learning_rate = 0.001
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transformations and DataLoader
    transform = Compose([Resize((128, 128)), ToTensor()])
    dataset = BoneSegmentationDataset(image_dir, mask_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load SAM 2 model
    predictor = load_sam2_model(config_path, checkpoint_path)
    predictor.model.to(device)

    # Freeze specific layers (optional)
    for name, param in predictor.model.named_parameters():
        if "image_encoder" in name:  # Freeze image encoder layers
            param.requires_grad = False

    # Loss and optimizer
    criterion = CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, predictor.model.parameters()), lr=learning_rate)

    # Fine-tune the model
    finetune_model(predictor, dataloader, optimizer, criterion, device, epochs)

if __name__ == "__main__":
    main()