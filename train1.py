import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from model import UNet
from data_prep import images, text_inputs  # Assuming your `data_prep.py` remains the same

# Hyperparameters
num_epochs = 20
learning_rate = 1e-4
BATCH_SIZE = 4

# Device Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model (Image Segmentation)
model = UNet(in_channels=3, out_channels=3).to(device)  # Assuming 3-channel segmentation output

# Loss Function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Image Transformation for Augmentation (Optional)
image_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Add flips, rotations, etc., for robustness
    transforms.ToTensor()  # Assuming data_prep doesn't already do this
])

# Datasets (Adapt later, but here's the segmentation task  setup) 
class ImageSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, images, text_inputs, targets, transform=None):
        self.images = images
        self.text_inputs = text_inputs
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        text_input = self.text_inputs[idx]
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, text_input, target

# Assuming 'target_images' variable exists - Replace this with your real target data!
image_target_pairs = list(zip(images, text_inputs, target_images)) 
dataset = ImageSegmentationDataset(images, text_inputs, target_images, transform=image_transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Training Loop
for epoch in range(num_epochs):
    for images, text_inputs, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        # Text Feature Processing (Placeholder)
        text_features = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").get_text_features(**text_inputs)

        optimizer.zero_grad()
        outputs = model(images, text_features)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# Model Saving (Optional)
torch.save(model.state_dict(), 'trained_segmentation_model.pth')

