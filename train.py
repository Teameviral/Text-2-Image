import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from torchvision import transforms
from PIL import Image
from torchvision.models import vgg16
import torchvision.transforms.functional as F
import torch.nn.functional as FF

from data_prep import load_dataset
from model import UNet, TextProcessingModule

# Hyperparameters
EPOCHS = 20
BATCH_SIZE = 4
LEARNING_RATE = 0.001
NUM_FOLDS = 5  # Number of folds for k-fold cross-validation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data Preparation
data_dir = "dataset"
images, text_inputs, image_paths = load_dataset(data_dir)

# K-Fold Preparation
kf = KFold(n_splits=NUM_FOLDS, shuffle=True)

# Load pre-trained VGG16 model for feature extraction
vgg16_model = vgg16(pretrained=True).features[:23].to(DEVICE).eval()

class YourCustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, captions):
        self.images = images
        self.captions = captions

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        caption = self.captions[idx]

        # Preprocess the image if needed
        transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Adjust size as needed
            transforms.ToTensor(),
        ])
        image = transform(image)

        return image, caption



# Outer Loop for K-Fold Cross-Validation
for fold, (train_idx, val_idx) in enumerate(kf.split(images)):
    # Prepare data for this fold
    train_images = [images[i] for i in train_idx]
    train_captions = [text_inputs[i] for i in train_idx]
    val_images = [images[i] for i in val_idx]
    val_captions = [text_inputs[i] for i in val_idx]

    train_dataset = YourCustomDataset(train_images, train_captions)
    val_dataset = YourCustomDataset(val_images, val_captions)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Model, Loss, Optimizer (for each fold)
    model = UNet(in_channels=3, out_channels=3, text_input_size=512, text_output_size=256).to(DEVICE)
    feature_loss = nn.MSELoss()  # You can experiment with different feature loss functions
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop (with regularization)
for epoch in range(EPOCHS):
    model.train()
    for batch_idx, (image, caption) in enumerate(train_loader):
        image, caption = image.to(DEVICE), caption.to(DEVICE)

        # Assuming text_features are obtained by processing captions
        text_features = TextProcessingModule(input_size=60, output_size=256)(caption)

        optimizer.zero_grad()
        output = model(image, text_features)
        loss = feature_loss(output, image)

        # Add regularization (e.g., L1/L2 weight decay)
        l1_reg = 0.001  # Example L1 regularization strength
        l1_penalty = sum(p.abs().sum() for p in model.parameters())
        loss += l1_reg * l1_penalty

        loss.backward()
        optimizer.step()

        # ... (Printing loss as before) ...

    # Validation Loop
    model.eval()
    with torch.no_grad():
        for val_image, val_caption in val_loader:
            val_image, val_caption = val_image.to(DEVICE), val_caption.to(DEVICE)
            val_text_features = TextProcessingModule(input_size=YOUR_INPUT_SIZE, output_size=YOUR_OUTPUT_SIZE)(val_caption)
            val_output = model(val_image, val_text_features)
            val_loss = feature_loss(val_output, val_image)
            # ... (Additional validation metrics or logging) ...

# Save the model for this fold

    torch.save(model.state_dict(), f"model_fold_{fold}.pth")
