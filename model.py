import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class TextProcessingModule(nn.Module):
    def __init__(self, input_size, output_size):
        super(TextProcessingModule, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, text_features):
        try:
            input_ids = text_features.get('input_ids')
            attention_mask = text_features.get('attention_mask')

            if input_ids is None or attention_mask is None:
                raise ValueError("Invalid format for text_features")

            combined_input = torch.cat((input_ids, attention_mask), dim=-1)
            combined_input = combined_input.float()
            reshaped_input = combined_input.view(combined_input.size(0), -1)

        except AttributeError:
            input_ids = text_features
            attention_mask = None
            reshaped_input = input_ids.view(input_ids.size(0), -1)

        output = self.fc(reshaped_input)
        return self.relu(output)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, text_input_size=512, text_output_size=256):
        super().__init__()

        # Encoder (Downsampling)
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.maxpool = nn.MaxPool2d(2)

        # Text Processing Module
        self.text_module = TextProcessingModule(text_input_size, text_output_size)

        # Initialize the linear layer with dynamic input size
        sample_text_features = {'input_ids': torch.zeros(1, 1, 30), 'attention_mask': torch.zeros(1, 1, 30)}
        sample_processed_text = self.text_module(sample_text_features)

        # Get the size of the processed text dynamically
        text_output_size_dynamic = sample_processed_text.view(sample_processed_text.size(0), -1).size(1)

        # Decoder (Upsampling)
        self.up1 = nn.ConvTranspose2d(512 + text_output_size_dynamic, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512 + text_output_size_dynamic, 256)  # Skip Connection
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)
        self.up4 = nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(64, out_channels)

        # Linear layer
        self.fc = nn.Linear(text_output_size_dynamic, out_channels)

    def forward(self, x, text_features):
        # Encoder steps
        x1 = self.down1(x)
        x2 = self.maxpool(x1)
        x3 = self.down2(x2)
        x4 = self.maxpool(x3)
        x5 = self.down3(x4)
        x6 = self.maxpool(x5)
        x7 = self.down4(x6)

        # Process text_features
        processed_text = self.text_module(text_features)

        # Expand processed_text dimensions to match x7
        processed_text = processed_text.unsqueeze(-1).unsqueeze(-1).expand_as(x7)

        # Concatenate processed_text with x7 (bottleneck)
        x7 = torch.cat([x7, processed_text], dim=1)

        # Decoder Steps
        x = self.up1(x7)
        x = torch.cat([x, x6], dim=1)  # Skip connection
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x5], dim=1)  # Skip connection
        x = self.conv2(x)
        x = self.up3(x)
        x = torch.cat([x, x4], dim=1)  # Skip connection
        x = self.conv3(x)
        x = self.up4(x)
        x = torch.cat([x, x3], dim=1)  # Skip connection
        x = self.conv4(x)

        return x
