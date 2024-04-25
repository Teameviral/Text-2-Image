import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Two Convolutions + ReLU block"""
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

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is not None:
            combined_input = torch.cat((input_ids, attention_mask), dim=-1)
        else:
            combined_input = input_ids

        combined_input = combined_input.float()  # Ensure correct data type
        reshaped_input = combined_input.view(combined_input.size(0), -1)
        output = self.fc(reshaped_input)
        return self.relu(output)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, text_input_size=512, text_output_size=256):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.maxpool = nn.MaxPool2d(2)
        self.text_module = TextProcessingModule(text_input_size, text_output_size)
        self.up1 = nn.ConvTranspose2d(512 + text_output_size, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512 + text_output_size, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)
        self.up4 = nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(64, out_channels)

    def forward(self, x, input_ids, attention_mask=None):
        x1 = self.down1(x)
        x2 = self.maxpool(x1)
        x3 = self.down2(x2)
        x4 = self.maxpool(x3)
        x5 = self.down3(x4)
        x6 = self.maxpool(x5)
        x7 = self.down4(x6)

        processed_text = self.text_module(input_ids, attention_mask)
        processed_text = processed_text.view(x7.size(0), -1, 1, 1).expand(-1, -1, x7.size(2), x7.size(3))

        x7 = torch.cat([x7, processed_text], dim=1)

        x = self.up1(x7)
        x = torch.cat([x, x6], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x5], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        x = torch.cat([x, x4], dim=1)
        x = self.conv3(x)
        x = self.up4(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv4(x)

        return x
