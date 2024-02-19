import os
import numpy as np
import PIL
from PIL import Image
import torch
from torchvision import transforms
from transformers import CLIPTokenizer  

# Hyperparameters 
IMAGE_SIZE = 128 
CAPTION_MAX_LENGTH = 30  
BATCH_SIZE = 4  # Start with a small batch size

# Paths
data_dir = "dataset"
images_dir = os.path.join(data_dir, "images")
captions_file = os.path.join(data_dir, "captions.txt")

# ImageNet statistics for normalization
imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess_image(image):  # 'image' should be a PIL Image object 
    """Preprocesses a PIL Image for the model"""
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    return transform(image) 



def tokenize_caption(caption):
    """Tokenizes a caption using a CLIP tokenizer"""
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    inputs = tokenizer(caption, padding="max_length", truncation=True, max_length=CAPTION_MAX_LENGTH, return_tensors="pt")
    return inputs


def load_dataset(data_dir):
    """Loads dataset with image verification and caption loading"""
    images_dir = os.path.join(data_dir, "images")
    captions_file = os.path.join(data_dir, "captions.txt")

    image_paths = []
    for filename in os.listdir(images_dir):
        filepath = os.path.join(images_dir, filename)
        try:
            img = Image.open(filepath).convert("RGB")  # Load as PIL Image
            image = preprocess_image(img)
            if torch.is_tensor(image):
                image = transforms.ToPILImage()(img) 
            image_paths.append(filepath)
        except (OSError, PIL.UnidentifiedImageError):
            print(f"Warning: Skipping invalid image file: {filepath}")

    captions = open(captions_file, "r").readlines()

    images = []
    text_inputs = []
    captions = open(captions_file, "r").readlines()

    #print("Image Paths:")  #  Added debugging
    #for path in image_paths:
        #print(path)

    #print("Captions:")  # Added debugging
    #for caption in captions:
        #print(caption.strip()) 

    assert len(image_paths) == len(captions), "Mismatch in number of images and captions"

    for image_path, caption in zip(image_paths, captions):
        image = preprocess_image(image_path)
        text_input = tokenize_caption(caption.strip())

        images.append(image)
        text_inputs.append(text_input)

    return images, text_inputs



# Example Usage (add data splitting later)
if __name__ == "__main__":
    images, text_inputs = load_dataset(data_dir)
