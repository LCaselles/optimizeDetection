import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import fire

def fuse_model(model):
    """
    Fuse Conv2d and BatchNorm2d layers in the model for optimization.

    Args:
        model (torch.nn.Module): The model to be fused.

    Returns:
        torch.nn.Module: The fused model.
    """
    for m in model.modules():
        if isinstance(m, nn.Sequential):
            if len(list(m.children())) == 2:
                if isinstance(m[0], nn.Conv2d) and isinstance(m[1], nn.BatchNorm2d):
                    torch.quantization.fuse_modules(m, [0, 1], inplace=True)
    return model

def load_calibration_data(path_img, num_samples=50):
    """
    Load and preprocess calibration images.

    Args:
        path_img (str): Path to the images directory.
        num_samples (int): Number of random samples to select.

    Returns:
        torch.Tensor: The preprocessed calibration data.
    """
    image_files = os.listdir(path_img)
    sampled_files = np.random.choice(image_files, num_samples, replace=False)
    inputs = [transform(Image.open(os.path.join(path_img, file))).unsqueeze(0) for file in sampled_files]
    return torch.cat(inputs)

def main(path_img="dataset/dataset/images/"):
    """
    Main function to load, process, and save the quantized model.

    Args:
        path_img (str): Path to the directory containing images for calibration.
    """
    # Define preprocessing transforms for images
    global transform
    transform = transforms.Compose([
        transforms.Resize((372, 1230)),
        transforms.ToTensor()
    ])
    
    # Load and preprocess calibration data
    calibration_inputs = load_calibration_data(path_img)

    # Load and prepare the model
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    model.eval()
    model = fuse_model(model)

    # Quantization configuration
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_prepared = torch.quantization.prepare(model)

    # Calibrate the model with dummy data
    model_prepared(calibration_inputs)
    model_quantized = torch.quantization.convert(model_prepared)

    # Dynamic quantization
    model_dynamic_quantized = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )

    # Save the models
    torch.save(model_dynamic_quantized.state_dict(), 'quantized_model.pth')
    torch.save(model.state_dict(), 'non_quantized_model.pth')
    torch.save(model_dynamic_quantized, 'quantized_model_full.pth')
    torch.save(model, 'non_quantized_model_full.pth')

if __name__ == "__main__":
    fire.Fire(main)
