import torch
import numpy as np
import os
import cv2
from train import RelightingModel
from utils import params
from tqdm import tqdm

def load_model(model_path, albedo_channels):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = RelightingModel(albedo_channels=albedo_channels)
    state_dict = torch.load(model_path, map_location=device)
    
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, device

def relight_images(model, distances, cosines, albedo, normals, device):
    with torch.no_grad():
        # Move inputs to device
        distances = distances.to(device)
        cosines = cosines.to(device)
        albedo = albedo.to(device)
        normals = normals.to(device)
        
        outputs = []
        # Process each light direction
        for i in range(distances.shape[1]):
            # Process batch through model
            output, _ = model(distances[:, i], cosines[:, i], albedo, normals)
            
            # Move to CPU and convert to numpy
            output = output.cpu().numpy()
            
            # Normalize like in train.py's save_comparison_images
            output = output[..., ::-1]  # BGR to RGB
            output = (output - output.min()) / (output.max() - output.min())
            
            outputs.append(output[0])  # Remove batch dimension
            
            # Clear GPU memory after each direction
            torch.cuda.empty_cache()
            
        return np.array(outputs)

def save_images(images, output_dir, acq_idx):
    print(f"Saving acquisition {acq_idx}")
    acq_dir = os.path.join(output_dir, f'acquisition_{acq_idx}')
    os.makedirs(acq_dir, exist_ok=True)
    
    for i, image in enumerate(images):
        # Convert to BGR for OpenCV
        image_bgr = (image[..., ::-1] * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(acq_dir, f'relit_image_{i}.png'), image_bgr)

def relight(model_path, distances, cosines, albedo, normals, output_dir):
    albedo_channels = albedo.shape[-1]
    model, device = load_model(model_path, albedo_channels)
    
    os.makedirs(output_dir, exist_ok=True)

    # Convert to tensors
    distances = torch.from_numpy(distances).float()
    cosines = torch.from_numpy(cosines).float()
    albedo = torch.from_numpy(albedo).float()
    normals = torch.from_numpy(normals).float()

    # Process one acquisition at a time
    for acq in tqdm(range(distances.shape[0])):
        try:
            torch.cuda.empty_cache()  # Clear GPU memory
            
            relit_images = relight_images(
                model,
                distances[acq:acq+1],
                cosines[acq:acq+1],
                albedo[acq:acq+1],
                normals[acq:acq+1],
                device
            )
            
            save_images(relit_images, output_dir, acq)
            
        except RuntimeError as e:
            print(f"Error processing acquisition {acq}: {str(e)}")

    print(f"Relit images saved in {output_dir}")