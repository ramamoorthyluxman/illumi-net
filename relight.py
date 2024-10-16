import torch
import numpy as np
import os
import cv2
from train import RelightingModel
from utils import params
from tqdm import tqdm

def load_model(model_path, albedo_channels):
    model = RelightingModel(albedo_channels=albedo_channels)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def relight_images(model, distances, cosines, albedo, normals):
    with torch.no_grad():
        outputs = []
        for i in tqdm(range(distances.shape[1]), desc="Relighting images", leave=False):
            distance = distances[0, i].unsqueeze(0)  # Shape: (1, 128, 128)
            cosine = cosines[0, i].unsqueeze(0)  # Shape: (1, 128, 128)
            albedo_input = albedo[0].unsqueeze(0)  # Shape: (1, 128, 128, 3)
            normals_input = normals[0].unsqueeze(0)  # Shape: (1, 128, 128, 3)
            
            
            output, _ = model(distance, cosine, albedo_input, normals_input)
                        
            output = output.squeeze(0).cpu().numpy()
            outputs.append(output)
    return np.array(outputs)

def save_images(images, output_dir, acq_idx):
    acq_dir = os.path.join(output_dir, f'acquisition_{acq_idx}')
    os.makedirs(acq_dir, exist_ok=True)
    
    for i, image in enumerate(images):
        image_clip =  np.clip(image, 0, 255)
        # Save the image
        cv2.imwrite(os.path.join(acq_dir, f'relit_image_{i}.png'),image_clip)

def relight(model_path, distances, cosines, albedo, normals, output_dir):
    distances = np.transpose(distances, (0,1,3,2))
    cosines = np.transpose(cosines, (0,1,3,2))
    albedo_channels = albedo.shape[-1]
    model = load_model(model_path, albedo_channels)
    
    os.makedirs(output_dir, exist_ok=True)

    # Convert numpy arrays to PyTorch tensors
    distances = torch.from_numpy(distances).float()
    cosines = torch.from_numpy(cosines).float()
    albedo = torch.from_numpy(albedo).float()
    normals = torch.from_numpy(normals).float()

    num_acquisitions = distances.shape[0]
    
    for acq in tqdm(range(num_acquisitions), desc="Processing acquisitions"):
        relit_images = relight_images(model, 
                                      distances[acq:acq+1], 
                                      cosines[acq:acq+1], 
                                      albedo[acq:acq+1], 
                                      normals[acq:acq+1])        
        
        save_images(relit_images, output_dir, acq)

    print(f"Relit images saved in {output_dir}")
