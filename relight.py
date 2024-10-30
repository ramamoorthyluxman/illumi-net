import torch
import numpy as np
import os
import cv2
from train import RelightingModel
from utils import params
from tqdm import tqdm

def load_model(model_path, albedo_channels):
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = RelightingModel(albedo_channels=albedo_channels)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

def relight_images(model, distances, cosines, albedo, normals, device):
    with torch.no_grad():
        outputs = []
        batch_size = 32  # Adjust based on your GPU memory
        
        # Calculate number of batches
        n_samples = distances.shape[1]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for batch in tqdm(range(n_batches), desc="Relighting images", leave=False):
            # Calculate batch indices
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, n_samples)
            
            # Prepare batch inputs
            distance_batch = distances[0, start_idx:end_idx].to(device)  # Move to GPU
            cosine_batch = cosines[0, start_idx:end_idx].to(device)
            albedo_batch = albedo[0].unsqueeze(0).expand(end_idx - start_idx, -1, -1, -1).to(device)
            normals_batch = normals[0].unsqueeze(0).expand(end_idx - start_idx, -1, -1, -1).to(device)
            
            # Process batch
            output_batch, _ = model(distance_batch, cosine_batch, albedo_batch, normals_batch)
            
            # Move results back to CPU and convert to numpy
            output_batch = output_batch.cpu().numpy()
            outputs.extend([img for img in output_batch])
            
            # Clear GPU cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    return np.array(outputs)

def save_images(images, output_dir, acq_idx):
    acq_dir = os.path.join(output_dir, f'acquisition_{acq_idx}')
    os.makedirs(acq_dir, exist_ok=True)
    
    # Use parallel processing for saving images
    from concurrent.futures import ThreadPoolExecutor
    
    def save_single_image(args):
        i, image = args
        cv2.imwrite(os.path.join(acq_dir, f'relit_image_{i}.png'), image)
    
    with ThreadPoolExecutor() as executor:
        executor.map(save_single_image, enumerate(images))

def relight(model_path, distances, cosines, albedo, normals, output_dir):
    albedo_channels = albedo.shape[-1]
    model, device = load_model(model_path, albedo_channels)
    
    os.makedirs(output_dir, exist_ok=True)

    # Convert numpy arrays to PyTorch tensors
    distances = torch.from_numpy(distances).float()
    cosines = torch.from_numpy(cosines).float()
    albedo = torch.from_numpy(albedo).float()
    normals = torch.from_numpy(normals).float()

    # Process acquisitions
    num_acquisitions = distances.shape[0]
    
    for acq in tqdm(range(num_acquisitions), desc="Processing acquisitions"):
        try:
            relit_images = relight_images(model, 
                                        distances[acq:acq+1], 
                                        cosines[acq:acq+1], 
                                        albedo[acq:acq+1], 
                                        normals[acq:acq+1],
                                        device)
            
            save_images(relit_images, output_dir, acq)
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"GPU OOM for acquisition {acq}. Trying with smaller batch size...")
                # Could implement fallback to smaller batch size here
            else:
                raise e

    print(f"Relit images saved in {output_dir}")

def get_memory_info():
    """Helper function to print GPU memory usage"""
    if torch.cuda.is_available():
        print("\nGPU Memory Usage:")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_properties(i).name}")
            print(f"Allocated: {torch.cuda.memory_allocated(i)/1e9:.2f}GB")
            print(f"Cached: {torch.cuda.memory_reserved(i)/1e9:.2f}GB")