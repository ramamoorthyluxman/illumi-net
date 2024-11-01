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
        total_images = distances.shape[1]
        current_idx = 0
        
        while current_idx < total_images:
            # Process in chunks of maximum size params.RTI_MAX_IMAGES_PER_CHUNK
            end_idx = min(current_idx + params.RTI_MAX_IMAGES_PER_CHUNK, total_images)
            chunk_size = end_idx - current_idx
            
            # Calculate number of batches for this chunk
            n_batches = (chunk_size + params.BATCH_SIZE - 1) // params.BATCH_SIZE
            
            chunk_outputs = []
            for batch in tqdm(range(n_batches), desc=f"Processing images {current_idx} to {end_idx-1}", leave=False):
                # Calculate batch indices within the chunk
                start_batch = batch * params.BATCH_SIZE
                end_batch = min((batch + 1) * params.BATCH_SIZE, chunk_size)
                
                # Calculate absolute indices
                abs_start = current_idx + start_batch
                abs_end = current_idx + end_batch
                
                # Prepare batch inputs
                distance_batch = distances[0, abs_start:abs_end].to(device)
                cosine_batch = cosines[0, abs_start:abs_end].to(device)
                albedo_batch = albedo[0].unsqueeze(0).expand(end_batch - start_batch, -1, -1, -1).to(device)
                normals_batch = normals[0].unsqueeze(0).expand(end_batch - start_batch, -1, -1, -1).to(device)
                
                # Process batch
                output_batch, _ = model(distance_batch, cosine_batch, albedo_batch, normals_batch)
                
                # Move results back to CPU and convert to numpy
                output_batch = output_batch.cpu().numpy()
                chunk_outputs.extend([img for img in output_batch])
                
                # Clear GPU cache after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Add chunk outputs to main outputs list
            outputs.extend(chunk_outputs)
            current_idx = end_idx
                
    return np.array(outputs)

def save_images(images, output_dir, acq_idx):
    print("Saving acq idx ", acq_idx)
    acq_dir = os.path.join(output_dir, f'acquisition_{acq_idx}')
    os.makedirs(acq_dir, exist_ok=True)
    
    # Use parallel processing for saving images
    from concurrent.futures import ThreadPoolExecutor
    
    def save_single_image(args):
        i, image = args
        print("Saving ", os.path.join(acq_dir, f'relit_image_{i}.png'))
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
    print(f"Processing {num_acquisitions} acquisitions...")
    
    for acq in tqdm(range(num_acquisitions), desc="Processing acquisitions"):
        try:
            relit_images = relight_images(model, 
                                        distances[acq:acq+1], 
                                        cosines[acq:acq+1], 
                                        albedo[acq:acq+1], 
                                        normals[acq:acq+1],
                                        device)
            
            save_images(relit_images, output_dir, acq)
            
            # Clear GPU memory after each acquisition
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"GPU OOM for acquisition {acq}. Trying with reduced batch size...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Temporarily reduce batch size for this acquisition
                original_batch_size = params.BATCH_SIZE
                params.BATCH_SIZE = max(1, params.BATCH_SIZE // 2)
                print(f"Reduced batch size to {params.BATCH_SIZE}")
                
                # Retry the current acquisition
                relit_images = relight_images(model, 
                                            distances[acq:acq+1], 
                                            cosines[acq:acq+1], 
                                            albedo[acq:acq+1], 
                                            normals[acq:acq+1],
                                            device)
                
                save_images(relit_images, output_dir, acq)
                
                # Restore original batch size
                params.BATCH_SIZE = original_batch_size
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