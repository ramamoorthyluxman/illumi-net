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
    """
    Relight images with proper tensor handling and shape management
    """
    with torch.no_grad():
        outputs = []
        batch_size = params.BATCH_SIZE  
        
        # Ensure correct shapes and data types
        distances = distances.float()
        cosines = cosines.float()
        albedo = albedo.float()
        normals = normals.float()
        
        # Calculate number of batches
        n_samples = distances.shape[1]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for batch in tqdm(range(n_batches), desc="Relighting images", leave=False):
            try:
                # Calculate batch indices
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)
                current_batch_size = end_idx - start_idx
                
                # Prepare batch inputs with correct shapes
                distance_batch = distances[0, start_idx:end_idx].to(device)
                cosine_batch = cosines[0, start_idx:end_idx].to(device)
                
                # Handle albedo and normals expansion correctly
                albedo_batch = albedo[0].expand(current_batch_size, -1, -1, -1).to(device)
                normals_batch = normals[0].expand(current_batch_size, -1, -1, -1).to(device)
                
                
                
                # Process batch
                output_batch, _ = model(distance_batch, cosine_batch, albedo_batch, normals_batch)
                
                # Move results back to CPU and append to outputs
                output_batch = output_batch.cpu().numpy()
                outputs.extend([img for img in output_batch])
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error in batch {batch}: {str(e)}")
                print(f"Last successful batch size: {len(outputs)}")
                raise e
                
    return np.array(outputs)

def relight(model_path, distances, cosines, albedo, normals, output_dir):
    """
    Main relighting function with improved error handling and memory management
    """
    try:
        # Get model input channels from albedo shape
        albedo_channels = albedo.shape[-1]
        model, device = load_model(model_path, albedo_channels)
        
        os.makedirs(output_dir, exist_ok=True)

        # Convert numpy arrays to PyTorch tensors with correct shape
        distances = torch.from_numpy(distances).float()
        cosines = torch.from_numpy(cosines).float()
        albedo = torch.from_numpy(albedo).float()
        normals = torch.from_numpy(normals).float()

        # Print input shapes for debugging
        print(f"Input shapes:")
        print(f"distances: {distances.shape}")
        print(f"cosines: {cosines.shape}")
        print(f"albedo: {albedo.shape}")
        print(f"normals: {normals.shape}")

        # Process acquisitions
        num_acquisitions = distances.shape[0]
        print(f"Processing {num_acquisitions} acquisitions...")
        
        for acq in tqdm(range(num_acquisitions), desc="Processing acquisitions"):
            try:
                # Get current GPU memory usage
                if torch.cuda.is_available():
                    get_memory_info()
                
                relit_images = relight_images(model, 
                                            distances[acq:acq+1], 
                                            cosines[acq:acq+1], 
                                            albedo[acq:acq+1], 
                                            normals[acq:acq+1],
                                            device)
                
                # Ensure the output directory exists
                acq_dir = os.path.join(output_dir, f'acquisition_{acq}')
                os.makedirs(acq_dir, exist_ok=True)
                
                # Save images
                for i, image in enumerate(relit_images):
                    output_path = os.path.join(acq_dir, f'relit_image_{i}.png')
                    # Convert to uint8 and ensure correct range
                    image = (image * 255).clip(0, 255).astype(np.uint8)
                    cv2.imwrite(output_path, image)
                
                # Clear GPU memory after each acquisition
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"GPU OOM for acquisition {acq}. Attempting recovery...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # Reduce batch size and retry
                    params.BATCH_SIZE = params.BATCH_SIZE // 2
                    if params.BATCH_SIZE < 1:
                        raise RuntimeError("Batch size too small, cannot continue")
                    print(f"Reduced batch size to {params.BATCH_SIZE}")
                    # Retry the current acquisition
                    acq -= 1
                else:
                    raise e
                    
    except Exception as e:
        print(f"Error during relighting: {str(e)}")
        raise e

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


def get_memory_info():
    """Helper function to print GPU memory usage"""
    if torch.cuda.is_available():
        print("\nGPU Memory Usage:")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_properties(i).name}")
            print(f"Allocated: {torch.cuda.memory_allocated(i)/1e9:.2f}GB")
            print(f"Cached: {torch.cuda.memory_reserved(i)/1e9:.2f}GB")