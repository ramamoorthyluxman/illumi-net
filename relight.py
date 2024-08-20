import torch
import numpy as np
from PIL import Image
import os
from train import RelightingModel
from utils import params

def load_model(model_path):
    model = RelightingModel(height=params.RTI_NET_PATCH_SIZE, width=params.RTI_NET_PATCH_SIZE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def prepare_input(distances, cosines, albedo, normals):
    # Convert numpy arrays to torch tensors and add batch dimension
    distances = torch.FloatTensor(distances).unsqueeze(0)
    cosines = torch.FloatTensor(cosines).unsqueeze(0)
    albedo = torch.FloatTensor(albedo).unsqueeze(0)
    normals = torch.FloatTensor(normals).unsqueeze(0)
    
    return distances, cosines, albedo, normals

def relight_image(model, distances, cosines, albedo, normals):
    with torch.no_grad():
        output, _ = model(distances, cosines, albedo, normals)
    return output.squeeze(0).cpu().numpy()

def save_image(image_array, output_path):
    # Ensure the output is in the range [0, 255]
    image_array = np.clip(image_array * 255, 0, 255).astype(np.uint8)
    image = Image.fromarray(image_array)
    image.save(output_path)

def test(distances, cosines, albedo, normals, model_path, output_dir):
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path).to(device)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    # Prepare input
    dist, cos, alb, norm = prepare_input(distances, cosines, albedo, normals)
    
    # Move input to the same device as the model
    dist = dist.to(device)
    cos = cos.to(device)
    alb = alb.to(device)
    norm = norm.to(device)
    
    # Relight the image
    relighted_image = relight_image(model, dist, cos, alb, norm)
    
    # Save the relighted image
    output_path = os.path.join(output_dir, f'relighted_image_{i}.png')
    save_image(relighted_image, output_path)
    print(f"Relighted image saved to {output_path}")

