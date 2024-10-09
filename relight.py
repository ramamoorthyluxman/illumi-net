import torch
import numpy as np
from PIL import Image
from train import RelightingModel
from utils import params

class relight:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)
        self.patch_size = params.RTI_NET_PATCH_SIZE

    def load_model(self, model_path):
        model = RelightingModel(height=self.patch_size, width=self.patch_size)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model

    def reconstruct(self, distance_matrix, cosine_matrix, normal_map, albedo):
        # Ensure inputs are the correct shape and type
        distance_matrix = torch.FloatTensor(distance_matrix).unsqueeze(0).to(self.device)
        cosine_matrix = torch.FloatTensor(cosine_matrix).unsqueeze(0).to(self.device)
        normal_map = torch.FloatTensor(normal_map).to(self.device)
        albedo = torch.FloatTensor(albedo).unsqueeze(0).to(self.device)

        m, n = distance_matrix.shape[1:]  # Get dimensions

        # Prepare output tensor
        output = torch.zeros((m, n, 3), device=self.device)

        with torch.no_grad():
            for i in range(0, m, self.patch_size):
                for j in range(0, n, self.patch_size):
                    # Extract patch
                    i_end = min(i + self.patch_size, m)
                    j_end = min(j + self.patch_size, n)
                    
                    dist_patch = distance_matrix[:, i:i_end, j:j_end]
                    cos_patch = cosine_matrix[:, i:i_end, j:j_end]
                    normal_patch = normal_map[i:i_end, j:j_end].unsqueeze(0)
                    albedo_patch = albedo[:, i:i_end, j:j_end]
                    
                    # Forward pass
                    patch_output, _ = self.model(dist_patch, cos_patch, albedo_patch, normal_patch)
                    
                    # Place the output patch in the full output image
                    output[i:i_end, j:j_end, :] = patch_output.squeeze(0)

        # Convert to numpy and ensure it's in the correct range
        output = output.cpu().numpy()
        output = np.clip(output, 0, 1)
        output = (output * 255).astype(np.uint8)

        return output

# Example usage
# if __name__ == "__main__":
#     model_path = "path/to/your/saved_models/relighting_model_final.pth"
#     reconstructor = ImageReconstructor(model_path)

#     # Load your test data (you'll need to implement this based on how your data is stored)
#     # Ensure these are numpy arrays of shape (m, n) for distance and cosine matrices,
#     # (m, n, 3) for normal map, and (m, n) for albedo
#     distance_matrix = np.random.rand(256, 256).astype(np.float32)
#     cosine_matrix = np.random.rand(256, 256).astype(np.float32)
#     normal_map = np.random.rand(256, 256, 3).astype(np.float32)
#     albedo = np.random.rand(256, 256).astype(np.float32)

#     # Reconstruct the image
#     reconstructed_image = reconstructor.reconstruct(distance_matrix, cosine_matrix, normal_map, albedo)

#     # Save the reconstructed image
#     Image.fromarray(reconstructed_image).save("reconstructed_image.png")
#     print("Reconstructed image saved as reconstructed_image.png")