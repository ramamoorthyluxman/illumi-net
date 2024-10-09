import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from utils import params
import random
import math
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import torchvision.models as models
import torchvision.transforms as transforms


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features.eval()
        self.vgg_layers = nn.ModuleList(vgg)
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
        
    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return output

class CombinedLoss(nn.Module):
    def __init__(self, lambda_mse=1.0, lambda_perceptual=0.1):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()
        self.lambda_mse = lambda_mse
        self.lambda_perceptual = lambda_perceptual
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, output, target):
        # MSE Loss
        mse_loss = self.mse_loss(output, target)
        
        # Perceptual Loss
        output_normalized = self.normalize(output.permute(0, 3, 1, 2))  # Change from NHWC to NCHW and normalize
        target_normalized = self.normalize(target.permute(0, 3, 1, 2))  # Change from NHWC to NCHW and normalize
        
        output_features = self.perceptual_loss(output_normalized)
        target_features = self.perceptual_loss(target_normalized)
        
        perceptual_loss = 0
        for key in output_features.keys():
            perceptual_loss += self.mse_loss(output_features[key], target_features[key])
        
        # Combined Loss
        total_loss = self.lambda_mse * mse_loss + self.lambda_perceptual * perceptual_loss
        
        return total_loss, mse_loss, perceptual_loss

class PatchRelightingDataset(Dataset):
    def __init__(self, distances, cosines, albedo, normals, targets):
        self.distances = torch.FloatTensor(distances)
        self.cosines = torch.FloatTensor(cosines)
        self.albedo = torch.ByteTensor(albedo)
        self.normals = torch.ByteTensor(normals)
        self.targets = torch.ByteTensor(targets)
        self.patch_size, self.patches_per_image = self.calculate_patches() 
        print("Adjusted patch size, number of patches: ", self.patch_size, self.patches_per_image)       
        self.patches = self._create_patches()  

    def calculate_patches(self):
        print("Calculating the patches")
        patch_height, patch_width = params.RTI_NET_PATCH_SIZE
        desired_patches = params.RTI_MAX_NUMBER_PATCHES
        _, _, image_height, image_width = self.distances.shape
        # Case 4: Patch size greater than image size
        patch_width = min(patch_width, image_width)
        patch_height = min(patch_height, image_height)

        # Calculate initial number of patches
        patches_x = max(1, image_width // patch_width)
        patches_y = max(1, image_height // patch_height)
        total_patches = patches_x * patches_y

        # Case 3: Desired patches is 0 (auto-calculate)
        if desired_patches == 0:
            new_width = patch_width
            new_height = patch_height
            for i in range(patch_width, 0, -1):
                if image_width % i == 0:
                    new_width = i
                    break
            for i in range(patch_height, 0, -1):
                if image_height % i == 0:
                    new_height = i
                    break
            return new_height, new_width, (image_width // new_width) * (image_height // new_height)

        # Case 1 & 2: Adjust patch size to fit desired patches or as close as possible
        if desired_patches <= total_patches:
            target_patches = max(1, min(desired_patches, total_patches))
        else:
            target_patches = desired_patches

        # Find the best divisors
        best_patches_x = 1
        best_patches_y = 1
        for i in range(1, target_patches + 1):
            if image_width % i == 0 and image_height % (target_patches // i) == 0:
                best_patches_x = i
                best_patches_y = target_patches // i
                if best_patches_x * best_patches_y == target_patches:
                    break

        new_width = image_width // best_patches_x
        new_height = image_height // best_patches_y

        # Ensure new patch sizes are not smaller than original
        if new_width < patch_width or new_height < patch_height:
            return patch_height, patch_width, total_patches

        return new_height, new_width, best_patches_x * best_patches_y

    def _create_patches(self):
        patches = []
        K, N, H, W = self.distances.shape
        grid_size = math.isqrt(self.patches_per_image)  # Square root of patches_per_image
        cell_h, cell_w = H // grid_size, W // grid_size
        
        for k in range(K):
            for n in range(N):
                for _ in range(self.patches_per_image):
                    # Randomly select a grid cell
                    i = random.randint(0, grid_size - 1)
                    j = random.randint(0, grid_size - 1)
                    
                    # Calculate the range for this grid cell
                    h_start = i * cell_h
                    h_end = min((i + 1) * cell_h, H) - self.patch_size[0]
                    w_start = j * cell_w
                    w_end = min((j + 1) * cell_w, W) - self.patch_size[1]
                    
                    # Randomly select a patch within this grid cell
                    h = random.randint(h_start, max(h_start, h_end))
                    w = random.randint(w_start, max(w_start, w_end))
                    
                    patches.append((k, n, h, w))
        
        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        k, n, i, j = self.patches[idx]
        patch_slice = slice(i, i + self.patch_size[0]), slice(j, j + self.patch_size[1])

        return {
            'distances': self.distances[k, n][patch_slice],
            'cosines': self.cosines[k, n][patch_slice],
            'albedo': self.albedo[k][patch_slice].float() / 255.0,
            'normals': self.normals[k][patch_slice].float(),
            'target': self.targets[k, n][patch_slice].float() / 255.0
        }

class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(ResidualBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.in1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.in2 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(out_channels)
        ) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.in2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class AttentionBlock2D(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock2D, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // 16)
        self.fc2 = nn.Linear(in_channels // 16, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gap = self.gap(x).view(x.size(0), -1)
        fc1 = self.fc1(gap)
        fc2 = self.fc2(torch.relu(fc1))
        attention = self.sigmoid(fc2).view(x.size(0), x.size(1), 1, 1)
        return x * attention

class RelightingModel(nn.Module):
    def __init__(self, height, width):
        super(RelightingModel, self).__init__()
        
        # Initial convolutions
        self.conv_distances = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv_cosines = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv_albedo = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv_normals = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        
        # Encoder with dilated convolutions
        self.encoder1 = ResidualBlock2D(192, 128, dilation=1)
        self.encoder2 = ResidualBlock2D(128, 256, dilation=2)
        self.encoder3 = ResidualBlock2D(256, 512, dilation=4)
        self.encoder4 = ResidualBlock2D(512, 1024, dilation=8)
        
        # Bridge
        self.bridge = AttentionBlock2D(1024)
        
        # Decoder
        self.decoder3 = ResidualBlock2D(1536, 512)
        self.decoder2 = ResidualBlock2D(768, 256)
        self.decoder1 = ResidualBlock2D(384, 128)
        
        # Output
        self.final_conv = nn.Conv2d(128, 3, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, distances, cosines, albedo, normals):
        # Reshape inputs to add channel dimension
        distances = distances.unsqueeze(1)
        cosines = cosines.unsqueeze(1)
        albedo = albedo.unsqueeze(1)
        normals = normals.permute(0, 3, 1, 2)

        # Initial processing
        x1 = self.conv_distances(distances)
        x2 = self.conv_cosines(cosines)
        x3 = self.conv_albedo(albedo)
        x4 = self.conv_normals(normals)
        
        x = torch.cat([x1, x2, x3, x4], dim=1)
        
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        
        # Bridge
        bridge = self.bridge(e4)
        
        # Decoder
        d3 = self.decoder3(torch.cat([self.upsample(bridge), e3], dim=1))
        d2 = self.decoder2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.upsample(d2), e1], dim=1))
        
        # Output
        out = self.final_conv(d1)
        
        # Reshape the output to match the target shape (N, H, W, 3)
        out = out.permute(0, 2, 3, 1)

        intermediates = {
            'e1': e1, 'e2': e2, 'e3': e3, 'e4': e4,
            'd3': d3, 'd2': d2, 'd1': d1,
            'final': out
        }
        return out, intermediates
    
def visualize_output(model, val_loader, device):
    # Get a random batch from the validation set
    batch = next(iter(val_loader))
    
    # Select a random sample from the batch
    idx = random.randint(0, batch['distances'].shape[0] - 1)
    
    distances = batch['distances'][idx:idx+1].to(device)
    cosines = batch['cosines'][idx:idx+1].to(device)
    albedo = batch['albedo'][idx:idx+1].to(device)
    normals = batch['normals'][idx:idx+1].to(device)
    target = batch['target'][idx:idx+1].to(device)

    model.eval()
    with torch.no_grad():
        output, _ = model(distances, cosines, albedo, normals)

    # Convert tensors to numpy arrays for plotting
    output = output.cpu().numpy()[0]
    target = target.cpu().numpy()[0]

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the ground truth
    ax1.imshow(target)
    ax1.set_title('Ground Truth')
    ax1.axis('off')

    # Plot the model output
    ax2.imshow(output)
    ax2.set_title('Model Output')
    ax2.axis('off')

    plt.tight_layout()
    return fig

def visualize_comparisons(model, val_loader, device, num_samples=10):
    model.eval()
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5*num_samples))
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_samples:
                break
            
            distances = batch['distances'].to(device)
            cosines = batch['cosines'].to(device)
            albedo = batch['albedo'].to(device)
            normals = batch['normals'].to(device)
            targets = batch['target'].to(device)

            outputs, _ = model(distances, cosines, albedo, normals)

            # Convert tensors to numpy arrays for plotting
            original = targets[0].cpu().numpy()
            reconstructed = outputs[0].cpu().numpy()

            # Plot original image
            axes[i, 0].imshow(original)
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')

            # Plot reconstructed image
            axes[i, 1].imshow(reconstructed)
            axes[i, 1].set_title('Reconstructed')
            axes[i, 1].axis('off')

    plt.tight_layout()
    return fig

def train_model(model, train_loader, val_loader, num_epochs=100, model_save_path='.'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = CombinedLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')

    # For live plotting
    train_losses = []
    val_losses = []
    epochs = []
    
    # For accumulating output
    output_text = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            distances = batch['distances'].to(device)
            cosines = batch['cosines'].to(device)
            albedo = batch['albedo'].to(device)
            normals = batch['normals'].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()
            outputs, _ = model(distances, cosines, albedo, normals)
            loss, mse, perceptual = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                distances = batch['distances'].to(device)
                cosines = batch['cosines'].to(device)
                albedo = batch['albedo'].to(device)
                normals = batch['normals'].to(device)
                targets = batch['target'].to(device)

                outputs, _ = model(distances, cosines, albedo, normals)
                loss, _, _ = criterion(outputs, targets)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        output_text.append(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Update plot data
        epochs.append(epoch + 1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Clear the previous output and create a new plot
        clear_output(wait=True)
        
        # Display accumulated text output
        print("\n".join(output_text))
        
        # Create and display the loss plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, train_losses, 'b-', label='Train Loss')
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        display(fig)
        plt.close(fig)  # Close the figure to free up memory
        
        # Visualize image comparisons every 10 epochs (or adjust as needed)
        if (epoch + 1) % 10 == 0:
            comparison_fig = visualize_comparisons(model, val_loader, device)
            display(comparison_fig)
            plt.close(comparison_fig)
        
        scheduler.step(val_loss)

        # Save model every N epochs
        if (epoch + 1) % params.RTI_NET_SAVE_MODEL_EVERY_N_EPOCHS == 0:
            save_path = os.path.join(model_save_path, f'relighting_model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)
            output_text.append(f"Model saved at epoch {epoch+1}")

    return model

def prepare_data(distances, cosines, albedo, normals, targets):
    # Splitting the data
    K = distances.shape[0]
    N = distances.shape[1]
    train_distances = []
    val_distances = []
    train_cosines = []
    val_cosines = []
    train_targets = []   
    val_targets = []

    for i in range(K):
        train_indices, val_indices = train_test_split(
            np.arange(N), 
            test_size=0.2, 
            random_state=42
        )
        train_distances.append(distances[i, train_indices, :, :])
        val_distances.append(distances[i, val_indices, :, :])
        train_cosines.append(cosines[i, train_indices, :, :])
        val_cosines.append(cosines[i, val_indices, :, :])
        train_targets.append(targets[i, train_indices, :, :])
        val_targets.append(targets[i, val_indices, :, :])

    print("Train distances shape: ", np.array(train_distances).shape,
          "Val distances shape: ", np.array(val_distances).shape,
          "Train cosines shape: ", np.array(train_cosines).shape,
          "Val cosines shape: ", np.array(val_cosines).shape,
          "Train targets shape: ", np.array(train_targets).shape,
          "Val targets shape: ", np.array(val_targets).shape)

    # Create datasets
    train_dataset = PatchRelightingDataset(
        train_distances, train_cosines,
        albedo, normals,
        train_targets
    )
    val_dataset = PatchRelightingDataset(
        val_distances, val_cosines,
        albedo, normals,
        val_targets
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=14)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=14)

    return train_loader, val_loader, train_indices, val_indices

def train(distances, cosines, albedo, normals, targets):

    distances = np.transpose(distances, (0,1,3,2))
    cosines = np.transpose(cosines, (0,1,3,2))

    print("Input shapes - distances: ", distances.shape, "cosines: ", cosines.shape, 
          "albedo: ", albedo.shape, "normals: ", normals.shape, "targets: ", targets.shape)
    
    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
    os.makedirs(model_save_path, exist_ok=True)

    # Prepare data
    train_loader, val_loader, train_indices, val_indices = prepare_data(distances, cosines, albedo, normals, targets)

    np.save(os.path.join(model_save_path, 'train_indices.npy'), train_indices)
    np.save(os.path.join(model_save_path, 'val_indices.npy'), val_indices)

    # Initialize the model
    model = RelightingModel(height=params.RTI_NET_PATCH_SIZE[0], width=params.RTI_NET_PATCH_SIZE[1])

    # Train the model
    trained_model = train_model(model, train_loader, val_loader, num_epochs=params.RTI_NET_EPOCHS, model_save_path=model_save_path)

    # Save the final model
    torch.save(trained_model.state_dict(), os.path.join(model_save_path, 'relighting_model_final.pth'))

    print("Training completed and model saved.")