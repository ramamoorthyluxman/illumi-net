# Train.py
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
import torchvision.models as models
import torchvision.transforms as transforms
import subprocess
import sys
from datetime import datetime

# Set visible GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = params.CUDA_VISIBLE_DEVICES  # Modify this according to your needs

def get_gpu_memory_stats():
    """
    Get the GPU memory usage stats using nvidia-smi
    Returns a string with memory usage info for each GPU
    """
    if not torch.cuda.is_available():
        return "GPU not available"
    
    try:
        gpu_stats = []
        for i in range(torch.cuda.device_count()):
            # Get memory usage in MB
            allocated = torch.cuda.memory_allocated(i) / 1024**2
            cached = torch.cuda.memory_reserved(i) / 1024**2
            
            # Get device properties
            prop = torch.cuda.get_device_properties(i)
            total_memory = prop.total_memory / 1024**2
            
            gpu_stats.append(f"GPU {i} ({prop.name}): "
                           f"{allocated:.1f}MB allocated, "
                           f"{cached:.1f}MB cached, "
                           f"{total_memory:.1f}MB total")
    except Exception as e:
        return f"Error getting GPU stats: {str(e)}"
    
    return "\n".join(gpu_stats)

def get_gpu_utilization():
    """
    Get GPU utilization using nvidia-smi
    Returns utilization percentage for each GPU
    """
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        utils = [int(x) for x in result.strip().split('\n')]
        return [f"GPU {i}: {util}%" for i, util in enumerate(utils)]
    except:
        return ["Could not get GPU utilization"]

def print_gpu_stats(epoch, batch_idx, num_batches):
    """
    Print comprehensive GPU statistics
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "="*80)
    print(f"GPU Stats at {current_time} (Epoch {epoch+1}, Batch {batch_idx}/{num_batches})")
    print("-"*80)
    print("Memory Usage:")
    print(get_gpu_memory_stats())
    print("\nGPU Utilization:")
    print("\n".join(get_gpu_utilization()))
    print("="*80 + "\n")


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
        self.lambda_mse = params.LAMBDA_MSE
        self.lambda_perceptual = params.LAMDA_PERCEPTUAL
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
        # Convert inputs to numpy arrays if they aren't already
        distances = np.array(distances)
        cosines = np.array(cosines)
        albedo = np.array(albedo)
        normals = np.array(normals)
        targets = np.array(targets)
        
        # Validate input shapes
        K, N, H, W = distances.shape
        assert cosines.shape == (K, N, H, W), f"Cosines shape {cosines.shape} doesn't match distances shape {distances.shape}"
        assert albedo.shape == (K, H, W, 3), f"Albedo shape {albedo.shape} doesn't match expected shape {(K, H, W, 3)}"
        assert normals.shape == (K, H, W, 3), f"Normals shape {normals.shape} doesn't match expected shape {(K, H, W, 3)}"
        assert targets.shape == (K, N, H, W, 3), f"Targets shape {targets.shape} doesn't match expected shape {(K, N, H, W, 3)}"
        
        # Convert to torch tensors
        self.distances = torch.FloatTensor(distances)
        self.cosines = torch.FloatTensor(cosines)
        self.albedo = torch.FloatTensor(albedo)
        self.normals = torch.FloatTensor(normals)
        self.targets = torch.FloatTensor(targets)
        
        # Calculate patch size and number of patches
        self.patch_size, self.patches_per_image = self.calculate_patches() 
        print("Adjusted patch size, number of patches: ", self.patch_size, self.patches_per_image)       
        
        # Create patches list
        self.patches = self._create_patches()  
        print(f"Created {len(self.patches)} total patches")

    def calculate_patches(self):
        print("Calculating the patches")
        patch_height, patch_width = params.RTI_NET_PATCH_SIZE
        desired_patches = params.RTI_MAX_NUMBER_PATCHES
        _, _, image_height, image_width = self.distances.shape
        
        # Ensure patch sizes are not larger than image dimensions and are divisible by 32
        patch_height = min(patch_height, image_height)
        patch_width = min(patch_width, image_width)
        patch_height = (patch_height // 32) * 32
        patch_width = (patch_width // 32) * 32
        
        # Ensure patches are not zero-sized
        assert patch_height > 0 and patch_width > 0, f"Invalid patch size: {patch_height}x{patch_width}"
        
        # Calculate maximum possible patches
        max_patches_y = image_height // patch_height
        max_patches_x = image_width // patch_width
        max_patches = max_patches_y * max_patches_x
        
        # Adjust desired patches if it exceeds maximum possible
        adjusted_patches = min(desired_patches, max_patches)
        assert adjusted_patches > 0, "No valid patches could be created"
        
        return [patch_height, patch_width], adjusted_patches

    def _create_patches(self):
        patches = []
        K, N, H, W = self.distances.shape
        grid_size = math.isqrt(self.patches_per_image)
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
                    
                    # Ensure valid ranges
                    h_start = max(0, h_start)
                    h_end = max(h_start, h_end)
                    w_start = max(0, w_start)
                    w_end = max(w_start, w_end)
                    
                    # Randomly select a patch within this grid cell
                    h = random.randint(h_start, h_end)
                    w = random.randint(w_start, w_end)
                    
                    patches.append((k, n, h, w))
        
        # Verify we have patches
        assert len(patches) > 0, "No patches were created"
        return patches

    def __len__(self):
        """Return the total number of patches"""
        return len(self.patches)

    def __getitem__(self, idx):
        """Get a specific patch by index"""
        if idx >= len(self.patches):
            raise IndexError(f"Index {idx} out of bounds for dataset with {len(self.patches)} patches")
            
        k, n, i, j = self.patches[idx]
        patch_slice = slice(i, i + self.patch_size[0]), slice(j, j + self.patch_size[1])
        
        try:
            item = {
                'distances': self.distances[k, n][patch_slice],
                'cosines': self.cosines[k, n][patch_slice],
                'albedo': self.albedo[k][patch_slice],
                'normals': self.normals[k][patch_slice],
                'target': self.targets[k, n][patch_slice]
            }
            
            # Validate output shapes
            pH, pW = self.patch_size
            assert item['distances'].shape == (pH, pW), f"Invalid distances shape: {item['distances'].shape}"
            assert item['cosines'].shape == (pH, pW), f"Invalid cosines shape: {item['cosines'].shape}"
            assert item['albedo'].shape == (pH, pW, 3), f"Invalid albedo shape: {item['albedo'].shape}"
            assert item['normals'].shape == (pH, pW, 3), f"Invalid normals shape: {item['normals'].shape}"
            assert item['target'].shape == (pH, pW, 3), f"Invalid target shape: {item['target'].shape}"
            
            return item
            
        except Exception as e:
            print(f"Error getting item {idx}: {e}")
            print(f"Patch info - k: {k}, n: {n}, i: {i}, j: {j}")
            raise e

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
    def __init__(self, albedo_channels):
        super(RelightingModel, self).__init__()
        
        # Initial convolutions
        self.conv_distances = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv_cosines = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv_albedo = nn.Conv2d(albedo_channels, 32, kernel_size=3, padding=1)
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
        self.upsample = nn.Upsample(size=None, scale_factor=2, mode='bilinear', align_corners=True)

    def ensure_size_match(self, x, target):
        """Ensure x matches target size"""
        if x.shape[2:] != target.shape[2:]:
            x = nn.functional.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=True)
        return x

    def forward(self, distances, cosines, albedo, normals):
        # Store original size
        original_size = (distances.shape[1], distances.shape[2])
        
        # Reshape inputs to add channel dimension
        distances = distances.unsqueeze(1)
        cosines = cosines.unsqueeze(1)
        albedo = albedo.permute(0, 3, 1, 2)
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
        
        # Decoder with size matching
        d3_up = self.upsample(bridge)
        d3_up = self.ensure_size_match(d3_up, e3)
        d3 = self.decoder3(torch.cat([d3_up, e3], dim=1))
        
        d2_up = self.upsample(d3)
        d2_up = self.ensure_size_match(d2_up, e2)
        d2 = self.decoder2(torch.cat([d2_up, e2], dim=1))
        
        d1_up = self.upsample(d2)
        d1_up = self.ensure_size_match(d1_up, e1)
        d1 = self.decoder1(torch.cat([d1_up, e1], dim=1))
        
        # Output with size matching to input
        out = self.final_conv(d1)
        out = nn.functional.interpolate(out, size=original_size, mode='bilinear', align_corners=True)
        
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
    originals = []
    reconstructed = []
    
    with torch.no_grad():
        for batch in val_loader:
            distances = batch['distances'].to(device)
            cosines = batch['cosines'].to(device)
            albedo = batch['albedo'].to(device)
            normals = batch['normals'].to(device)
            targets = batch['target'].to(device)

            outputs, _ = model(distances, cosines, albedo, normals)

            # Convert tensors to numpy arrays and change from BGR to RGB
            originals.extend([img[:,:,::-1] for img in targets.cpu().numpy()])
            reconstructed.extend([img[:,:,::-1] for img in outputs.cpu().numpy()])

            if len(originals) >= num_samples:
                break

    # Trim to the desired number of samples
    originals = originals[:num_samples]
    reconstructed = reconstructed[:num_samples]

    return originals, reconstructed

def save_comparison_images(model, val_loader, device, epoch, model_save_path, num_samples=10):
    model.eval()
    
    with torch.no_grad():
        for batch in val_loader:
            distances = batch['distances'].to(device)
            cosines = batch['cosines'].to(device)
            albedo = batch['albedo'].to(device)
            normals = batch['normals'].to(device)
            targets = batch['target'].to(device)

            outputs, _ = model(distances, cosines, albedo, normals)

            # Convert tensors to numpy arrays
            originals = targets.cpu().numpy()
            reconstructed = outputs.cpu().numpy()

            # Calculate number of rows needed
            num_rows = math.ceil(min(num_samples, originals.shape[0]) / 2)

            # Create a single figure for all samples
            fig, axes = plt.subplots(num_rows, 4, figsize=(20, 5*num_rows))
            fig.suptitle(f'Comparison at Epoch {epoch}', fontsize=16)

            for i in range(min(num_samples, originals.shape[0])):
                row = i // 2
                col = (i % 2) * 2

                # Normalize and convert original image from BGR to RGB
                original_img = originals[i]
                original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())
                original_img = original_img[..., ::-1]  # BGR to RGB

                # Normalize and convert reconstructed image from BGR to RGB
                reconstructed_img = reconstructed[i]
                reconstructed_img = (reconstructed_img - reconstructed_img.min()) / (reconstructed_img.max() - reconstructed_img.min())
                reconstructed_img = reconstructed_img[..., ::-1]  # BGR to RGB

                # Plot original image
                axes[row, col].imshow(original_img)
                axes[row, col].set_title(f'Original {i+1}')
                axes[row, col].axis('off')

                # Plot reconstructed image
                axes[row, col+1].imshow(reconstructed_img)
                axes[row, col+1].set_title(f'Reconstructed {i+1}')
                axes[row, col+1].axis('off')

            # Remove any unused subplots
            for i in range(min(num_samples, originals.shape[0]), num_rows*2):
                row = i // 2
                col = (i % 2) * 2
                axes[row, col].axis('off')
                axes[row, col+1].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(model_save_path, f'comparison_epoch_{epoch}.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)

            break  # Only process one batch

def plot_losses(train_losses, val_losses, epoch, model_save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Losses up to Epoch {epoch}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_path, f'loss_plot_epoch_{epoch}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def train_model(model, train_loader, val_loader, num_epochs=100, model_save_path='.'):
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"Using {torch.cuda.device_count()} GPUs")
        # Use DataParallel if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        device = torch.device("cuda")        
    else:
        print("Using CPU")
        device = torch.device("cpu")
    model = model.to(device)
    criterion = CombinedLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=params.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # Print GPU stats at the start of each epoch
        print_gpu_stats(epoch, 0, len(train_loader))

        for batch_idx, batch in enumerate(train_loader):
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

            # # Print GPU stats every N batches
            # if batch_idx % params.BATCH_SIZE == 0:  # Adjust frequency as needed
            #     print_gpu_stats(epoch, batch_idx, len(train_loader))
                
            # # Clear cache periodically if needed
            # if batch_idx % params.BATCH_SIZE == 0:  # Adjust frequency as needed
            #     torch.cuda.empty_cache()

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

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print("Epoch {}/{}, Train Loss: {:.4f}, Val Loss: {:.4f}".format(
            epoch+1, num_epochs, train_loss, val_loss))
        
        # Save comparison images every N epochs
        if (epoch + 1) % params.RTI_NET_SAVE_MODEL_EVERY_N_EPOCHS == 0:
            save_comparison_images(model, val_loader, device, epoch + 1, model_save_path)
            plot_losses(train_losses, val_losses, epoch + 1, model_save_path)

        scheduler.step(val_loss)

        # Save model every N epochs
        if (epoch + 1) % params.RTI_NET_SAVE_MODEL_EVERY_N_EPOCHS == 0:
            save_path = os.path.join(model_save_path, 'relighting_model_epoch_{}.pth'.format(epoch+1))
            torch.save(model.state_dict(), save_path)
            print("Model saved at epoch {}".format(epoch+1))

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
        train_targets.append(targets[i, train_indices, :, :, :])
        val_targets.append(targets[i, val_indices, :, :, :])

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
    train_loader = DataLoader(train_dataset, 
                              batch_size=params.BATCH_SIZE, 
                              shuffle=params.TRAIN_SHUFFLE, 
                              num_workers=params.NUM_WORKERS,
                              persistent_workers=params.PERSISTENT_WORKER,
                              pin_memory=params.PIN_MEMORY,
                              prefetch_factor=params.PREFETCH_FACTOR)
    val_loader = DataLoader(val_dataset, 
                            batch_size=params.BATCH_SIZE, 
                            shuffle=params.VAL_SHUFFLE, 
                            num_workers=params.NUM_WORKERS,
                            persistent_workers=params.PERSISTENT_WORKER,
                            pin_memory=params.PIN_MEMORY,
                            prefetch_factor=params.PREFETCH_FACTOR)

    return train_loader, val_loader, train_indices, val_indices

def train(distances, cosines, albedo, normals, targets):
    print("Going to train..")    
    
    distances = np.array(distances)
    cosines = np.array(cosines)
    albedo = np.array(albedo)
    normals = np.array(normals)
    targets = np.array(targets)

    print("Input shapes - distances: ", distances.shape, "cosines: ", cosines.shape, 
          "albedo: ", albedo.shape, "normals: ", normals.shape, "targets: ", targets.shape)
    
    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
    os.makedirs(model_save_path, exist_ok=True)

    # Prepare data
    print("Preparing data..")
    train_loader, val_loader, train_indices, val_indices = prepare_data(distances, cosines, albedo, normals, targets)

    np.save(os.path.join(model_save_path, 'train_indices.npy'), train_indices)
    np.save(os.path.join(model_save_path, 'val_indices.npy'), val_indices)

    # Initialize the model
    print("Initializing model..")
    albedo_channels = albedo.shape[-1]
    model = RelightingModel(albedo_channels=albedo_channels)

    # Train the model
    print("Training the model..")
    trained_model = train_model(model, train_loader, val_loader, num_epochs=params.RTI_NET_EPOCHS, model_save_path=model_save_path)

    # Save the final model
    torch.save(trained_model.state_dict(), os.path.join(model_save_path, 'relighting_model_final.pth'))

    print("Training completed and model saved.")
