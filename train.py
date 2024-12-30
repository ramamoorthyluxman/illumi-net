# System and basic imports
import os
import sys
import random
import math
from datetime import datetime
import subprocess
import shutil

# Set matplotlib backend first
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Scientific computing imports
import numpy as np
from sklearn.model_selection import train_test_split

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

# Local imports
from utils import params

# Set visible GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = params.CUDA_VISIBLE_DEVICES

def get_gpu_memory_stats():
    """Get GPU memory usage stats"""
    if not torch.cuda.is_available():
        return "GPU not available"
    
    try:
        gpu_stats = []
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**2
            cached = torch.cuda.memory_reserved(i) / 1024**2
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
    """Get GPU utilization percentage"""
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
    """Print GPU statistics"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "="*80)
    print(f"GPU Stats at {current_time} (Epoch {epoch+1}, Batch {batch_idx}/{num_batches})")
    print("-"*80)
    print("Memory Usage:")
    print(get_gpu_memory_stats())
    print("\nGPU Utilization:")
    print("\n".join(get_gpu_utilization()))
    print("="*80 + "\n")

class PatchRelightingDataset(Dataset):
    def __init__(self, distances, cosines, albedo, normals, azimuths, targets):
        # Convert inputs to numpy arrays
        distances = np.array(distances)
        cosines = np.array(cosines)
        albedo = np.array(albedo)
        normals = np.array(normals)
        targets = np.array(targets)
        azimuths = np.array(azimuths)  # Shape: [K, N]
        
        # Validate input shapes
        K, N, H, W = distances.shape
        assert cosines.shape == (K, N, H, W), f"Cosines shape {cosines.shape} doesn't match distances shape {distances.shape}"
        assert albedo.shape == (K, H, W, 3), f"Albedo shape {albedo.shape} doesn't match expected shape {(K, H, W, 3)}"
        assert normals.shape == (K, H, W, 3), f"Normals shape {normals.shape} doesn't match expected shape {(K, H, W, 3)}"
        assert targets.shape == (K, N, H, W, 3), f"Targets shape {targets.shape} doesn't match expected shape {(K, N, H, W, 3)}"
        assert azimuths.shape == (K, N), f"Azimuths shape {azimuths.shape} doesn't match expected shape {(K, N)}"
        
        # Convert to torch tensors
        self.distances = torch.FloatTensor(distances)
        self.cosines = torch.FloatTensor(cosines)
        self.albedo = torch.FloatTensor(albedo)
        self.normals = torch.FloatTensor(normals)
        self.targets = torch.FloatTensor(targets)
        self.azimuths = torch.FloatTensor(azimuths)
        
        # Calculate patch size and number of patches
        self.patch_size, self.patches_per_image = self.calculate_patches() 
        print("Adjusted patch size, number of patches: ", self.patch_size, self.patches_per_image)       
        
        # Create patches list with content validation
        self.patches = self._create_valid_patches()  
        print(f"Created {len(self.patches)} valid patches after filtering")

    # Continuation of PatchRelightingDataset
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
        
        assert patch_height > 0 and patch_width > 0, f"Invalid patch size: {patch_height}x{patch_width}"
        
        max_patches_y = image_height // patch_height
        max_patches_x = image_width // patch_width
        max_patches = max_patches_y * max_patches_x
        
        adjusted_patches = min(desired_patches, max_patches)
        assert adjusted_patches > 0, "No valid patches could be created"
        
        return [patch_height, patch_width], adjusted_patches

    def is_valid_patch(self, k, n, i, j):
        patch_slice = slice(i, i + self.patch_size[0]), slice(j, j + self.patch_size[1])
        
        target_patch = self.targets[k, n][patch_slice]
        
        threshold = params.PATCH_PIX_VAL_THRESHOLD
        non_black_pixels = torch.sum(torch.max(target_patch, dim=2)[0] > threshold)
        total_pixels = self.patch_size[0] * self.patch_size[1]
        non_black_ratio = non_black_pixels / total_pixels
        
        normal_patch = self.normals[k][patch_slice]
        albedo_patch = self.albedo[k][patch_slice]
        
        has_valid_normals = torch.sum(torch.abs(normal_patch) > threshold) > 0
        has_valid_albedo = torch.sum(torch.abs(albedo_patch) > threshold) > 0
        
        required_ratio = params.NON_BLACK_PIX_RATIO_MIN
        
        return (non_black_ratio > required_ratio and 
                has_valid_normals and 
                has_valid_albedo)

    def _create_valid_patches(self):
        patches = []
        attempted_patches = 0
        max_attempts = self.patches_per_image * 3
        K, N, H, W = self.distances.shape
        grid_size = math.isqrt(self.patches_per_image)
        cell_h, cell_w = H // grid_size, W // grid_size
        
        print("Creating valid patches...")
        
        for k in range(K):
            for n in range(N):
                valid_patches_count = 0
                attempts = 0
                
                while valid_patches_count < self.patches_per_image and attempts < max_attempts:
                    i = random.randint(0, grid_size - 1)
                    j = random.randint(0, grid_size - 1)
                    
                    h_start = i * cell_h
                    h_end = min((i + 1) * cell_h, H) - self.patch_size[0]
                    w_start = j * cell_w
                    w_end = min((j + 1) * cell_w, W) - self.patch_size[1]
                    
                    h_start = max(0, h_start)
                    h_end = max(h_start, h_end)
                    w_start = max(0, w_start)
                    w_end = max(w_start, w_end)
                    
                    h = random.randint(h_start, h_end)
                    w = random.randint(w_start, w_end)
                    
                    if self.is_valid_patch(k, n, h, w):
                        patches.append((k, n, h, w))
                        valid_patches_count += 1
                    
                    attempts += 1
                    attempted_patches += 1
                    
                    if attempts % 100 == 0:
                        print(f"Attempted {attempts} patches, found {valid_patches_count} valid patches")
                
                if valid_patches_count < self.patches_per_image:
                    print(f"Warning: Could only find {valid_patches_count} valid patches for image {k}, view {n}")
        
        print(f"Created {len(patches)} valid patches out of {attempted_patches} attempts")
        assert len(patches) > 0, "No valid patches were created"
        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
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
                'azimuth': self.azimuths[k, n],
                'target': self.targets[k, n][patch_slice]
            }
            
            # Additional validation
            if torch.isnan(item['target']).any():
                raise ValueError("NaN values found in target patch")
            if torch.all(item['target'] == 0):
                raise ValueError("Completely black patch found")
            
            # Validate output shapes
            pH, pW = self.patch_size
            assert item['distances'].shape == (pH, pW), f"Invalid distances shape: {item['distances'].shape}"
            assert item['cosines'].shape == (pH, pW), f"Invalid cosines shape: {item['cosines'].shape}"
            assert item['albedo'].shape == (pH, pW, 3), f"Invalid albedo shape: {item['albedo'].shape}"
            assert item['normals'].shape == (pH, pW, 3), f"Invalid normals shape: {item['normals'].shape}"
            assert item['target'].shape == (pH, pW, 3), f"Invalid target shape: {item['target'].shape}"
            assert isinstance(item['azimuth'].item(), float), "Invalid azimuth type"
            
            return item
            
        except Exception as e:
            print(f"Error getting item {idx}: {e}")
            print(f"Patch info - k: {k}, n: {n}, i: {i}, j: {j}")
            raise e

# Network Building Blocks
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

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        
    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_pool, max_pool], dim=1)
        attention = torch.sigmoid(self.conv(attention))
        return x * attention
    
class RelightingModel(nn.Module):
    def __init__(self, albedo_channels):
        super(RelightingModel, self).__init__()
        
        # Initial convolutions for image-sized features
        self.conv_distances = nn.Conv2d(1, 96, kernel_size=3, padding=1)
        self.conv_cosines = nn.Conv2d(1, 96, kernel_size=3, padding=1)
        self.conv_albedo = nn.Conv2d(albedo_channels, 48, kernel_size=3, padding=1)
        self.conv_normals = nn.Conv2d(3, 48, kernel_size=3, padding=1)
        
        # Angle Processing Network
        self.angle_encoder = nn.Sequential(
            nn.Linear(1, 32),      # Single angle input
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # Batch normalization layers
        self.bn_distances = nn.BatchNorm2d(96)
        self.bn_cosines = nn.BatchNorm2d(96)
        self.bn_albedo = nn.BatchNorm2d(48)
        self.bn_normals = nn.BatchNorm2d(48)
        
        # Activation function
        self.prelu = nn.PReLU()
        
        # Multi-scale processing
        self.conv_distances_small = nn.Conv2d(96, 48, kernel_size=1)
        self.conv_distances_large = nn.Conv2d(96, 48, kernel_size=5, padding=2)
        self.conv_cosines_small = nn.Conv2d(96, 48, kernel_size=1)
        self.conv_cosines_large = nn.Conv2d(96, 48, kernel_size=5, padding=2)
        
        # Encoder (increased channels to accommodate angle features)
        total_channels = 48*6 + 64  # Original channels (48*6) + angle features (64)
        self.encoder1 = ResidualBlock2D(total_channels, 128)
        self.encoder2 = ResidualBlock2D(128, 256)
        self.encoder3 = ResidualBlock2D(256, 512)
        self.encoder4 = ResidualBlock2D(512, 1024)
        
        # Bridge
        self.bridge = AttentionBlock2D(1024)
        
        # Decoder
        self.decoder3 = ResidualBlock2D(1536, 512)  # 1024 + 512 = 1536
        self.decoder2 = ResidualBlock2D(768, 256)   # 512 + 256 = 768
        self.decoder1 = ResidualBlock2D(384, 128)   # 256 + 128 = 384
        
        # Final convolutions
        self.final_conv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        
        self.final_conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU()
        )
        
        self.final_conv3 = nn.Conv2d(32, 3, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

    def ensure_size_match(self, x, target):
        if x.shape[2:] != target.shape[2:]:
            x = nn.functional.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=True)
        return x

    def forward(self, distances, cosines, albedo, normals, azimuth):
        # Store original size
        original_size = (distances.shape[1], distances.shape[2])
        
        # Process angular features
        azimuth = azimuth.unsqueeze(1)  # Add feature dimension [B, 1]
        angle_features = self.angle_encoder(azimuth)  # Shape: [B, 64]
        
        # Expand angle features to match spatial dimensions
        B = azimuth.shape[0]
        angle_features = angle_features.view(B, -1, 1, 1).expand(-1, -1, original_size[0], original_size[1])
        
        # Process other inputs
        distances = distances.unsqueeze(1)
        cosines = cosines.unsqueeze(1)
        albedo = albedo.permute(0, 3, 1, 2)
        normals = normals.permute(0, 3, 1, 2)
        
        # Initial processing with batch norm
        x1 = self.prelu(self.bn_distances(self.conv_distances(distances)))
        x2 = self.prelu(self.bn_cosines(self.conv_cosines(cosines)))
        x3 = self.prelu(self.bn_albedo(self.conv_albedo(albedo)))
        x4 = self.prelu(self.bn_normals(self.conv_normals(normals)))
        
        # Multi-scale processing
        x1_small = self.conv_distances_small(x1)
        x1_large = self.conv_distances_large(x1)
        x2_small = self.conv_cosines_small(x2)
        x2_large = self.conv_cosines_large(x2)
        
        # Concatenate all features including angle features
        x = torch.cat([x1_small, x1_large, x2_small, x2_large, x3, x4, angle_features], dim=1)
        
        # Add dropout
        x = self.dropout(x)
        
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        
        # Bridge with attention
        bridge = self.bridge(e4)
        
        # Decoder with skip connections
        d3_up = self.upsample(bridge)
        d3_up = self.ensure_size_match(d3_up, e3)
        d3 = self.decoder3(torch.cat([d3_up, e3], dim=1))
        
        d2_up = self.upsample(d3)
        d2_up = self.ensure_size_match(d2_up, e2)
        d2 = self.decoder2(torch.cat([d2_up, e2], dim=1))
        
        d1_up = self.upsample(d2)
        d1_up = self.ensure_size_match(d1_up, e1)
        d1 = self.decoder1(torch.cat([d1_up, e1], dim=1))
        
        # Final convolutions
        out = self.final_conv1(d1)
        out = self.final_conv2(out)
        out = self.final_conv3(out)
        
        # Ensure output size matches input size
        out = nn.functional.interpolate(out, size=original_size, mode='bilinear', align_corners=True)
        
        # Reshape output to match target shape (N, H, W, 3)
        out = out.permute(0, 2, 3, 1)
        
        intermediates = {
            'e1': e1, 'e2': e2, 'e3': e3, 'e4': e4,
            'd3': d3, 'd2': d2, 'd1': d1,
            'final': out
        }
        
        return out, intermediates

class CombinedLoss(nn.Module):
    def __init__(self, use_perceptual=True, input_channels=3):
        super(CombinedLoss, self).__init__()
        self.use_perceptual = use_perceptual
        self.input_channels = input_channels
        
        # Basic losses
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # Initialize VGG for perceptual loss if needed
        if self.use_perceptual:
            vgg = models.vgg16(pretrained=True)
            if input_channels != 3:
                self.channel_adapt = nn.Conv2d(input_channels, 3, kernel_size=1)
            else:
                self.channel_adapt = nn.Identity()
            
            self.vgg_features = nn.Sequential(
                *list(vgg.features)[:4]
            ).eval()
            
            for param in self.vgg_features.parameters():
                param.requires_grad = False
            
        # Pooling layers for reuse
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        # Sobel filters
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                                    dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                                    dtype=torch.float32).view(1, 1, 3, 3))
        
    def compute_intensity_loss(self, output, target):
        """Compute intensity matching loss"""
        output_intensity = torch.mean(output, dim=[2, 3])  # Global average pooling
        target_intensity = torch.mean(target, dim=[2, 3])
        return F.mse_loss(output_intensity, target_intensity)
        
    def normalize_vgg(self, x):
        """Normalize input to VGG expected range"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        return (x - mean) / std
    
    def compute_gradient_sobel(self, x):
        """Compute gradients using Sobel filters"""
        gradients = []
        for c in range(x.shape[1]):
            gx = F.conv2d(x[:, c:c+1], self.sobel_x, padding=1)
            gy = F.conv2d(x[:, c:c+1], self.sobel_y, padding=1)
            gradients.append(torch.sqrt(gx**2 + gy**2))
        return torch.cat(gradients, dim=1)
    
    def compute_perceptual_loss(self, output, target):
        """Compute perceptual loss using VGG features"""
        if not self.use_perceptual:
            return torch.tensor(0.0).to(output.device)
            
        output = self.channel_adapt(output)
        target = self.channel_adapt(target)
        
        output = output.clamp(0, 1)
        target = target.clamp(0, 1)
        
        output = self.normalize_vgg(output)
        target = self.normalize_vgg(target)
        
        output_features = self.vgg_features(output)
        target_features = self.vgg_features(target)
        
        return F.mse_loss(output_features, target_features)
    
    def compute_range_losses(self, output, target):
        """Compute losses for different intensity ranges"""
        losses = {}
        
        dark_mask = (target <= 0.2).float()
        losses['dark'] = torch.mean((output * dark_mask - target * dark_mask) ** 2)
        
        mid_mask = ((target > 0.2) & (target < 0.8)).float()
        losses['mid'] = torch.mean((output * mid_mask - target * mid_mask) ** 2)
        
        bright_mask = (target >= 0.8).float()
        losses['bright'] = 2.0 * torch.mean((output * bright_mask - target * bright_mask) ** 2)
        
        large_diff_mask = (torch.abs(output - target) > 0.1).float()
        losses['large_dev'] = 2.0 * torch.mean((output * large_diff_mask - target * large_diff_mask) ** 2)
        
        return losses
    
    def forward(self, output, target):
        if output.dim() == 4 and output.shape[-1] == self.input_channels:
            output = output.permute(0, 3, 1, 2)
            target = target.permute(0, 3, 1, 2)
        
        losses = {}

        # Add intensity matching loss
        losses['intensity'] = self.compute_intensity_loss(output, target)
        
        # Basic losses
        losses['mse'] = self.mse_loss(output, target)
        losses['l1'] = self.l1_loss(output, target)
        
        # Highlight-aware loss
        bright_mask = (target > 0.7).float()
        losses['highlight'] = F.mse_loss(output * bright_mask, target * bright_mask)
        
        # Gradient loss
        output_grad = self.compute_gradient_sobel(output)
        target_grad = self.compute_gradient_sobel(target)
        losses['gradient'] = F.mse_loss(output_grad, target_grad)
        
        # Local contrast loss
        output_mean = self.avg_pool(output)
        target_mean = self.avg_pool(target)
        output_contrast = output - output_mean
        target_contrast = target - target_mean
        losses['contrast'] = F.mse_loss(output_contrast, target_contrast)
        
        # Specular highlight loss
        max_pooled = self.max_pool(target)
        specular_mask = (target > 0.8 * max_pooled).float()
        losses['specular'] = F.mse_loss(output * specular_mask, target * specular_mask)
        
        # Perceptual loss
        if self.use_perceptual:
            losses['perceptual'] = self.compute_perceptual_loss(output, target)

        # Range-specific losses
        range_losses = self.compute_range_losses(output, target)
        losses.update(range_losses)

        # Total loss
        total_loss = (
            params.LAMBDA_MSE * losses['mse'] +
            params.LAMBDA_L1 * losses['l1'] +
            params.LAMBDA_HIGHLIGHT * losses['highlight'] +
            params.LAMBDA_GRADIENT * losses['gradient'] +
            params.LAMBDA_CONTRAST * losses['contrast'] +
            params.LAMBDA_SPECULAR * losses['specular'] +
            params.LAMBDA_DARK * losses['dark'] +
            params.LAMBDA_MID * losses['mid'] +
            params.LAMBDA_BRIGHT * losses['bright'] +
            params.LAMBDA_LARGE_DEV * losses['large_dev'] +
            5.0 * losses['intensity']  # Add high weight for intensity matching
        )
        
        if self.use_perceptual:
            total_loss += params.LAMBDA_PERCEPTUAL * losses['perceptual']
        
        return total_loss, losses

def train_model(model, train_loader, val_loader, num_epochs=100, model_save_path='.'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    criterion = CombinedLoss(use_perceptual=True, input_channels=3).to(device)
    
    # Initialize optimizer and scheduler based on params
    optimizer = None
    scheduler = None
    if params.OPTIMIZER == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=params.LEARNING_RATE,
            weight_decay=params.WEIGHT_DECAY
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )

    elif params.OPTIMIZER == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=params.LEARNING_RATE,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=params.WEIGHT_DECAY,
            amsgrad=True  # Uses the maximum of past squared gradients
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=params.EPOCHS_PER_CYCLE,
            T_mult=2
        )
    elif params.OPTIMIZER == "Lion":
        # Requires: pip install lion-pytorch
        from lion_pytorch import Lion
        optimizer = Lion(
            model.parameters(),
            lr=params.LEARNING_RATE,
            weight_decay=params.WEIGHT_DECAY,
            beta1=0.9,
            beta2=0.99
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True,
            min_lr=1e-6
        )
    elif params.OPTIMIZER == "RAdam":
        optimizer = optim.RAdam(
            model.parameters(),
            lr=params.LEARNING_RATE,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=params.WEIGHT_DECAY
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True,
            min_lr=1e-6
        )
    elif params.OPTIMIZER == "AdaBelief":
    # Requires: pip install adabelief-pytorch
        from adabelief_pytorch import AdaBelief
        optimizer = AdaBelief(
            model.parameters(),
            lr=params.LEARNING_RATE,
            eps=1e-16,
            betas=(0.9, 0.999),
            weight_decay=params.WEIGHT_DECAY,
            weight_decouple=True,
            rectify=False
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True,
            min_lr=1e-6
        )

    elif params.OPTIMIZER == "SGD":
        # SGD with momentum and Nesterov acceleration
        optimizer = optim.SGD(
            model.parameters(),
            lr=params.LEARNING_RATE,
            momentum=0.9,  # High momentum for stable training
            weight_decay=params.WEIGHT_DECAY,
            nesterov=True  # Nesterov accelerated gradient for better convergence
        )
        
        # Option 1: OneCycleLR - works very well with SGD
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=params.LEARNING_RATE,
            epochs=params.RTI_NET_EPOCHS,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,  # Spend 30% of time warming up
            div_factor=25,  # initial_lr = max_lr/25
            final_div_factor=1e4,  # final_lr = initial_lr/1e4
            three_phase=True,  # Use three-phase learning rate schedule
            anneal_strategy='cos'  # Cosine annealing
        )
    
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        print_gpu_stats(epoch, 0, len(train_loader))

        for batch_idx, batch in enumerate(train_loader):
            distances = batch['distances'].to(device)
            cosines = batch['cosines'].to(device)
            albedo = batch['albedo'].to(device)
            normals = batch['normals'].to(device)
            azimuth = batch['azimuth'].to(device)
            targets = batch['target'].to(device)

            # Check for NaN values
            if (torch.isnan(distances).any() or torch.isnan(cosines).any() or 
                torch.isnan(albedo).any() or torch.isnan(normals).any() or 
                torch.isnan(targets).any() or torch.isnan(azimuth).any()):
                print(f"Warning: NaN detected in input batch {batch_idx}")
                continue

            optimizer.zero_grad()
            outputs, _ = model(distances, cosines, albedo, normals, azimuth)
            tot_loss, losses = criterion(outputs, targets)
            tot_loss.backward()
            optimizer.step()
            train_loss += tot_loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                distances = batch['distances'].to(device)
                cosines = batch['cosines'].to(device)
                albedo = batch['albedo'].to(device)
                normals = batch['normals'].to(device)
                azimuth = batch['azimuth'].to(device)
                targets = batch['target'].to(device)

                outputs, _ = model(distances, cosines, albedo, normals, azimuth)
                loss, _ = criterion(outputs, targets)
                val_loss += loss.item()

        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save visualizations and model
        if (epoch + 1) % params.RTI_NET_SAVE_MODEL_EVERY_N_EPOCHS == 0:
            save_comparison_images(model, val_loader, device, epoch + 1, model_save_path)
            plot_losses(train_losses, val_losses, epoch + 1, model_save_path)
            torch.save(model.state_dict(), 
                      os.path.join(model_save_path, f'relighting_model_epoch_{epoch+1}.pth'))

        scheduler.step(val_loss)

    return model

def save_comparison_images(model, val_loader, device, epoch, model_save_path, num_samples=8):
    model.eval()
    
    with torch.no_grad():
        # Get a batch of data
        batch = next(iter(val_loader))
        
        # Sort batch by azimuth to get diverse lighting conditions
        azimuths = batch['azimuth']
        sorted_indices = torch.argsort(azimuths)
        
        # Take evenly spaced samples to get diverse lighting directions
        step = len(sorted_indices) // num_samples
        selected_indices = sorted_indices[::step][:num_samples]
        
        # Select the data using these indices
        distances = batch['distances'][selected_indices].to(device)
        cosines = batch['cosines'][selected_indices].to(device)
        albedo = batch['albedo'][selected_indices].to(device)
        normals = batch['normals'][selected_indices].to(device)
        azimuth = batch['azimuth'][selected_indices].to(device)
        targets = batch['target'][selected_indices].to(device)
        
        # Get model outputs
        outputs, _ = model(distances, cosines, albedo, normals, azimuth)
        
        # Convert tensors to numpy arrays
        originals = targets.cpu().numpy()
        reconstructed = outputs.cpu().numpy()
        azimuths = azimuth.cpu().numpy()
        
        # Calculate number of rows needed (num_samples / 2 rounded up)
        num_rows = (num_samples + 1) // 2
        
        # Create figure with 2 comparisons (4 images) per row
        fig, axes = plt.subplots(num_rows, 4, figsize=(20, 4*num_rows))
        fig.suptitle(f'Different Lighting Directions at Epoch {epoch}', fontsize=16)
        
        for i in range(num_samples):
            row = i // 2  # Integer division to get row index
            col = (i % 2) * 2  # Multiply by 2 because each comparison takes 2 columns
            
            # Normalize and convert original image
            original_img = originals[i]
            original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())
            original_img = original_img[..., ::-1]  # BGR to RGB
            
            # Normalize and convert reconstructed image
            reconstructed_img = reconstructed[i]
            reconstructed_img = (reconstructed_img - reconstructed_img.min()) / (reconstructed_img.max() - reconstructed_img.min())
            reconstructed_img = reconstructed_img[..., ::-1]  # BGR to RGB
            
            # Plot original image
            axes[row, col].imshow(original_img)
            axes[row, col].set_title(f'Original (Az: {azimuths[i]:.2f})')
            axes[row, col].axis('off')
            
            # Plot reconstructed image
            axes[row, col + 1].imshow(reconstructed_img)
            axes[row, col + 1].set_title(f'Reconstructed (Az: {azimuths[i]:.2f})')
            axes[row, col + 1].axis('off')
            
        # Hide any empty subplots in the last row if num_samples is odd
        if num_samples % 2 == 1:
            axes[-1, -2].axis('off')
            axes[-1, -1].axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(model_save_path, f'comparison_epoch_{epoch}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

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
    plt.savefig(os.path.join(model_save_path, f'loss_plot_epoch_{epoch}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def prepare_data(distances, cosines, albedo, normals, azimuths, targets):
    # Splitting the data
    K = distances.shape[0]
    N = distances.shape[1]
    train_distances = []
    val_distances = []
    train_cosines = []
    val_cosines = []
    train_targets = []   
    val_targets = []
    train_azimuths = []
    val_azimuths = []

    for i in range(K):
        train_indices, val_indices = train_test_split(
            np.arange(N), 
            test_size=params.TRAIN_VAL_SPLIT, 
            random_state=42
        )
        train_distances.append(distances[i, train_indices, :, :])
        val_distances.append(distances[i, val_indices, :, :])
        train_cosines.append(cosines[i, train_indices, :, :])
        val_cosines.append(cosines[i, val_indices, :, :])
        train_targets.append(targets[i, train_indices, :, :, :])
        val_targets.append(targets[i, val_indices, :, :, :])
        train_azimuths.append(azimuths[i, train_indices])
        val_azimuths.append(azimuths[i, val_indices])

    print("Train distances shape: ", np.array(train_distances).shape,
          "\nVal distances shape: ", np.array(val_distances).shape,
          "\nTrain cosines shape: ", np.array(train_cosines).shape,
          "\nVal cosines shape: ", np.array(val_cosines).shape,
          "\nTrain targets shape: ", np.array(train_targets).shape,
          "\nVal targets shape: ", np.array(val_targets).shape,
          "\nTrain azimuths shape: ", np.array(train_azimuths).shape,
          "\nVal azimuths shape: ", np.array(val_azimuths).shape)

    # Create datasets
    train_dataset = PatchRelightingDataset(
        train_distances, train_cosines,
        albedo, normals,
        train_azimuths, train_targets
    )
    val_dataset = PatchRelightingDataset(
        val_distances, val_cosines,
        albedo, normals,
        val_azimuths, val_targets
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

def create_numbered_folder(base_path):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = "saved_models"
    existing_folders = [d for d in os.listdir(os.path.dirname(base_path)) 
                       if os.path.isdir(os.path.join(os.path.dirname(base_path), d)) 
                       and d.startswith(base_name)]
    
    max_num = -1
    for folder in existing_folders:
        try:
            num = int(folder.split('_')[2])
            max_num = max(max_num, num)
        except (IndexError, ValueError):
            continue
    
    new_num = str(max_num + 1).zfill(2)
    folder_name = f"{base_name}_{new_num}_{current_time}"
    full_path = os.path.join(os.path.dirname(base_path), folder_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path

def train(distances, cosines, albedo, normals, azimuths, targets):
    print("Going to train..")    
    
    # Convert inputs to numpy arrays
    distances = np.array(distances)
    cosines = np.array(cosines)
    albedo = np.array(albedo)
    normals = np.array(normals)
    azimuths = np.array(azimuths)
    targets = np.array(targets)

    print("Input shapes:",
          "\ndistances:", distances.shape,
          "\ncosines:", cosines.shape,
          "\nalbedo:", albedo.shape,
          "\nnormals:", normals.shape,
          "\nazimuths:", azimuths.shape,
          "\ntargets:", targets.shape)
    
    model_save_path = params.RTI_MODEL_SAVE_DIR
    model_save_path = create_numbered_folder(model_save_path)
    shutil.copy2('./utils/params.py', model_save_path)

    # Prepare data
    print("Preparing data..")
    train_loader, val_loader, train_indices, val_indices = prepare_data(
        distances, cosines, albedo, normals, azimuths, targets
    )

    np.save(os.path.join(model_save_path, 'train_indices.npy'), train_indices)
    np.save(os.path.join(model_save_path, 'val_indices.npy'), val_indices)

    # Initialize model
    print("Initializing model..")
    albedo_channels = albedo.shape[-1]
    model = RelightingModel(albedo_channels=albedo_channels)

    # Train model
    print("Training the model..")
    trained_model = train_model(model, train_loader, val_loader, 
                              num_epochs=params.RTI_NET_EPOCHS, 
                              model_save_path=model_save_path)

    # Save final model
    torch.save(trained_model.state_dict(), os.path.join(model_save_path, 'relighting_model_final.pth'))
    print("Training completed and model saved.")