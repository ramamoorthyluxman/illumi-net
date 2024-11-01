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
import torchvision.transforms as transforms

# Local imports
from utils import params

# Set visible GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = params.CUDA_VISIBLE_DEVICES

class ReflectanceNet(nn.Module):
    def __init__(self, albedo_channels):
        super(ReflectanceNet, self).__init__()
        
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Initial feature extraction - simplified but wider
        self.distance_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.cosine_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.albedo_encoder = nn.Sequential(
            nn.Conv2d(albedo_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.normal_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Main processing
        self.main_process = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Specular attention
        self.specular_attention = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        # Final reconstruction
        self.reconstruction = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Initialize weights
        self.apply(init_weights)
        
    def forward(self, distances, cosines, albedo, normals):
        # Ensure correct dimensions and normalize inputs
        distances = torch.clamp(distances.unsqueeze(1), 0, 1)
        cosines = torch.clamp(cosines.unsqueeze(1), -1, 1)
        albedo = torch.clamp(albedo.permute(0, 3, 1, 2), 0, 1)
        
        # Normalize normals
        normals = normals.permute(0, 3, 1, 2)
        normals_norm = torch.norm(normals, dim=1, keepdim=True).clamp(min=1e-6)
        normals = normals / normals_norm
        
        # Extract features
        dist_features = self.distance_encoder(distances)
        cos_features = self.cosine_encoder(cosines)
        albedo_features = self.albedo_encoder(albedo)
        normal_features = self.normal_encoder(normals)
        
        # Combine features
        features = torch.cat([
            dist_features, cos_features, 
            albedo_features, normal_features
        ], dim=1)
        
        # Process features
        processed = self.main_process(features)
        
        # Generate specular attention mask
        specular_mask = self.specular_attention(processed)
        
        # Final reconstruction
        output = self.reconstruction(processed)
        
        # Ensure output is in valid range
        output = output.clamp(0, 1).permute(0, 2, 3, 1)
        
        return output, {'specular_mask': specular_mask}

class ReflectanceLoss(nn.Module):
    def __init__(self):
        super(ReflectanceLoss, self).__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, output, target, specular_mask=None):
        # Base reconstruction loss
        mse_loss = self.mse(output, target)
        
        # Gradient loss for preserving edges
        dy_out, dx_out = torch.gradient(output, dim=(1, 2))
        dy_target, dx_target = torch.gradient(target, dim=(1, 2))
        gradient_loss = F.mse_loss(torch.sqrt(dx_out**2 + dy_out**2), 
                                 torch.sqrt(dx_target**2 + dy_target**2))
        
        # Specular-aware loss
        if specular_mask is not None:
            specular_regions = (target > 0.8).float()  # Identify bright regions in target
            specular_loss = F.mse_loss(
                output * specular_regions,
                target * specular_regions
            )
        else:
            specular_loss = torch.tensor(0.0, device=output.device)
        
        # Combined loss
        total_loss = (
            mse_loss + 
            0.5 * gradient_loss + 
            2.0 * specular_loss
        )
        
        return total_loss

class PatchDataset(Dataset):
    def __init__(self, distances, cosines, albedo, normals, targets):
        self.distances = torch.FloatTensor(distances)
        self.cosines = torch.FloatTensor(cosines)
        self.albedo = torch.FloatTensor(albedo)
        self.normals = torch.FloatTensor(normals)
        self.targets = torch.FloatTensor(targets)
        
        # Normalize inputs to prevent numerical issues
        self.normalize_inputs()
        
        self.patch_size = params.RTI_NET_PATCH_SIZE
        self.patches = self._create_patches()
        print(f"Created dataset with {len(self.patches)} valid patches")
    
    def normalize_inputs(self):
        """Normalize inputs to prevent numerical issues"""
        # Normalize targets to [0, 1]
        if torch.max(self.targets) > 0:
            self.targets = self.targets / torch.max(self.targets)
        
        # Normalize distances
        if torch.max(self.distances) > 0:
            self.distances = self.distances / torch.max(self.distances)
        
        # Ensure cosines are in [-1, 1]
        self.cosines = torch.clamp(self.cosines, -1, 1)
        
        # Normalize albedo to [0, 1]
        if torch.max(self.albedo) > 0:
            self.albedo = self.albedo / torch.max(self.albedo)
        
        # Normalize normals to unit length
        norm = torch.norm(self.normals, dim=-1, keepdim=True)
        self.normals = torch.where(norm > 0, self.normals / (norm + 1e-6), self.normals)
    
    def _is_valid_patch(self, k, n, i, j):
        """Check if a patch is valid based on content"""
        # Get the target patch
        patch = self.targets[k, n, i:i+self.patch_size[0], j:j+self.patch_size[1]]
        
        # Calculate percentage of non-zero pixels (across all channels)
        non_zero_mask = torch.any(patch > 0.01, dim=-1)  # Check across color channels
        non_zero_ratio = torch.sum(non_zero_mask).item() / (patch.shape[0] * patch.shape[1])
        
        # Check if patch has some variation
        patch_std = torch.std(patch)
        
        # More lenient criteria:
        # 1. At least 10% non-zero pixels
        # 2. Some variation in the patch
        is_valid = non_zero_ratio > 0.05 and patch_std > 0.005
        
        if not is_valid:
            return False, f"Invalid patch at {k},{n},{i},{j}: non_zero_ratio={non_zero_ratio:.3f}, std={patch_std:.3f}"
        return True, ""
    
    def _create_patches(self):
        """Create list of valid patches"""
        K, N, H, W = self.distances.shape
        patches = []
        
        # Calculate stride (75% overlap for more patches)
        stride_h = self.patch_size[0] // 4
        stride_w = self.patch_size[1] // 4
        
        total_patches = 0
        invalid_reasons = []
        
        for k in range(K):
            for n in range(N):
                for i in range(0, H - self.patch_size[0] + 1, stride_h):
                    for j in range(0, W - self.patch_size[1] + 1, stride_w):
                        total_patches += 1
                        is_valid, reason = self._is_valid_patch(k, n, i, j)
                        if is_valid:
                            patches.append((k, n, i, j))
                        else:
                            invalid_reasons.append(reason)
        
        print(f"Patch Statistics:")
        print(f"Total patches considered: {total_patches}")
        print(f"Valid patches: {len(patches)} ({len(patches)/total_patches*100:.1f}%)")
        
        if len(invalid_reasons) > 0:
            print("\nSample of invalid patch reasons:")
            for reason in invalid_reasons[:5]:  # Show first 5 reasons
                print(reason)
        
        if len(patches) < params.BATCH_SIZE:
            raise ValueError(f"Only found {len(patches)} valid patches, need at least {params.BATCH_SIZE} for a single batch")
            
        return patches
    
    def __len__(self):
        return len(self.patches)
    
      
    def __getitem__(self, idx):
        k, n, i, j = self.patches[idx]
        patch_slice = slice(i, i+self.patch_size[0]), slice(j, j+self.patch_size[1])
        
        try:
            item = {
                'distances': self.distances[k, n][patch_slice],
                'cosines': self.cosines[k, n][patch_slice],
                'albedo': self.albedo[k][patch_slice],
                'normals': self.normals[k][patch_slice],
                'target': self.targets[k, n][patch_slice]
            }
            
            # Additional validation
            for key, value in item.items():
                if torch.isnan(value).any():
                    print(f"Warning: NaN values found in {key} for patch {k},{n},{i},{j}")
                    # Replace NaN values with zeros
                    item[key] = torch.nan_to_num(value, 0.0)
            
            return item
            
        except Exception as e:
            print(f"Error loading patch {k},{n},{i},{j}: {str(e)}")
            # Return a neighboring valid patch instead
            alternate_idx = (idx + 1) % len(self.patches)
            return self.__getitem__(alternate_idx)


def train_model(model, train_loader, val_loader, num_epochs, model_save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = ReflectanceLoss().to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,  # Lower initial learning rate
        weight_decay=1e-5,
        eps=1e-8
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        for batch in train_loader:
            distances = batch['distances'].to(device)
            cosines = batch['cosines'].to(device)
            albedo = batch['albedo'].to(device)
            normals = batch['normals'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            outputs, intermediates = model(distances, cosines, albedo, normals)
            loss = criterion(outputs, targets, intermediates.get('specular_mask'))
            
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
                batch_count += 1
        
        # Calculate average training loss
        if batch_count > 0:
            train_loss /= batch_count
        else:
            print("Warning: No valid batches in training epoch")
            continue
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batch_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                distances = batch['distances'].to(device)
                cosines = batch['cosines'].to(device)
                albedo = batch['albedo'].to(device)
                normals = batch['normals'].to(device)
                targets = batch['target'].to(device)
                
                outputs, intermediates = model(distances, cosines, albedo, normals)
                loss = criterion(outputs, targets, intermediates.get('specular_mask'))
                
                if not torch.isnan(loss):
                    val_loss += loss.item()
                    val_batch_count += 1
        
        # Calculate average validation loss
        if val_batch_count > 0:
            val_loss /= val_batch_count
        else:
            print("Warning: No valid batches in validation")
            val_loss = float('inf')
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Update learning rate scheduler
        if not math.isnan(val_loss):
            scheduler.step(val_loss)
        
        # Save model and visualizations periodically
        if (epoch + 1) % params.RTI_NET_SAVE_MODEL_EVERY_N_EPOCHS == 0:
            # Only save if we have valid losses
            if not math.isnan(train_loss) and not math.isnan(val_loss):
                save_visualizations(model, val_loader, device, epoch + 1, model_save_path)
                torch.save(model.state_dict(), 
                          os.path.join(model_save_path, f'reflectance_model_epoch_{epoch+1}.pth'))
                
                # Save loss plot
                plt.figure(figsize=(10, 5))
                plt.plot(train_losses, label='Train Loss')
                plt.plot(val_losses, label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training and Validation Losses')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(model_save_path, f'loss_plot_epoch_{epoch+1}.png'))
                plt.close()
        
        # Early stopping condition
        if len(train_losses) > 10 and all(math.isnan(x) for x in train_losses[-5:]):
            print("Training stopped due to consecutive NaN losses")
            break
    
    return model

def save_visualizations(model, val_loader, device, epoch, save_path):
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        distances = batch['distances'].to(device)
        cosines = batch['cosines'].to(device)
        albedo = batch['albedo'].to(device)
        normals = batch['normals'].to(device)
        targets = batch['target'].to(device)
        
        outputs, intermediates = model(distances, cosines, albedo, normals)
        
        # Save sample images
        for i in range(min(3, outputs.shape[0])):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            ax1.imshow(targets[i].cpu().numpy())
            ax1.set_title('Target')
            ax1.axis('off')
            
            ax2.imshow(outputs[i].cpu().numpy())
            ax2.set_title('Output')
            ax2.axis('off')
            
            ax3.imshow(intermediates['specular_mask'][i, 0].cpu().numpy(), cmap='hot')
            ax3.set_title('Specular Mask')
            ax3.axis('off')
            
            plt.savefig(os.path.join(save_path, f'visualization_epoch_{epoch}_sample_{i}.png'))
            plt.close()

def prepare_data(distances, cosines, albedo, normals, targets):
    """
    Prepare data loaders with proper splitting strategy based on data dimensions.
    """
    K, N = distances.shape[:2]
    
    if K == 1:
        # When we have only one sample, split along N (number of lighting conditions)
        print("Single sample detected, splitting along lighting conditions dimension")
        train_indices, val_indices = train_test_split(range(N), test_size=0.2, random_state=42)
        
        # Create datasets by splitting along N
        train_dataset = PatchDataset(
            distances[:, train_indices], 
            cosines[:, train_indices],
            albedo,  # Albedo doesn't have N dimension
            normals,  # Normals don't have N dimension
            targets[:, train_indices]
        )
        
        val_dataset = PatchDataset(
            distances[:, val_indices], 
            cosines[:, val_indices],
            albedo,  # Albedo doesn't have N dimension
            normals,  # Normals don't have N dimension
            targets[:, val_indices]
        )
    else:
        # When we have multiple samples, split along K
        print("Multiple samples detected, splitting along sample dimension")
        train_indices, val_indices = train_test_split(range(K), test_size=0.2, random_state=42)
        
        train_dataset = PatchDataset(
            distances[train_indices], 
            cosines[train_indices],
            albedo[train_indices], 
            normals[train_indices],
            targets[train_indices]
        )
        
        val_dataset = PatchDataset(
            distances[val_indices], 
            cosines[val_indices],
            albedo[val_indices], 
            normals[val_indices],
            targets[val_indices]
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=params.BATCH_SIZE,
        shuffle=True,
        num_workers=params.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=params.BATCH_SIZE,
        shuffle=False,
        num_workers=params.NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    return train_loader, val_loader

def train(distances, cosines, albedo, normals, targets):
    print("Starting training process...")
    
    # Convert inputs to numpy arrays
    distances = np.array(distances)
    cosines = np.array(cosines)
    albedo = np.array(albedo)
    normals = np.array(normals)
    targets = np.array(targets)
    
    # Create save directory
    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(model_save_path, f"reflectance_model_{timestamp}")
    os.makedirs(model_save_path, exist_ok=True)
    
    # Save training parameters
    shutil.copy2('./utils/params.py', model_save_path)
    
    print("Input shapes:")
    print(f"Distances: {distances.shape}")
    print(f"Cosines: {cosines.shape}")
    print(f"Albedo: {albedo.shape}")
    print(f"Normals: {normals.shape}")
    print(f"Targets: {targets.shape}")
    
    # Prepare data loaders
    print("Preparing data loaders...")
    train_loader, val_loader = prepare_data(distances, cosines, albedo, normals, targets)
    
    # Initialize model
    print("Initializing model...")
    model = ReflectanceNet(albedo_channels=albedo.shape[-1])
    
    # Print model summary
    print("\nModel Architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Training setup
    print("\nStarting training...")
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=params.RTI_NET_EPOCHS,
        model_save_path=model_save_path
    )
    
    # Save final model
    final_model_path = os.path.join(model_save_path, 'reflectance_model_final.pth')
    torch.save(trained_model.state_dict(), final_model_path)
    print(f"\nTraining completed. Final model saved at: {final_model_path}")
    
    return trained_model

def get_gpu_memory_stats():
    """Get GPU memory usage statistics"""
    if not torch.cuda.is_available():
        return "GPU not available"
    
    try:
        gpu_stats = []
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**2
            reserved = torch.cuda.memory_reserved(i) / 1024**2
            
            props = torch.cuda.get_device_properties(i)
            gpu_stats.append(
                f"GPU {i} ({props.name}): "
                f"{allocated:.1f}MB allocated, "
                f"{reserved:.1f}MB reserved"
            )
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
    """Print comprehensive GPU statistics"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "="*80)
    print(f"GPU Stats at {current_time} (Epoch {epoch+1}, Batch {batch_idx}/{num_batches})")
    print("-"*80)
    print("Memory Usage:")
    print(get_gpu_memory_stats())
    print("\nGPU Utilization:")
    print("\n".join(get_gpu_utilization()))
    print("="*80 + "\n")

# if __name__ == "__main__":
#     # Example usage
#     try:
#         # Assuming data is loaded from somewhere
#         # distances, cosines, albedo, normals, targets = load_data()
        
#         # Set random seeds for reproducibility
#         random.seed(42)
#         np.random.seed(42)
#         torch.manual_seed(42)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(42)
        
#         # Train model
#         model = train(distances, cosines, albedo, normals, targets)
        
#     except Exception as e:
#         print(f"Error during training: {str(e)}")
#         raise