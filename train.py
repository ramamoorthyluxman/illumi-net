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

class PatchRelightingDataset(Dataset):
    def __init__(self, distances, cosines, albedo, normals, targets, same_illumination=False):
        self.distances = torch.FloatTensor(distances)
        self.cosines = torch.FloatTensor(cosines)
        self.albedo = torch.ByteTensor(albedo)
        self.normals = torch.ByteTensor(normals)
        self.targets = torch.ByteTensor(targets)
        self.patch_size = params.RTI_NET_PATCH_SIZE
        self.patches_per_image = params.RTI_NET_PATCHES_PER_IMAGE
        self.same_illumination = same_illumination
        
        self.patches = self._create_patches()

    def _create_patches(self):
        patches = []
        if self.same_illumination:
            H, W = self.distances.shape
            K = self.albedo.shape[0]
            N = self.targets.shape[1]
        else:
            K, N, H, W = self.distances.shape
        
        for k in range(K):
            # Generate patch locations for this acquisition
            acquisition_patches = []
            for _ in range(self.patches_per_image):
                h = random.randint(0, H - self.patch_size)
                w = random.randint(0, W - self.patch_size)
                acquisition_patches.append((h, w))
            
            # Use the same patches for all images in this acquisition
            for n in range(N):
                for h, w in acquisition_patches:
                    patches.append((k, n, h, w))
        
        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        k, n, i, j = self.patches[idx]
        patch_slice = slice(i, i + self.patch_size), slice(j, j + self.patch_size)

        if self.same_illumination:
            distances = self.distances[patch_slice]
            cosines = self.cosines[patch_slice]
        else:
            distances = self.distances[k, n][patch_slice]
            cosines = self.cosines[k, n][patch_slice]

        albedo = self.albedo[k][patch_slice].float() / 255.0
        normals = self.normals[k][patch_slice].float() / 255.0
        target = self.targets[k, n][patch_slice].float() / 255.0

        return {
            'distances': distances,
            'cosines': cosines,
            'albedo': albedo,
            'normals': normals,
            'target': target
        }

class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(ResidualBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else None
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
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

def train_model(model, train_loader, val_loader, num_epochs=100, model_save_path='.'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_loss = float('inf')

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
            loss = criterion(outputs, targets)
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
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        scheduler.step()

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))

        # Save model every N epochs
        if (epoch + 1) % params.RTI_NET_SAVE_MODEL_EVERY_N_EPOCHS == 0:
            save_path = os.path.join(model_save_path, f'relighting_model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch+1}")

    return model

def prepare_data(distances, cosines, albedo, normals, targets, same_illumination=False):
    K = albedo.shape[0]
    
    train_indices, val_indices = train_test_split(
        np.arange(K), 
        test_size=0.2, 
        random_state=42
    )

    # Create datasets
    train_dataset = PatchRelightingDataset(
        distances, cosines,
        albedo[train_indices], normals[train_indices],
        targets[train_indices],
        same_illumination=same_illumination
    )
    val_dataset = PatchRelightingDataset(
        distances, cosines,
        albedo[val_indices], normals[val_indices],
        targets[val_indices],
        same_illumination=same_illumination
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=14)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=14)

    return train_loader, val_loader

def train(distances, cosines, albedo, normals, targets):
    distances = np.transpose(distances)
    cosines = np.transpose(cosines)

    same_illumination = params.SAME_ILLUMINATION

    model_save_path = os.path.join(os.path.dirname(__file__), 'saved_models')

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    print("Input shapes - distances: ", distances.shape, "cosines: ", cosines.shape, 
          "albedo: ", albedo.shape, "normals: ", normals.shape, "targets: ", targets.shape)

    # Prepare data
    print("Preparing data...")
    train_loader, val_loader = prepare_data(distances, cosines, albedo, normals, targets, same_illumination)

    # Initialize the model
    print("Initializing model...")
    model = RelightingModel(height=params.RTI_NET_PATCH_SIZE, width=params.RTI_NET_PATCH_SIZE)

    # Train the model
    print("Training model...")
    trained_model = train_model(model, train_loader, val_loader, num_epochs=params.RTI_NET_EPOCHS, model_save_path=model_save_path)

    # Save the final model
    torch.save(trained_model.state_dict(), os.path.join(model_save_path, 'relighting_model_final.pth'))

    print("Training completed and model saved.")

# if __name__ == "__main__":
#     # Example usage (replace with your actual data loading)
#     K, N, H, W = 5, 10, 256, 256  # Adjust these values as needed
#     same_illumination = True  # Set this according to your needs

#     if same_illumination:
#         distances = np.random.rand(H, W).astype(np.float32)  # Single distance matrix for all K and N
#         cosines = np.random.rand(H, W).astype(np.float32)    # Single cosine matrix for all K and N
#     else:
#         distances = np.random.rand(K, N, H, W).astype(np.float32)
#         cosines = np.random.rand(K, N, H, W).astype(np.float32)

#     albedo = np.random.randint(0, 256, (K, H, W), dtype=np.uint8)
#     normals = np.random.rand(K, H, W, 3).astype(np.float32)
#     targets = np.random.randint(0, 256, (K, N, H, W, 3), dtype=np.uint8)

#     train(distances, cosines, albedo, normals, targets)