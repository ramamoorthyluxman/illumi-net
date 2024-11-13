# IllumiNet: Neural Relighting for Cultural Heritage

## Overview
IllumiNet is an advanced deep learning framework designed for relighting cultural heritage artifacts using Reflectance Transformation Imaging (RTI) techniques. The project implements a novel neural network architecture that combines photometric stereo principles with modern deep learning approaches to achieve high-quality relighting results.

## Features
- Neural relighting of cultural heritage objects
- Support for both collimated and point-source lighting models
- Automatic surface normal and albedo estimation
- Distance and cosine matrix computation for light interaction
- Patch-based training for efficient processing of high-resolution images
- Multi-scale feature processing with residual and attention mechanisms
- Comprehensive loss function incorporating perceptual and physics-based components
- Hyperparameter optimization capabilities
- Real-time GPU memory monitoring and utilization tracking

## Requirements

### Hardware
- CUDA-capable GPU(s) with at least 8GB VRAM
- Sufficient RAM for dataset handling (16GB minimum recommended)
- Storage space for dataset and model checkpoints

### Software Dependencies
```bash
# Core dependencies
numpy
torch
torchvision
matplotlib
opencv-python (cv2)
scikit-learn
tqdm

# Optional dependencies
lion-pytorch  # For Lion optimizer
adabelief-pytorch  # For AdaBelief optimizer
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/illumi-net.git
cd illumi-net
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
illumi-net/
├── main.py                # Main entry point
├── train.py               # Training implementation
├── relight.py             # Relighting implementation
├── utils/
│   ├── params.py          # Configuration parameters
│   ├── dataset_class.py   # Dataset handling
│   └── compute_normals.py # Normal map computation
├── saved_models/          # Model checkpoints
└── data/                  # Dataset storage
```

## Configuration

### Key Parameters in `params.py`

#### Dataset Parameters
- `ACQ_PATHS`: List of paths to acquisition folders
- `MAX_NB_IMAGES_PER_ACQ`: Maximum number of images per acquisition
- `SURFACE_PHYSICAL_SIZE`: Physical dimensions of the surface [width, height]
- `COLLIMATED_LIGHT`: Toggle between collimated and point-source lighting

#### Training Parameters
- `RTI_NET_EPOCHS`: Number of training epochs
- `RTI_NET_PATCH_SIZE`: Size of image patches for training [height, width]
- `RTI_MAX_NUMBER_PATCHES`: Maximum number of patches per image
- `BATCH_SIZE`: Training batch size
- `LEARNING_RATE`: Initial learning rate
- `OPTIMIZER`: Choice of optimizer ("Adam", "AdamW", "Lion", "RAdam", "AdaBelief", "SGD")

#### Loss Function Weights
- `LAMBDA_MSE`: MSE loss weight
- `LAMBDA_L1`: L1 loss weight
- `LAMBDA_HIGHLIGHT`: Highlight preservation weight
- `LAMBDA_GRADIENT`: Gradient consistency weight
- `LAMBDA_SPECULAR`: Specular reflection weight
- `LAMBDA_CONTRAST`: Local contrast weight
- `LAMBDA_PERCEPTUAL`: Perceptual loss weight

## Usage

### Training

1. Configure your dataset paths (list of acquisition paths) and parameters in `params.py`:
```python
ACQ_PATHS = ['/path/to/your/dataset']
TRAINING = True
```

2. Run training:
```bash
python main.py
```

### Relighting

1. Update `params.py` with model path and output settings:
```python
TRAINING = False
RTI_MODEL_PATH = '/path/to/saved/model.pth'
```

2. Run relighting:
```bash
python main.py
```

## Network Architecture

### 1. Input Processing Networks

The network utilizes multiple specialized components for processing different types of input data:

#### 1.1 Distance Feature Network
```
Input: [B, 1, H, W] (Distance matrices)
└── Conv2D(in=1, out=96, k=3, p=1)
    └── BatchNorm2D(96)
    └── PReLU
        ├── SmallScaleConv(in=96, out=48, k=1)
        └── LargeScaleConv(in=96, out=48, k=5, p=2)
```

#### 1.2 Cosine Feature Network
```
Input: [B, 1, H, W] (Cosine matrices)
└── Conv2D(in=1, out=96, k=3, p=1)
    └── BatchNorm2D(96)
    └── PReLU
        ├── SmallScaleConv(in=96, out=48, k=1)
        └── LargeScaleConv(in=96, out=48, k=5, p=2)
```

#### 1.3 Albedo Processing Network
```
Input: [B, 3, H, W] (RGB albedo maps)
└── Conv2D(in=3, out=48, k=3, p=1)
    └── BatchNorm2D(48)
    └── PReLU
```

#### 1.4 Normal Processing Network
```
Input: [B, 3, H, W] (Surface normals)
└── Conv2D(in=3, out=48, k=3, p=1)
    └── BatchNorm2D(48)
    └── PReLU
```

#### 1.5 Angular Feature Network
```
Input: [B, 1] (Azimuth angles)
└── Linear(in=1, out=32)
    └── ReLU
    └── Linear(in=32, out=64)
        └── ReLU
        └── Linear(in=64, out=64)
            └── Spatial Expansion to [B, 64, H, W]
```

### 2. Feature Integration

The processed features are concatenated along the channel dimension:
```
Combined Features = [
    Distance Small-Scale (48),
    Distance Large-Scale (48),
    Cosine Small-Scale (48),
    Cosine Large-Scale (48),
    Albedo Features (48),
    Normal Features (48),
    Angular Features (64)
]
Total Channels: 352
```

### 3. Core Network Architecture

#### 3.1 Encoder Path
```
Input: [B, 352, H, W]
└── ResBlock1 (352 → 128) + Dropout(0.2)
    └── MaxPool2D
    └── ResBlock2 (128 → 256)
        └── MaxPool2D
        └── ResBlock3 (256 → 512)
            └── MaxPool2D
            └── ResBlock4 (512 → 1024)
                └── AttentionBridge
```

#### 3.2 Residual Block Structure
```
ResidualBlock2D:
    Input
    ├── Conv2D → InstanceNorm2D → ReLU
    │   └── Conv2D → InstanceNorm2D
    └── Identity/1x1Conv (if channel change)
    └── Add → ReLU
```

#### 3.3 Attention Mechanisms

##### Spatial Attention
```
Input Feature Maps
└── Average Pooling Branch
    └── Max Pooling Branch
        └── Concatenate
            └── Conv2D(2→1, k=7, p=3)
                └── Sigmoid
                    └── Multiply with Input
```

##### Channel Attention (SE Block)
```
Input Feature Maps
└── Global Average Pooling
    └── FC(in→in//16)
        └── ReLU
        └── FC(in//16→in)
            └── Sigmoid
                └── Scale Features
```

#### 3.4 Decoder Path
```
Bridge Features [B, 1024, H/8, W/8]
└── Upsample + Concat(Skip3)
    └── ResBlock3 (1536 → 512)
        └── Upsample + Concat(Skip2)
        └── ResBlock2 (768 → 256)
            └── Upsample + Concat(Skip1)
            └── ResBlock1 (384 → 128)
```

### 4. Output Generation
```
Features [B, 128, H, W]
└── Conv2D(128→64) → BatchNorm → PReLU
    └── Conv2D(64→32) → BatchNorm → PReLU
        └── Conv2D(32→3)
            └── Reshape to [B, H, W, 3]
```

## Implementation Details

### 1. Feature Dimensions Through the Network

```
Input Stages:
- Distance/Cosine: [B, 1, H, W] → [B, 96, H, W] → [B, 48*2, H, W]
- Albedo/Normals: [B, C, H, W] → [B, 48, H, W]
- Angular: [B, 1] → [B, 64, H, W]

Encoder Stages:
Level 1: [B, 352, H, W] → [B, 128, H, W]
Level 2: [B, 128, H/2, W/2] → [B, 256, H/2, W/2]
Level 3: [B, 256, H/4, W/4] → [B, 512, H/4, W/4]
Level 4: [B, 512, H/8, W/8] → [B, 1024, H/8, W/8]

Decoder Stages:
Level 3: [B, 1536, H/4, W/4] → [B, 512, H/4, W/4]
Level 2: [B, 768, H/2, W/2] → [B, 256, H/2, W/2]
Level 1: [B, 384, H, W] → [B, 128, H, W]

Output: [B, 3, H, W] → [B, H, W, 3]
```

### 2. Key Architectural Features

#### Multi-Scale Processing
- Dual-path processing for distance and cosine features
- Small-scale (1x1 conv) for local features
- Large-scale (5x5 conv) for contextual information

#### Skip Connections
- ResNet-style connections within blocks
- U-Net-style connections between encoder and decoder
- Helps maintain fine details and gradient flow

#### Normalization Strategy
- Instance Normalization in residual blocks for style-invariant features
- Batch Normalization in main pathway for stable training

#### Attention Integration
- Spatial attention for focusing on relevant image regions
- Channel attention for feature recalibration
- Bridge attention for global context

## Memory Considerations

For an input image of size H×W:
1. Feature maps at original resolution: O(HW)
2. Multi-scale features: O(HW/4 + HW/16 + HW/64)
3. Skip connections: Additional O(HW)

Total memory complexity: O(HW)

## Training Details

### Patch-Based Training
- Input images are divided into patches of size [64×64]
- Patches are filtered based on content validity
- Valid patches must contain:
  - Sufficient non-black pixels (>50%)
  - Valid normal information
  - Valid albedo information

### Gradient Flow
- PReLU activations prevent dying gradients
- Residual connections assist gradient propagation
- Instance normalization helps with feature distribution

### Loss Function Components
The combined loss function includes:
- MSE and L1 losses
- Perceptual loss using VGG features
- Gradient-domain loss
- Highlight-aware loss
- Local contrast loss
- Range-specific losses for dark, mid, and bright regions

## Data Preparation

1. **Dataset Structure**
```
dataset/
├── acquisition1/
│   ├── images/
│   ├── normal_map.npy
│   ├── albedo.png
│   ├── distance_matrices.npy
│   └── angles_matrices.npy
└── acquisition2/
    └── ...
```

2. Required files per acquisition:
   - `.lp` files containing light positions
   - Input images
   - Normal maps and albedo (computed or provided)
   - Distance and angle matrices

## Visualization Tools

The project includes tools for:
- Distance and cosine matrix heatmaps
- Training progress monitoring
- Comparison visualizations
- GPU memory usage tracking

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{illuminet2024,
  title={IllumiNet: Neural Relighting for Cultural Heritage Preservation},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## Contact

For questions or support, please open an issue in the GitHub repository or contact the maintainers at l.ramamoorthy01gmail.com.