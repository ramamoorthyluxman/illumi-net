# Neural Network for Image Relighting with Multi-Input Processing and Hybrid Architecture

## Table of Contents
1. [Introduction](#introduction)
2. [Model Architecture](#model-architecture)
3. [Data Processing](#data-processing)
4. [Loss Functions](#loss-functions)
5. [Training Process](#training-process)
6. [Usage](#usage)
7. [Dependencies](#dependencies)
8. [Detailed Network Architecture Explanation](#detailed-network-architecture-explanation)

## Introduction

This project implements a deep learning model for image relighting using a combination of convolutional neural networks and attention mechanisms. The model takes as input distance matrices, cosine matrices, surface albedos, and surface normals to produce relit images.

## Model Architecture

The RelightingModel class defines the neural network architecture:

1. **Input Processing**:
   - Separate convolutional layers for distances, cosines, albedo, and normals.
   - Distances and cosines use 1-channel input, albedo uses variable channels, and normals use 3 channels.

2. **Encoder**:
   - Four ResidualBlock2D layers with increasing dilation rates (1, 2, 4, 8).
   - Each block increases the number of channels (128, 256, 512, 1024).
   - MaxPooling is applied between encoder blocks.

3. **Bridge**:
   - An AttentionBlock2D to focus on important features.

4. **Decoder**:
   - Three ResidualBlock2D layers.
   - Features from the encoder are concatenated using skip connections.
   - Upsampling is done using bilinear interpolation.

5. **Output**:
   - A final convolutional layer to produce the 3-channel output image.

### Key Components:

- **ResidualBlock2D**: Implements residual connections with dilated convolutions.
- **AttentionBlock2D**: Applies self-attention mechanism to focus on important features.

## Data Processing

The PatchRelightingDataset class handles data preparation:

1. Converts input data to PyTorch tensors.
2. Calculates optimal patch sizes based on image dimensions and desired number of patches.
3. Creates patches from the input data for efficient training.

## Loss Functions

The model uses a combination of losses:

1. **MSE Loss**: Mean Squared Error between the output and target images.
2. **Perceptual Loss**: Uses a pre-trained VGG16 network to compare high-level features.
3. **Combined Loss**: A weighted sum of MSE and Perceptual losses.

## Training Process

The training process involves:

1. Preparing data loaders for training and validation sets.
2. Initializing the model, loss functions, and optimizer (Adam).
3. Training loop:
   - Forward pass
   - Loss calculation
   - Backpropagation
   - Optimization step
4. Validation after each epoch.
5. Learning rate scheduling using ReduceLROnPlateau.
6. Saving model checkpoints and visualization outputs at regular intervals.

## Usage

To train the model:

1. Prepare your input data (distances, cosines, albedo, normals, targets).
2. Call the `train` function with your data.
3. The function will handle data preparation, model initialization, and training.

```python
trained_model = train(distances, cosines, albedo, normals, targets)
```

## Dependencies

- PyTorch
- NumPy
- Matplotlib
- scikit-learn
- torchvision

Make sure to install these dependencies before running the code.

## Visualization

The training process includes:

1. Saving comparison images of original and reconstructed samples.
2. Plotting training and validation losses.

These visualizations are saved at regular intervals to monitor the training progress.

## Detailed Network Architecture Explanation

The relighting model implemented here is a hybrid architecture that combines elements from several popular network designs. It's primarily based on the U-Net architecture, but incorporates modern improvements like residual connections, dilated convolutions, and attention mechanisms.

### 1. Overall Structure: Modified U-Net

The network follows a U-Net-like encoder-decoder structure:

- It has a contracting path (encoder) that captures context.
- An expanding path (decoder) that enables precise localization.
- Skip connections that transfer information across the network.

However, unlike a traditional U-Net:
- It uses residual blocks instead of plain convolutions.
- It incorporates dilated convolutions in the encoder.
- It adds an attention mechanism at the bottleneck.

### 2. Encoder: Dilated Residual Network

The encoder is composed of ResidualBlock2D modules with increasing dilation rates:

```python
self.encoder1 = ResidualBlock2D(192, 128, dilation=1)
self.encoder2 = ResidualBlock2D(128, 256, dilation=2)
self.encoder3 = ResidualBlock2D(256, 512, dilation=4)
self.encoder4 = ResidualBlock2D(512, 1024, dilation=8)
```

- **Residual Connections**: These allow the network to learn residual functions, making it easier to train deep networks.
- **Dilated Convolutions**: These increase the receptive field without losing resolution, capturing multi-scale information.

This design is inspired by networks like DeepLab for semantic segmentation, which use dilated convolutions to capture multi-scale context.

### 3. Bottleneck: Attention Mechanism

At the bottleneck, an AttentionBlock2D is used:

```python
self.bridge = AttentionBlock2D(1024)
```

This attention mechanism is similar to the Squeeze-and-Excitation (SE) block:
- It uses global average pooling to capture channel-wise statistics.
- Two fully connected layers learn channel-wise attention weights.
- These weights are applied to the input feature map.

This allows the network to focus on the most important features, similar to the attention mechanism in transformers, but applied in a convolutional context.

### 4. Decoder: Upsampling with Skip Connections

The decoder uses a series of upsampling operations and ResidualBlock2D modules:

```python
self.decoder3 = ResidualBlock2D(1536, 512)
self.decoder2 = ResidualBlock2D(768, 256)
self.decoder1 = ResidualBlock2D(384, 128)
```

- **Skip Connections**: Features from the encoder are concatenated with the upsampled features, allowing the network to combine low-level and high-level information.
- **Residual Blocks**: These are used in the decoder as well, maintaining the benefits of residual learning throughout the network.

### 5. Input Processing

The network processes multiple input types separately before combining them:

```python
self.conv_distances = nn.Conv2d(1, 64, kernel_size=3, padding=1)
self.conv_cosines = nn.Conv2d(1, 64, kernel_size=3, padding=1)
self.conv_albedo = nn.Conv2d(albedo_channels, 32, kernel_size=3, padding=1)
self.conv_normals = nn.Conv2d(3, 32, kernel_size=3, padding=1)
```

This allows the network to learn specialized features for each input type before combining them.

### 6. Output Generation

The final output is generated using a 1x1 convolution:

```python
self.final_conv = nn.Conv2d(128, 3, kernel_size=1)
```

This projects the final feature map to the desired 3-channel output.

### Comparison to Other Architectures

1. **U-Net**: The overall encoder-decoder structure with skip connections is similar to U-Net, but with significant modifications.

2. **ResNet**: The use of residual blocks throughout the network is inspired by ResNet architectures.

3. **DeepLab**: The use of dilated convolutions in the encoder is similar to DeepLab's atrous spatial pyramid pooling.

4. **Squeeze-and-Excitation Networks**: The attention mechanism is similar to SE blocks, adding channel-wise attention.

5. **Transformers**: While not using self-attention like transformers, the attention block incorporates a form of global context similar to the global attention in vision transformers.

In summary, this network is a hybrid architecture that combines the strengths of U-Net (encoder-decoder with skip connections), ResNet (residual learning), dilated