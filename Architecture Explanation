1. Base Architecture Choice: UNet
We chose UNet as the base architecture because:

Its encoder-decoder structure is excellent for image-to-image translation tasks
Skip connections help preserve spatial information which is crucial for defect preservation
The architecture can handle both local and global features effectively
It has proven success in medical image segmentation where preserving small details is critical

2. Architectural Modifications
2.1 Channel Attention Mechanism
Added CBAM-style attention because:

Helps the model focus on relevant features in both clean and defect areas
Adaptively recalibrates channel-wise feature responses
Particularly useful for defect regions which might have subtle features

pythonCopyclass ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        # Using both max and average pooling provides comprehensive feature information
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
2.2 Skip Connections
Modified the original UNet skip connections because:

Added controllable skip connections with [False, False, False, False] configuration
This modification helps balance feature propagation
Prevents excessive low-level feature transfer which could affect defect restoration
Allows the model to selectively use features from encoder layers

2.3 Defect-Aware Processing
Added mask attention mechanism because:

Helps the model process defect and non-defect regions differently
Preserves defect characteristics while still denoising the overall image
Allows for adaptive processing based on defect location

pythonCopyself.mask_attention = nn.Sequential(
    nn.Conv2d(1, 64, 3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 1, 1),
    nn.Sigmoid()
)
3. Loss Function Design
We designed a custom multi-component loss function because different aspects of the restoration need different types of supervision:
3.1 Components and Their Purposes
pythonCopytotal_loss = (0.25 * defect_loss +     
              0.30 * ssim_loss +        
              0.20 * perceptual_loss +  
              0.15 * edge_loss +        
              0.10 * basic_loss)

Defect Loss (weight: 0.25):

Focuses on preserving defect regions
Uses weighted mask to give more importance to defect areas
Helps maintain defect characteristics


SSIM Loss (weight: 0.30):

Ensures structural similarity with ground truth
Particularly important for overall image quality
Highest weight because it balances structure and perception


Perceptual Loss (weight: 0.20):

Uses VGG features to capture high-level content
Helps maintain perceptual quality
Important for natural-looking results


Edge Loss (weight: 0.15):

Preserves edge information
Critical for defect boundaries
Uses Sobel filters for edge detection


Basic Loss (weight: 0.10):

Combination of L1 and MSE loss
Provides pixel-level supervision
Lower weight as it's less important than structural features



4. Training Strategy

Used AdamW optimizer with learning rate 1e-4
Implemented CosineAnnealingWarmRestarts scheduler for better convergence
Added dropout (0.15) for regularization
Early stopping with patience of 25 epochs
Batch size of 8 for stable training

5. Results and Metrics
Our model achieved:

Overall PSNR: 29.89 dB
Overall SSIM: 0.76
Defect PSNR: 42.95 dB
Defect SSIM: 0.99

These results show:

Strong defect preservation (high defect metrics)
Good overall image restoration
Balanced performance between cleanup and defect preservation

6. References

UNet: "U-Net: Convolutional Networks for Biomedical Image Segmentation" by Ronneberger et al.
CBAM: "CBAM: Convolutional Block Attention Module" by Woo et al.
[Add other papers you referenced]

7. Implementation
The complete implementation is available at: https://github.com/I-am-VarunM/DeNoisingProject-KLATencor
Model weights: https://drive.google.com/file/d/1izxVWqllWJ5hQe9SDaDji94gqBzTIlvo/view?usp=sharing
