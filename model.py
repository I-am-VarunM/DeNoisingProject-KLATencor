import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
class ChannelAttention(nn.Module):
    """
    Channel attention module for focusing on important features
    """
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False)
        )
        
    def forward(self, x):
        # Generate attention using both average and max pooling
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        
        # Combine and apply sigmoid
        attention = torch.sigmoid(avg_out + max_out)
        return x * attention

def plot_training_curves(history):
    """
    Plot training metrics over time with improved error handling
    """
    plt.figure(figsize=(15, 10))
    
    # Plot training loss
    if isinstance(history.get('train_loss', []), list) and history['train_loss']:
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    
    # Extract and plot validation metrics
    if 'val_metrics' in history and history['val_metrics']:
        epochs = range(len(history['val_metrics']))
        
        # Extract PSNR values
        psnr_values = [m.get('psnr', 0) for m in history['val_metrics']]
        defect_psnr = [m.get('defect_psnr', 0) for m in history['val_metrics']]
        
        # Extract SSIM values
        ssim_values = [m.get('ssim', 0) for m in history['val_metrics']]
        defect_ssim = [m.get('defect_ssim', 0) for m in history['val_metrics']]
        
        # Plot PSNR
        plt.subplot(2, 2, 2)
        plt.plot(epochs, psnr_values, label='Overall PSNR')
        plt.plot(epochs, defect_psnr, label='Defect PSNR')
        plt.title('PSNR Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR (dB)')
        plt.legend()
        
        # Plot SSIM
        plt.subplot(2, 2, 3)
        plt.plot(epochs, ssim_values, label='Overall SSIM')
        plt.plot(epochs, defect_ssim, label='Defect SSIM')
        plt.title('SSIM Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()
def visualize_predictions(model, dataloader, device, num_samples=4):
    """
    Visualize model predictions with defect overlay
    """
    model.eval()
    fig, axes = plt.subplots(4, num_samples, figsize=(20, 16))
    
    with torch.no_grad():
        for i, (degraded, clean, mask) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            degraded = degraded.to(device)
            clean = clean.to(device)
            mask = mask.to(device)
            
            if mask.shape[1] == 3:
                mask = mask.mean(dim=1, keepdim=True)
            
            outputs = model(degraded, mask)
            
            # Convert to numpy for visualization
            degraded_np = degraded[0].cpu().numpy().transpose(1, 2, 0)
            clean_np = clean[0].cpu().numpy().transpose(1, 2, 0)
            output_np = outputs[0].cpu().numpy().transpose(1, 2, 0)
            mask_np = mask[0].cpu().numpy().transpose(1, 2, 0)
            
            # Plot results
            axes[0, i].imshow(degraded_np)
            axes[0, i].set_title('Degraded')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(clean_np)
            axes[1, i].imshow(mask_np, alpha=0.3, cmap='Reds')
            axes[1, i].set_title('Clean + Defects')
            axes[1, i].axis('off')
            
            axes[2, i].imshow(output_np)
            axes[2, i].imshow(mask_np, alpha=0.3, cmap='Reds')
            axes[2, i].set_title('Restored + Defects')
            axes[2, i].axis('off')
            
            # Plot difference map
            diff = np.abs(clean_np - output_np).mean(axis=2)
            diff = diff * mask_np.squeeze()
            axes[3, i].imshow(diff, cmap='hot')
            axes[3, i].set_title('Difference in Defects')
            axes[3, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.close()

def save_checkpoint(model, optimizer, epoch, metrics, filename):
    """
    Save model checkpoint with all necessary information
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename):
    """
    Load model checkpoint and restore training state
    """
    try:
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics']
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        return 0, None

def calculate_model_size(model):
    """
    Calculate and print model size and parameters
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        
    size_all_mb = (param_size + buffer_size) / 1024**2
    return {
        'parameters': sum(p.numel() for p in model.parameters()),
        'size_mb': size_all_mb,
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
def calculate_ssim(img1, img2, window_size=11):
    """
    Calculate SSIM with improved error handling and stability
    """
    try:
        # Convert inputs to grayscale if they're RGB
        if img1.ndim == 3 and img1.shape[2] == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        if img2.ndim == 3 and img2.shape[2] == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        # Constants for numerical stability
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        # Ensure proper data type and range
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        # Gaussian kernel
        kernel = cv2.getGaussianKernel(window_size, 1.5)
        window = np.outer(kernel, kernel.transpose())
        
        # Means
        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Variances and covariance
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
        
        # Add small epsilon for numerical stability
        eps = 1e-12
        sigma1_sq = np.maximum(sigma1_sq, eps)
        sigma2_sq = np.maximum(sigma2_sq, eps)
        
        # Calculate SSIM
        num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = num / den
        
        # Handle any invalid values
        ssim_map = np.nan_to_num(ssim_map, nan=0.0, posinf=1.0, neginf=0.0)
        
        return float(np.mean(ssim_map))
        
    except Exception as e:
        print(f"Error in SSIM calculation: {str(e)}")
        return 0.0

class PriorityFocusedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.vgg = self._load_vgg()
        
    def _load_vgg(self):
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:23]
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        return vgg
    
    def _ssim_loss(self, pred, target):
        """Calculate SSIM-based loss with adjusted window size"""
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        # Use larger window size for better structural similarity
        kernel_size = 5
        sigma = 1.5
        
        # Gaussian kernel
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        kernel = kernel.to(pred.device)
        
        # Calculate mean
        mu_x = F.conv2d(pred, kernel, padding=kernel_size//2, groups=pred.shape[1])
        mu_y = F.conv2d(target, kernel, padding=kernel_size//2, groups=target.shape[1])
        
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        # Calculate variance and covariance
        sigma_x_sq = F.conv2d(pred ** 2, kernel, padding=kernel_size//2, groups=pred.shape[1]) - mu_x_sq
        sigma_y_sq = F.conv2d(target ** 2, kernel, padding=kernel_size//2, groups=target.shape[1]) - mu_y_sq
        sigma_xy = F.conv2d(pred * target, kernel, padding=kernel_size//2, groups=pred.shape[1]) - mu_xy
        
        # SSIM formula
        num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        den = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
        
        ssim = num / (den + 1e-8)
        return 1 - ssim.mean()
    
    def _create_gaussian_kernel(self, kernel_size, sigma):
        """Create Gaussian kernel for SSIM calculation"""
        coords = torch.arange(kernel_size).float() - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel = g.unsqueeze(0) * g.unsqueeze(1)
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
    
    def _defect_focused_loss(self, pred, target, mask):
        # Reduce defect region weight from 19 to 12
        weighted_mask = mask * 11 + 1
        
        # Combined loss for defect regions with adjusted weights
        mse_loss = torch.mean(weighted_mask * (pred - target) ** 2)
        ssim_loss = self._ssim_loss(pred * mask, target * mask)
        l1_loss = torch.mean(weighted_mask * torch.abs(pred - target))
        
        # Adjust weights to focus more on overall image quality
        return 0.3 * mse_loss + 0.3 * ssim_loss + 0.4 * l1_loss
    
    def forward(self, pred, target, mask):
        # Component losses with adjusted weights
        defect_loss = self._defect_focused_loss(pred, target, mask)
        perceptual_loss = self.mse_loss(self.vgg(pred), self.vgg(target))
        ssim_loss = self._ssim_loss(pred, target)
        
        # New edge-preservation loss
        edge_loss = self._edge_preservation_loss(pred, target)
        
        # Basic reconstruction loss with L1 and MSE
        basic_loss = 0.6 * self.mse_loss(pred, target) + 0.4 * self.l1_loss(pred, target)
        
        # Adjusted weights to prioritize overall image quality
        total_loss = (0.25 * defect_loss +     # Reduced from 0.4
                     0.30 * ssim_loss +        # Increased from 0.2
                     0.20 * perceptual_loss +  # Kept same
                     0.15 * edge_loss +        # Added edge preservation
                     0.10 * basic_loss)        # Reduced from 0.2
        
        return total_loss
    
    def _edge_preservation_loss(self, pred, target):
        """Edge preservation loss for better structural details"""
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                             device=pred.device).float().unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                             device=pred.device).float().unsqueeze(0).unsqueeze(0)
        
        # Calculate edges
        pred_edges_x = F.conv2d(pred, sobel_x.repeat(3,1,1,1), padding=1, groups=3)
        pred_edges_y = F.conv2d(pred, sobel_y.repeat(3,1,1,1), padding=1, groups=3)
        target_edges_x = F.conv2d(target, sobel_x.repeat(3,1,1,1), padding=1, groups=3)
        target_edges_y = F.conv2d(target, sobel_y.repeat(3,1,1,1), padding=1, groups=3)
        
        pred_edges = torch.sqrt(pred_edges_x ** 2 + pred_edges_y ** 2 + 1e-6)
        target_edges = torch.sqrt(target_edges_x ** 2 + target_edges_y ** 2 + 1e-6)
        
        return F.l1_loss(pred_edges, target_edges)
def evaluate_and_visualize(model, val_loader, device):
    """Complete evaluation function that generates all required visualizations and metrics"""
    print("Generating object-wise metrics and plots...")
    object_metrics, overall_metrics = calculate_and_plot_object_metrics(model, val_loader, device)
    
    print("\nGenerating sample outputs...")
    visualize_sample_outputs(model, val_loader, device)
    
    # Save detailed metrics to file
    with open('detailed_metrics.txt', 'w') as f:
        f.write("Object-wise Metrics:\n")
        f.write("-" * 50 + "\n")
        for obj in sorted(object_metrics.keys()):
            f.write(f"\n{obj}:\n")
            for metric, value in object_metrics[obj].items():
                f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\n" + "-" * 50 + "\n")
        f.write("\nOverall Averages:\n")
        for metric, value in overall_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    return object_metrics, overall_metrics
def evaluate_defect_regions(pred, target, mask):
    """
    Calculate metrics specifically for defect regions with improved handling
    """
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()

    # Create binary mask and ensure proper shapes
    defect_regions = (mask > 0.5).astype(np.float32)
    
    # Ensure proper scaling
    pred = (pred * 255).astype(np.uint8)
    target = (target * 255).astype(np.uint8)
    
    # Calculate metrics only in defect regions
    masked_pred = pred * defect_regions
    masked_target = target * defect_regions
    
    defect_psnr = calculate_psnr(masked_target, masked_pred)
    defect_ssim = calculate_ssim(masked_target, masked_pred)
    
    return defect_psnr, defect_ssim

def calculate_psnr(original, denoised):
    """
    Calculate PSNR with improved handling of edge cases
    """
    mse = np.mean((original - denoised) ** 2)
    if mse < 1e-10:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

class EarlyStopping:
    """
    Early stopping handler with improved state management
    """
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state = None
    
    def __call__(self, val_loss, model_state=None):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state = model_state
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_state = model_state
            self.counter = 0
        return self.early_stop

def train_model(model, train_loader, val_loader, num_epochs=150, device='cuda'):
    """
    Training loop with improved NaN handling and stability
    """
    criterion = PriorityFocusedLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=15,        # Increased from 10
        T_mult=2, 
        eta_min=1e-7   # Reduced from 1e-6
    )
    early_stopping = EarlyStopping(
        patience=20,    # Increased from 15
        min_delta=1e-5  # Reduced from 1e-4
    )
    
    # Initialize tracking
    best_metrics = {'defect_ssim': 0, 'epoch': 0}
    history = {'train_loss': [], 'val_metrics': [], 'defect_metrics': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            degraded, clean, mask = [x.to(device) for x in batch]
            
            # Ensure proper mask dimensions
            if mask.shape[1] == 3:
                mask = mask.mean(dim=1, keepdim=True)
            
            optimizer.zero_grad()
            outputs = model(degraded, mask)
            loss = criterion(outputs, clean, mask)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"NaN loss detected in epoch {epoch}")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_metrics = evaluate_model(model, val_loader, device)
        
        # Handle NaN in metrics
        defect_ssim = val_metrics.get('defect_ssim', 0)
        if np.isnan(defect_ssim):
            defect_ssim = 0
        
        # Update learning rate with safe value
        scheduler.step(epoch)  # Use epoch number instead of metric
        
        # Save best model
        if defect_ssim > best_metrics['defect_ssim']:
            best_metrics = {
                'defect_ssim': defect_ssim,
                'epoch': epoch,
                'state_dict': model.state_dict()
            }
            torch.save(best_metrics, 'best_model.pth')
        
        # Early stopping check
        if early_stopping(1 - defect_ssim):
            print(f"Early stopping triggered at epoch {epoch}")
            break
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_metrics'].append(val_metrics)
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Validation Metrics:")
        for metric_name, value in val_metrics.items():
            print(f"{metric_name}: {value:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    return model, history, best_metrics
# Complementary classes and functions for the denoising implementation

class DenoisingDataset(Dataset):
    """
    Dataset class with improved image handling and error checking
    """
    def __init__(self, root_dir, transform=None, mode='Train'):
        self.root_dir = root_dir
        self.transform = transform
        self.degraded_images = []
        self.clean_images = []
        self.defect_masks = []
        
        self._load_dataset(mode)
        
    def _load_dataset(self, mode):
        """Load dataset with improved error handling"""
        for obj_class in sorted(os.listdir(self.root_dir)):
            obj_path = os.path.join(self.root_dir, obj_class)
            if not os.path.isdir(obj_path):
                continue
                
            degraded_path = os.path.join(obj_path, mode, 'Degraded_image')
            clean_path = os.path.join(obj_path, mode, 'GT_clean_image')
            mask_path = os.path.join(obj_path, mode, 'Defect_mask')
            
            if not all(os.path.exists(p) for p in [degraded_path, clean_path, mask_path]):
                print(f"Warning: Missing directories for {obj_class}")
                continue
                
            self._load_images(degraded_path, clean_path, mask_path)
            
        print(f"Loaded {len(self.degraded_images)} images for {mode}")
    
    def _load_images(self, degraded_path, clean_path, mask_path):
        """Load individual images with validation"""
        for root, _, files in os.walk(degraded_path):
            for img_name in sorted(f for f in files if f.endswith('.png')):
                degraded_img = os.path.join(root, img_name)
                relative_path = os.path.relpath(degraded_img, degraded_path)
                clean_img = os.path.join(clean_path, relative_path)
                mask_img = os.path.join(mask_path, os.path.splitext(relative_path)[0] + '_mask.png')
                
                if not all(os.path.exists(p) for p in [degraded_img, clean_img, mask_img]):
                    continue
                    
                self.degraded_images.append(degraded_img)
                self.clean_images.append(clean_img)
                self.defect_masks.append(mask_img)
    
    def __len__(self):
        return len(self.degraded_images)
    
    def __getitem__(self, idx):
        try:
            degraded_image = Image.open(self.degraded_images[idx]).convert('RGB')
            clean_image = Image.open(self.clean_images[idx]).convert('RGB')
            defect_mask = Image.open(self.defect_masks[idx]).convert('L')
            
            if self.transform:
                degraded_image = self.transform(degraded_image)
                clean_image = self.transform(clean_image)
                defect_mask = self.transform(defect_mask)
            
            return degraded_image, clean_image, defect_mask
            
        except Exception as e:
            print(f"Error loading image at index {idx}: {str(e)}")
            # Return a valid but empty tensor in case of error
            return torch.zeros(3, 256, 256), torch.zeros(3, 256, 256), torch.zeros(1, 256, 256)

class ImprovedUNet(nn.Module):
    """
    Enhanced UNet with attention mechanisms and skip connections
    """
    def __init__(self, skip_connections=None):
        super().__init__()
        self.skip_connections = [True, True, True, True] if skip_connections is None else skip_connections
        
        # Encoder
        self.enc1 = self._conv_block(4, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Attention modules
        self.att1 = ChannelAttention(64)
        self.att2 = ChannelAttention(128)
        self.att3 = ChannelAttention(256)
        self.att4 = ChannelAttention(512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(1024 if self.skip_connections[3] else 512, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512 if self.skip_connections[2] else 256, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256 if self.skip_connections[1] else 128, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128 if self.skip_connections[0] else 64, 64)
        
        self.final = nn.Conv2d(64, 3, kernel_size=1)
        
        # Defect attention
        self.mask_attention = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
    
    def _conv_block(self, in_ch, out_ch):
        """Improved convolution block with regularization"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
    
    def forward(self, x, mask):
        # Generate attention weights
        mask_weights = self.mask_attention(mask)
        
        # Concatenate input and mask
        x = torch.cat([x, mask], dim=1)
        
        # Encoder path
        e1 = self.att1(self.enc1(x))
        e2 = self.att2(self.enc2(self.pool(e1)))
        e3 = self.att3(self.enc3(self.pool(e2)))
        e4 = self.att4(self.enc4(self.pool(e3)))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder path with skip connections
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1) if self.skip_connections[3] else d4)
        
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1) if self.skip_connections[2] else d3)
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1) if self.skip_connections[1] else d2)
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1) if self.skip_connections[0] else d1)
        
        # Final output with mask attention
        out = self.final(d1)
        return out * mask_weights + out * (1 - mask_weights)

def evaluate_model(model, dataloader, device):
    """
    Evaluate model with improved error handling
    """
    model.eval()
    metrics = {
        'psnr': [],
        'ssim': [],
        'defect_psnr': [],
        'defect_ssim': []
    }
    
    with torch.no_grad():
        for degraded, clean, mask in dataloader:
            degraded = degraded.to(device)
            clean = clean.to(device)
            mask = mask.to(device)
            
            if mask.shape[1] == 3:
                mask = mask.mean(dim=1, keepdim=True)
            
            try:
                outputs = model(degraded, mask)
                
                # Process each image in batch
                for i in range(outputs.shape[0]):
                    output_np = outputs[i].cpu().numpy().transpose(1, 2, 0)
                    clean_np = clean[i].cpu().numpy().transpose(1, 2, 0)
                    mask_np = mask[i].cpu().numpy().transpose(1, 2, 0)
                    
                    # Ensure proper value range
                    output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
                    clean_np = np.clip(clean_np * 255, 0, 255).astype(np.uint8)
                    
                    # Calculate metrics with error handling
                    try:
                        psnr = calculate_psnr(clean_np, output_np)
                        ssim = calculate_ssim(clean_np, output_np)
                        
                        if not (np.isnan(psnr) or np.isnan(ssim)):
                            metrics['psnr'].append(psnr)
                            metrics['ssim'].append(ssim)
                        
                        # Calculate defect metrics
                        defect_psnr, defect_ssim = evaluate_defect_regions(
                            output_np/255.0, clean_np/255.0, mask_np
                        )
                        
                        if not (np.isnan(defect_psnr) or np.isnan(defect_ssim)):
                            metrics['defect_psnr'].append(defect_psnr)
                            metrics['defect_ssim'].append(defect_ssim)
                            
                    except Exception as e:
                        print(f"Error calculating metrics: {str(e)}")
                        continue
                        
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                continue
    
    # Calculate averages with safety checks
    return {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in metrics.items()}

def visualize_sample_outputs(model, val_loader, device, num_samples=5):
    """Visualize sample outputs with metrics"""
    model.eval()
    fig = plt.figure(figsize=(20, 4*num_samples))
    
    with torch.no_grad():
        for idx, (degraded, clean, mask) in enumerate(val_loader):
            if idx >= num_samples:
                break
                
            degraded = degraded.to(device)
            clean = clean.to(device)
            mask = mask.to(device)
            
            if mask.shape[1] == 3:
                mask = mask.mean(dim=1, keepdim=True)
            
            outputs = model(degraded, mask)
            
            # Get the object class name
            obj_path = val_loader.dataset.degraded_images[idx * val_loader.batch_size]
            object_class = obj_path.split('/')[-4]
            
            # Process first image in batch
            degraded_np = degraded[0].cpu().numpy().transpose(1, 2, 0)
            clean_np = clean[0].cpu().numpy().transpose(1, 2, 0)
            output_np = outputs[0].cpu().numpy().transpose(1, 2, 0)
            mask_np = mask[0].cpu().numpy().transpose(1, 2, 0)
            
            # Calculate metrics for this sample
            psnr = calculate_psnr(clean_np * 255, output_np * 255)
            ssim = calculate_ssim(clean_np * 255, output_np * 255)
            defect_psnr, defect_ssim = evaluate_defect_regions(output_np, clean_np, mask_np)
            
            # Create subplot for this sample
            plt.subplot(num_samples, 4, idx*4 + 1)
            plt.imshow(degraded_np)
            plt.title(f'{object_class}\nDegraded')
            plt.axis('off')
            
            plt.subplot(num_samples, 4, idx*4 + 2)
            plt.imshow(clean_np)
            plt.imshow(mask_np, alpha=0.3, cmap='Reds')
            plt.title('Ground Truth\nwith Defects')
            plt.axis('off')
            
            plt.subplot(num_samples, 4, idx*4 + 3)
            plt.imshow(output_np)
            plt.imshow(mask_np, alpha=0.3, cmap='Reds')
            plt.title('Restored\nwith Defects')
            plt.axis('off')
            
            plt.subplot(num_samples, 4, idx*4 + 4)
            metrics_text = f'PSNR: {psnr:.2f}dB\nSSIM: {ssim:.3f}\nDefect PSNR: {defect_psnr:.2f}dB\nDefect SSIM: {defect_ssim:.3f}'
            plt.text(0.1, 0.5, metrics_text, fontsize=10, transform=plt.gca().transAxes)
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_outputs.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_metrics(history, best_epoch, output_dir):
    """
    Save training metrics and generate visualization plots with improved handling
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics to file
    with open(os.path.join(output_dir, 'metrics.txt'), 'a') as f:
        f.write(f"\nBest Epoch: {best_epoch}\n")
        f.write("\nFinal Metrics:\n")
        
        # Handle training loss
        if 'train_loss' in history:
            final_loss = history['train_loss'][-1] if isinstance(history['train_loss'], list) else history['train_loss']
            f.write(f"Training Loss: {final_loss:.4f}\n")
        
        # Handle validation metrics
        if 'val_metrics' in history and history['val_metrics']:
            final_metrics = history['val_metrics'][-1] if isinstance(history['val_metrics'], list) else history['val_metrics']
            for metric_name, value in final_metrics.items():
                if isinstance(value, (int, float)):
                    f.write(f"{metric_name}: {value:.4f}\n")
                else:
                    f.write(f"{metric_name}: {value}\n")
    
    # Plot training curves
    plt.figure(figsize=(15, 10))
    
    # Plot training loss
    if 'train_loss' in history and isinstance(history['train_loss'], list):
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    
    # Plot validation metrics
    if 'val_metrics' in history and history['val_metrics']:
        # Extract metrics over time
        epochs = range(len(history['val_metrics']))
        metrics_dict = {}
        
        # Organize metrics by type
        for epoch_metrics in history['val_metrics']:
            for metric_name, value in epoch_metrics.items():
                if metric_name not in metrics_dict:
                    metrics_dict[metric_name] = []
                metrics_dict[metric_name].append(value)
        
        # Plot PSNR
        if 'psnr' in metrics_dict:
            plt.subplot(2, 2, 2)
            plt.plot(epochs, metrics_dict['psnr'], label='Overall PSNR')
            if 'defect_psnr' in metrics_dict:
                plt.plot(epochs, metrics_dict['defect_psnr'], label='Defect PSNR')
            plt.title('PSNR Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('PSNR (dB)')
            plt.legend()
        
        # Plot SSIM
        if 'ssim' in metrics_dict:
            plt.subplot(2, 2, 3)
            plt.plot(epochs, metrics_dict['ssim'], label='Overall SSIM')
            if 'defect_ssim' in metrics_dict:
                plt.plot(epochs, metrics_dict['defect_ssim'], label='Defect SSIM')
            plt.title('SSIM Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('SSIM')
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()
def calculate_and_plot_object_metrics(model, val_loader, device):
    """Calculate and visualize metrics for each object class"""
    model.eval()
    metrics_dict = {}
    
    with torch.no_grad():
        for batch_idx, (degraded, clean, mask) in enumerate(val_loader):
            # Extract object class from path
            obj_path = val_loader.dataset.degraded_images[batch_idx * val_loader.batch_size]
            object_class = obj_path.split('/')[-4]  # Adjust split index based on your path structure
            
            if object_class not in metrics_dict:
                metrics_dict[object_class] = {
                    'psnr': [], 'ssim': [],
                    'defect_psnr': [], 'defect_ssim': []
                }
            
            # Process batch
            degraded, clean, mask = degraded.to(device), clean.to(device), mask.to(device)
            if mask.shape[1] == 3:
                mask = mask.mean(dim=1, keepdim=True)
            
            outputs = model(degraded, mask)
            
            # Calculate metrics
            for i in range(outputs.shape[0]):
                if (batch_idx * val_loader.batch_size + i) >= len(val_loader.dataset):
                    break
                    
                output_np = outputs[i].cpu().numpy().transpose(1, 2, 0)
                clean_np = clean[i].cpu().numpy().transpose(1, 2, 0)
                mask_np = mask[i].cpu().numpy().transpose(1, 2, 0)
                
                # Scale to 0-255 range
                output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
                clean_np = np.clip(clean_np * 255, 0, 255).astype(np.uint8)
                
                # Calculate metrics
                psnr = calculate_psnr(clean_np, output_np)
                ssim = calculate_ssim(clean_np, output_np)
                defect_psnr, defect_ssim = evaluate_defect_regions(
                    output_np/255.0, clean_np/255.0, mask_np
                )
                
                metrics_dict[object_class]['psnr'].append(psnr)
                metrics_dict[object_class]['ssim'].append(ssim)
                metrics_dict[object_class]['defect_psnr'].append(defect_psnr)
                metrics_dict[object_class]['defect_ssim'].append(defect_ssim)
    
    # Calculate averages
    avg_metrics = {
        obj: {metric: np.mean(values) for metric, values in metrics.items()}
        for obj, metrics in metrics_dict.items()
    }
    
    # Create visualization
    objects = sorted(avg_metrics.keys())
    metrics = ['psnr', 'ssim', 'defect_psnr', 'defect_ssim']
    
    plt.figure(figsize=(20, 15))
    for idx, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, idx)
        values = [avg_metrics[obj][metric] for obj in objects]
        
        # Create bar plot
        bars = plt.bar(objects, values)
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Object-wise {metric.upper()}')
        plt.xlabel('Object Class')
        plt.ylabel(metric.upper())
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}',
                    ha='center', va='bottom')
        
        # Add average line
        avg = np.mean(values)
        plt.axhline(y=avg, color='r', linestyle='--', label=f'Average: {avg:.2f}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('object_wise_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print overall averages
    print("\nOverall Validation Metrics:")
    overall_averages = {}
    for metric in metrics:
        avg = np.mean([m[metric] for m in avg_metrics.values()])
        overall_averages[metric] = avg
        print(f"Average {metric.upper()}: {avg:.4f}")
    
    return avg_metrics, overall_averages
def main():
    """
    Main execution function with improved error handling
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'denoising_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Data loading
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    try:
        train_dataset = DenoisingDataset('Denoising_Dataset_train_val', transform, 'Train')
        val_dataset = DenoisingDataset('Denoising_Dataset_train_val', transform, 'Val')
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
        
        # Model initialization
        model = ImprovedUNet(skip_connections=[False, True, True, True]).to(device)
        
        # Training
        model, history, best_metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=150,
            device=device
        )
        
        # Save results
        if history and best_metrics:
            save_metrics(history, best_metrics['epoch'], output_dir)
            plot_training_curves(history)
            
            print(f"Training completed. Best model saved at epoch {best_metrics['epoch']}")
            print(f"Best defect SSIM: {best_metrics['defect_ssim']:.4f}")
        else:
            print("Warning: Training completed but no metrics were recorded")
            
    except Exception as e:
        print(f"Error in training process: {str(e)}")
        raise

if __name__ == '__main__':
    main()