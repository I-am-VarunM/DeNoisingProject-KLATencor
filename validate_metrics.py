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
def calculate_ssim(img1, img2, window_size=11):
    """Calculate SSIM between two images"""
    # Convert to grayscale if RGB
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1.astype('float32'), cv2.COLOR_RGB2GRAY)
    if img2.ndim == 3:
        img2 = cv2.cvtColor(img2.astype('float32'), cv2.COLOR_RGB2GRAY)
    
    # Constants for stability
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
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
    
    # SSIM calculation
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(np.mean(ssim_map))

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def evaluate_defect_regions(pred, target, mask):
    """Calculate metrics specifically for defect regions"""
    # Create binary mask
    mask_binary = (mask > 0.5).astype(np.float32)
    
    # Apply mask to predictions and targets
    masked_pred = pred * mask_binary
    masked_target = target * mask_binary
    
    # Calculate metrics
    defect_psnr = calculate_psnr(masked_target * 255, masked_pred * 255)
    defect_ssim = calculate_ssim(masked_target * 255, masked_pred * 255)
    
    return defect_psnr, defect_ssim

def plot_metrics(avg_metrics, object_order=None):
    """Create bar plots for all metrics with specified object order"""
    if object_order is None:
        object_order = sorted(avg_metrics.keys())
    
    metrics = ['psnr', 'ssim', 'defect_psnr', 'defect_ssim']
    metric_names = ['PSNR (dB)', 'SSIM', 'Defect PSNR (dB)', 'Defect SSIM']
    
    plt.figure(figsize=(20, 15))
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names), 1):
        plt.subplot(2, 2, idx)
        
        values = [avg_metrics[obj][metric] for obj in object_order]
        
        # Create bars
        bars = plt.bar(range(len(values)), values)
        plt.xticks(range(len(values)), object_order, rotation=45, ha='right')
        plt.title(f'Object-wise {metric_name}')
        plt.xlabel('Object Class')
        plt.ylabel(metric_name)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
        
        # Add average line
        avg = np.mean(values)
        plt.axhline(y=avg, color='r', linestyle='--', label=f'Average: {avg:.2f}')
        plt.legend()
        
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('object_wise_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_sample_outputs(model, val_loader, device, num_samples=5):
    """Generate and save sample outputs with metrics"""
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
            
            # Get object class
            obj_path = val_loader.dataset.degraded_images[idx * val_loader.batch_size]
            object_class = obj_path.split('/')[-4]
            
            # Process first image in batch
            degraded_np = degraded[0].cpu().numpy().transpose(1, 2, 0)
            clean_np = clean[0].cpu().numpy().transpose(1, 2, 0)
            output_np = outputs[0].cpu().numpy().transpose(1, 2, 0)
            mask_np = mask[0].cpu().numpy().transpose(1, 2, 0)
            
            # Calculate metrics
            psnr = calculate_psnr(clean_np * 255, output_np * 255)
            ssim = calculate_ssim(clean_np * 255, output_np * 255)
            defect_psnr, defect_ssim = evaluate_defect_regions(output_np, clean_np, mask_np)
            
            # Create subplot
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
            metrics_text = (f'PSNR: {psnr:.2f}dB\n'
                          f'SSIM: {ssim:.3f}\n'
                          f'Defect PSNR: {defect_psnr:.2f}dB\n'
                          f'Defect SSIM: {defect_ssim:.3f}')
            plt.text(0.1, 0.5, metrics_text, fontsize=10)
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_outputs.png', dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_and_visualize(model, val_loader, device):
    """Main evaluation function with fixed path processing"""
    model.eval()
    metrics_dict = {}
    
    print("Evaluating model...")
    print(f"Total batches in val_loader: {len(val_loader)}")
    
    # First, let's check what objects we actually have in the dataset
    dataset_objects = set()
    for path in val_loader.dataset.degraded_images:
        # Extract object class using correct index
        parts = path.split('/')
        object_class = parts[1].lower()  # Changed from -4 to 1
        dataset_objects.add(object_class)
    
    print(f"\nFound {len(dataset_objects)} objects in dataset: {sorted(dataset_objects)}")
    
    with torch.no_grad():
        for batch_idx, (degraded, clean, mask) in enumerate(tqdm(val_loader)):
            try:
                # Get object class
                obj_path = val_loader.dataset.degraded_images[batch_idx * val_loader.batch_size]
                object_class = obj_path.split('/')[1].lower()  # Changed from -4 to 1
                
                if batch_idx == 0:
                    print(f"Processing first batch, object class: {object_class}")
                
                if object_class not in metrics_dict:
                    metrics_dict[object_class] = {
                        'psnr': [], 'ssim': [],
                        'defect_psnr': [], 'defect_ssim': []
                    }
                
                # Process batch
                degraded = degraded.to(device)
                clean = clean.to(device)
                mask = mask.to(device)
                
                if mask.shape[1] == 3:
                    mask = mask.mean(dim=1, keepdim=True)
                
                outputs = model(degraded, mask)
                
                # Calculate metrics for each image
                for i in range(outputs.shape[0]):
                    if (batch_idx * val_loader.batch_size + i) >= len(val_loader.dataset):
                        break
                    
                    # Get correct object class for this specific image
                    img_path = val_loader.dataset.degraded_images[batch_idx * val_loader.batch_size + i]
                    img_object_class = img_path.split('/')[1].lower()
                    
                    output_np = outputs[i].cpu().numpy().transpose(1, 2, 0)
                    clean_np = clean[i].cpu().numpy().transpose(1, 2, 0)
                    mask_np = mask[i].cpu().numpy().transpose(1, 2, 0)
                    
                    # Calculate metrics
                    psnr = calculate_psnr(clean_np * 255, output_np * 255)
                    ssim = calculate_ssim(clean_np * 255, output_np * 255)
                    defect_psnr, defect_ssim = evaluate_defect_regions(output_np, clean_np, mask_np)
                    
                    # Store metrics under correct object class
                    if img_object_class not in metrics_dict:
                        metrics_dict[img_object_class] = {
                            'psnr': [], 'ssim': [],
                            'defect_psnr': [], 'defect_ssim': []
                        }
                    
                    metrics_dict[img_object_class]['psnr'].append(psnr)
                    metrics_dict[img_object_class]['ssim'].append(ssim)
                    metrics_dict[img_object_class]['defect_psnr'].append(defect_psnr)
                    metrics_dict[img_object_class]['defect_ssim'].append(defect_ssim)
                    
                    if batch_idx == 0 and i == 0:
                        print(f"First image metrics - PSNR: {psnr:.2f}, SSIM: {ssim:.3f}")
                
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {str(e)}")
                continue
    
    print("\nCollected metrics for objects:", list(metrics_dict.keys()))
    
    # Calculate averages
    avg_metrics = {}
    for obj in metrics_dict:
        if metrics_dict[obj]['psnr']:  # Check if we have any metrics
            avg_metrics[obj] = {
                metric: np.mean(values) for metric, values in metrics_dict[obj].items()
            }
    
    if not avg_metrics:
        print("ERROR: No metrics were collected! Please check the dataset structure.")
        return None, None
    
    print("\nCalculated averages for objects:", list(avg_metrics.keys()))
    
    # Plot only if we have data
    if avg_metrics:
        print("\nGenerating visualizations...")
        object_order = sorted(avg_metrics.keys())  # Ensure alphabetical order
        plot_metrics(avg_metrics, object_order)
        plot_sample_outputs(model, val_loader, device)
        
        # Save detailed metrics
        with open('detailed_metrics.txt', 'w') as f:
            f.write("Object-wise Metrics:\n")
            f.write("-" * 50 + "\n")
            
            for obj in object_order:
                f.write(f"\n{obj}:\n")
                for metric in ['psnr', 'ssim', 'defect_psnr', 'defect_ssim']:
                    f.write(f"{metric}: {avg_metrics[obj][metric]:.4f}\n")
            
            # Overall averages
            f.write("\n" + "-" * 50 + "\n")
            f.write("\nOverall Averages:\n")
            overall_metrics = {}
            for metric in ['psnr', 'ssim', 'defect_psnr', 'defect_ssim']:
                avg = np.mean([m[metric] for m in avg_metrics.values()])
                overall_metrics[metric] = avg
                f.write(f"{metric}: {avg:.4f}\n")
    
        return avg_metrics, overall_metrics
    else:
        print("No metrics to visualize!")
        return None, None


def main():
    """Main execution function with improved error handling"""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data loading
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    print("Loading validation dataset...")
    try:
        val_dataset = DenoisingDataset('Denoising_Dataset_train_val', transform, 'Val')
        print(f"Dataset loaded with {len(val_dataset)} images")
        
        # Print first few paths to verify structure
        print("\nFirst few dataset paths:")
        for i in range(min(3, len(val_dataset.degraded_images))):
            print(val_dataset.degraded_images[i])
        
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
        print(f"DataLoader created with {len(val_loader)} batches")
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return
    
    # Load model
    print("\nLoading model...")
    model = ImprovedUNet(skip_connections=[False, True, True, True]).to(device)
    
    try:
        saved_model = torch.load('best_model.pth')
        model.load_state_dict(saved_model['state_dict'], strict=False)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Run evaluation
    print("\nStarting evaluation...")
    avg_metrics, overall_metrics = evaluate_and_visualize(model, val_loader, device)
    
    if avg_metrics and overall_metrics:
        print("\nEvaluation completed successfully!")
    else:
        print("\nEvaluation failed to generate metrics!")

if __name__ == '__main__':
    main()