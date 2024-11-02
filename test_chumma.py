import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

# Step 1: Define the model architecture
class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 16, channels, 1)
        )
        
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attention = torch.sigmoid(avg_out + max_out)
        return x * attention

class ImprovedUNet(nn.Module):
    def __init__(self, skip_connections=None):
        super().__init__()
        self.skip_connections = [True, True, True, True] if skip_connections is None else skip_connections
        
        # Encoder
        self.enc1 = self.conv_block(4, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Channel attention
        self.att1 = ChannelAttention(64)
        self.att2 = ChannelAttention(128)
        self.att3 = ChannelAttention(256)
        self.att4 = ChannelAttention(512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        self.pool = nn.MaxPool2d(2)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024 if self.skip_connections[3] else 512, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512 if self.skip_connections[2] else 256, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256 if self.skip_connections[1] else 128, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128 if self.skip_connections[0] else 64, 64)
        
        self.final = nn.Conv2d(64, 3, 1)
        self.mask_attention = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
    
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, mask):
        # Attention weights
        mask_weights = self.mask_attention(mask)
        x = torch.cat([x, mask], dim=1)
        
        # Encoder
        e1 = self.att1(self.enc1(x))
        e2 = self.att2(self.enc2(self.pool(e1)))
        e3 = self.att3(self.enc3(self.pool(e2)))
        e4 = self.att4(self.enc4(self.pool(e3)))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], 1) if self.skip_connections[3] else d4)
        
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], 1) if self.skip_connections[2] else d3)
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], 1) if self.skip_connections[1] else d2)
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], 1) if self.skip_connections[0] else d1)
        
        out = self.final(d1)
        return out * mask_weights + out * (1 - mask_weights)

# Step 2: Define the Dataset class
class SimpleTestDataset(torch.utils.data.Dataset):
    """Simple dataset for direct test images"""
    def __init__(self, test_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        
        # Get all image files in the test directory
        for file in os.listdir(test_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(test_dir, file))
        
        self.image_paths.sort()  # Sort for consistent ordering
        print(f"\nFound {len(self.image_paths)} test images:")
        for path in self.image_paths:
            print(f"- {os.path.basename(path)}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        name = os.path.basename(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, name

def test_single_images():
    """Test function for direct images"""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'test_results'
    test_dir = 'test'  # Directory containing test images
    
    print(f"\nStarting test process...")
    print(f"Using device: {device}")
    print(f"Test directory: {test_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Verify test directory
    if not os.path.exists(test_dir):
        print(f"Error: Test directory '{test_dir}' not found!")
        return
    
    # Load model
    print("\nLoading model...")
    try:
        model = ImprovedUNet(skip_connections=[False, True, True, True]).to(device)
        checkpoint = torch.load('best_model.pth', map_location=device)
        
        # Try different state dict keys
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Prepare transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Create dataset and dataloader
    try:
        test_dataset = SimpleTestDataset(test_dir, transform)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False
        )
    except Exception as e:
        print(f"Error preparing dataset: {str(e)}")
        return
    
    # Test loop
    print("\nProcessing images...")
    with torch.no_grad():
        for image, name in tqdm(test_loader):
            # Move to device
            image = image.to(device)
            
            # Create dummy mask (all zeros)
            dummy_mask = torch.zeros((1, 1, 256, 256), device=device)
            
            # Get prediction
            restored = model(image, dummy_mask)
            
            # Convert to numpy and prepare for saving
            restored_np = restored.cpu().numpy()[0].transpose(1, 2, 0)
            restored_np = np.clip(restored_np * 255, 0, 255).astype(np.uint8)
            
            # Save restored image
            output_path = os.path.join(output_dir, f"restored_{name[0]}")
            cv2.imwrite(output_path, cv2.cvtColor(restored_np, cv2.COLOR_RGB2BGR))
            
            # Create and save comparison
            input_np = image.cpu().numpy()[0].transpose(1, 2, 0)
            input_np = np.clip(input_np * 255, 0, 255).astype(np.uint8)
            
            plt.figure(figsize=(10, 5))
            
            plt.subplot(121)
            plt.imshow(input_np)
            plt.title('Input Image')
            plt.axis('off')
            
            plt.subplot(122)
            plt.imshow(restored_np)
            plt.title('Restored Image')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"comparison_{name[0]}"))
            plt.close()
    
    print("\nTesting completed!")
    print(f"Results saved in '{output_dir}'")
    print("\nRestored images saved as:")
    for name in os.listdir(output_dir):
        print(f"- {name}")

if __name__ == '__main__':
    test_single_images()