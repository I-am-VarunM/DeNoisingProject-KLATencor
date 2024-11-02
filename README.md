# Image Restoration with Defect Preservation

This project implements a deep learning solution for image restoration while preserving critical defects. The implementation uses an improved UNet architecture with attention mechanisms.

## Project Structure
```
.
├── model.py              # Training implementation file
├── validate_metrics.py   # Validation and metrics calculation
├── best_model.pth        # Saved model weights (after training)
├── requirements.txt      # Required packages
└── README.md            # This file
```

## Requirements

Install the required packages:
```bash
pip install torch torchvision
pip install Pillow opencv-python numpy
pip install matplotlib tqdm
```

## Dataset Structure
Your dataset should be organized as follows:
```
Denoising_Dataset_train_val/
├── bottle/
│   ├── Train/
│   │   ├── Degraded_image/
│   │   ├── GT_clean_image/
│   │   └── Defect_mask/
│   └── Val/
│       ├── Degraded_image/
│       ├── GT_clean_image/
│       └── Defect_mask/
├── cable/
└── ...
```

## Training

To train the model, run:
```bash
python model.py
```

### Adjustable Parameters in model.py

1. Model Architecture Parameters (ImprovedUNet class):
```python
# In ImprovedUNet.__init__
self.enc1 = self._conv_block(4, 64)      # Initial channels
dropout_rate = 0.15                       # Dropout rate
```

2. Training Parameters:
```python
# Batch size and epochs
batch_size = 8
num_epochs = 150

# Optimizer parameters
lr = 1e-4
weight_decay = 2e-4

# Learning rate scheduler
scheduler_patience = 10
scheduler_factor = 0.5
min_lr = 1e-6

# Early stopping
early_stopping_patience = 25
min_delta = 1e-4
```

3. Loss Function Weights (PriorityFocusedLoss class):
```python
# In forward method
defect_loss_weight = 0.25
ssim_loss_weight = 0.30
perceptual_loss_weight = 0.20
edge_loss_weight = 0.15
basic_loss_weight = 0.10
```

## Validation

To validate the model and generate metrics, run:
```bash
python validate_metrics.py
```

### Adjustable Parameters in validate_metrics.py

1. Visualization Parameters:
```python
# Number of sample outputs to generate
num_samples = 5

# Figure size for plots
figsize = (20, 15)
```

2. Metric Calculation Parameters:
```python
# SSIM calculation
window_size = 11    # SSIM window size
```

## Output Files

After running validate_metrics.py, you'll get:
1. `object_wise_metrics.png`: Bar charts showing PSNR/SSIM for each object
2. `sample_outputs.png`: Sample output visualizations
3. `detailed_metrics.txt`: Detailed metrics for each object

## Parameter Tuning Guide

1. To improve overall image quality (PSNR/SSIM):
   - Decrease `defect_loss_weight`
   - Increase `ssim_loss_weight`
   - Increase `basic_loss_weight`

2. To improve defect preservation:
   - Increase `defect_loss_weight`
   - Increase dropout rate
   - Increase weight decay

3. To balance training/validation performance:
   - Adjust dropout rate
   - Modify weight decay
   - Tune learning rate and scheduler parameters

4. To modify training time/convergence:
   - Adjust `num_epochs`
   - Modify scheduler parameters
   - Change early stopping patience

## Contributing

Feel free to submit issues and enhancement requests!

## References

- UNet Architecture: [Original Paper](https://arxiv.org/abs/1505.04597)
- MVTec Dataset: [Dataset Link](https://www.mvtec.com/company/research/datasets/mvtec-ad)
