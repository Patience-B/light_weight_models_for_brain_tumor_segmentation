import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import mlflow
import mlflow.pytorch
import time
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

# -------------------- MEMORY MANAGEMENT --------------------
def clear_memory():
    """Clear memory to prevent OOM errors"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -------------------- LIGHTWEIGHT 3D UNET MODEL --------------------
class ConvBlock3D(nn.Module):
    """Lightweight 3D convolution block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class LightweightUNet3D(nn.Module):
    """Very lightweight 3D U-Net for low-memory systems"""
    def __init__(self, in_channels=4, out_channels=4, base_channels=8):
        super().__init__()
        
        # Encoder (downsampling path)
        self.enc1 = ConvBlock3D(in_channels, base_channels)
        self.enc2 = ConvBlock3D(base_channels, base_channels*2)
        self.enc3 = ConvBlock3D(base_channels*2, base_channels*4)
        
        # Bottleneck
        self.bottleneck = ConvBlock3D(base_channels*4, base_channels*4)
        
        # Decoder (upsampling path)
        self.dec3 = ConvBlock3D(base_channels*8, base_channels*2)  # *8 due to skip connection
        self.dec2 = ConvBlock3D(base_channels*4, base_channels)    # *4 due to skip connection
        self.dec1 = ConvBlock3D(base_channels*2, base_channels)    # *2 due to skip connection
        
        # Final output layer
        self.final = nn.Conv3d(base_channels, out_channels, kernel_size=1)
        
        # Pooling
        self.pool = nn.MaxPool3d(2)
        
    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3))
        
        # Decoder path with skip connections
        d3 = F.interpolate(b, size=e3.shape[2:], mode='trilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = F.interpolate(d3, size=e2.shape[2:], mode='trilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = F.interpolate(d2, size=e1.shape[2:], mode='trilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        return self.final(d1)

# -------------------- MEMORY-EFFICIENT DATASET --------------------
class LightweightGliomaDataset(Dataset):
    """Memory-efficient dataset with patch extraction"""
    def __init__(self, img_dir, img_list, mask_dir, mask_list, patch_size=(64, 64, 64), 
                 max_patches_per_volume=4):
        self.img_paths = [os.path.join(img_dir, f) for f in sorted(img_list)]
        self.mask_paths = [os.path.join(mask_dir, f) for f in sorted(mask_list)]
        self.patch_size = patch_size
        self.max_patches = max_patches_per_volume
        
    def __len__(self):
        return len(self.img_paths) * self.max_patches
    
    def extract_patch(self, volume, patch_size, patch_idx):
        """Extract a specific patch from volume"""
        c, d, h, w = volume.shape
        pd, ph, pw = patch_size
        
        # Calculate patch positions
        patches_per_dim = 2  # 2x2x2 = 8 patches max
        z_idx = (patch_idx // 4) % patches_per_dim
        y_idx = (patch_idx // 2) % patches_per_dim
        x_idx = patch_idx % patches_per_dim
        
        # Calculate start positions
        start_d = min(z_idx * (d // patches_per_dim), d - pd)
        start_h = min(y_idx * (h // patches_per_dim), h - ph)
        start_w = min(x_idx * (w // patches_per_dim), w - pw)
        
        # Ensure we don't go out of bounds
        start_d = max(0, start_d)
        start_h = max(0, start_h)
        start_w = max(0, start_w)
        
        return volume[:, start_d:start_d+pd, start_h:start_h+ph, start_w:start_w+pw]
    
    def __getitem__(self, idx):
        volume_idx = idx // self.max_patches
        patch_idx = idx % self.max_patches
        
        # Load volume
        X = np.load(self.img_paths[volume_idx])  # (D, H, W, C)
        Y = np.load(self.mask_paths[volume_idx]) # (D, H, W, C)
        
        # Convert to tensor and move channels first
        X = torch.tensor(X, dtype=torch.float32).permute(3, 0, 1, 2)  # (C, D, H, W)
        Y = torch.tensor(Y, dtype=torch.float32).permute(3, 0, 1, 2)  # (C, D, H, W)
        
        # Extract patches
        X = self.extract_patch(X, self.patch_size, patch_idx)
        Y = self.extract_patch(Y, self.patch_size, patch_idx)
        
        # Normalize input
        X = (X - X.mean()) / (X.std() + 1e-8)
        
        return X, Y

# -------------------- DICE SCORE CALCULATION --------------------
def dice_score(pred, target, num_classes, smooth=1e-6):
    """Calculate dice score for segmentation"""
    pred = torch.argmax(pred, dim=1)  # (B, D, H, W)
    if target.ndim == 5:
        target = torch.argmax(target, dim=1)
    
    dice = 0.0
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        dice += (2. * intersection + smooth) / (union + smooth)
    return dice / num_classes

# -------------------- DATA LOADING --------------------
DATA_PATH = "glioma split data"

def validate_dir(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory not found: {path}")

# Directory paths
train_img_dir = os.path.join(DATA_PATH, "train/images/")
train_mask_dir = os.path.join(DATA_PATH, "train/masks/")
val_img_dir = os.path.join(DATA_PATH, "val/images/")
val_mask_dir = os.path.join(DATA_PATH, "val/masks/")
test_img_dir = os.path.join(DATA_PATH, "test/images/")
test_mask_dir = os.path.join(DATA_PATH, "test/masks/")

# Validate all directories
for dir_path in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, test_img_dir, test_mask_dir]:
    validate_dir(dir_path)

def get_sorted_files(directory):
    return sorted([f for f in os.listdir(directory) if f.endswith('.npy')])

train_img_list = get_sorted_files(train_img_dir)
train_mask_list = get_sorted_files(train_mask_dir)
val_img_list = get_sorted_files(val_img_dir)
val_mask_list = get_sorted_files(val_mask_dir)
test_img_list = get_sorted_files(test_img_dir)
test_mask_list = get_sorted_files(test_mask_dir)

# -------------------- TRAINING SETUP --------------------
print("Setting up training...")

# MLflow experiment setup
mlflow.set_experiment("lightweight-glioma-segmentation")

# Device setup - use CPU for stability on limited hardware
device = torch.device("cpu")
print(f"Using device: {device}")

# Create lightweight model
model = LightweightUNet3D(in_channels=4, out_channels=4, base_channels=8).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# Optimized hyperparameters for low-end hardware
epochs = 30
batch_size = 1  # Very small batch size for memory safety
learning_rate = 0.001
patch_size = (64, 64, 64)  # Smaller patches
max_patches_per_volume = 2  # Fewer patches per volume

# Loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, verbose=True)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3)


# Early stopping parameters
best_dice = 0.0
patience = 10
counter = 0

# Create datasets
print("Creating datasets...")
train_dataset = LightweightGliomaDataset(
    train_img_dir, train_img_list, train_mask_dir, train_mask_list, 
    patch_size=patch_size, max_patches_per_volume=max_patches_per_volume
)
val_dataset = LightweightGliomaDataset(
    val_img_dir, val_img_list, val_mask_dir, val_mask_list, 
    patch_size=patch_size, max_patches_per_volume=max_patches_per_volume
)

# Create data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=0,  # No multiprocessing to save memory
    pin_memory=False
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=0, 
    pin_memory=False
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# -------------------- TRAINING LOOP --------------------
print("Starting training...")

with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_params({
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "model": "LightweightUNet3D",
        "base_channels": 8,
        "patch_size": patch_size,
        "max_patches_per_volume": max_patches_per_volume,
        "total_parameters": total_params,
        "device": str(device)
    })
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        epoch_loss = 0.0
        train_batches = 0
        
        for batch_idx, (X, Y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            X, Y = X.to(device), Y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass (no mixed precision for CPU)
            output = model(X)
            
            # Prepare target
            if Y.ndim == 5:
                Y = torch.argmax(Y, dim=1)
            
            # Calculate loss
            loss = loss_fn(output, Y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            train_batches += 1
            
            # Clear memory periodically
            if batch_idx % 10 == 0:
                clear_memory()
        
        avg_train_loss = epoch_loss / train_batches
        
        # Validation phase
        model.eval()
        val_dice = 0.0
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for X, Y in val_loader:
                X, Y = X.to(device), Y.to(device)
                
                output = model(X)
                
                if Y.ndim == 5:
                    Y = torch.argmax(Y, dim=1)
                
                # Calculate metrics
                loss = loss_fn(output, Y)
                dice = dice_score(output, Y, num_classes=4)
                
                val_loss += loss.item()
                val_dice += dice.item()
                val_batches += 1
                
                # Clear memory
                clear_memory()
        
        avg_val_loss = val_loss / val_batches
        avg_val_dice = val_dice / val_batches
        epoch_time = time.time() - start_time
        
        # Log metrics
        mlflow.log_metrics({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_dice": avg_val_dice,
            "epoch_time_sec": epoch_time
        }, step=epoch+1)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Dice: {avg_val_dice:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        print("-" * 50)
        
        # Learning rate scheduling
        scheduler.step(avg_val_dice)
        
        # Early stopping and model saving
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            counter = 0
            # Save best model
            torch.save(model.state_dict(), "best_lightweight_model.pth")
            mlflow.log_artifact("best_lightweight_model.pth")
            print(f"New best model saved! Dice: {best_dice:.4f}")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Clear memory at end of epoch
        clear_memory()
    
    # Save final model
    final_model_path = "lightweight_glioma_final.pth"
    torch.save(model.state_dict(), final_model_path)
    mlflow.log_artifact(final_model_path)
    
    print(f"\nTraining completed!")
    print(f"Best validation Dice score: {best_dice:.4f}")
    print(f"Final model saved as: {final_model_path}")

# -------------------- OPTIONAL: TEST EVALUATION --------------------
def evaluate_on_test_set():
    """Optional function to evaluate on test set"""
    print("\nEvaluating on test set...")
    
    # Load best model
    model.load_state_dict(torch.load("best_lightweight_model.pth", map_location=device))
    model.eval()
    
    # Create test dataset
    test_dataset = LightweightGliomaDataset(
        test_img_dir, test_img_list, test_mask_dir, test_mask_list, 
        patch_size=patch_size, max_patches_per_volume=max_patches_per_volume
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    test_dice = 0.0
    test_batches = 0
    
    with torch.no_grad():
        for X, Y in tqdm(test_loader, desc="Testing"):
            X, Y = X.to(device), Y.to(device)
            output = model(X)
            
            if Y.ndim == 5:
                Y = torch.argmax(Y, dim=1)
            
            dice = dice_score(output, Y, num_classes=4)
            test_dice += dice.item()
            test_batches += 1
            
            clear_memory()
    
    avg_test_dice = test_dice / test_batches
    print(f"Test Dice Score: {avg_test_dice:.4f}")
    
    return avg_test_dice

# Uncomment the line below to run test evaluation
# evaluate_on_test_set()
