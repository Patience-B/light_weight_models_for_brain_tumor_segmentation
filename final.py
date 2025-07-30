import os
import re
import gc
import json
import time
import shutil
import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm
import mlflow
import mlflow.pytorch

# -------------------- CONFIG --------------------
# Training data (with masks)
TRAIN_IMG_DIR = "glioma/images"
TRAIN_MASK_DIR = "glioma/masks"

# Validation data (no masks, images only)
VAL_IMG_DIR = "glioma_val/images"

# Manifest mapping val npy -> reference NIfTI path (for header/affine & naming)
VAL_MANIFEST_JSON = os.path.join(os.path.dirname(VAL_IMG_DIR), "val_manifest.json")

# Crop used during preprocessing (Z, Y, X). Must match how val .npy were created.
CROP_SLICES = (slice(13, 141), slice(56, 184), slice(56, 184))

# If your task is Meningioma RT (4-digit case id + 1-digit timepoint), set this True.
IS_RT_TASK = False

# Output directory for predictions & zip
PRED_DIR = "submissions_pred_nii"
ZIP_NAME = "submission.zip"

# Training hyperparams
EPOCHS = 50
BATCH_SIZE = 1
LEARNING_RATE = 1e-3
PATCH_SIZE = (64, 64, 64)
MAX_PATCHES_PER_VOLUME = 2
NUM_CLASSES = 4

# Device (CPU for low-memory systems)
DEVICE = torch.device("cpu")


# -------------------- MEMORY MANAGEMENT --------------------
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# -------------------- LIGHTWEIGHT 3D UNET MODEL --------------------
class ConvBlock3D(nn.Module):
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
    def __init__(self, in_channels=4, out_channels=4, base_channels=8):
        super().__init__()
        self.enc1 = ConvBlock3D(in_channels, base_channels)
        self.enc2 = ConvBlock3D(base_channels, base_channels*2)
        self.enc3 = ConvBlock3D(base_channels*2, base_channels*4)
        self.bottleneck = ConvBlock3D(base_channels*4, base_channels*4)
        self.dec3 = ConvBlock3D(base_channels*8, base_channels*2)
        self.dec2 = ConvBlock3D(base_channels*4, base_channels)
        self.dec1 = ConvBlock3D(base_channels*2, base_channels)
        self.final = nn.Conv3d(base_channels, out_channels, kernel_size=1)
        self.pool = nn.MaxPool3d(2)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = F.interpolate(b, size=e3.shape[2:], mode='trilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, size=e2.shape[2:], mode='trilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[2:], mode='trilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.final(d1)


# -------------------- DATASET --------------------
class LightweightGliomaDataset(Dataset):
    """
    Memory-efficient dataset with simple grid patch extraction.
    Expects:
      - X .npy: (D, H, W, C)
      - Y .npy: (D, H, W, C) one-hot mask (training only)
    """
    def __init__(self, img_dir, img_list, mask_dir, mask_list,
                 patch_size=(64, 64, 64), max_patches_per_volume=4):
        self.img_paths = [os.path.join(img_dir, f) for f in sorted(img_list)]
        self.mask_paths = [os.path.join(mask_dir, f) for f in sorted(mask_list)]
        assert len(self.img_paths) == len(self.mask_paths), \
            "Train images and masks count mismatch."
        self.patch_size = patch_size
        self.max_patches = max_patches_per_volume

    def __len__(self):
        return len(self.img_paths) * self.max_patches

    @staticmethod
    def extract_patch(volume, patch_size, patch_idx):
        # volume: (C, D, H, W)
        c, d, h, w = volume.shape
        pd, ph, pw = patch_size
        patches_per_dim = 2  # 2x2x2 grid
        z_idx = (patch_idx // 4) % patches_per_dim
        y_idx = (patch_idx // 2) % patches_per_dim
        x_idx = patch_idx % patches_per_dim
        start_d = min(z_idx * (d // patches_per_dim), d - pd)
        start_h = min(y_idx * (h // patches_per_dim), h - ph)
        start_w = min(x_idx * (w // patches_per_dim), w - pw)
        start_d = max(0, start_d); start_h = max(0, start_h); start_w = max(0, start_w)
        return volume[:, start_d:start_d+pd, start_h:start_h+ph, start_w:start_w+pw]

    def __getitem__(self, idx):
        volume_idx = idx // self.max_patches
        patch_idx = idx % self.max_patches
        X = np.load(self.img_paths[volume_idx])  # (D, H, W, C)
        Y = np.load(self.mask_paths[volume_idx]) # (D, H, W, C) one-hot
        X = torch.tensor(X, dtype=torch.float32).permute(3, 0, 1, 2)  # (C, D, H, W)
        Y = torch.tensor(Y, dtype=torch.float32).permute(3, 0, 1, 2)  # (C, D, H, W)
        X = self.extract_patch(X, self.patch_size, patch_idx)
        Y = self.extract_patch(Y, self.patch_size, patch_idx)
        # normalize per-patch
        X = (X - X.mean()) / (X.std() + 1e-8)
        return X, Y


# -------------------- LOSSES & METRICS --------------------
def dice_loss(pred, target, num_classes, smooth=1e-6):
    """
    pred: (B, C, D, H, W) logits
    target: class indices (B, D, H, W) or one-hot (B, C, D, H, W)
    """
    pred = F.softmax(pred, dim=1)
    if target.ndim == 5:
        target_one_hot = target
    else:
        target_one_hot = torch.zeros_like(pred)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)
    dice_loss_total = 0.0
    for cls in range(num_classes):
        pred_cls = pred[:, cls]
        target_cls = target_one_hot[:, cls]
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_loss_total += 1 - dice
    return dice_loss_total / num_classes

def combined_loss(pred, target, num_classes, alpha=0.5, smooth=1e-6):
    if target.ndim == 5:
        target_ce = torch.argmax(target, dim=1)
    else:
        target_ce = target
    ce_loss = F.cross_entropy(pred, target_ce)
    d_loss = dice_loss(pred, target_ce, num_classes, smooth)
    return alpha * ce_loss + (1 - alpha) * d_loss

def dice_score(pred, target, num_classes, smooth=1e-6):
    """
    Returns mean multiclass Dice across classes.
    pred: logits (B, C, D, H, W)
    target: one-hot (B, C, D, H, W) or class indices (B, D, H, W)
    """
    pred_lbl = torch.argmax(pred, dim=1)  # (B, D, H, W)
    if target.ndim == 5:
        target_lbl = torch.argmax(target, dim=1)
    else:
        target_lbl = target
    dice = 0.0
    for cls in range(num_classes):
        p = (pred_lbl == cls).float()
        t = (target_lbl == cls).float()
        intersection = (p * t).sum()
        union = p.sum() + t.sum()
        dice += (2. * intersection + smooth) / (union + smooth)
    return dice / num_classes


# -------------------- HELPERS --------------------
def validate_dir(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory not found: {path}")

def get_sorted_files(directory):
    return sorted([f for f in os.listdir(directory) if f.endswith('.npy')])

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def parse_case_timepoint_from_path(ref_path, is_rt=False):
    """
    Extract case ID and timepoint from a reference NIfTI filename.
    Expected patterns (examples):
      - BraTS style: .../xxxxx-yyy_*mod*.nii.gz  -> five-digit ID + 3-digit TP
      - RT style:    .../dddd-t_*mod*.nii.gz     -> four-digit ID + 1-digit TP
    Returns '#####-###' or '####-#'
    """
    fname = os.path.basename(ref_path)
    # First, try strict patterns:
    if not is_rt:
        m = re.search(r'(\d{5})-(\d{3})', fname)
        if m:
            return f"{m.group(1)}-{m.group(2)}"
    else:
        m = re.search(r'(\d{4})-(\d)', fname)
        if m:
            return f"{m.group(1)}-{m.group(2)}"
    # Fallback: grab last two numeric groups in name
    nums = re.findall(r'\d+', fname)
    if len(nums) >= 2:
        if not is_rt:
            return f"{nums[-2].zfill(5)}-{nums[-1].zfill(3)}"
        else:
            return f"{nums[-2].zfill(4)}-{nums[-1]}"
    raise ValueError(f"Could not parse case/timepoint from: {fname}")


# -------------------- DATA LOADING (TRAIN) --------------------
print("Setting up training...")
validate_dir(TRAIN_IMG_DIR); validate_dir(TRAIN_MASK_DIR)

train_img_list = get_sorted_files(TRAIN_IMG_DIR)
train_mask_list = get_sorted_files(TRAIN_MASK_DIR)

print(f"Found {len(train_img_list)} training images and {len(train_mask_list)} masks.")

# -------------------- MODEL, OPTIM, SCHED --------------------
mlflow.set_experiment("lightweight-glioma-segmentation")
device = DEVICE
print(f"Using device: {device}")

model = LightweightUNet3D(in_channels=4, out_channels=NUM_CLASSES, base_channels=8).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3)

# Early stopping based on TRAIN dice (no val masks)
best_train_dice = 0.0
patience = 10
counter = 0

# -------------------- DATALOADERS --------------------
print("Creating datasets...")
train_dataset = LightweightGliomaDataset(
    TRAIN_IMG_DIR, train_img_list, TRAIN_MASK_DIR, train_mask_list,
    patch_size=PATCH_SIZE, max_patches_per_volume=MAX_PATCHES_PER_VOLUME
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=False
)

print(f"Training samples (patches): {len(train_dataset)}")

# -------------------- TRAINING LOOP --------------------
print("Starting training...")

with mlflow.start_run():
    mlflow.log_params({
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "model": "LightweightUNet3D",
        "base_channels": 8,
        "patch_size": PATCH_SIZE,
        "max_patches_per_volume": MAX_PATCHES_PER_VOLUME,
        "total_parameters": total_params,
        "device": str(device),
        "loss_function": "combined_ce_dice",
        "val_has_masks": False
    })

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        n_batches = 0

        for batch_idx, (X, Y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = combined_loss(out, Y, num_classes=NUM_CLASSES, alpha=0.5)
            loss.backward()
            optimizer.step()

            # metrics
            with torch.no_grad():
                train_dice_batch = dice_score(out, Y, num_classes=NUM_CLASSES).item()

            epoch_loss += loss.item()
            epoch_dice += train_dice_batch
            n_batches += 1

            if batch_idx % 10 == 0:
                clear_memory()

        avg_train_loss = epoch_loss / max(1, n_batches)
        avg_train_dice = epoch_dice / max(1, n_batches)
        epoch_time = time.time() - start_time

        # Log metrics
        mlflow.log_metrics({
            "train_loss": avg_train_loss,
            "train_dice": avg_train_dice,
            "epoch_time_sec": epoch_time
        }, step=epoch+1)

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Train Dice: {avg_train_dice:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        print("-" * 50)

        # LR scheduling & early stopping on TRAIN dice (best available without val labels)
        scheduler.step(avg_train_dice)

        if avg_train_dice > best_train_dice:
            best_train_dice = avg_train_dice
            counter = 0
            torch.save(model.state_dict(), "best_lightweight_model.pth")
            mlflow.log_artifact("best_lightweight_model.pth")
            print(f"New best model saved! Train Dice: {best_train_dice:.4f}")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        clear_memory()

    # Save final model
    final_model_path = "lightweight_glioma_final.pth"
    torch.save(model.state_dict(), final_model_path)
    mlflow.log_artifact(final_model_path)
    print("\nTraining completed!")
    print(f"Best TRAIN Dice: {best_train_dice:.4f}")
    print(f"Final model saved as: {final_model_path}")


# -------------------- INFERENCE ON VAL (NO MASKS) --------------------
def load_manifest(manifest_json):
    if not os.path.exists(manifest_json):
        raise FileNotFoundError(
            f"Manifest not found: {manifest_json}\n"
            "Please create a JSON mapping like:\n"
            "{\n"
            '  "image_0.npy": {"ref_nii": "/path/to/case/xxxxx-yyy_t1c.nii.gz"},\n'
            '  "image_1.npy": {"ref_nii": "/path/to/case/....."}\n'
            "}\n"
        )
    with open(manifest_json, "r") as f:
        return json.load(f)

def run_inference_and_save_nii(model, val_img_dir, manifest_json, pred_out_dir,
                               crop_slices, is_rt_task=False):
    ensure_dir(pred_out_dir)
    manifest = load_manifest(manifest_json)
    model.eval()

    # Reverse mapping for crop indices
    zsl, ysl, xsl = crop_slices  # each is a slice(start, stop)

    with torch.no_grad():
        for fname in sorted(os.listdir(val_img_dir)):
            if not fname.endswith(".npy"):
                continue

            fpath = os.path.join(val_img_dir, fname)
            if fname not in manifest or "ref_nii" not in manifest[fname]:
                raise KeyError(f"Missing 'ref_nii' for {fname} in manifest {manifest_json}")

            ref_nii_path = manifest[fname]["ref_nii"]
            ref_img = nib.load(ref_nii_path)
            ref_shape = ref_img.shape  # (X, Y, Z) or (H, W, D) depending on orientation
            affine = ref_img.affine
            header = ref_img.header

            # Load preprocessed val tensor
            X = np.load(fpath)  # (D, H, W, C)
            X_t = torch.tensor(X, dtype=torch.float32).permute(3, 0, 1, 2)  # (C, D, H, W)
            # normalize per-volume (same as training's per-patch standardization is ok at inference)
            X_t = (X_t - X_t.mean()) / (X_t.std() + 1e-8)
            X_t = X_t.unsqueeze(0).to(DEVICE)  # (1, C, D, H, W)

            # Forward
            logits = model(X_t)
            pred_lbl = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)  # (D, H, W)

            # Map training label 3 back to 4 (BraTS-style)
            pred_lbl[pred_lbl == 3] = 4

            # Embed back into original canvas with zeros elsewhere
            # Be careful with axis order: our npy is (D,H,W). Many NIfTI are (X,Y,Z) ~ (W,H,D).
            # We will construct in (D,H,W) then transpose to match ref_img if needed
            # The safest way: create array in ref_img shape order (ref_img.get_fdata() returns (X,Y,Z))
            # Our preproc crop was (Z, Y, X) in CROP_SLICES.

            # Initialize full-size volume in reference orientation
            full_mask = np.zeros(ref_shape, dtype=np.uint8)  # (X, Y, Z) order typical in nib

            # We need pred_lbl as (Z, Y, X) to align with crop_slices indexing order
            # Current pred_lbl is (D, H, W) == (Z, Y, X) already.
            # Place it back:
            full_mask[xsl, ysl, zsl] = np.transpose(pred_lbl, (2, 1, 0))  # (Z,Y,X)->(X,Y,Z)

            # Create NIfTI using original header/affine (preserves spacing, origin, orientation)
            pred_img = nib.Nifti1Image(full_mask, affine=affine, header=header)

            # Filename according to challenge spec
            case_tp = parse_case_timepoint_from_path(ref_nii_path, is_rt=is_rt_task)
            out_fname = f"{case_tp}.nii.gz"
            out_path = os.path.join(pred_out_dir, out_fname)

            nib.save(pred_img, out_path)
            print(f"[OK] Saved: {out_path}")

    print(f"\nAll predictions saved to: {pred_out_dir}")

def make_submission_zip(pred_dir, zip_name):
    base = os.path.splitext(zip_name)[0]
    if os.path.exists(zip_name):
        os.remove(zip_name)
    shutil.make_archive(base, 'zip', pred_dir)
    print(f"Submission archive created: {zip_name}")


# ---- run inference + package ----
print("\nRunning inference on validation set and saving NIfTI predictions...")
validate_dir(VAL_IMG_DIR)
run_inference_and_save_nii(
    model, VAL_IMG_DIR, VAL_MANIFEST_JSON, PRED_DIR,
    crop_slices=CROP_SLICES, is_rt_task=IS_RT_TASK
)

make_submission_zip(PRED_DIR, ZIP_NAME)
print("Done.")
