"""
nnU-Net 3d_fullres training on Apple Silicon MPS
Bypasses nnU-Net's data loader (crashes on macOS) while keeping:
- Same PlainConvUNet architecture (6 stages, 30.4M params)
- Same preprocessed ACDC data (.b2nd)
- Same Dice + CE loss
- Same LR schedule (poly)
- Checkpointing + resume
"""
import torch
import torch.nn.functional as F
import numpy as np
import blosc2
import pickle
import glob
import os
import sys
import json
import time
import random
from pathlib import Path
from torch.nn import Conv3d, InstanceNorm3d, LeakyReLU
from dynamic_network_architectures.architectures.unet import PlainConvUNet

# ─── Config ───
PATCH_SIZE = [20, 256, 224]
BATCH_SIZE = 2
NUM_EPOCHS = 1000
INITIAL_LR = 0.01
WEIGHT_DECAY = 3e-5
VAL_EVERY = 50
DEVICE = "mps"
NUM_CLASSES = 4  # bg + RV + MYO + LV

BASE_DIR = os.path.expanduser("~/nnunet/preprocessed/Dataset027_ACDC")
DATA_DIR = os.path.join(BASE_DIR, "nnUNetPlans_3d_fullres")
RESULTS_DIR = os.path.expanduser("~/nnunet/results/Dataset027_ACDC/mps_training")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Data Loading ───
def load_case(case_id):
    """Load a single preprocessed case (.b2nd + _seg.b2nd)"""
    img = blosc2.open(os.path.join(DATA_DIR, f"{case_id}.b2nd"))[:]
    seg = blosc2.open(os.path.join(DATA_DIR, f"{case_id}_seg.b2nd"))[:]
    return img, seg  # (1, D, H, W), (1, D, H, W)

def pad_or_crop(arr, target_shape):
    """Pad or crop array to target_shape (D, H, W)"""
    result = np.zeros((arr.shape[0], *target_shape), dtype=arr.dtype)
    # Compute overlap
    slices_src = []
    slices_dst = []
    for i in range(3):
        src_size = arr.shape[i + 1]
        tgt_size = target_shape[i]
        if src_size >= tgt_size:
            # Crop: random offset
            offset = random.randint(0, src_size - tgt_size)
            slices_src.append(slice(offset, offset + tgt_size))
            slices_dst.append(slice(0, tgt_size))
        else:
            # Pad: center
            offset = (tgt_size - src_size) // 2
            slices_src.append(slice(0, src_size))
            slices_dst.append(slice(offset, offset + src_size))
    
    result[:, slices_dst[0], slices_dst[1], slices_dst[2]] = \
        arr[:, slices_src[0], slices_src[1], slices_src[2]]
    return result

def get_batch(case_ids, batch_size, patch_size):
    """Load a random batch with crop/pad to patch_size"""
    selected = random.choices(case_ids, k=batch_size)
    images, targets = [], []
    for cid in selected:
        img, seg = load_case(cid)
        img = pad_or_crop(img, patch_size)
        seg = pad_or_crop(seg, patch_size)
        # Simple augmentation: random flip
        for axis in [1, 2, 3]:
            if random.random() > 0.5:
                img = np.flip(img, axis=axis).copy()
                seg = np.flip(seg, axis=axis).copy()
        images.append(img)
        targets.append(seg)
    
    data = torch.from_numpy(np.stack(images)).float()
    target = torch.from_numpy(np.stack(targets)).long().squeeze(1)
    return data, target

# ─── Loss: Dice + CE ───
def dice_loss(pred, target, num_classes=4):
    """Soft Dice loss per class"""
    pred_soft = F.softmax(pred, dim=1)
    dice = 0.0
    for c in range(1, num_classes):  # skip background
        pred_c = pred_soft[:, c]
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice += 1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)
    return dice / (num_classes - 1)

def compute_dice_scores(pred, target, num_classes=4):
    """Compute Dice per class for monitoring"""
    pred_argmax = pred.argmax(dim=1)
    scores = {}
    class_names = {1: "RV", 2: "MYO", 3: "LV"}
    for c in range(1, num_classes):
        pred_c = (pred_argmax == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        scores[class_names[c]] = (2.0 * intersection / (union + 1e-5)).item()
    return scores

# ─── Poly LR Schedule ───
def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs) ** exponent

# ─── Main ───
def main():
    print("=" * 60)
    print("nnU-Net ACDC Training — Apple Silicon MPS")
    print("=" * 60)
    
    # Load splits
    splits_path = os.path.join(BASE_DIR, "splits_final.json")
    with open(splits_path) as f:
        splits = json.load(f)
    
    train_ids = splits[0]["train"]
    val_ids = splits[0]["val"]
    print(f"Train: {len(train_ids)} cases, Val: {len(val_ids)} cases")
    
    # Build network (same as nnU-Net 3d_fullres plan)
    network = PlainConvUNet(
        input_channels=1, n_stages=6,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        conv_op=Conv3d,
        kernel_sizes=[[1,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],
        strides=[[1,1,1],[1,2,2],[2,2,2],[2,2,2],[1,2,2],[1,2,2]],
        n_conv_per_stage=[2,2,2,2,2,2],
        n_conv_per_stage_decoder=[2,2,2,2,2],
        conv_bias=True, norm_op=InstanceNorm3d,
        norm_op_kwargs={'eps':1e-05, 'affine':True},
        dropout_op=None, dropout_op_kwargs=None,
        nonlin=LeakyReLU, nonlin_kwargs={'inplace':True},
        num_classes=NUM_CLASSES
    ).to(DEVICE)
    
    n_params = sum(p.numel() for p in network.parameters()) / 1e6
    print(f"Network: {n_params:.1f}M params on {DEVICE}")
    
    optimizer = torch.optim.SGD(network.parameters(), lr=INITIAL_LR,
                                 momentum=0.99, weight_decay=WEIGHT_DECAY,
                                 nesterov=True)
    
    # Resume?
    start_epoch = 0
    best_val_dice = 0.0
    ckpt_path = os.path.join(RESULTS_DIR, "checkpoint_latest.pth")
    if os.path.exists(ckpt_path):
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        network.load_state_dict(ckpt["network"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_dice = ckpt.get("best_val_dice", 0.0)
        print(f"  Resumed at epoch {start_epoch}, best Dice={best_val_dice:.4f}")
    
    # Log file
    log_path = os.path.join(RESULTS_DIR, "training_log.csv")
    if start_epoch == 0:
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,train_dice_ce,lr,val_dice_rv,val_dice_myo,val_dice_lv,val_dice_mean,time_min\n")
    
    steps_per_epoch = 250  # same as nnU-Net default
    t_start_total = time.time()
    
    print(f"\nStarting training from epoch {start_epoch}...")
    for epoch in range(start_epoch, NUM_EPOCHS):
        t_epoch = time.time()
        network.train()
        
        # Poly LR
        lr = poly_lr(epoch, NUM_EPOCHS, INITIAL_LR)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            data, target = get_batch(train_ids, BATCH_SIZE, PATCH_SIZE)
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            
            optimizer.zero_grad()
            output = network(data)
            
            ce = F.cross_entropy(output, target)
            dl = dice_loss(output, target)
            loss = ce + dl
            
            loss.backward()
            torch.mps.synchronize()
            optimizer.step()
            torch.mps.synchronize()
            
            epoch_loss += loss.item()
        
        epoch_loss /= steps_per_epoch
        epoch_time = (time.time() - t_epoch) / 60
        
        # Validation
        val_dice_rv, val_dice_myo, val_dice_lv, val_dice_mean = 0, 0, 0, 0
        if (epoch + 1) % VAL_EVERY == 0 or epoch == 0:
            network.eval()
            all_scores = {"RV": [], "MYO": [], "LV": []}
            with torch.no_grad():
                for vid in val_ids[:10]:  # subset for speed
                    data, target = get_batch([vid], 1, PATCH_SIZE)
                    data = data.to(DEVICE)
                    target = target.to(DEVICE)
                    output = network(data)
                    torch.mps.synchronize()
                    scores = compute_dice_scores(output, target)
                    for k in scores:
                        all_scores[k].append(scores[k])
            
            val_dice_rv = np.mean(all_scores["RV"])
            val_dice_myo = np.mean(all_scores["MYO"])
            val_dice_lv = np.mean(all_scores["LV"])
            val_dice_mean = (val_dice_rv + val_dice_myo + val_dice_lv) / 3
            
            print(f"  VAL Dice — RV={val_dice_rv:.4f}  MYO={val_dice_myo:.4f}  LV={val_dice_lv:.4f}  mean={val_dice_mean:.4f}")
            
            # Save best
            if val_dice_mean > best_val_dice:
                best_val_dice = val_dice_mean
                torch.save({"network": network.state_dict(), "epoch": epoch,
                             "best_val_dice": best_val_dice},
                            os.path.join(RESULTS_DIR, "checkpoint_best.pth"))
                print(f"  ★ New best model! Dice={best_val_dice:.4f}")
        
        # Print
        total_min = (time.time() - t_start_total) / 60
        print(f"Epoch {epoch:4d}/{NUM_EPOCHS} | loss={epoch_loss:.4f} | lr={lr:.6f} | "
              f"{epoch_time:.1f}min | total={total_min:.0f}min")
        
        # Save checkpoint
        torch.save({"network": network.state_dict(), "optimizer": optimizer.state_dict(),
                     "epoch": epoch, "best_val_dice": best_val_dice},
                    ckpt_path)
        
        # Log
        with open(log_path, "a") as f:
            f.write(f"{epoch},{epoch_loss:.6f},{epoch_loss:.6f},{lr:.8f},"
                    f"{val_dice_rv:.4f},{val_dice_myo:.4f},{val_dice_lv:.4f},{val_dice_mean:.4f},"
                    f"{total_min:.2f}\n")
        
        # Early stop check
        if val_dice_myo >= 0.90:
            print(f"\n🎉 SLO atteint! MYO Dice={val_dice_myo:.4f} >= 0.90")
            break

    print(f"\nTraining terminé! Best Dice={best_val_dice:.4f}")
    print(f"Checkpoint: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
