"""
Transfer learning M&Ms depuis checkpoint ACDC.
- Charge les poids ACDC pré-entraînés
- Fine-tune sur M&Ms avec LR réduit
- Preprocessing on-the-fly (resample + z-score)
"""
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import os
import json
import time
import random
from torch.nn import Conv3d, InstanceNorm3d, LeakyReLU
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from scipy.ndimage import zoom

PATCH_SIZE = [12, 256, 224]
BATCH_SIZE = 2
NUM_EPOCHS = 200
INITIAL_LR = 0.002       # LR réduit pour fine-tuning (vs 0.01 ACDC)
WEIGHT_DECAY = 3e-5
VAL_EVERY = 10
STEPS_PER_EPOCH = 50
WARMUP_EPOCHS = 3
NUM_CLASSES = 4
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

MNMS_DIR = os.path.expanduser("~/nnunet/raw/Dataset028_MnMs_v2")
ACDC_CKPT = os.path.expanduser("~/nnunet/results/Dataset027_ACDC/mps_training/checkpoint_best.pth")
RESULTS_DIR = os.path.expanduser("~/nnunet/results/Dataset028_MnMs/transfer_training")
os.makedirs(RESULTS_DIR, exist_ok=True)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

def preprocess_case(img_path, lbl_path, target_spacing=(1.5625, 1.5625, 10.0)):
    """Resample + z-score."""
    img_nii = nib.load(img_path)
    lbl_nii = nib.load(lbl_path)
    img = img_nii.get_fdata().astype(np.float32)
    lbl = lbl_nii.get_fdata().astype(np.int16)
    spacing = np.array(img_nii.header.get_zooms()[:3])

    # Resample
    scale = spacing / np.array(target_spacing)
    if not np.allclose(scale, 1.0, atol=0.05):
        img = zoom(img, scale, order=3)
        lbl = zoom(lbl, scale, order=0)

    # Z-score
    mean_v = img.mean()
    std_v = img.std() + 1e-8
    img = (img - mean_v) / std_v

    # Transpose to (C, Z, H, W)
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))[np.newaxis]  # (1, Z, H, W)
        lbl = np.transpose(lbl, (2, 0, 1))[np.newaxis]

    return img, lbl

def pad_or_crop(arr, target_shape, offsets=None):
    result = np.zeros((arr.shape[0], *target_shape), dtype=arr.dtype)
    slices_src, slices_dst = [], []
    computed_offsets = []
    for i in range(3):
        s, t = arr.shape[i+1], target_shape[i]
        if s >= t:
            off = offsets[i] if offsets else random.randint(0, s - t)
            slices_src.append(slice(off, off + t))
            slices_dst.append(slice(0, t))
            computed_offsets.append(off)
        else:
            off = (t - s) // 2
            slices_src.append(slice(0, s))
            slices_dst.append(slice(off, off + s))
            computed_offsets.append(0)
    result[:, slices_dst[0], slices_dst[1], slices_dst[2]] = \
        arr[:, slices_src[0], slices_src[1], slices_src[2]]
    if offsets is None:
        return result, computed_offsets
    return result

def dice_loss(pred, target):
    pred_soft = F.softmax(pred, dim=1)
    dice = 0.0
    for c in range(1, NUM_CLASSES):
        pc = pred_soft[:, c]
        tc = (target == c).float()
        inter = (pc * tc).sum()
        dice += 1.0 - (2.0 * inter + 1e-5) / (pc.sum() + tc.sum() + 1e-5)
    return dice / (NUM_CLASSES - 1)

def compute_dice(pred, target):
    pa = pred.argmax(dim=1)
    names = {1: "RV", 2: "MYO", 3: "LV"}
    scores = {}
    for c in range(1, NUM_CLASSES):
        pc = (pa == c).float()
        tc = (target == c).float()
        scores[names[c]] = (2.0 * (pc * tc).sum() / (pc.sum() + tc.sum() + 1e-5)).item()
    return scores

def poly_lr(epoch, max_ep, init_lr, warmup=3):
    if epoch < warmup:
        return init_lr * (epoch + 1) / warmup
    return init_lr * (1 - epoch / max_ep) ** 0.9

def main():
    print("=" * 60)
    print("CDT Phase 02 — M&Ms Transfer Learning")
    print(f"  Device: {DEVICE}, LR: {INITIAL_LR} (fine-tune)")
    print(f"  Checkpoint ACDC: {ACDC_CKPT}")
    print("=" * 60)

    # Load case list
    img_dir = os.path.join(MNMS_DIR, "imagesTr")
    lbl_dir = os.path.join(MNMS_DIR, "labelsTr")
    all_cases = sorted([f.replace("_0000.nii.gz", "") for f in os.listdir(img_dir) if f.endswith("_0000.nii.gz")])

    random.seed(42)
    random.shuffle(all_cases)
    split = int(0.8 * len(all_cases))
    train_ids = all_cases[:split]
    val_ids = all_cases[split:]
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

    # Preload
    print("\nPreprocessing et chargement en RAM...")
    t0 = time.time()
    train_cache, val_cache = {}, {}
    for i, cid in enumerate(train_ids):
        img, lbl = preprocess_case(
            os.path.join(img_dir, f"{cid}_0000.nii.gz"),
            os.path.join(lbl_dir, f"{cid}.nii.gz")
        )
        lbl = np.clip(lbl, 0, 3)
        train_cache[cid] = (img, lbl)
        if (i + 1) % 30 == 0:
            print(f"  Train: {i+1}/{len(train_ids)}")
    for i, cid in enumerate(val_ids):
        img, lbl = preprocess_case(
            os.path.join(img_dir, f"{cid}_0000.nii.gz"),
            os.path.join(lbl_dir, f"{cid}.nii.gz")
        )
        lbl = np.clip(lbl, 0, 3)
        val_cache[cid] = (img, lbl)
    print(f"Chargé en {time.time()-t0:.1f}s")

    # Network
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

    # Load ACDC checkpoint (transfer learning)
    ckpt = torch.load(ACDC_CKPT, map_location=DEVICE, weights_only=False)
    network.load_state_dict(ckpt["network"])
    print(f"Loaded ACDC checkpoint (epoch {ckpt['epoch']}, dice={ckpt.get('best_val_dice', 'N/A')})")

    optimizer = torch.optim.SGD(network.parameters(), lr=INITIAL_LR,
                                 momentum=0.99, weight_decay=WEIGHT_DECAY, nesterov=True)

    log_path = os.path.join(RESULTS_DIR, "training_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,loss,lr,dice_rv,dice_myo,dice_lv,dice_mean,time_min\n")

    t_total = time.time()
    best_dice = 0.0
    print(f"\nTraining epoch 0 -> {NUM_EPOCHS}...\n")

    for epoch in range(NUM_EPOCHS):
        t_ep = time.time()
        network.train()
        lr = poly_lr(epoch, NUM_EPOCHS, INITIAL_LR, WARMUP_EPOCHS)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        ep_loss = 0.0
        for step in range(STEPS_PER_EPOCH):
            selected = random.choices(train_ids, k=BATCH_SIZE)
            images, targets = [], []
            for cid in selected:
                img, seg = train_cache[cid]
                img, offsets = pad_or_crop(img, PATCH_SIZE)
                seg = pad_or_crop(seg, PATCH_SIZE, offsets=offsets)
                for axis in [1, 2, 3]:
                    if random.random() > 0.5:
                        img = np.flip(img, axis=axis).copy()
                        seg = np.flip(seg, axis=axis).copy()
                images.append(img)
                targets.append(seg)

            data = torch.from_numpy(np.stack(images)).float().to(DEVICE)
            target = torch.from_numpy(np.stack(targets)).long().squeeze(1).to(DEVICE)

            optimizer.zero_grad()
            out = network(data)
            loss = F.cross_entropy(out, target) + dice_loss(out, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 12.0)
            if DEVICE == "mps":
                torch.mps.synchronize()
            optimizer.step()
            ep_loss += loss.item()

        ep_loss /= STEPS_PER_EPOCH
        ep_min = (time.time() - t_ep) / 60
        tot_min = (time.time() - t_total) / 60

        d_rv = d_myo = d_lv = d_mean = 0.0
        if (epoch + 1) % VAL_EVERY == 0 or epoch == 0:
            network.eval()
            all_s = {"RV": [], "MYO": [], "LV": []}
            with torch.no_grad():
                for vid in val_ids[:20]:
                    img, seg = val_cache[vid]
                    img_t, _ = pad_or_crop(img, PATCH_SIZE)
                    seg_t = pad_or_crop(seg, PATCH_SIZE, offsets=[0,0,0])
                    d = torch.from_numpy(img_t[np.newaxis]).float().to(DEVICE)
                    t = torch.from_numpy(seg_t[np.newaxis]).long().squeeze(1).to(DEVICE)
                    out = network(d)
                    if DEVICE == "mps":
                        torch.mps.synchronize()
                    for k, v in compute_dice(out, t).items():
                        all_s[k].append(v)
            d_rv = np.mean(all_s["RV"])
            d_myo = np.mean(all_s["MYO"])
            d_lv = np.mean(all_s["LV"])
            d_mean = (d_rv + d_myo + d_lv) / 3
            print(f"  VAL -> RV={d_rv:.4f}  MYO={d_myo:.4f}  LV={d_lv:.4f}  mean={d_mean:.4f}")
            if d_mean > best_dice:
                best_dice = d_mean
                torch.save({"network": network.state_dict(), "epoch": epoch,
                             "best_val_dice": best_dice}, os.path.join(RESULTS_DIR, "checkpoint_best.pth"))
                print(f"  * New best! Dice={best_dice:.4f}")

        print(f"Epoch {epoch:4d}/{NUM_EPOCHS} | loss={ep_loss:.4f} | lr={lr:.6f} | "
              f"{ep_min:.1f}min | total={tot_min:.0f}min")

        torch.save({"network": network.state_dict(), "optimizer": optimizer.state_dict(),
                     "epoch": epoch, "best_val_dice": best_dice},
                    os.path.join(RESULTS_DIR, "checkpoint_latest.pth"))

        with open(log_path, "a") as f:
            f.write(f"{epoch},{ep_loss:.6f},{lr:.8f},{d_rv:.4f},{d_myo:.4f},{d_lv:.4f},{d_mean:.4f},{tot_min:.2f}\n")

        if d_myo >= 0.90:
            print(f"\nSLO atteint! MYO={d_myo:.4f}")
            break

    print(f"\nTraining termine! Best Dice: {best_dice:.4f}")
    print(f"Checkpoints: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
