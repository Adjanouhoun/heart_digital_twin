"""
CDT Phase 02 — ACDC Session 3 (Kaggle T4)
Resume depuis epoch 64, objectif MYO >= 0.90
Ajouter comme input : output du notebook ACDC session 2
"""
import torch
import torch.nn.functional as F
import numpy as np
import os, json, time, random, shutil, glob

try:
    from dynamic_network_architectures.architectures.unet import PlainConvUNet
except ImportError:
    os.system("pip install dynamic-network-architectures blosc2 -q")
    from dynamic_network_architectures.architectures.unet import PlainConvUNet

from torch.nn import Conv3d, InstanceNorm3d, LeakyReLU

PATCH_SIZE = [20, 256, 224]
BATCH_SIZE = 4
NUM_EPOCHS = 200
INITIAL_LR = 0.01
WEIGHT_DECAY = 3e-5
VAL_EVERY = 10
STEPS_PER_EPOCH = 250
WARMUP_EPOCHS = 0
NUM_CLASSES = 4
DEVICE = "cuda"
TARGET_SPACING = (1.5625, 1.5625, 5.0)
MYO_SLO = 0.90

ACDC_INPUT = None
CKPT_PATH = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "checkpoint_best.pth" in files:
        CKPT_PATH = os.path.join(root, "checkpoint_best.pth")
    if "dataset.json" in files:
        with open(os.path.join(root, "dataset.json")) as f:
            ds = json.load(f)
        if "ACDC" in ds.get("name", ""):
            ACDC_INPUT = root

if ACDC_INPUT is None:
    candidates = glob.glob("/kaggle/input/**/imagesTr", recursive=True)
    if candidates:
        ACDC_INPUT = os.path.dirname(candidates[0])

assert ACDC_INPUT, "ACDC dataset not found"
assert CKPT_PATH, "Checkpoint not found"
print(f"ACDC: {ACDC_INPUT}")
print(f"Checkpoint: {CKPT_PATH}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

import blosc2
from scipy.ndimage import zoom

def load_b2nd(path):
    return blosc2.open(path)[:]

def preprocess_nifti(img_path, lbl_path):
    import nibabel as nib
    img = nib.load(img_path).get_fdata().astype(np.float32)
    lbl = nib.load(lbl_path).get_fdata().astype(np.int16)
    img = (img - img.mean()) / (img.std() + 1e-8)
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))[np.newaxis]
        lbl = np.transpose(lbl, (2, 0, 1))[np.newaxis]
    lbl = np.clip(lbl, 0, 3)
    return img, lbl

def pad_or_crop(arr, target_shape, offsets=None):
    result = np.zeros((arr.shape[0], *target_shape), dtype=arr.dtype)
    computed_offsets = []
    for i in range(3):
        s, t = arr.shape[i+1], target_shape[i]
        if s >= t:
            off = offsets[i] if offsets else random.randint(0, s - t)
            src_sl = slice(off, off + t)
            dst_sl = slice(0, t)
            computed_offsets.append(off)
        else:
            off = (t - s) // 2
            src_sl = slice(0, s)
            dst_sl = slice(off, off + s)
            computed_offsets.append(0)
        if i == 0: s0, d0 = src_sl, dst_sl
        elif i == 1: s1, d1 = src_sl, dst_sl
        else: s2, d2 = src_sl, dst_sl
    result[:, d0, d1, d2] = arr[:, s0, s1, s2]
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
    return {names[c]: (2.0 * ((pa==c).float() * (target==c).float()).sum() / ((pa==c).float().sum() + (target==c).float().sum() + 1e-5)).item() for c in range(1, NUM_CLASSES)}

def poly_lr(epoch, max_ep, init_lr):
    return init_lr * (1 - epoch / max_ep) ** 0.9

img_dir = os.path.join(ACDC_INPUT, "imagesTr")
lbl_dir = os.path.join(ACDC_INPUT, "labelsTr")
sample = os.listdir(img_dir)[0]
EXT = ".nii.gz" if sample.endswith(".nii.gz") else ".nii"
IMG_SUFFIX = f"_0000{EXT}"
LBL_SUFFIX = EXT
all_cases = sorted([f.replace(IMG_SUFFIX, "") for f in os.listdir(img_dir) if f.endswith(IMG_SUFFIX)])
random.seed(42)
random.shuffle(all_cases)
split = int(0.8 * len(all_cases))
train_ids, val_ids = all_cases[:split], all_cases[split:]
print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

print("Preprocessing...")
t0 = time.time()
train_cache, val_cache = {}, {}
for i, cid in enumerate(train_ids):
    train_cache[cid] = preprocess_nifti(
        os.path.join(img_dir, f"{cid}{IMG_SUFFIX}"),
        os.path.join(lbl_dir, f"{cid}{LBL_SUFFIX}"))
    if (i+1) % 20 == 0: print(f"  Train: {i+1}/{len(train_ids)}")
for cid in val_ids:
    val_cache[cid] = preprocess_nifti(
        os.path.join(img_dir, f"{cid}{IMG_SUFFIX}"),
        os.path.join(lbl_dir, f"{cid}{LBL_SUFFIX}"))
print(f"Charge en {time.time()-t0:.1f}s")

network = PlainConvUNet(input_channels=1, n_stages=6, features_per_stage=[32,64,128,256,320,320], conv_op=Conv3d,
    kernel_sizes=[[1,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]], strides=[[1,1,1],[1,2,2],[2,2,2],[2,2,2],[1,2,2],[1,2,2]],
    n_conv_per_stage=[2,2,2,2,2,2], n_conv_per_stage_decoder=[2,2,2,2,2], conv_bias=True, norm_op=InstanceNorm3d,
    norm_op_kwargs={'eps':1e-05,'affine':True}, dropout_op=None, dropout_op_kwargs=None,
    nonlin=LeakyReLU, nonlin_kwargs={'inplace':True}, num_classes=NUM_CLASSES).to(DEVICE)

ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
network.load_state_dict(ckpt["network"])
start_epoch = ckpt.get("epoch", 0) + 1
best_dice = ckpt.get("best_val_dice", 0.0)
print(f"Resumed epoch {start_epoch}, best={best_dice:.4f}")

optimizer = torch.optim.SGD(network.parameters(), lr=INITIAL_LR, momentum=0.99, weight_decay=WEIGHT_DECAY, nesterov=True)
if "optimizer" in ckpt:
    optimizer.load_state_dict(ckpt["optimizer"])

RESULTS_DIR = "/kaggle/working"
log_path = os.path.join(RESULTS_DIR, "training_log.csv")
with open(log_path, "w") as f:
    f.write("epoch,loss,lr,dice_rv,dice_myo,dice_lv,dice_mean,time_min\n")

t_total = time.time()
for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
    network.train()
    lr = poly_lr(epoch, start_epoch + NUM_EPOCHS, INITIAL_LR)
    for pg in optimizer.param_groups: pg['lr'] = lr
    ep_loss = 0.0
    for step in range(STEPS_PER_EPOCH):
        selected = random.choices(train_ids, k=BATCH_SIZE)
        images, targets = [], []
        for cid in selected:
            img, seg = train_cache[cid]
            img, offsets = pad_or_crop(img, PATCH_SIZE)
            seg = pad_or_crop(seg, PATCH_SIZE, offsets=offsets)
            for ax in [1,2,3]:
                if random.random() > 0.5:
                    img = np.flip(img, axis=ax).copy()
                    seg = np.flip(seg, axis=ax).copy()
            images.append(img); targets.append(seg)
        data = torch.from_numpy(np.stack(images)).float().to(DEVICE)
        target = torch.from_numpy(np.stack(targets)).long().squeeze(1).to(DEVICE)
        optimizer.zero_grad()
        out = network(data)
        loss = F.cross_entropy(out, target) + dice_loss(out, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 12.0)
        optimizer.step()
        ep_loss += loss.item()
        if step % 50 == 0: print(f"  Epoch {epoch} step {step}/{STEPS_PER_EPOCH}: loss={loss.item():.4f}")
    ep_loss /= STEPS_PER_EPOCH
    tot_min = (time.time() - t_total) / 60
    d_rv = d_myo = d_lv = d_mean = 0.0
    if (epoch+1) % VAL_EVERY == 0:
        network.eval()
        all_s = {"RV":[], "MYO":[], "LV":[]}
        with torch.no_grad():
            for vid in val_ids:
                img, seg = val_cache[vid]
                img_t, _ = pad_or_crop(img, PATCH_SIZE)
                seg_t = pad_or_crop(seg, PATCH_SIZE, offsets=[0,0,0])
                out = network(torch.from_numpy(img_t[np.newaxis]).float().to(DEVICE))
                for k, v in compute_dice(out, torch.from_numpy(seg_t[np.newaxis]).long().squeeze(1).to(DEVICE)).items():
                    all_s[k].append(v)
        d_rv, d_myo, d_lv = np.mean(all_s["RV"]), np.mean(all_s["MYO"]), np.mean(all_s["LV"])
        d_mean = (d_rv + d_myo + d_lv) / 3
        print(f"  VAL -> RV={d_rv:.4f} MYO={d_myo:.4f} LV={d_lv:.4f} mean={d_mean:.4f}")
        if d_mean > best_dice:
            best_dice = d_mean
            torch.save({"network": network.state_dict(), "optimizer": optimizer.state_dict(),
                         "epoch": epoch, "best_val_dice": best_dice},
                        os.path.join(RESULTS_DIR, "checkpoint_best.pth"))
            print(f"  * New best! {best_dice:.4f}")
        if d_myo >= MYO_SLO:
            print(f"\nSLO MYO atteint! {d_myo:.4f}")
            break
    print(f"Epoch {epoch}/{start_epoch+NUM_EPOCHS} | loss={ep_loss:.4f} | lr={lr:.6f} | total={tot_min:.0f}min")
    torch.save({"network": network.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch, "best_val_dice": best_dice},
                os.path.join(RESULTS_DIR, "checkpoint_latest.pth"))
    with open(log_path, "a") as f:
        f.write(f"{epoch},{ep_loss:.6f},{lr:.8f},{d_rv:.4f},{d_myo:.4f},{d_lv:.4f},{d_mean:.4f},{tot_min:.2f}\n")
    if tot_min > 500:
        print(f"\nLimite Kaggle ({tot_min:.0f}min)")
        break

print(f"\nBest Dice: {best_dice:.4f}")
for f in ["checkpoint_best.pth", "checkpoint_latest.pth", "training_log.csv"]:
    src = os.path.join(RESULTS_DIR, f)
    if os.path.exists(src): print(f"  Output: {f}")
