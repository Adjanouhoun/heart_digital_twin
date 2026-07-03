"""nnU-Net ACDC Training — MPS — v2 final (preload + clipping + warmup)"""
import torch
import torch.nn.functional as F
import numpy as np
import blosc2
import os
import json
import time
import random
from torch.nn import Conv3d, InstanceNorm3d, LeakyReLU
from dynamic_network_architectures.architectures.unet import PlainConvUNet

PATCH_SIZE = [20, 256, 224]
BATCH_SIZE = 2
NUM_EPOCHS = 1000
INITIAL_LR = 0.01
WEIGHT_DECAY = 3e-5
VAL_EVERY = 10
STEPS_PER_EPOCH = 50
WARMUP_EPOCHS = 5
DEVICE = "mps"
NUM_CLASSES = 4

BASE_DIR = os.path.expanduser("~/nnunet/preprocessed/Dataset027_ACDC")
DATA_DIR = os.path.join(BASE_DIR, "nnUNetPlans_3d_fullres")
RESULTS_DIR = os.path.expanduser("~/nnunet/results/Dataset027_ACDC/mps_training")
os.makedirs(RESULTS_DIR, exist_ok=True)

def preload_all_cases(case_ids):
    cache = {}
    for i, cid in enumerate(case_ids):
        img = blosc2.open(os.path.join(DATA_DIR, f"{cid}.b2nd"))[:]
        seg = blosc2.open(os.path.join(DATA_DIR, f"{cid}_seg.b2nd"))[:]
        cache[cid] = (img, seg)
        if (i + 1) % 20 == 0:
            print(f"  Chargé {i+1}/{len(case_ids)} cas...")
    return cache

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

def get_batch(cache, case_ids, batch_size, patch_size):
    selected = random.choices(case_ids, k=batch_size)
    images, targets = [], []
    for cid in selected:
        img, seg = cache[cid]
        img, offsets = pad_or_crop(img, patch_size)
        seg = pad_or_crop(seg, patch_size, offsets=offsets)
        for axis in [1, 2, 3]:
            if random.random() > 0.5:
                img = np.flip(img, axis=axis).copy()
                seg = np.flip(seg, axis=axis).copy()
        images.append(img)
        targets.append(seg)
    return (torch.from_numpy(np.stack(images)).float(),
            torch.from_numpy(np.stack(targets)).long().squeeze(1))

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

def poly_lr(epoch, max_ep, init_lr, warmup=5):
    if epoch < warmup:
        return init_lr * (epoch + 1) / warmup
    return init_lr * (1 - epoch / max_ep) ** 0.9

def main():
    print("=" * 60)
    print("nnU-Net ACDC — Apple Silicon MPS (v2 final)")
    print("  warmup=5, grad_clip=12, aligned crop")
    print("=" * 60)

    with open(os.path.join(BASE_DIR, "splits_final.json")) as f:
        splits = json.load(f)
    train_ids = splits[0]["train"]
    val_ids = splits[0]["val"]
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

    print("\nPréchargement des données en RAM...")
    t0 = time.time()
    train_cache = preload_all_cases(train_ids)
    val_cache = preload_all_cases(val_ids)
    print(f"Chargé en {time.time()-t0:.1f}s\n")

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

    print(f"Network: {sum(p.numel() for p in network.parameters())/1e6:.1f}M params")

    optimizer = torch.optim.SGD(network.parameters(), lr=INITIAL_LR,
                                 momentum=0.99, weight_decay=WEIGHT_DECAY, nesterov=True)

    start_epoch, best_val_dice = 0, 0.0
    ckpt_path = os.path.join(RESULTS_DIR, "checkpoint_latest.pth")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        network.load_state_dict(ckpt["network"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_dice = ckpt.get("best_val_dice", 0.0)
        print(f"Resumed epoch {start_epoch}, best={best_val_dice:.4f}")

    log_path = os.path.join(RESULTS_DIR, "training_log.csv")
    if start_epoch == 0:
        with open(log_path, "w") as f:
            f.write("epoch,loss,lr,dice_rv,dice_myo,dice_lv,dice_mean,time_min\n")

    t_total = time.time()
    print(f"\nTraining epoch {start_epoch} → {NUM_EPOCHS}...\n")

    for epoch in range(start_epoch, NUM_EPOCHS):
        t_ep = time.time()
        network.train()
        lr = poly_lr(epoch, NUM_EPOCHS, INITIAL_LR, WARMUP_EPOCHS)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        ep_loss = 0.0
        for step in range(STEPS_PER_EPOCH):
            data, target = get_batch(train_cache, train_ids, BATCH_SIZE, PATCH_SIZE)
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            out = network(data)
            loss = F.cross_entropy(out, target) + dice_loss(out, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 12.0)
            torch.mps.synchronize()
            optimizer.step()
            ep_loss += loss.item()

            if step % 10 == 0:
                print(f"  Epoch {epoch} step {step}: loss={loss.item():.4f}")

        ep_loss /= STEPS_PER_EPOCH
        ep_min = (time.time() - t_ep) / 60
        tot_min = (time.time() - t_total) / 60

        d_rv = d_myo = d_lv = d_mean = 0.0
        if (epoch + 1) % VAL_EVERY == 0 or epoch == 0:
            network.eval()
            all_s = {"RV": [], "MYO": [], "LV": []}
            with torch.no_grad():
                for vid in val_ids[:10]:
                    d, t = get_batch(val_cache, [vid], 1, PATCH_SIZE)
                    out = network(d.to(DEVICE))
                    torch.mps.synchronize()
                    for k, v in compute_dice(out, t.to(DEVICE)).items():
                        all_s[k].append(v)
            d_rv = np.mean(all_s["RV"])
            d_myo = np.mean(all_s["MYO"])
            d_lv = np.mean(all_s["LV"])
            d_mean = (d_rv + d_myo + d_lv) / 3
            print(f"  VAL → RV={d_rv:.4f}  MYO={d_myo:.4f}  LV={d_lv:.4f}  mean={d_mean:.4f}")
            if d_mean > best_val_dice:
                best_val_dice = d_mean
                torch.save({"network": network.state_dict(), "epoch": epoch,
                             "best_val_dice": best_val_dice},
                            os.path.join(RESULTS_DIR, "checkpoint_best.pth"))
                print(f"  ★ Best! Dice={best_val_dice:.4f}")

        print(f"Epoch {epoch:4d}/{NUM_EPOCHS} | loss={ep_loss:.4f} | lr={lr:.6f} | "
              f"{ep_min:.1f}min | total={tot_min:.0f}min")

        torch.save({"network": network.state_dict(), "optimizer": optimizer.state_dict(),
                     "epoch": epoch, "best_val_dice": best_val_dice}, ckpt_path)

        with open(log_path, "a") as f:
            f.write(f"{epoch},{ep_loss:.6f},{lr:.8f},{d_rv:.4f},{d_myo:.4f},{d_lv:.4f},{d_mean:.4f},{tot_min:.2f}\n")

        if d_myo >= 0.90:
            print(f"\n🎉 SLO atteint! MYO={d_myo:.4f} >= 0.90")
            break

    print(f"\nDone! Best Dice={best_val_dice:.4f}")

if __name__ == "__main__":
    main()
