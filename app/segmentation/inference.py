"""
CDT Segmentation Inference — nnU-Net v2 checkpoint
Usage en pipeline : NIfTI → segmentation → masque RV/MYO/LV
"""
import os
import numpy as np
import nibabel as nib
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)

MODELS_DIR = os.environ.get("CDT_MODELS_DIR", "/app/models/nnunet")


def segment_nifti(nifti_path: str, output_path: str = None) -> np.ndarray:
    """Segmente une IRM cardiaque NIfTI.
    
    Args:
        nifti_path: chemin vers le NIfTI d'entrée
        output_path: chemin de sortie (optionnel)
    
    Returns:
        masque de segmentation (0=BG, 1=RV, 2=MYO, 3=LV)
    """
    checkpoint = os.path.join(MODELS_DIR, "checkpoint_best.pth")
    
    if not os.path.exists(checkpoint):
        logger.warning("segmentation.no_checkpoint", path=checkpoint)
        return _fallback_segmentation(nifti_path, output_path)
    
    try:
        return _nnunet_inference(nifti_path, output_path, checkpoint)
    except Exception as e:
        logger.warning("segmentation.nnunet_failed", error=str(e))
        return _fallback_segmentation(nifti_path, output_path)


def _nnunet_inference(nifti_path: str, output_path: str, checkpoint: str) -> np.ndarray:
    """Inference nnU-Net v2."""
    import torch
    from torch.nn import Conv3d, InstanceNorm3d, LeakyReLU
    
    try:
        from dynamic_network_architectures.architectures.unet import PlainConvUNet
    except ImportError:
        raise ImportError("dynamic_network_architectures not installed")
    
    img_nii = nib.load(nifti_path)
    img = img_nii.get_fdata().astype(np.float32)
    affine = img_nii.affine
    spacing = img_nii.header.get_zooms()[:3]
    
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    
    img = (img - img.mean()) / (img.std() + 1e-8)
    
    device = "cpu"
    
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    
    if "network_weights" in ckpt:
        weights = ckpt["network_weights"]
    elif "network" in ckpt:
        weights = ckpt["network"]
    else:
        weights = ckpt
    
    network = PlainConvUNet(
        input_channels=1, n_stages=6,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        conv_op=Conv3d,
        kernel_sizes=[[1,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],
        strides=[[1,1,1],[1,2,2],[2,2,2],[2,2,2],[1,2,2],[1,2,2]],
        n_conv_per_stage=[2,2,2,2,2,2],
        n_conv_per_stage_decoder=[2,2,2,2,2],
        conv_bias=True,
        norm_op=InstanceNorm3d,
        norm_op_kwargs={'eps': 1e-05, 'affine': True},
        dropout_op=None, dropout_op_kwargs=None,
        nonlin=LeakyReLU, nonlin_kwargs={'inplace': True},
        num_classes=4,
    ).to(device)
    
    network.load_state_dict(weights)
    network.eval()
    
    D, H, W = img.shape
    patch = [20, 256, 224]
    pD, pH, pW = patch
    
    pad_d = max(0, pD - D)
    pad_h = max(0, pH - H)
    pad_w = max(0, pW - W)
    img_pad = np.pad(img, (
        (pad_d // 2, pad_d - pad_d // 2),
        (pad_h // 2, pad_h - pad_h // 2),
        (pad_w // 2, pad_w - pad_w // 2)
    ), mode='constant')
    
    d0 = (img_pad.shape[0] - pD) // 2
    h0 = (img_pad.shape[1] - pH) // 2
    w0 = (img_pad.shape[2] - pW) // 2
    
    patch_data = img_pad[d0:d0+pD, h0:h0+pH, w0:w0+pW]
    tensor = torch.from_numpy(patch_data[np.newaxis, np.newaxis]).float().to(device)
    
    with torch.no_grad():
        out = network(tensor)
        pred = out.argmax(dim=1).squeeze().cpu().numpy()
    
    seg = np.zeros_like(img, dtype=np.uint8)
    seg_region = pred[:min(pD,D), :min(pH,H), :min(pW,W)]
    seg[:seg_region.shape[0], :seg_region.shape[1], :seg_region.shape[2]] = seg_region
    
    if img.ndim == 3 and img_nii.get_fdata().ndim == 3:
        seg = np.transpose(seg, (1, 2, 0))
    
    if output_path:
        nib.save(nib.Nifti1Image(seg, affine), output_path)
        logger.info("segmentation.saved", path=output_path, 
                     labels=np.unique(seg).tolist())
    
    logger.info("segmentation.complete",
                shape=seg.shape,
                rv=int((seg == 1).sum()),
                myo=int((seg == 2).sum()),
                lv=int((seg == 3).sum()))
    
    return seg


def _fallback_segmentation(nifti_path: str, output_path: str = None) -> np.ndarray:
    """Segmentation fallback basee sur le seuillage."""
    img_nii = nib.load(nifti_path)
    img = img_nii.get_fdata().astype(np.float32)
    
    seg = np.zeros_like(img, dtype=np.uint8)
    thresh = img.mean()
    seg[img > thresh * 1.5] = 3
    seg[img > thresh * 0.8] = 2
    
    if output_path:
        nib.save(nib.Nifti1Image(seg, img_nii.affine), output_path)
    
    logger.warning("segmentation.fallback_used")
    return seg
