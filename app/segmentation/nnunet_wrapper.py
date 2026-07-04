"""
Segmenteur cardiaque — wrapper nnU-Net v2.

CONTRATS D'INTERFACE (Regle 4 — golden rules) :

1. Labels (CardiacLabel) — DOIT correspondre à dataset.json utilisé à l'entraînement
   (cf. scripts/generate_meshes_acdc.py, cellule 3 des notebooks Colab) :
       0 = background, 1 = RV, 2 = MYO, 3 = LV
   ATTENTION : ceci est l'INVERSE de l'ancienne convention utilisée dans ce fichier
   (LV=1, RV=3), qui ne correspondait à RIEN de ce que le modèle entraîné produit.
   Tout code aval (gmsh_mesher.py notamment) doit utiliser cette même convention.

2. Spacing d'entrainement — le réseau a été entraîné sur des volumes rééchantillonnés
   à (1.5625, 1.5625, 5.0) mm (spacing ACDC natif, anisotrope). Le pipeline
   d'ingestion (app/segmentation/preprocessor.py) rééchantillonne lui à 1.0mm³
   isotrope pour un usage générique multi-dataset. CE FICHIER NE SUPPOSE PAS que
   son entrée est déjà au bon spacing : NNUNetV2Segmenter re-rééchantillonne
   lui-même en interne vers le spacing d'entraînement, à partir du spacing_mm
   réellement fourni par l'appelant, puis renvoie le masque au spacing d'origine.

3. Checkpoint — format attendu : dict avec clés "network" (state_dict) et
   optionnellement "epoch", "best_val_dice" (pour traçabilité / MLflow).
"""
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional
import os
import time
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class CardiacLabel(IntEnum):
    BACKGROUND = 0
    RV_CAVITY  = 1
    MYOCARDIUM = 2
    LV_CAVITY  = 3
    LGE_SCAR   = 4   # EMIDEC uniquement, absent du modèle ACDC de base


@dataclass
class SegmentationResult:
    mask: np.ndarray
    probabilities: Optional[np.ndarray]
    spacing_mm: tuple
    volume_lv_ml: float
    volume_rv_ml: float
    volume_myo_ml: float
    volume_scar_ml: float
    scar_burden_pct: float
    dice_lv: Optional[float] = None
    dice_rv: Optional[float] = None
    dice_myo: Optional[float] = None
    model_name: str = "nnunet-v2-cardiac-v1.0"
    inference_time_s: float = 0.0


def _volumes_from_mask(mask_3d: np.ndarray, spacing_mm: tuple) -> dict:
    vox = (spacing_mm[0] * spacing_mm[1] * spacing_mm[2]) / 1000.0  # mm^3 -> mL
    return {
        "lv":   float(np.sum(mask_3d == CardiacLabel.LV_CAVITY)  * vox),
        "rv":   float(np.sum(mask_3d == CardiacLabel.RV_CAVITY)  * vox),
        "myo":  float(np.sum(mask_3d == CardiacLabel.MYOCARDIUM) * vox),
        "scar": float(np.sum(mask_3d == CardiacLabel.LGE_SCAR)   * vox),
    }


# ────────────────────────────────────────────────────────────────────────────
# Segmenteur réel — nnU-Net v2 (PlainConvUNet), checkpoint entraîné Kaggle/Colab
# ────────────────────────────────────────────────────────────────────────────

class NNUNetV2Segmenter:
    """
    Segmenteur clinique — charge un checkpoint entraîné et fait l'inférence
    par sliding window, exactement comme scripts/generate_meshes_acdc.py
    (mêmes hyperparamètres architecture — NE PAS MODIFIER sans réentraîner).
    """

    TRAINING_SPACING_MM = (1.5625, 1.5625, 5.0)
    PATCH_SIZE = (20, 256, 224)
    NUM_CLASSES = 4  # background, RV, MYO, LV (pas de SCAR dans le modele ACDC de base)

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        import torch
        self._torch = torch
        self.checkpoint_path = checkpoint_path
        self.device = device

        network = self._build_network(device)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        network.load_state_dict(ckpt["network"])
        network.eval()
        self.network = network
        self.epoch = ckpt.get("epoch", "?")
        self.best_val_dice = ckpt.get("best_val_dice", "?")
        self.model_name = f"nnunet-v2-acdc-epoch{self.epoch}-dice{self.best_val_dice}"

        logger.info("nnunet_segmenter.loaded",
                    checkpoint=checkpoint_path, epoch=self.epoch,
                    best_val_dice=self.best_val_dice, device=device)

    def _build_network(self, device):
        from torch.nn import Conv3d, InstanceNorm3d, LeakyReLU
        from dynamic_network_architectures.architectures.unet import PlainConvUNet
        return PlainConvUNet(
            input_channels=1, n_stages=6,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=Conv3d,
            kernel_sizes=[[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            strides=[[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2], [1, 2, 2]],
            n_conv_per_stage=[2, 2, 2, 2, 2, 2],
            n_conv_per_stage_decoder=[2, 2, 2, 2, 2],
            conv_bias=True, norm_op=InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-05, "affine": True},
            dropout_op=None, dropout_op_kwargs=None,
            nonlin=LeakyReLU, nonlin_kwargs={"inplace": True},
            num_classes=self.NUM_CLASSES,
        ).to(device)

    def predict(self, volume: np.ndarray, spacing_mm: tuple) -> SegmentationResult:
        """
        volume     : (D, H, W) déjà normalisé (z-score) au spacing `spacing_mm` fourni
                     (peu importe lequel — on rééchantillonne nous-mêmes).
        spacing_mm : spacing RÉEL du volume passé en entrée (ex: 1.0mm³ isotrope
                     si venant de app.segmentation.preprocessor).
        """
        import SimpleITK as sitk
        t0 = time.time()

        # 1. Rééchantillonner vers le spacing d'entraînement (contrat #2 ci-dessus)
        img = sitk.GetImageFromArray(volume.astype(np.float32))
        img.SetSpacing(spacing_mm)
        resampled_img, resampled_arr = self._resample(img, self.TRAINING_SPACING_MM)

        # 2. Inférence sliding window
        seg_training_space = self._sliding_window_inference(resampled_arr)

        # 3. Revenir au spacing d'origine (pour être cohérent avec le volume d'entrée)
        seg_label_img = sitk.GetImageFromArray(seg_training_space.astype(np.uint8))
        seg_label_img.SetSpacing(self.TRAINING_SPACING_MM)
        seg_label_img.SetOrigin(resampled_img.GetOrigin())
        mask_3d = self._resample_label_to_reference(seg_label_img, img)

        vols = _volumes_from_mask(mask_3d, spacing_mm)
        scar_burden = (vols["scar"] / (vols["myo"] + vols["scar"]) * 100) if (vols["myo"] + vols["scar"]) > 0 else 0.0
        duration = time.time() - t0

        logger.info("nnunet_segmenter.predicted",
                    duration_ms=round(duration * 1000),
                    lv_ml=round(vols["lv"], 1), rv_ml=round(vols["rv"], 1),
                    myo_ml=round(vols["myo"], 1))

        return SegmentationResult(
            mask=mask_3d, probabilities=None, spacing_mm=spacing_mm,
            volume_lv_ml=round(vols["lv"], 2), volume_rv_ml=round(vols["rv"], 2),
            volume_myo_ml=round(vols["myo"], 2), volume_scar_ml=round(vols["scar"], 2),
            scar_burden_pct=round(scar_burden, 2),
            model_name=self.model_name,
            inference_time_s=round(duration, 3),
        )

    def _resample(self, img, target_spacing):
        import SimpleITK as sitk
        original_spacing = img.GetSpacing()
        original_size = img.GetSize()
        new_size = [int(round(osz * ospc / tspc))
                    for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)]
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(img.GetDirection())
        resampler.SetOutputOrigin(img.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(0)
        resampler.SetInterpolator(sitk.sitkBSpline)
        resampled = resampler.Execute(img)
        return resampled, sitk.GetArrayFromImage(resampled).astype(np.float32)

    def _resample_label_to_reference(self, label_img, reference_img):
        import SimpleITK as sitk
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_img)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        resampled = resampler.Execute(label_img)
        return sitk.GetArrayFromImage(resampled).astype(np.uint8)

    def _sliding_window_inference(self, volume: np.ndarray, overlap: float = 0.5) -> np.ndarray:
        torch = self._torch
        F = torch.nn.functional
        pD, pH, pW = self.PATCH_SIZE
        D, H, W = volume.shape

        pad_d, pad_h, pad_w = max(0, pD - D), max(0, pH - H), max(0, pW - W)
        if pad_d or pad_h or pad_w:
            volume = np.pad(volume, (
                (pad_d // 2, pad_d - pad_d // 2),
                (pad_h // 2, pad_h - pad_h // 2),
                (pad_w // 2, pad_w - pad_w // 2)), mode="constant", constant_values=0)
        D2, H2, W2 = volume.shape

        step_d = max(1, int(pD * (1 - overlap)))
        step_h = max(1, int(pH * (1 - overlap)))
        step_w = max(1, int(pW * (1 - overlap)))

        d_starts = list(range(0, max(1, D2 - pD + 1), step_d))
        h_starts = list(range(0, max(1, H2 - pH + 1), step_h))
        w_starts = list(range(0, max(1, W2 - pW + 1), step_w))
        if d_starts[-1] + pD < D2: d_starts.append(D2 - pD)
        if h_starts[-1] + pH < H2: h_starts.append(H2 - pH)
        if w_starts[-1] + pW < W2: w_starts.append(W2 - pW)

        prediction = np.zeros((self.NUM_CLASSES, D2, H2, W2), dtype=np.float32)
        count = np.zeros((D2, H2, W2), dtype=np.float32)

        with torch.no_grad():
            for d0 in d_starts:
                for h0 in h_starts:
                    for w0 in w_starts:
                        patch = volume[d0:d0 + pD, h0:h0 + pH, w0:w0 + pW]
                        tensor = torch.from_numpy(patch[np.newaxis, np.newaxis]).float().to(self.device)
                        prob = F.softmax(self.network(tensor), dim=1).cpu().numpy()[0]
                        prediction[:, d0:d0 + pD, h0:h0 + pH, w0:w0 + pW] += prob
                        count[d0:d0 + pD, h0:h0 + pH, w0:w0 + pW] += 1

        count = np.maximum(count, 1e-8)
        prediction /= count[np.newaxis]

        if pad_d or pad_h or pad_w:
            prediction = prediction[:,
                         pad_d // 2: pad_d // 2 + D,
                         pad_h // 2: pad_h // 2 + H,
                         pad_w // 2: pad_w // 2 + W]
        return prediction.argmax(axis=0).astype(np.uint8)


# ────────────────────────────────────────────────────────────────────────────
# Fallback démo — NON CLINIQUE, pour que le pipeline tourne sans checkpoint
# ────────────────────────────────────────────────────────────────────────────

class DemoSegmenter:
    """
    Segmenteur de démonstration adapté aux volumes ACDC.
    Segmentation par seuillage + morphologie coupe par coupe.
    NON CLINIQUE — pour tests pipeline uniquement.
    """

    def predict(self, volume: np.ndarray, spacing_mm: tuple) -> SegmentationResult:
        t0 = time.time()
        shape = volume.shape

        if shape[2] < shape[0] and shape[2] < shape[1]:
            vol = np.transpose(volume, (2, 0, 1))
            acdc_format = True
        else:
            vol = volume
            acdc_format = False

        Z, H, W = vol.shape
        mask_3d = np.zeros((Z, H, W), dtype=np.uint8)
        for z in range(Z):
            mask_3d[z] = self._segment_slice(vol[z])

        if acdc_format:
            mask_3d = np.transpose(mask_3d, (1, 2, 0))

        vols = _volumes_from_mask(mask_3d, spacing_mm)
        scar_burden = (vols["scar"] / (vols["myo"] + vols["scar"]) * 100) if (vols["myo"] + vols["scar"]) > 0 else 0.0
        duration = time.time() - t0

        logger.info("demo_segmenter.predicted",
                    shape=list(mask_3d.shape), duration_ms=round(duration * 1000),
                    lv_ml=round(vols["lv"], 1))

        return SegmentationResult(
            mask=mask_3d, probabilities=None, spacing_mm=spacing_mm,
            volume_lv_ml=round(vols["lv"], 2), volume_rv_ml=round(vols["rv"], 2),
            volume_myo_ml=round(vols["myo"], 2), volume_scar_ml=round(vols["scar"], 2),
            scar_burden_pct=round(scar_burden, 2),
            model_name="demo-slice-heuristic-NOT-FOR-PRODUCTION",
            inference_time_s=round(duration, 3),
        )

    def _segment_slice(self, slice_2d: np.ndarray) -> np.ndarray:
        from scipy import ndimage
        H, W = slice_2d.shape
        mask = np.zeros((H, W), dtype=np.uint8)

        if slice_2d.std() < 0.1:
            return mask

        p25 = np.percentile(slice_2d, 25)
        p50 = np.percentile(slice_2d, 50)
        p75 = np.percentile(slice_2d, 75)

        cardiac_mask = slice_2d > p50
        cardiac_mask = ndimage.binary_fill_holes(cardiac_mask)
        cardiac_mask = ndimage.binary_erosion(cardiac_mask, iterations=2)
        cardiac_mask = ndimage.binary_dilation(cardiac_mask, iterations=2)

        labeled, n_components = ndimage.label(cardiac_mask)
        if n_components == 0:
            return mask

        component_sizes = [(labeled == i).sum() for i in range(1, n_components + 1)]
        sorted_components = sorted(enumerate(component_sizes, 1), key=lambda x: -x[1])

        if len(sorted_components) >= 1:
            lv_label = sorted_components[0][0]
            lv_region = (labeled == lv_label)
            lv_bright = lv_region & (slice_2d > p75)
            if lv_bright.sum() > 10:
                mask[lv_bright] = CardiacLabel.LV_CAVITY
            lv_dilated = ndimage.binary_dilation(lv_bright, iterations=4)
            myo_region = lv_dilated & ~lv_bright & (slice_2d > p25)
            if myo_region.sum() > 5:
                mask[myo_region] = CardiacLabel.MYOCARDIUM

        if len(sorted_components) >= 2:
            rv_label = sorted_components[1][0]
            rv_region = (labeled == rv_label) & (slice_2d > p50)
            if rv_region.sum() > 10:
                mask[rv_region] = CardiacLabel.RV_CAVITY

        return mask


def dice_score(pred: np.ndarray, target: np.ndarray, label: int) -> float:
    pred_bin = (pred == label).astype(np.float32)
    target_bin = (target == label).astype(np.float32)
    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum()
    if union == 0:
        return 1.0
    return float(2.0 * intersection / union)


_segmenter_singleton = None


def get_segmenter(model_dir=None):
    """
    Retourne le segmenteur réel si un checkpoint valide est configuré
    (variable d'env NNUNET_CHECKPOINT_PATH ou argument model_dir), sinon
    retombe sur DemoSegmenter avec un avertissement explicite (jamais de
    fallback silencieux).
    """
    global _segmenter_singleton
    if _segmenter_singleton is not None:
        return _segmenter_singleton

    checkpoint_path = model_dir or os.environ.get("NNUNET_CHECKPOINT_PATH")
    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            _segmenter_singleton = NNUNetV2Segmenter(checkpoint_path, device="cpu")
            logger.info("segmenter.real_model_loaded", checkpoint=checkpoint_path)
            return _segmenter_singleton
        except Exception as e:
            logger.error("segmenter.real_model_failed", checkpoint=checkpoint_path, error=str(e))

    logger.warning("segmenter.demo_mode",
                    msg="⚠️ DemoSegmenter — NON CLINIQUE — aucun checkpoint valide trouvé",
                    checked_path=checkpoint_path)
    _segmenter_singleton = DemoSegmenter()
    return _segmenter_singleton
