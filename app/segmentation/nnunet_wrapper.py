from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class CardiacLabel(IntEnum):
    BACKGROUND = 0
    LV_CAVITY  = 1
    MYOCARDIUM = 2
    RV_CAVITY  = 3
    LGE_SCAR   = 4


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


class DemoSegmenter:
    """
    Segmenteur de démonstration adapté aux volumes ACDC.
    Les IRM ACDC sont en format (H, W, Z) avec spacing anisotropique.
    Segmentation par seuillage + morphologie coupe par coupe.
    NON CLINIQUE — pour tests pipeline uniquement.
    """

    def predict(self, volume: np.ndarray, spacing_mm: tuple) -> SegmentationResult:
        import time
        t0 = time.time()

        # Détecter l'orientation : ACDC = (H, W, Z) avec Z petit (8-15 coupes)
        # nnU-Net attend (D, H, W) mais ACDC donne (H, W, Z)
        shape = volume.shape

        # Si le dernier axe est le plus petit → format ACDC (H, W, Z)
        if shape[2] < shape[0] and shape[2] < shape[1]:
            # Transposer en (Z, H, W) pour traitement coupe par coupe
            vol = np.transpose(volume, (2, 0, 1))
            acdc_format = True
        else:
            vol = volume
            acdc_format = False

        Z, H, W = vol.shape
        mask_3d = np.zeros((Z, H, W), dtype=np.uint8)

        for z in range(Z):
            slice_2d = vol[z]
            mask_3d[z] = self._segment_slice(slice_2d)

        # Remettre dans le format original
        if acdc_format:
            mask_3d = np.transpose(mask_3d, (1, 2, 0))  # (H, W, Z)

        # Calculer les volumes
        vox = (spacing_mm[0] * spacing_mm[1] * spacing_mm[2]) / 1000.0
        vol_lv   = float(np.sum(mask_3d == CardiacLabel.LV_CAVITY)  * vox)
        vol_rv   = float(np.sum(mask_3d == CardiacLabel.RV_CAVITY)  * vox)
        vol_myo  = float(np.sum(mask_3d == CardiacLabel.MYOCARDIUM) * vox)
        vol_scar = float(np.sum(mask_3d == CardiacLabel.LGE_SCAR)   * vox)
        scar_burden = (vol_scar / (vol_myo + vol_scar) * 100) if (vol_myo + vol_scar) > 0 else 0.0

        duration = time.time() - t0
        logger.info("demo_segmenter.predicted",
                    shape=list(mask_3d.shape),
                    duration_ms=round(duration * 1000),
                    lv_ml=round(vol_lv, 1))

        return SegmentationResult(
            mask=mask_3d, probabilities=None, spacing_mm=spacing_mm,
            volume_lv_ml=round(vol_lv, 2), volume_rv_ml=round(vol_rv, 2),
            volume_myo_ml=round(vol_myo, 2), volume_scar_ml=round(vol_scar, 2),
            scar_burden_pct=round(scar_burden, 2),
            model_name="demo-slice-heuristic-NOT-FOR-PRODUCTION",
            inference_time_s=round(duration, 3),
        )

    def _segment_slice(self, slice_2d: np.ndarray) -> np.ndarray:
        """
        Segmentation heuristique d'une coupe 2D cardiaque.
        Utilise le seuillage Otsu + analyse de composantes connexes.
        """
        from scipy import ndimage

        H, W = slice_2d.shape
        mask = np.zeros((H, W), dtype=np.uint8)

        # Ignorer les coupes vides (hors champ cardiaque)
        if slice_2d.std() < 0.1:
            return mask

        # Seuillage : zone brillante = sang (LV/RV), zone intermédiaire = myocarde
        # Les valeurs sont z-score normalisées
        p25 = np.percentile(slice_2d, 25)
        p50 = np.percentile(slice_2d, 50)
        p75 = np.percentile(slice_2d, 75)
        p90 = np.percentile(slice_2d, 90)

        # Masque binaire des structures cardiaques (> médiane)
        cardiac_mask = slice_2d > p50

        # Nettoyer les petits artefacts
        cardiac_mask = ndimage.binary_fill_holes(cardiac_mask)
        cardiac_mask = ndimage.binary_erosion(cardiac_mask, iterations=2)
        cardiac_mask = ndimage.binary_dilation(cardiac_mask, iterations=2)

        # Labellisation des composantes connexes
        labeled, n_components = ndimage.label(cardiac_mask)

        if n_components == 0:
            return mask

        # Trier par taille (plus grande = ventricule)
        component_sizes = [(labeled == i).sum() for i in range(1, n_components + 1)]
        sorted_components = sorted(enumerate(component_sizes, 1), key=lambda x: -x[1])

        # Composante la plus grande → LV (zone brillante centrale)
        if len(sorted_components) >= 1:
            lv_label = sorted_components[0][0]
            lv_region = (labeled == lv_label)

            # LV cavity = zone très brillante dans la région principale
            lv_bright = lv_region & (slice_2d > p75)
            if lv_bright.sum() > 10:
                mask[lv_bright] = CardiacLabel.LV_CAVITY

            # Myocarde = anneau autour du LV (zone intermédiaire)
            lv_dilated = ndimage.binary_dilation(lv_bright, iterations=4)
            myo_region = lv_dilated & ~lv_bright & (slice_2d > p25)
            if myo_region.sum() > 5:
                mask[myo_region] = CardiacLabel.MYOCARDIUM

        # Deuxième composante → RV (adjacent au LV, plus grande)
        if len(sorted_components) >= 2:
            rv_label = sorted_components[1][0]
            rv_region = (labeled == rv_label) & (slice_2d > p50)
            if rv_region.sum() > 10:
                mask[rv_region] = CardiacLabel.RV_CAVITY

        return mask


def dice_score(pred: np.ndarray, target: np.ndarray, label: int) -> float:
    pred_bin   = (pred == label).astype(np.float32)
    target_bin = (target == label).astype(np.float32)
    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum()
    if union == 0:
        return 1.0
    return float(2.0 * intersection / union)


def get_segmenter(model_dir=None):
    logger.warning("segmenter.demo_mode", msg="⚠️ DemoSegmenter — non clinique")
    return DemoSegmenter()
