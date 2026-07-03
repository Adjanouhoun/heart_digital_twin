import io
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PreprocessingResult:
    volume: np.ndarray
    spacing_mm: tuple
    origin: tuple
    direction: tuple
    nifti_bytes: bytes
    original_spacing: tuple
    original_size: tuple
    preprocessing_log: list


class CardiacMRIPreprocessor:

    def __init__(self):
        self._target_spacing = 1.0
        self._clip_low = 0.5
        self._clip_high = 99.5

    def preprocess_nifti_bytes(self, nifti_bytes: bytes) -> PreprocessingResult:
        log = []
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
            f.write(nifti_bytes)
            tmp_path = f.name
        image = sitk.ReadImage(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        log.append(f"NIfTI chargé: {original_size} voxels, spacing={original_spacing}")
        return self._process_sitk_image(image, log)

    def preprocess_dicom_series(self, dicom_dir: Path) -> PreprocessingResult:
        log = []
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
        if not dicom_names:
            raise ValueError(f"Aucun fichier DICOM trouvé dans {dicom_dir}")
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        log.append(f"DICOM chargé: {original_size} voxels, spacing={original_spacing}")
        return self._process_sitk_image(image, log)

    def _process_sitk_image(self, image, log):
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        image = sitk.Cast(image, sitk.sitkFloat32)
        image = sitk.DICOMOrient(image, "RAS")
        log.append("Orientation RAS appliquée")
        image = self._resample_isotropic(image, self._target_spacing)
        new_size = image.GetSize()
        log.append(f"Rééchantillonné → {new_size} @ {self._target_spacing}mm³")
        image = self._n4_bias_correction(image)
        log.append("Correction N4ITK appliquée")
        volume = sitk.GetArrayFromImage(image).astype(np.float32)
        volume = self._normalize_intensity(volume)
        log.append(f"Normalisé: min={volume.min():.3f} max={volume.max():.3f}")
        norm_image = sitk.GetImageFromArray(volume)
        norm_image.SetSpacing(image.GetSpacing())
        norm_image.SetOrigin(image.GetOrigin())
        norm_image.SetDirection(image.GetDirection())
        nifti_bytes = self._to_nifti_bytes(norm_image)
        log.append(f"NIfTI sérialisé: {len(nifti_bytes) / 1024:.0f} KB")
        return PreprocessingResult(
            volume=volume,
            spacing_mm=image.GetSpacing(),
            origin=image.GetOrigin(),
            direction=image.GetDirection(),
            nifti_bytes=nifti_bytes,
            original_spacing=original_spacing,
            original_size=original_size,
            preprocessing_log=log,
        )

    def _resample_isotropic(self, image, target_spacing):
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        new_size = [int(round(original_size[i] * original_spacing[i] / target_spacing)) for i in range(3)]
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing([target_spacing] * 3)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(float(sitk.GetArrayViewFromImage(image).min()))
        resampler.SetInterpolator(sitk.sitkBSpline)
        return resampler.Execute(image)

    def _n4_bias_correction(self, image):
        try:
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
            mask = sitk.OtsuThreshold(image, 0, 1, 200)
            return corrector.Execute(image, mask)
        except Exception as e:
            logger.warning("n4_bias.failed", error=str(e))
            return image

    def _normalize_intensity(self, volume):
        p_low = np.percentile(volume, self._clip_low)
        p_high = np.percentile(volume, self._clip_high)
        volume = np.clip(volume, p_low, p_high)
        mean, std = volume.mean(), volume.std()
        if std > 1e-8:
            volume = (volume - mean) / std
        else:
            volume = volume - mean
        return volume.astype(np.float32)

    def _to_nifti_bytes(self, image):
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
            tmp_path = f.name
        sitk.WriteImage(image, tmp_path, useCompression=True)
        data = Path(tmp_path).read_bytes()
        Path(tmp_path).unlink(missing_ok=True)
        return data
