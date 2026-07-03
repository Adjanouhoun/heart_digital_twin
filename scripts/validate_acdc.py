"""
Validation Phase 02 sur dataset ACDC.
Mesure les Dice scores sur 10 patients (end-diastole).
SLOs : Dice MYO >= 0.90, Dice LV >= 0.90, Dice RV >= 0.85

Usage :
    python scripts/validate_acdc.py --acdc_dir ~/Downloads/ACDC/database/training --n_patients 10
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Labels ACDC ───────────────────────────────────────────────────────────────
ACDC_LV  = 3   # Cavité LV
ACDC_MYO = 2   # Myocarde
ACDC_RV  = 1   # Cavité RV

# ── Labels CDT ────────────────────────────────────────────────────────────────
CDT_LV   = 1
CDT_MYO  = 2
CDT_RV   = 3


def dice_score(pred: np.ndarray, gt: np.ndarray, label_pred: int, label_gt: int) -> float:
    """Calcule le Dice entre prédiction et ground truth."""
    pred_bin = (pred == label_pred).astype(np.float32)
    gt_bin   = (gt   == label_gt).astype(np.float32)
    intersection = (pred_bin * gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum()
    if union == 0:
        return 1.0
    return float(2.0 * intersection / union)


def load_nifti(path: Path) -> tuple[np.ndarray, tuple]:
    """Charge un fichier NIfTI et retourne (array, spacing)."""
    img = nib.load(str(path))
    data = img.get_fdata().astype(np.float32)
    zooms = img.header.get_zooms()
    spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    return data, spacing


def validate_patient(patient_dir: Path, segmenter) -> dict:
    """
    Valide le pipeline sur un patient ACDC.
    Utilise le frame end-diastole (_frame01 ou frame avec index le plus bas).
    """
    patient_id = patient_dir.name

    # Trouver les fichiers
    nifti_files = sorted(patient_dir.glob("*_frame*[0-9].nii.gz"))
    gt_files    = sorted(patient_dir.glob("*_frame*_gt.nii.gz"))

    if not nifti_files or not gt_files:
        return {"patient": patient_id, "error": "Fichiers manquants"}

    # Prendre le premier frame (end-diastole)
    nifti_path = nifti_files[0]
    gt_path    = gt_files[0]

    print(f"\n{'='*60}")
    print(f"Patient : {patient_id}")
    print(f"IRM     : {nifti_path.name}")
    print(f"GT      : {gt_path.name}")

    # Charger le volume et la ground truth
    volume, spacing = load_nifti(nifti_path)
    gt, _           = load_nifti(gt_path)
    gt = gt.astype(np.uint8)

    # Normaliser le volume (z-score)
    mean, std = volume.mean(), volume.std()
    if std > 1e-8:
        volume = (volume - mean) / std

    # Si 4D → prendre le premier volume
    if volume.ndim == 4:
        volume = volume[..., 0]
    if gt.ndim == 4:
        gt = gt[..., 0]

    print(f"Volume  : {volume.shape} @ spacing={tuple(round(s,2) for s in spacing)}")

    # Segmentation
    t0 = time.time()
    result = segmenter.predict(volume, spacing)
    duration = time.time() - t0

    # Calcul des Dice scores
    dice_lv  = dice_score(result.mask, gt, CDT_LV,  ACDC_LV)
    dice_myo = dice_score(result.mask, gt, CDT_MYO, ACDC_MYO)
    dice_rv  = dice_score(result.mask, gt, CDT_RV,  ACDC_RV)

    # SLOs
    slo_lv  = "✅" if dice_lv  >= 0.90 else "❌"
    slo_myo = "✅" if dice_myo >= 0.90 else "❌"
    slo_rv  = "✅" if dice_rv  >= 0.85 else "❌"

    print(f"Dice LV  : {dice_lv:.4f}  {slo_lv}  (SLO ≥ 0.90)")
    print(f"Dice MYO : {dice_myo:.4f}  {slo_myo}  (SLO ≥ 0.90)")
    print(f"Dice RV  : {dice_rv:.4f}  {slo_rv}  (SLO ≥ 0.85)")
    print(f"Durée    : {duration:.2f}s")
    print(f"LV={result.volume_lv_ml:.1f}mL  MYO={result.volume_myo_ml:.1f}mL  SCAR={result.scar_burden_pct:.1f}%")

    return {
        "patient": patient_id,
        "nifti": str(nifti_path.name),
        "shape": list(volume.shape),
        "spacing": list(spacing),
        "dice_lv": round(dice_lv, 4),
        "dice_myo": round(dice_myo, 4),
        "dice_rv": round(dice_rv, 4),
        "volume_lv_ml": result.volume_lv_ml,
        "volume_myo_ml": result.volume_myo_ml,
        "scar_burden_pct": result.scar_burden_pct,
        "duration_s": round(duration, 2),
        "slo_lv": dice_lv >= 0.90,
        "slo_myo": dice_myo >= 0.90,
        "slo_rv": dice_rv >= 0.85,
    }


def main():
    parser = argparse.ArgumentParser(description="Validation ACDC Phase 02")
    parser.add_argument("--acdc_dir", type=Path,
                        default=Path.home() / "Downloads/ACDC/database/training")
    parser.add_argument("--n_patients", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("reports/acdc_validation.json"))
    args = parser.parse_args()

    print("=" * 60)
    print("CDT Phase 02 — Validation ACDC")
    print(f"Dataset : {args.acdc_dir}")
    print(f"Patients : {args.n_patients}")
    print("=" * 60)

    # Charger le segmenteur
    from app.segmentation.nnunet_wrapper import get_segmenter
    segmenter = get_segmenter()
    print(f"Segmenteur : {segmenter.__class__.__name__}")

    # Lister les patients
    patient_dirs = sorted([
        d for d in args.acdc_dir.iterdir()
        if d.is_dir() and d.name.startswith("patient")
    ])[:args.n_patients]

    print(f"Patients trouvés : {len(patient_dirs)}")

    # Validation
    results = []
    for patient_dir in patient_dirs:
        try:
            r = validate_patient(patient_dir, segmenter)
            results.append(r)
        except Exception as e:
            print(f"❌ Erreur {patient_dir.name}: {e}")
            results.append({"patient": patient_dir.name, "error": str(e)})

    # Rapport final
    valid = [r for r in results if "error" not in r]
    print(f"\n{'='*60}")
    print("RAPPORT FINAL — ACDC Validation")
    print(f"{'='*60}")

    if valid:
        mean_lv  = np.mean([r["dice_lv"]  for r in valid])
        mean_myo = np.mean([r["dice_myo"] for r in valid])
        mean_rv  = np.mean([r["dice_rv"]  for r in valid])
        mean_dur = np.mean([r["duration_s"] for r in valid])

        slo_lv_ok  = sum(r["slo_lv"]  for r in valid)
        slo_myo_ok = sum(r["slo_myo"] for r in valid)
        slo_rv_ok  = sum(r["slo_rv"]  for r in valid)
        n = len(valid)

        print(f"\nDice LV  moyen : {mean_lv:.4f}  ({slo_lv_ok}/{n} patients ≥ 0.90)")
        print(f"Dice MYO moyen : {mean_myo:.4f}  ({slo_myo_ok}/{n} patients ≥ 0.90)")
        print(f"Dice RV  moyen : {mean_rv:.4f}  ({slo_rv_ok}/{n} patients ≥ 0.85)")
        print(f"Durée moyenne  : {mean_dur:.2f}s/patient")

        print(f"\nSLOs Phase 02 :")
        print(f"  Dice MYO ≥ 0.90 : {'✅' if mean_myo >= 0.90 else '❌'}  ({mean_myo:.4f})")
        print(f"  Dice LV  ≥ 0.90 : {'✅' if mean_lv  >= 0.90 else '❌'}  ({mean_lv:.4f})")
        print(f"  Dice RV  ≥ 0.85 : {'✅' if mean_rv  >= 0.85 else '❌'}  ({mean_rv:.4f})")
        print(f"  Pipeline < 10min: {'✅' if mean_dur < 600 else '❌'}  ({mean_dur:.1f}s)")

        summary = {
            "n_patients": n,
            "mean_dice_lv":  round(mean_lv,  4),
            "mean_dice_myo": round(mean_myo, 4),
            "mean_dice_rv":  round(mean_rv,  4),
            "mean_duration_s": round(mean_dur, 2),
            "slo_myo_passed": mean_myo >= 0.90,
            "slo_lv_passed":  mean_lv  >= 0.90,
            "slo_rv_passed":  mean_rv  >= 0.85,
            "model": segmenter.__class__.__name__,
        }
    else:
        summary = {"error": "Aucun patient traité avec succès"}

    # Sauvegarder le rapport
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"summary": summary, "patients": results}
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=lambda o: int(o) if isinstance(o, bool) else float(o) if hasattr(o, "__float__") else str(o))
    print(f"\nRapport sauvegardé → {args.output}")


if __name__ == "__main__":
    main()
# patch appliqué en fin de fichier — ne rien ajouter ici
# patch appliqué en fin de fichier — ne rien ajouter ici
