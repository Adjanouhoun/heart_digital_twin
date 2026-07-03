"""
Convertit le dataset MM-WHS (IRM cardiaque uniquement) au format nnU-Net v2.
Dataset ID : 029 (MM-WHS MRI)

Labels MM-WHS :
  0  = Background
  1  = LV myocardium
  2  = LV blood pool
  3  = RV blood pool
  4  = LA blood pool
  5  = RA blood pool
  6  = Aorta
  7  = Pulmonary artery

On garde uniquement LV/MYO/RV pour cohérence avec ACDC/M&Ms :
  0  = Background
  1  = LV blood pool  (label 2 → 1)
  2  = LV myocardium  (label 1 → 2)
  3  = RV blood pool  (label 3 → 3)

Usage :
  python scripts/prepare_mmwhs_nnunet.py --mmwhs_dir ~/Downloads/mmwhs
"""
import json
import shutil
import argparse
import numpy as np
from pathlib import Path

def remap_labels(gt_array: np.ndarray) -> np.ndarray:
    """Remapper les labels MM-WHS → labels CDT (cohérent avec ACDC)."""
    out = np.zeros_like(gt_array)
    out[gt_array == 2] = 1  # LV blood pool → LV cavity
    out[gt_array == 1] = 2  # LV myocardium → Myocardium
    out[gt_array == 3] = 3  # RV blood pool → RV cavity
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmwhs_dir", type=Path,
                        default=Path.home() / "Downloads/mmwhs")
    args = parser.parse_args()

    OUT_DIR = Path.home() / "nnunet/raw/Dataset029_MMWHS"
    (OUT_DIR / "imagesTr").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "labelsTr").mkdir(parents=True, exist_ok=True)

    cases = []

    # Chercher les fichiers IRM MR (pas CT)
    mr_files = sorted(args.mmwhs_dir.rglob("*mr_train_*.nii.gz"))
    mr_files = [f for f in mr_files if "label" not in f.name]

    print(f"Fichiers IRM MM-WHS trouvés : {len(mr_files)}")

    try:
        import nibabel as nib
    except ImportError:
        print("nibabel requis : pip install nibabel")
        return

    for mr_path in mr_files:
        # Trouver le label correspondant
        label_path = mr_path.parent / mr_path.name.replace(
            "mr_train_", "mr_train_label_"
        )
        if not label_path.exists():
            # Essayer autre convention de nommage
            label_path = Path(str(mr_path).replace("image", "label"))

        if not label_path.exists():
            print(f"⚠️  Label manquant pour {mr_path.name} — ignoré")
            continue

        patient_id = mr_path.stem.replace(".nii", "").replace("mr_train_", "mmwhs_")

        # Copier l'image directement
        dst_img = OUT_DIR / "imagesTr" / f"{patient_id}_0000.nii.gz"
        shutil.copy2(mr_path, dst_img)

        # Remapper les labels et sauvegarder
        gt_nib = nib.load(str(label_path))
        gt_array = np.array(gt_nib.dataobj).astype(np.uint8)
        gt_remapped = remap_labels(gt_array)
        gt_new = nib.Nifti1Image(gt_remapped, gt_nib.affine, gt_nib.header)

        dst_gt = OUT_DIR / "labelsTr" / f"{patient_id}.nii.gz"
        nib.save(gt_new, str(dst_gt))

        cases.append(patient_id)
        print(f"✅ {patient_id}")

    dataset_json = {
        "channel_names": {"0": "MRI"},
        "labels": {
            "background": 0,
            "LV": 1,
            "MYO": 2,
            "RV": 3,
        },
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
        "name": "MMWHS_MRI",
        "description": "Multi-Modality Whole Heart Segmentation — MRI only",
        "reference": "http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mmwhs/",
        "licence": "Research only",
        "release": "1.0",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }

    with open(OUT_DIR / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\n✅ Dataset MM-WHS MRI préparé : {len(cases)} cas")
    print(f"📁 {OUT_DIR}")

if __name__ == "__main__":
    main()
