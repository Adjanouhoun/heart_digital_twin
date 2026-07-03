"""
Convertit le dataset M&Ms au format nnU-Net v2.
Dataset ID : 028 (M&Ms)

Labels M&Ms :
  0 = Background
  1 = LV cavity
  2 = Myocardium
  3 = RV cavity

Structure source M&Ms :
  MnMs_dataset/
    Training/
      Labeled/
        {vendor}/
          {patient}/
            {patient}_sa_gt.nii.gz   ← segmentation
            {patient}_sa.nii.gz      ← IRM

Usage :
  python scripts/prepare_mnms_nnunet.py --mnms_dir ~/Downloads/MnMs_dataset
"""
import json
import shutil
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnms_dir", type=Path,
                        default=Path.home() / "Downloads/MnMs_dataset")
    args = parser.parse_args()

    OUT_DIR = Path.home() / "nnunet/raw/Dataset028_MnMs"
    (OUT_DIR / "imagesTr").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "labelsTr").mkdir(parents=True, exist_ok=True)

    cases = []
    labeled_dir = args.mnms_dir / "Training" / "Labeled"

    if not labeled_dir.exists():
        # Essayer structure alternative
        labeled_dir = args.mnms_dir / "Labeled"
    if not labeled_dir.exists():
        labeled_dir = args.mnms_dir

    # Chercher tous les fichiers _sa.nii.gz récursivement
    nifti_files = sorted(labeled_dir.rglob("*_sa.nii.gz"))
    # Exclure les GT
    nifti_files = [f for f in nifti_files if "_gt" not in f.name]

    print(f"Fichiers IRM trouvés : {len(nifti_files)}")

    for nifti_path in nifti_files:
        gt_path = nifti_path.parent / nifti_path.name.replace("_sa.nii.gz", "_sa_gt.nii.gz")

        if not gt_path.exists():
            print(f"⚠️  GT manquant pour {nifti_path.name} — ignoré")
            continue

        # Extraire l'ID patient
        patient_id = nifti_path.stem.replace("_sa", "").replace(".nii", "")

        dst_img = OUT_DIR / "imagesTr" / f"{patient_id}_0000.nii.gz"
        dst_gt  = OUT_DIR / "labelsTr" / f"{patient_id}.nii.gz"

        shutil.copy2(nifti_path, dst_img)
        shutil.copy2(gt_path, dst_gt)
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
        "name": "MnMs",
        "description": "Multi-Centre Multi-Vendor Cardiac Segmentation Challenge",
        "reference": "https://www.ub.edu/mnms/",
        "licence": "CC BY-NC-SA 4.0",
        "release": "1.0",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }

    with open(OUT_DIR / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\n✅ Dataset M&Ms préparé : {len(cases)} cas")
    print(f"📁 {OUT_DIR}")

if __name__ == "__main__":
    main()
