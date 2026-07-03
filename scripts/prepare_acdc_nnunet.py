"""
Convertit le dataset ACDC au format nnU-Net v2.
Structure cible :
  ~/nnunet/raw/Dataset027_ACDC/
    dataset.json
    imagesTr/  ← volumes IRM (.nii.gz)
    labelsTr/  ← segmentations GT (.nii.gz)

Labels ACDC → nnU-Net :
  0 = Background
  1 = RV cavity
  2 = Myocardium
  3 = LV cavity
"""
import json
import shutil
import os
from pathlib import Path

ACDC_DIR = Path.home() / "Downloads/ACDC/database/training"
OUT_DIR  = Path.home() / "nnunet/raw/Dataset027_ACDC"

(OUT_DIR / "imagesTr").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "labelsTr").mkdir(parents=True, exist_ok=True)

patient_dirs = sorted([d for d in ACDC_DIR.iterdir() if d.name.startswith("patient")])

cases = []
for patient_dir in patient_dirs:
    patient_id = patient_dir.name  # patient001, patient002...

    # Prendre les deux frames (ED + ES) pour maximiser les données
    frames = sorted(patient_dir.glob("*_frame*[0-9].nii.gz"))
    gts    = sorted(patient_dir.glob("*_frame*_gt.nii.gz"))

    for nifti_path, gt_path in zip(frames, gts):
        # Extraire le numéro de frame
        frame_num = nifti_path.stem.split("_frame")[1].split(".")[0]
        case_id = f"{patient_id}_frame{frame_num}"

        # Copier l'image (nnU-Net v2 : _0000 = channel 0)
        dst_img = OUT_DIR / "imagesTr" / f"{case_id}_0000.nii.gz"
        dst_gt  = OUT_DIR / "labelsTr" / f"{case_id}.nii.gz"

        shutil.copy2(nifti_path, dst_img)
        shutil.copy2(gt_path, dst_gt)
        cases.append(case_id)
        print(f"✅ {case_id}")

# dataset.json requis par nnU-Net v2
dataset_json = {
    "channel_names": {"0": "MRI"},
    "labels": {
        "background": 0,
        "RV":  1,
        "MYO": 2,
        "LV":  3,
    },
    "numTraining": len(cases),
    "file_ending": ".nii.gz",
    "name": "ACDC",
    "description": "Automated Cardiac Diagnosis Challenge",
    "reference": "https://acdc.creatis.insa-lyon.fr",
    "licence": "CC BY-NC-ND 4.0",
    "release": "1.0",
    "overwrite_image_reader_writer": "SimpleITKIO",
}

with open(OUT_DIR / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=2)

print(f"\n✅ Dataset ACDC préparé : {len(cases)} cas")
print(f"📁 {OUT_DIR}")
