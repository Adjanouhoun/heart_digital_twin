"""
Convertit le dataset EMIDEC (LGE-IRM cicatrices) au format nnU-Net v2.
Dataset ID : 030 (EMIDEC — ScarSeg)

Labels EMIDEC :
  0 = Background
  1 = Myocarde sain
  2 = Cicatrice (infarctus)
  3 = Cavité LV

Usage :
  python scripts/prepare_emidec_nnunet.py --emidec_dir ~/Downloads/EMIDEC
  
Téléchargement : https://emidec.com/
"""
import json
import shutil
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emidec_dir", type=Path,
                        default=Path.home() / "Downloads/EMIDEC")
    args = parser.parse_args()

    OUT_DIR = Path.home() / "nnunet/raw/Dataset030_EMIDEC"
    (OUT_DIR / "imagesTr").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "labelsTr").mkdir(parents=True, exist_ok=True)

    cases = []
    nifti_files = sorted(args.emidec_dir.rglob("*_img.nii.gz"))

    print(f"Fichiers LGE-IRM EMIDEC trouvés : {len(nifti_files)}")

    for nifti_path in nifti_files:
        gt_path = Path(str(nifti_path).replace("_img.nii.gz", "_gt.nii.gz"))
        if not gt_path.exists():
            print(f"⚠️  GT manquant pour {nifti_path.name}")
            continue

        patient_id = nifti_path.stem.replace("_img", "").replace(".nii", "")
        dst_img = OUT_DIR / "imagesTr" / f"{patient_id}_0000.nii.gz"
        dst_gt  = OUT_DIR / "labelsTr" / f"{patient_id}.nii.gz"

        shutil.copy2(nifti_path, dst_img)
        shutil.copy2(gt_path, dst_gt)
        cases.append(patient_id)
        print(f"✅ {patient_id}")

    dataset_json = {
        "channel_names": {"0": "LGE-MRI"},
        "labels": {
            "background": 0,
            "myocardium_healthy": 1,
            "scar": 2,
            "lv_cavity": 3,
        },
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
        "name": "EMIDEC",
        "description": "Emidec — LGE-MRI Cardiac Scar Segmentation",
        "reference": "https://emidec.com/",
        "release": "1.0",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }

    with open(OUT_DIR / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\n✅ Dataset EMIDEC préparé : {len(cases)} cas")
    print(f"📁 {OUT_DIR}")

if __name__ == "__main__":
    main()
