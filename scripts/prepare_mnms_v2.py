"""
Prépare M&Ms pour transfer learning depuis ACDC.
- Extrait les frames ED/ES annotés (4D → 3D)
- Remappe les labels : M&Ms (1=LV,2=MYO,3=RV) → ACDC (1=RV,2=MYO,3=LV)
- Dataset ID : 028
"""
import nibabel as nib
import numpy as np
import json
import os
from pathlib import Path

def remap_labels(seg):
    """M&Ms: 1=LV,2=MYO,3=RV → ACDC: 1=RV,2=MYO,3=LV"""
    out = np.zeros_like(seg)
    out[seg == 1] = 3   # LV → 3
    out[seg == 2] = 2   # MYO → 2 (same)
    out[seg == 3] = 1   # RV → 1
    return out

def main():
    SRC = Path.home() / "nnunet/raw/Dataset028_MnMs"
    OUT = Path.home() / "nnunet/raw/Dataset028_MnMs_v2"
    (OUT / "imagesTr").mkdir(parents=True, exist_ok=True)
    (OUT / "labelsTr").mkdir(parents=True, exist_ok=True)

    img_dir = SRC / "imagesTr"
    lbl_dir = SRC / "labelsTr"

    cases = []
    img_files = sorted(img_dir.glob("*_0000.nii.gz"))

    print(f"Fichiers IRM trouvés : {len(img_files)}")

    for img_path in img_files:
        patient_id = img_path.name.replace("_0000.nii.gz", "")
        lbl_path = lbl_dir / f"{patient_id}.nii.gz"

        if not lbl_path.exists():
            print(f"  ⚠ GT manquant pour {patient_id}")
            continue

        img_nii = nib.load(str(img_path))
        lbl_nii = nib.load(str(lbl_path))
        img_data = img_nii.get_fdata()
        lbl_data = lbl_nii.get_fdata()

        # Trouver les frames annotés
        if img_data.ndim == 4:
            for t in range(lbl_data.shape[3]):
                frame_lbl = lbl_data[:,:,:,t]
                if (frame_lbl > 0).sum() > 100:
                    frame_img = img_data[:,:,:,t]
                    frame_lbl = remap_labels(frame_lbl)

                    case_id = f"{patient_id}_frame{t:02d}"

                    # Sauvegarder comme 3D
                    affine = img_nii.affine
                    img_3d = nib.Nifti1Image(frame_img.astype(np.float32), affine)
                    lbl_3d = nib.Nifti1Image(frame_lbl.astype(np.int16), affine)

                    nib.save(img_3d, str(OUT / "imagesTr" / f"{case_id}_0000.nii.gz"))
                    nib.save(lbl_3d, str(OUT / "labelsTr" / f"{case_id}.nii.gz"))

                    cases.append(case_id)
                    print(f"  ✅ {case_id} (frame {t})")
        else:
            # Déjà 3D
            lbl_data = remap_labels(lbl_data)
            nib.save(nib.Nifti1Image(img_data.astype(np.float32), img_nii.affine),
                     str(OUT / "imagesTr" / f"{patient_id}_0000.nii.gz"))
            nib.save(nib.Nifti1Image(lbl_data.astype(np.int16), lbl_nii.affine),
                     str(OUT / "labelsTr" / f"{patient_id}.nii.gz"))
            cases.append(patient_id)
            print(f"  ✅ {patient_id} (3D)")

    dataset_json = {
        "channel_names": {"0": "MRI"},
        "labels": {
            "background": 0,
            "RV": 1,
            "MYO": 2,
            "LV": 3,
        },
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
        "name": "MnMs_v2",
        "description": "M&Ms remapped to ACDC label convention for transfer learning",
    }
    with open(OUT / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\n✅ Dataset M&Ms v2 préparé : {len(cases)} cas (ED+ES)")
    print(f"   Labels remappés : 1=RV, 2=MYO, 3=LV (comme ACDC)")
    print(f"   📁 {OUT}")

if __name__ == "__main__":
    main()
