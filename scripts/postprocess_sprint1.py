"""Post-traitement hôte du Sprint 1 avec EndoCavityVolume."""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.solver.mechanics.endo_cavity_volume import EndoCavityVolume


ARTIFACT_DIR = ROOT / "sprint_artifacts/sprint1"
SEG_PATH = ROOT / "reports/meshes_acdc/segmentations/patient001_seg.nii.gz"

fields = np.load(ARTIFACT_DIR / "fields.npz")
nodes = fields["nodes"]
elements = fields["elements"]
displacement = fields["displacement_mm"]

seg_img = nib.load(str(SEG_PATH))
segmentation = np.asarray(seg_img.dataobj)
spacing_mm = np.asarray(seg_img.header.get_zooms()[:3], dtype=float)
cavity = EndoCavityVolume(
    segmentation, spacing_mm, nodes, elements, target_mm=5.0
)
volume_result = cavity.ejection_fraction(displacement)

result_path = ARTIFACT_DIR / "result.json"
with result_path.open() as stream:
    summary = json.load(stream)
summary["cavity"] = {
    "status": "complete",
    "V_ed_mL": float(volume_result["V_ed"]),
    "V_es_mL": float(volume_result["V_es"]),
    "EF_pct": float(volume_result["EF_pct"]),
    "reference_EF_pct": 23.65,
    "absolute_error_points": float(abs(volume_result["EF_pct"] - 23.65)),
}
with result_path.open("w") as stream:
    json.dump(summary, stream, indent=2)
print(json.dumps(summary, indent=2))
