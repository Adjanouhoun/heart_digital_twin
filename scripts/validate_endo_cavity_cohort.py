"""Régénère la validation EDV EndoCavityVolume sur quatre patients ACDC."""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.solver.mechanics.endo_cavity_volume import EndoCavityVolume


MESH_DIR = ROOT / "reports/meshes_acdc/meshes"
SEG_DIR = ROOT / "reports/meshes_acdc/segmentations"
PATIENTS = ("patient001", "patient002", "patient005", "patient008")


def read_pts(path):
    with path.open() as stream:
        count = int(stream.readline())
        return np.array([
            list(map(float, stream.readline().split())) for _ in range(count)
        ], dtype=np.float64)


def read_elem(path):
    with path.open() as stream:
        count = int(stream.readline())
        return np.array([
            list(map(int, stream.readline().split()[1:5])) for _ in range(count)
        ], dtype=np.int64)


results = []
for patient in PATIENTS:
    prefix = MESH_DIR / f"{patient}_coarse5_fixed"
    nodes = read_pts(prefix.with_suffix(".pts"))
    elements = read_elem(prefix.with_suffix(".elem"))
    image = nib.load(str(SEG_DIR / f"{patient}_seg.nii.gz"))
    mask = np.asarray(image.dataobj)
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=float)
    estimator = EndoCavityVolume(mask, spacing, nodes, elements, target_mm=5.0)
    estimated = estimator.volume(np.zeros_like(estimator.endo_verts))
    voxel_volume = float(np.prod(spacing))
    ground_truth = float(np.count_nonzero(mask == 3) * voxel_volume / 1000.0)
    relative_error = 100.0 * (estimated - ground_truth) / ground_truth
    results.append({
        "patient": patient,
        "estimated_edv_mL": estimated,
        "ground_truth_edv_mL": ground_truth,
        "relative_error_pct": relative_error,
        "absolute_relative_error_pct": abs(relative_error),
    })

report = {
    "patients": results,
    "max_absolute_relative_error_pct": max(
        row["absolute_relative_error_pct"] for row in results
    ),
    "mean_absolute_relative_error_pct": float(np.mean([
        row["absolute_relative_error_pct"] for row in results
    ])),
}
output = ROOT / "sprint_artifacts/sprint1/endo_cavity_cohort.json"
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w") as stream:
    json.dump(report, stream, indent=2)
print(json.dumps(report, indent=2))
