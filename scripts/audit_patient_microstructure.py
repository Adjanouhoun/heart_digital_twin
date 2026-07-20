"""Audit Sprint 2 du champ de microstructure de patient001.

Compare le proxy transmural utilisé par ``LDRBFiberGenerator`` aux distances
physiques vers les cavités et le fond de la segmentation. Aucun champ n'est
modifié et aucun solveur mécanique n'est lancé.
"""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
MESH = ROOT / "reports/meshes_acdc/meshes"
SEG_PATH = ROOT / "reports/meshes_acdc/segmentations/patient001_seg.nii.gz"
PATIENT = "patient001_coarse5_fixed"
OUT = ROOT / "sprint_artifacts/sprint2/patient001_microstructure_audit.json"


def read_pts(path):
    with path.open() as stream:
        count = int(stream.readline())
        return np.array([list(map(float, stream.readline().split()))
                         for _ in range(count)], dtype=np.float64)


nodes = read_pts(MESH / f"{PATIENT}.pts")
field = np.loadtxt(MESH / f"{PATIENT}_fibers_ldrb.lon", skiprows=1)
image = nib.load(SEG_PATH)
seg = np.asarray(image.dataobj)
spacing = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)

# Distance physique à la cavité la plus proche (RV=1, LV=3) et au fond
# extérieur. distance_transform_edt calcule la distance jusqu'au zéro le plus
# proche ; les masques sont donc inversés pour viser chaque interface.
cavity = (seg == 1) | (seg == 3)
background = seg == 0
d_endo_grid = ndimage.distance_transform_edt(~cavity, sampling=spacing)
d_epi_grid = ndimage.distance_transform_edt(~background, sampling=spacing)
coords = (nodes / spacing).T
d_endo = ndimage.map_coordinates(d_endo_grid, coords, order=1, mode="nearest")
d_epi = ndimage.map_coordinates(d_epi_grid, coords, order=1, mode="nearest")
denom = d_endo + d_epi
valid = denom > 1e-8
t_seg = np.full(len(nodes), np.nan)
t_seg[valid] = d_endo[valid] / denom[valid]

center = nodes.mean(axis=0)
radius = np.linalg.norm(nodes - center, axis=1)
t_proxy = (radius - radius.min()) / (radius.max() - radius.min())

f = field[:, :3]
s = field[:, 3:6]
alpha_proxy_deg = 60.0 - 120.0 * t_proxy
alpha_seg_deg = 60.0 - 120.0 * t_seg
delta_t = t_proxy[valid] - t_seg[valid]
delta_alpha = alpha_proxy_deg[valid] - alpha_seg_deg[valid]

summary = {
    "patient": PATIENT,
    "source_contract": "rule-based (LDRB) fibre generation",
    "implementation": {
        "apicobasal_coordinate": "normalised global z",
        "transmural_coordinate": "normalised distance to global node centroid",
        "laplace_problems_solved": 0,
        "element_tags_used_by_transmural_solver": False,
    },
    "counts": {
        "nodes": int(len(nodes)),
        "valid_segmentation_distance_nodes": int(valid.sum()),
    },
    "basis_qc": {
        "fiber_norm_max_abs_error": float(np.max(np.abs(np.linalg.norm(f, axis=1) - 1))),
        "sheet_norm_max_abs_error": float(np.max(np.abs(np.linalg.norm(s, axis=1) - 1))),
        "fiber_sheet_dot_max_abs": float(np.max(np.abs(np.sum(f * s, axis=1)))),
    },
    "transmural_proxy_vs_segmentation": {
        "pearson_correlation": float(np.corrcoef(t_proxy[valid], t_seg[valid])[0, 1]),
        "mean_absolute_coordinate_error": float(np.mean(np.abs(delta_t))),
        "p95_absolute_coordinate_error": float(np.percentile(np.abs(delta_t), 95)),
        "nodes_coordinate_error_gt_0_25_pct": float(np.mean(np.abs(delta_t) > 0.25) * 100),
        "mean_absolute_helix_angle_error_deg": float(np.mean(np.abs(delta_alpha))),
        "p95_absolute_helix_angle_error_deg": float(np.percentile(np.abs(delta_alpha), 95)),
        "nodes_helix_angle_error_gt_30deg_pct": float(np.mean(np.abs(delta_alpha) > 30) * 100),
    },
    "verdict": "not_a_validated_LDRB_field",
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
