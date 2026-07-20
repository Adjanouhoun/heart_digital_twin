"""Audit géométrique des marqueurs de facettes utilisés pour Bayer 2012."""
import json
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


ROOT = Path(__file__).resolve().parents[1]
MESH_DIR = ROOT / "reports/meshes_acdc/meshes"
PATIENT = "patient001_coarse5_fixed"
SEG_PATH = ROOT / "reports/meshes_acdc/segmentations/patient001_seg.nii.gz"
OUT = ROOT / "sprint_artifacts/sprint2/patient001_bayer_marker_audit.json"
NAMES = {10: "base", 20: "rv", 30: "lv", 40: "epi"}


def read_pts(path):
    with path.open() as stream:
        count = int(stream.readline())
        return np.array([list(map(float, stream.readline().split()))
                         for _ in range(count)], dtype=np.float64)


def read_elem(path):
    with path.open() as stream:
        count = int(stream.readline())
        return np.array([list(map(int, stream.readline().split()[1:5]))
                         for _ in range(count)], dtype=np.int64)


nodes = read_pts(MESH_DIR / f"{PATIENT}.pts")
elements = read_elem(MESH_DIR / f"{PATIENT}.elem")

# Une facette apparaissant une seule fois est sur la frontière extérieure du
# domaine tétraédrique.
owners = defaultdict(list)
for element_index, tet in enumerate(elements):
    for omitted in range(4):
        owners[tuple(sorted(np.delete(tet, omitted)))].append(element_index)
faces = np.array([face for face, cells in owners.items() if len(cells) == 1],
                 dtype=np.int64)
xyz = nodes[faces]
centroids = xyz.mean(axis=1)
areas = 0.5 * np.linalg.norm(
    np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1
)

image = nib.load(SEG_PATH)
seg = np.asarray(image.dataobj)
spacing = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
distance_grids = {
    20: ndimage.distance_transform_edt(seg != 1, sampling=spacing),
    30: ndimage.distance_transform_edt(seg != 3, sampling=spacing),
    40: ndimage.distance_transform_edt(seg != 0, sampling=spacing),
}
coords = (centroids / spacing).T
candidate_tags = np.array((20, 30, 40), dtype=np.int32)
distances = np.column_stack([
    ndimage.map_coordinates(distance_grids[tag], coords, order=1, mode="nearest")
    for tag in candidate_tags
])
sorted_distances = np.sort(distances, axis=1)
nearest = sorted_distances[:, 0]
margin = sorted_distances[:, 1] - sorted_distances[:, 0]
tags = candidate_tags[np.argmin(distances, axis=1)]
base = centroids[:, 2] > nodes[:, 2].max() - 5.0
tags[base] = 10


def marker_components(selected_faces):
    if len(selected_faces) == 0:
        return []
    edge_owner = defaultdict(list)
    for local_index, face in enumerate(selected_faces):
        for i, j in ((0, 1), (0, 2), (1, 2)):
            edge_owner[tuple(sorted((int(face[i]), int(face[j]))))].append(local_index)
    rows, cols = [], []
    for linked in edge_owner.values():
        for i in linked:
            for j in linked:
                if i != j:
                    rows.append(i)
                    cols.append(j)
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)),
                       shape=(len(selected_faces), len(selected_faces)))
    count, labels = connected_components(graph, directed=False)
    return sorted((int(x) for x in np.bincount(labels)), reverse=True)


report = {
    "patient": PATIENT,
    "segmentation_spacing_mm": spacing.tolist(),
    "boundary_facets": int(len(faces)),
    "markers": {},
}
for tag, name in NAMES.items():
    mask = tags == tag
    components = marker_components(faces[mask])
    item = {
        "facets": int(mask.sum()),
        "area_mm2": float(areas[mask].sum()),
        "components": int(len(components)),
        "component_sizes_desc": components[:10],
        "z_range_mm": [float(centroids[mask, 2].min()),
                       float(centroids[mask, 2].max())],
    }
    if tag != 10:
        item.update({
            "assigned_distance_mm": {
                "median": float(np.median(nearest[mask])),
                "p95": float(np.percentile(nearest[mask], 95)),
                "max": float(nearest[mask].max()),
            },
            "second_choice_margin_mm": {
                "median": float(np.median(margin[mask])),
                "p05": float(np.percentile(margin[mask], 5)),
                "lt_1mm_pct": float(np.mean(margin[mask] < 1.0) * 100),
                "lt_2_5mm_pct": float(np.mean(margin[mask] < 2.5) * 100),
            },
        })
    report["markers"][name] = item

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
