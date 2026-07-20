"""Audit élémentaire des champs historique et Bayer 2012 du Sprint 2."""
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MESH_DIR = ROOT / "reports/meshes_acdc/meshes"
PATIENT = "patient001_coarse5_fixed"
OUT = ROOT / "sprint_artifacts/sprint2/patient001_element_microstructure_audit.json"


def read_elem(path):
    with path.open() as stream:
        count = int(stream.readline())
        return np.array([list(map(int, stream.readline().split()[1:5]))
                         for _ in range(count)], dtype=np.int64)


def angle_deg(a, b, unoriented=False):
    dot = np.sum(a * b, axis=1)
    if unoriented:
        dot = np.abs(dot)
    return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))


elements = read_elem(MESH_DIR / f"{PATIENT}.elem")

# Paires d'éléments partageant une facette.
face_owner = {}
neighbor_pairs = []
for element_index, tet in enumerate(elements):
    for omitted in range(4):
        face = tuple(sorted(np.delete(tet, omitted)))
        if face in face_owner:
            neighbor_pairs.append((face_owner[face], element_index))
        else:
            face_owner[face] = element_index
neighbor_pairs = np.asarray(neighbor_pairs, dtype=np.int64)

fields = {
    "historical_proxy": MESH_DIR / f"{PATIENT}_fibers_ldrb.lon",
    "bayer2012": MESH_DIR / f"{PATIENT}_fibers_bayer2012.lon",
}
report = {
    "patient": PATIENT,
    "tetrahedra": int(len(elements)),
    "interior_facet_pairs": int(len(neighbor_pairs)),
    "fields": {},
}

for name, path in fields.items():
    field = np.loadtxt(path, skiprows=1)
    fiber_average = field[elements, :3].mean(axis=1)
    sheet_average = field[elements, 3:6].mean(axis=1)
    fiber_average_norm = np.linalg.norm(fiber_average, axis=1)
    sheet_average_norm = np.linalg.norm(sheet_average, axis=1)
    fiber = fiber_average / fiber_average_norm[:, None]
    sheet = sheet_average / sheet_average_norm[:, None]
    dot_fs = np.abs(np.sum(fiber * sheet, axis=1))
    fiber_jumps = angle_deg(
        fiber[neighbor_pairs[:, 0]], fiber[neighbor_pairs[:, 1]], unoriented=True
    )
    sheet_jumps = angle_deg(
        sheet[neighbor_pairs[:, 0]], sheet[neighbor_pairs[:, 1]], unoriented=True
    )
    report["fields"][name] = {
        "fiber_average_norm": {
            "min": float(fiber_average_norm.min()),
            "p01": float(np.percentile(fiber_average_norm, 1)),
            "median": float(np.median(fiber_average_norm)),
            "lt_0_5_pct": float(np.mean(fiber_average_norm < 0.5) * 100),
            "lt_0_1_count": int(np.sum(fiber_average_norm < 0.1)),
        },
        "sheet_average_norm": {
            "min": float(sheet_average_norm.min()),
            "p01": float(np.percentile(sheet_average_norm, 1)),
            "median": float(np.median(sheet_average_norm)),
            "lt_0_5_pct": float(np.mean(sheet_average_norm < 0.5) * 100),
            "lt_0_1_count": int(np.sum(sheet_average_norm < 0.1)),
        },
        "normalized_element_fiber_sheet_abs_dot": {
            "median": float(np.median(dot_fs)),
            "p95": float(np.percentile(dot_fs, 95)),
            "max": float(dot_fs.max()),
            "gt_0_1_pct": float(np.mean(dot_fs > 0.1) * 100),
        },
        "neighbor_fiber_axis_jump_deg": {
            "median": float(np.median(fiber_jumps)),
            "p95": float(np.percentile(fiber_jumps, 95)),
            "max": float(fiber_jumps.max()),
            "gt_30_pct": float(np.mean(fiber_jumps > 30) * 100),
            "gt_60_pct": float(np.mean(fiber_jumps > 60) * 100),
        },
        "neighbor_sheet_axis_jump_deg": {
            "median": float(np.median(sheet_jumps)),
            "p95": float(np.percentile(sheet_jumps, 95)),
            "max": float(sheet_jumps.max()),
            "gt_30_pct": float(np.mean(sheet_jumps > 30) * 100),
            "gt_60_pct": float(np.mean(sheet_jumps > 60) * 100),
        },
    }

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
