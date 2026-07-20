"""Audit spatial P1 des champs mécaniques patient sauvegardés.

Le solveur utilise un déplacement P2. Les archives ne contiennent que les
valeurs aux noeuds du maillage : les gradients reconstruits ici sont donc des
gradients affines P1 diagnostiques, et non une reproduction du J interne P2.
"""
import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--case-1", required=True, type=Path)
parser.add_argument("--case-2", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--base-depth-mm", type=float, default=5.0)
args = parser.parse_args()


def element_kinematics(archive):
    data = np.load(archive)
    x = data["nodes"]
    t = data["elements"]
    y = data["deformed_nodes"]
    X = np.stack((x[t[:, 1]] - x[t[:, 0]],
                  x[t[:, 2]] - x[t[:, 0]],
                  x[t[:, 3]] - x[t[:, 0]]), axis=2)
    Y = np.stack((y[t[:, 1]] - y[t[:, 0]],
                  y[t[:, 2]] - y[t[:, 0]],
                  y[t[:, 3]] - y[t[:, 0]]), axis=2)
    F = Y @ np.linalg.inv(X)
    J = np.linalg.det(F)
    stretches = np.linalg.svd(F, compute_uv=False)
    return x, t, F, J, stretches


x1, elements1, _, j1, stretches1 = element_kinematics(args.case_1)
x2, elements2, _, j2, stretches2 = element_kinematics(args.case_2)
if not np.array_equal(elements1, elements2) or not np.allclose(x1, x2):
    raise SystemExit("Les deux archives ne partagent pas le même maillage.")

centroids = x1[elements1].mean(axis=1)
reference_edges = np.stack(
    [np.linalg.norm(x1[elements1[:, i]] - x1[elements1[:, j]], axis=1)
     for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))],
    axis=1,
)
reference_matrices = np.stack(
    (x1[elements1[:, 1]] - x1[elements1[:, 0]],
     x1[elements1[:, 2]] - x1[elements1[:, 0]],
     x1[elements1[:, 3]] - x1[elements1[:, 0]]), axis=2,
)
reference_condition = np.linalg.cond(reference_matrices)
reference_volume = np.abs(np.linalg.det(reference_matrices)) / 6.0
zmax = float(x1[:, 2].max())
base_mask_nodes = x1[:, 2] > zmax - args.base_depth_mm
base_nodes = x1[base_mask_nodes]
order = np.lexsort((base_nodes[:, 1], base_nodes[:, 0]))
anchor_a = base_nodes[order[0]]
distances = np.linalg.norm(base_nodes[:, :2] - anchor_a[:2], axis=1)
anchor_b = base_nodes[int(np.argmax(distances))]

distance_to_base = np.maximum(0.0, zmax - args.base_depth_mm - centroids[:, 2])
distance_to_a = np.linalg.norm(centroids - anchor_a, axis=1)
distance_to_b = np.linalg.norm(centroids - anchor_b, axis=1)
distance_to_anchor = np.minimum(distance_to_a, distance_to_b)


def case_summary(j, stretches):
    order_j = np.argsort(j)
    critical_count = max(1, int(np.ceil(0.01 * len(j))))
    critical = order_j[:critical_count]
    lowest = order_j[:10]
    nonpositive = j <= 0
    return {
        "p1_reconstructed_j": {
            "min": float(j.min()),
            "p01": float(np.percentile(j, 1)),
            "p05": float(np.percentile(j, 5)),
            "median": float(np.median(j)),
            "max": float(j.max()),
            "count_nonpositive": int(np.count_nonzero(j <= 0)),
        },
        "principal_stretches": {
            "minimum_over_elements": float(stretches.min()),
            "maximum_over_elements": float(stretches.max()),
        },
        "reference_quality_of_nonpositive_elements": {
            "count": int(nonpositive.sum()),
            "condition_number_median": (
                float(np.median(reference_condition[nonpositive]))
                if nonpositive.any() else None
            ),
            "condition_number_over_50_fraction": (
                float(np.mean(reference_condition[nonpositive] > 50.0))
                if nonpositive.any() else None
            ),
            "volume_mm3_median": (
                float(np.median(reference_volume[nonpositive]))
                if nonpositive.any() else None
            ),
        },
        "lowest_1_percent": {
            "element_count": int(critical_count),
            "centroid_z_mm_range": [float(centroids[critical, 2].min()),
                                     float(centroids[critical, 2].max())],
            "distance_to_base_plane_mm": {
                "min": float(distance_to_base[critical].min()),
                "median": float(np.median(distance_to_base[critical])),
            },
            "distance_to_nearest_anchor_mm": {
                "min": float(distance_to_anchor[critical].min()),
                "median": float(np.median(distance_to_anchor[critical])),
            },
            "fraction_inside_basal_bc_layer": float(
                np.mean(centroids[critical, 2] > zmax - args.base_depth_mm)
            ),
        },
        "ten_lowest_elements": [
            {
                "element": int(i),
                "J": float(j[i]),
                "centroid_mm": centroids[i].tolist(),
                "distance_to_base_plane_mm": float(distance_to_base[i]),
                "distance_to_nearest_anchor_mm": float(distance_to_anchor[i]),
            }
            for i in lowest
        ],
    }


report = {
    "method": "affine_P1_reconstruction_from_saved_nodal_displacements",
    "limitation": (
        "Diagnostic spatial uniquement: le J interne du solveur provient du "
        "champ P2 et ne peut pas être reproduit exactement avec cette archive."
    ),
    "mesh": {"nodes": int(len(x1)), "elements": int(len(elements1))},
    "reference_mesh_quality": {
        "tetra_volume_mm3": {
            "min": float(reference_volume.min()),
            "p01": float(np.percentile(reference_volume, 1)),
            "median": float(np.median(reference_volume)),
        },
        "edge_length_mm": {
            "min": float(reference_edges.min()),
            "max": float(reference_edges.max()),
        },
        "edge_matrix_condition_number": {
            "median": float(np.median(reference_condition)),
            "p95": float(np.percentile(reference_condition, 95)),
            "p99": float(np.percentile(reference_condition, 99)),
            "max": float(reference_condition.max()),
            "fraction_over_50": float(np.mean(reference_condition > 50.0)),
        },
    },
    "boundary_conditions": {
        "base_depth_mm": args.base_depth_mm,
        "base_z_threshold_mm": zmax - args.base_depth_mm,
        "anchor_a_mm": anchor_a.tolist(),
        "anchor_b_mm": anchor_b.tolist(),
    },
    "case_1_kPa": case_summary(j1, stretches1),
    "case_2_kPa": case_summary(j2, stretches2),
    "comparison": {
        "pearson_j": float(np.corrcoef(j1, j2)[0, 1]),
        "lowest_1_percent_overlap_fraction": float(
            len(set(np.argsort(j1)[:max(1, int(np.ceil(.01 * len(j1))))]) &
                set(np.argsort(j2)[:max(1, int(np.ceil(.01 * len(j2))))])) /
            max(1, int(np.ceil(.01 * len(j1))))
        ),
    },
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
