"""Génère un champ LDRB Bayer 2012 sans écraser le champ historique.

Les facettes LV/RV/EPI sont classées par leur distance physique aux labels de
la segmentation ACDC. La base reprend la profondeur basale de 5 mm employée
par la mécanique. Les quatre problèmes de Laplace sont résolus par
``fenicsx-ldrb``.
"""
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
from scipy import ndimage

import basix.ufl
import dolfinx
from mpi4py import MPI
import ldrb
import ufl


ROOT = Path("/cdt")
MESH_DIR = ROOT / "reports/meshes_acdc/meshes"
PATIENT = "patient001_coarse5_fixed"
SEG_PATH = ROOT / "reports/meshes_acdc/segmentations/patient001_seg.nii.gz"
OUT_LON = MESH_DIR / f"{PATIENT}_fibers_bayer2012.lon"
OUT_REPORT = ROOT / "sprint_artifacts/sprint2/patient001_bayer2012_generation.json"
MARKERS = {"base": [10], "rv": [20], "lv": [30], "epi": [40]}


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


started = time.time()
print("stage=load_inputs", flush=True)
nodes = read_pts(MESH_DIR / f"{PATIENT}.pts")
elements = read_elem(MESH_DIR / f"{PATIENT}.elem")
domain = ufl.Mesh(ufl.VectorElement("Lagrange", ufl.tetrahedron, 1))
mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, elements, nodes, domain)
print(f"stage=mesh_created cells={len(elements)} nodes={len(nodes)}", flush=True)
tdim = mesh.topology.dim
mesh.topology.create_connectivity(tdim - 1, 0)
mesh.topology.create_connectivity(tdim - 1, tdim)
exterior = dolfinx.mesh.exterior_facet_indices(mesh.topology)
centroids = dolfinx.mesh.compute_midpoints(mesh, tdim - 1, exterior)

image = nib.load(SEG_PATH)
seg = np.asarray(image.dataobj)
spacing = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
distance_grids = {
    20: ndimage.distance_transform_edt(seg != 1, sampling=spacing),
    30: ndimage.distance_transform_edt(seg != 3, sampling=spacing),
    40: ndimage.distance_transform_edt(seg != 0, sampling=spacing),
}
sample_coords = (centroids / spacing).T
distances = np.column_stack([
    ndimage.map_coordinates(distance_grids[tag], sample_coords,
                            order=1, mode="nearest")
    for tag in (20, 30, 40)
])
values = np.array((20, 30, 40), dtype=np.int32)[np.argmin(distances, axis=1)]
base = centroids[:, 2] > nodes[:, 2].max() - 5.0
values[base] = 10
print("stage=facets_classified", flush=True)

order = np.argsort(exterior)
facet_tags = dolfinx.mesh.meshtags(
    mesh, tdim - 1, exterior[order].astype(np.int32), values[order]
)
counts = {str(tag): int(np.sum(values == tag)) for tag in (10, 20, 30, 40)}
if any(count == 0 for count in counts.values()):
    raise RuntimeError(f"Marqueurs LDRB incomplets: {counts}")
print(f"stage=facet_tags_created counts={counts}", flush=True)

scalar_solutions = ldrb.scalar_laplacians(
    mesh=mesh, ffun=facet_tags, markers=MARKERS
)


def remap_scalar(function):
    coords = function.function_space.tabulate_dof_coordinates()
    coord_to_dof = {tuple(np.round(c, 6)): i for i, c in enumerate(coords)}
    remap = np.array([coord_to_dof[tuple(np.round(n, 6))] for n in nodes])
    return function.x.array[remap]


def nodal_gradient(phi):
    gradient = np.zeros((len(nodes), 3), dtype=np.float64)
    weights = np.zeros(len(nodes), dtype=np.float64)
    for tet in elements:
        xyz = nodes[tet]
        matrix = np.array([xyz[1] - xyz[0], xyz[2] - xyz[0],
                           xyz[3] - xyz[0]]).T
        determinant = np.linalg.det(matrix)
        if abs(determinant) <= 1e-14:
            continue
        delta = np.array([phi[tet[1]] - phi[tet[0]],
                          phi[tet[2]] - phi[tet[0]],
                          phi[tet[3]] - phi[tet[0]]])
        cell_gradient = np.linalg.solve(matrix.T, delta)
        volume = abs(determinant) / 6.0
        gradient[tet] += cell_gradient * volume
        weights[tet] += volume
    valid_nodes = weights > 1e-14
    gradient[valid_nodes] /= weights[valid_nodes, None]
    return gradient


scalar_data = {name: remap_scalar(function)
               for name, function in scalar_solutions.items()}
gradient_data = {
    f"{name}_gradient": nodal_gradient(values).reshape(-1)
    for name, values in scalar_data.items() if name != "lv_rv"
}
gradient_qc = {}
for name in ("lv_gradient", "rv_gradient", "epi_gradient", "apex_gradient"):
    norms = np.linalg.norm(gradient_data[name].reshape(-1, 3), axis=1)
    gradient_qc[name] = {
        "min": float(norms.min()),
        "p01": float(np.percentile(norms, 1)),
        "zero_count": int(np.sum(norms <= 1e-12)),
    }
print(f"stage=gradient_qc values={gradient_qc}", flush=True)

print("stage=ldrb_start", flush=True)
system = ldrb.ldrb.compute_fiber_sheet_system(
    lv_scalar=scalar_data["lv"],
    rv_scalar=scalar_data["rv"],
    epi_scalar=scalar_data["epi"],
    lv_rv_scalar=scalar_data["lv_rv"],
    lv_gradient=gradient_data["lv_gradient"],
    rv_gradient=gradient_data["rv_gradient"],
    epi_gradient=gradient_data["epi_gradient"],
    apex_gradient=gradient_data["apex_gradient"],
    alpha_endo_lv=40.0,
    alpha_epi_lv=-50.0,
    alpha_endo_rv=40.0,
    alpha_epi_rv=-50.0,
    alpha_endo_sept=40.0,
    alpha_epi_sept=-50.0,
    beta_endo_lv=-65.0,
    beta_epi_lv=25.0,
    beta_endo_rv=-65.0,
    beta_epi_rv=25.0,
    beta_endo_sept=-65.0,
    beta_epi_sept=25.0,
)
print("stage=ldrb_complete", flush=True)

fibers = system.f0.reshape(-1, 3)
sheets = system.s0.reshape(-1, 3)
field = np.column_stack((fibers, sheets))

with OUT_LON.open("w") as stream:
    stream.write("2\n")
    np.savetxt(stream, field, fmt="%.9f")

report = {
    "patient": PATIENT,
    "algorithm": "Bayer2012_Laplace_Dirichlet",
    "implementation": {
        "package": "fenicsx-ldrb==0.1.4",
        "scalar_laplacians": "package",
        "nodal_gradient_projection": "element_gradient_volume_weighted",
        "bayer_orientation_rules": "package",
    },
    "dolfinx": dolfinx.__version__,
    "angles_deg": {"alpha_endo": 40, "alpha_epi": -50,
                   "beta_endo": -65, "beta_epi": 25},
    "facet_markers": counts,
    "n_exterior_facets": int(len(exterior)),
    "gradient_qc": gradient_qc,
    "field_finite": bool(np.isfinite(field).all()),
    "fiber_norm_max_abs_error": float(
        np.max(np.abs(np.linalg.norm(fibers, axis=1) - 1.0))
    ),
    "sheet_norm_max_abs_error": float(
        np.max(np.abs(np.linalg.norm(sheets, axis=1) - 1.0))
    ),
    "fiber_sheet_dot_max_abs": float(
        np.max(np.abs(np.sum(fibers * sheets, axis=1)))
    ),
    "duration_seconds": time.time() - started,
    "output": str(OUT_LON.relative_to(ROOT)),
}
OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
OUT_REPORT.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
