"""Sprint 1 : continuation orthotrope complete, sans pression, T_max=30 kPa."""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/cdt")

from app.solver.mechanics.fenicsx_solver import FenicsxSolver, MechanicsParameters


ROOT = Path("/cdt")
MESH_DIR = ROOT / "reports/meshes_acdc/meshes"
PATIENT = "patient001_coarse5_fixed"
ARTIFACT_DIR = ROOT / "sprint_artifacts/sprint1"
CHECKPOINT_DIR = ARTIFACT_DIR / "checkpoints"


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


def read_lon(path):
    with path.open() as stream:
        header = stream.readline().strip()
    if header != "2":
        raise ValueError(f"Format LDRB attendu: header 2; recu {header!r}.")
    values = np.loadtxt(path, skiprows=1, dtype=np.float64)
    if values.shape[1] != 6:
        raise ValueError(f"Six composantes LDRB attendues; recu {values.shape}.")
    return values


ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

nodes = read_pts(MESH_DIR / f"{PATIENT}.pts")
elements = read_elem(MESH_DIR / f"{PATIENT}.elem")
microstructure = read_lon(MESH_DIR / f"{PATIENT}_fibers_ldrb.lon")

params = MechanicsParameters(
    T_max_kPa=30.0,
    p_endo_kPa=0.0,
    easy_iteration_threshold=8,
    checkpoint_dir=str(CHECKPOINT_DIR),
)

started = time.time()
result = FenicsxSolver().simulate(
    params=params,
    nodes=nodes,
    elements=elements,
    fibers=microstructure,
    activation_times_ms=np.zeros(len(nodes)),
    twin_id="patient001",
    job_id="sprint1_orthotropic_tmax30_no_pressure",
)
elapsed = time.time() - started

deformed = nodes + result.displacement_mm
height_ed = float(np.ptp(nodes[:, 2]))
height_es = float(np.ptp(deformed[:, 2]))
apex_index = int(np.argmin(nodes[:, 2]))
apex_dz = float(result.displacement_mm[apex_index, 2])

summary = {
    "sprint": 1,
    "patient": PATIENT,
    "configuration": {
        "T_max_kPa": 30.0,
        "p_endo_kPa": 0.0,
        "nodes": int(len(nodes)),
        "tetrahedra": int(len(elements)),
        "microstructure_components": int(microstructure.shape[1]),
    },
    "solver": {
        "version": result.solver_version,
        "converged": bool(result.converged),
        "load_fraction": float(result.load_fraction),
        "iterations": int(result.n_iterations),
        "min_jacobian": float(result.min_jacobian),
        "domain_errors": int(result.domain_errors_total),
        "resumed_from_checkpoint": bool(result.resumed_from_checkpoint),
        "duration_seconds": float(elapsed),
    },
    "geometry": {
        "height_ed_mm": height_ed,
        "height_es_mm": height_es,
        "height_change_mm": height_es - height_ed,
        "apex_dz_mm": apex_dz,
        "max_displacement_mm": float(
            np.linalg.norm(result.displacement_mm, axis=1).max()
        ),
        "median_displacement_mm": float(
            np.median(np.linalg.norm(result.displacement_mm, axis=1))
        ),
        "endo_radial_disp_mm": float(result.endo_radial_disp_mm),
        "epi_radial_disp_mm": float(result.epi_radial_disp_mm),
    },
    "cavity": {"status": "pending_host_postprocessing"},
}

with (ARTIFACT_DIR / "result.json").open("w") as stream:
    json.dump(summary, stream, indent=2)
np.savez_compressed(
    ARTIFACT_DIR / "fields.npz",
    nodes=nodes,
    elements=elements,
    microstructure=microstructure,
    displacement_mm=result.displacement_mm,
)

print(json.dumps(summary, indent=2), flush=True)
if not result.converged:
    raise SystemExit("Sprint 1 incomplet : la continuation n'a pas converge.")
