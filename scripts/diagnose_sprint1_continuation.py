"""Teste un seul palier plus large depuis une copie du checkpoint Sprint 1."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/cdt")

from app.solver.mechanics.fenicsx_solver import FenicsxSolver, MechanicsParameters


ROOT = Path("/cdt")
MESH_DIR = ROOT / "reports/meshes_acdc/meshes"
PATIENT = "patient001_coarse5_fixed"
SOURCE = (ROOT / "sprint_artifacts/sprint1/checkpoints/"
          "patient001_sprint1_orthotropic_tmax30_no_pressure_mech.npz")
DIAG_DIR = ROOT / "sprint_artifacts/sprint1/diagnostic_dlam_5e4"
TWIN_ID = "patient001"
JOB_ID = "diagnostic_dlam_5e4"
DEST = DIAG_DIR / f"{TWIN_ID}_{JOB_ID}_mech.npz"


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
    values = np.loadtxt(path, skiprows=1, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"Champ LDRB (N,6) attendu; recu {values.shape}.")
    return values


DIAG_DIR.mkdir(parents=True, exist_ok=True)
source = np.load(SOURCE)
source_lam = float(source["lam"])
np.savez(
    DEST,
    w_array=source["w_array"],
    lam=source_lam,
    dlam=5.0e-4,
    n_steps=int(source["n_steps"]),
    min_J=float(source["min_J"]),
)

nodes = read_pts(MESH_DIR / f"{PATIENT}.pts")
elements = read_elem(MESH_DIR / f"{PATIENT}.elem")
microstructure = read_lon(MESH_DIR / f"{PATIENT}_fibers_ldrb.lon")

params = MechanicsParameters(
    T_max_kPa=30.0,
    p_endo_kPa=0.0,
    checkpoint_dir=str(DIAG_DIR),
    max_continuation_steps=int(source["n_steps"]) + 1,
)
result = FenicsxSolver().simulate(
    params=params,
    nodes=nodes,
    elements=elements,
    fibers=microstructure,
    activation_times_ms=np.zeros(len(nodes)),
    twin_id=TWIN_ID,
    job_id=JOB_ID,
)

print({
    "source_lam": source_lam,
    "requested_increment": 5.0e-4,
    "accepted_lam": result.load_fraction,
    "accepted": result.load_fraction > source_lam,
    "iterations": result.n_iterations,
    "min_jacobian": result.min_jacobian,
    "domain_errors": result.domain_errors_total,
})
