"""Controle court de compilation/convergence de la mecanique orthotrope.

Ce script ne valide pas la physiologie et ne lance pas un DoE. Il execute
exactement deux paliers sans pression sur le maillage patient001 corrige.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/cdt")

from app.solver.mechanics.fenicsx_solver import FenicsxSolver, MechanicsParameters


ROOT = Path("/cdt")
MESH_DIR = ROOT / "reports/meshes_acdc/meshes"
PATIENT = "patient001_coarse5_fixed"


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
    return np.loadtxt(path, skiprows=1, dtype=np.float64)


nodes = read_pts(MESH_DIR / f"{PATIENT}.pts")
elements = read_elem(MESH_DIR / f"{PATIENT}.elem")
microstructure = read_lon(MESH_DIR / f"{PATIENT}_fibers_ldrb.lon")

params = MechanicsParameters(
    T_max_kPa=30.0,
    p_endo_kPa=0.0,
    max_continuation_steps=2,
    checkpoint_dir=None,
)
result = FenicsxSolver().simulate(
    params=params,
    nodes=nodes,
    elements=elements,
    fibers=microstructure,
    activation_times_ms=np.zeros(len(nodes)),
    twin_id="patient001",
    job_id="orthotropic_smoke",
)

print({
    "converged_full_load": result.converged,
    "load_fraction": result.load_fraction,
    "iterations": result.n_iterations,
    "min_jacobian": result.min_jacobian,
    "solver_version": result.solver_version,
})

if result.load_fraction <= 0.0 or not np.isfinite(result.min_jacobian):
    raise SystemExit("ECHEC: aucun palier orthotrope valide.")
