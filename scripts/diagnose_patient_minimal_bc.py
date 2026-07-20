"""Sprint 2 : smoke test patient à faible charge avec appui basal minimal.

Ce diagnostic à faible charge n'est pas une simulation physiologique. Il vérifie la
stabilité et le sens de la réponse avant d'autoriser un nouveau run long.
"""
import json
from pathlib import Path
import sys
import time
import argparse

import numpy as np

ROOT = Path("/cdt")
sys.path.insert(0, str(ROOT))

from app.solver.mechanics.fenicsx_solver import FenicsxSolver, MechanicsParameters


MESH_DIR = ROOT / "reports/meshes_acdc/meshes"
PATIENT = "patient001_coarse5_fixed"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--field",
    default=f"{PATIENT}_fibers_ldrb.lon",
    help="Nom du fichier de microstructure dans le dossier des maillages.",
)
parser.add_argument("--max-steps", type=int, default=500)
parser.add_argument(
    "--tmax-kpa",
    type=float,
    default=1.0,
    help="Tension active maximale du diagnostic, en kPa.",
)
parser.add_argument(
    "--output",
    default="patient001_minimal_bc_tmax1.json",
    help="Nom du rapport JSON dans sprint_artifacts/sprint2.",
)
args = parser.parse_args()
OUT = ROOT / "sprint_artifacts/sprint2" / args.output
FIELDS_OUT = OUT.with_suffix(".npz")


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
fibers = np.loadtxt(MESH_DIR / args.field, skiprows=1)

params = MechanicsParameters(
    T_max_kPa=args.tmax_kpa,
    p_endo_kPa=0.0,
    base_bc_mode="normal_with_rigid_pins",
    dlam_init=0.2,
    dlam_max=0.5,
    easy_iteration_threshold=8,
    max_continuation_steps=args.max_steps,
    checkpoint_dir=None,
)
started = time.time()
result = FenicsxSolver().simulate(
    params=params,
    nodes=nodes,
    elements=elements,
    fibers=fibers,
    activation_times_ms=np.zeros(len(nodes)),
    twin_id="patient001",
    job_id=f"sprint2_minimal_bc_tmax{args.tmax_kpa:g}",
)
deformed = nodes + result.displacement_mm
apex = int(np.argmin(nodes[:, 2]))
summary = {
    "purpose": "diagnostic_non_physiological",
    "patient": PATIENT,
    "T_max_kPa": args.tmax_kpa,
    "microstructure": args.field,
    "base_bc_mode": params.base_bc_mode,
    "converged": bool(result.converged),
    "load_fraction": float(result.load_fraction),
    "iterations": int(result.n_iterations),
    "min_jacobian": float(result.min_jacobian),
    "domain_errors": int(result.domain_errors_total),
    "duration_seconds": time.time() - started,
    "height_change_mm": float(np.ptp(deformed[:, 2]) - np.ptp(nodes[:, 2])),
    "apex_dz_mm": float(result.displacement_mm[apex, 2]),
    "max_displacement_mm": float(np.linalg.norm(result.displacement_mm, axis=1).max()),
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(summary, indent=2) + "\n")
np.savez_compressed(
    FIELDS_OUT,
    nodes=nodes,
    elements=elements,
    displacement_mm=result.displacement_mm,
    deformed_nodes=deformed,
)
print(json.dumps(summary, indent=2))
if not result.converged:
    raise SystemExit("Le smoke test patient n'a pas atteint la charge complète.")
