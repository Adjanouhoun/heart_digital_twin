"""Génère le manifeste reproductible des entrées du Sprint 1."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.solver.mechanics.run_manifest import build_manifest, write_manifest


MESH_DIR = ROOT / "reports/meshes_acdc/meshes"
PATIENT = "patient001_coarse5_fixed"

parameters = {
    "patient": PATIENT,
    "T_max_kPa": 30.0,
    "p_endo_kPa": 0.0,
    "easy_iteration_threshold": 8,
    "dlam_init": 1.0e-4,
    "dlam_min": 1.0e-6,
    "dlam_max": 0.05,
    "j_min_accept": 0.1,
    "fenicsx_version": "0.7.3",
}
inputs = {
    "mesh_nodes": MESH_DIR / f"{PATIENT}.pts",
    "mesh_elements": MESH_DIR / f"{PATIENT}.elem",
    "ldrb_microstructure": MESH_DIR / f"{PATIENT}_fibers_ldrb.lon",
    "segmentation": ROOT / "reports/meshes_acdc/segmentations/patient001_seg.nii.gz",
    "mechanics_solver": ROOT / "app/solver/mechanics/fenicsx_solver.py",
    "cavity_volume": ROOT / "app/solver/mechanics/endo_cavity_volume.py",
    "run_script": ROOT / "scripts/run_sprint1_orthotropic_full.py",
    "postprocess_script": ROOT / "scripts/postprocess_sprint1.py",
}

manifest = build_manifest(ROOT, parameters, inputs)
output = ROOT / "sprint_artifacts/sprint1/manifest.json"
write_manifest(output, manifest)
print(output)
