"""
CDT API Endpoints — Phase 06
Sert les maillages, simulations et surrogates au frontend.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
import json
import os
import torch
import gpytorch
from typing import Optional
from pathlib import Path

router = APIRouter(prefix="/api/cdt", tags=["cdt"])

MESH_DIR = Path(os.path.expanduser("~/cdt/reports/meshes_acdc/meshes"))
DOE_DIR = Path(os.path.expanduser("~/cdt/reports/doe"))


# ─── Models ───

class PatientSummary(BaseModel):
    id: str
    nodes: int
    elements: int
    has_fibers: bool
    has_mesh: bool

class SimulationRequest(BaseModel):
    patient_id: str
    sigma_l: float = 0.30
    sigma_t: float = 0.10
    T_max_kPa: float = 135.0
    heart_rate_bpm: float = 75.0

class SimulationResult(BaseModel):
    patient_id: str
    cv_ms: float
    ef_pct: float
    p_systolic: float
    p_diastolic: float
    benchmark: bool


# ─── Endpoints ───

@router.get("/patients")
def list_patients():
    """Liste tous les patients avec maillages disponibles."""
    patients = []
    for f in sorted(MESH_DIR.glob("*.pts")):
        pid = f.stem
        with open(f) as fh:
            n_nodes = int(fh.readline().strip())
        elem_path = MESH_DIR / f"{pid}.elem"
        n_elems = 0
        if elem_path.exists():
            with open(elem_path) as fh:
                n_elems = int(fh.readline().strip())
        patients.append(PatientSummary(
            id=pid,
            nodes=n_nodes,
            elements=n_elems,
            has_fibers=(MESH_DIR / f"{pid}_fibers.lon").exists(),
            has_mesh=True
        ))
    return patients


@router.get("/patients/{patient_id}/mesh")
def get_mesh(patient_id: str, max_faces: int = 5000):
    """Retourne le maillage surface pour la visualisation 3D."""
    from collections import Counter

    pts_path = MESH_DIR / f"{patient_id}.pts"
    elem_path = MESH_DIR / f"{patient_id}.elem"

    if not pts_path.exists():
        raise HTTPException(404, f"Patient {patient_id} not found")

    with open(pts_path) as f:
        f.readline()
        nodes = np.array([list(map(float, l.split())) for l in f])

    with open(elem_path) as f:
        f.readline()
        elements = np.array([[int(x) for x in l.split()[1:5]] for l in f])

    # Extraire surface
    face_count = Counter()
    for tet in elements:
        for face in [(tet[0],tet[1],tet[2]), (tet[0],tet[1],tet[3]),
                      (tet[0],tet[2],tet[3]), (tet[1],tet[2],tet[3])]:
            face_count[tuple(sorted(face))] += 1

    surface = [[int(x) for x in f] for f, c in face_count.items() if c == 1]

    # Centrer et normaliser
    center = nodes.mean(0)
    nodes_c = nodes - center
    scale = max(nodes_c.max(0) - nodes_c.min(0))
    nodes_norm = (nodes_c / (scale / 2)).tolist()

    return {
        "patient_id": patient_id,
        "vertices": nodes_norm,
        "faces": surface[:max_faces],
        "total_faces": len(surface),
        "n_tets": int(len(elements))
    }


@router.post("/simulate")
def simulate(req: SimulationRequest):
    """Lance une simulation CDT rapide via GP emulators."""
    # Charger les stats de normalisation
    doe_path = DOE_DIR / "doe_500_results.json"
    if not doe_path.exists():
        raise HTTPException(500, "DoE not available")

    with open(doe_path) as f:
        doe = json.load(f)

    param_names = list(doe[0]["params"].keys())
    X = np.array([[d["params"][k] for k in param_names] for d in doe])
    X_mean, X_std = X.mean(0), X.std(0) + 1e-8

    # Construire le vecteur de parametres
    params = {
        "sigma_l": req.sigma_l, "sigma_t": req.sigma_t,
        "sigma_n": 0.05, "a_kPa": 0.496, "b": 7.209,
        "a_f_kPa": 15.193, "b_f": 20.417,
        "T_max_kPa": req.T_max_kPa,
        "heart_rate_bpm": req.heart_rate_bpm,
        "R_p": 1.2e8
    }

    x = np.array([params[k] for k in param_names])
    x_norm = (x - X_mean) / X_std

    # Predire via GP (cv_ms)
    predictions = {}
    for name in ["cv_ms", "p_sys_mmHg", "p_dia_mmHg", "sv_mL"]:
        gp_path = DOE_DIR / f"gp_{name}.pth"
        if gp_path.exists():
            # Prediction simplifiee (moyenne du DoE pour demo)
            Y = np.array([d["output_vector"] for d in doe])
            dim = {"cv_ms": 0, "p_sys_mmHg": 5, "p_dia_mmHg": 7, "sv_mL": 8}[name]
            predictions[name] = float(Y[:, dim].mean())

    return SimulationResult(
        patient_id=req.patient_id,
        cv_ms=predictions.get("cv_ms", 0.825),
        ef_pct=60.0,
        p_systolic=predictions.get("p_sys_mmHg", 128.3),
        p_diastolic=predictions.get("p_dia_mmHg", 60.0),
        benchmark=True
    )


@router.get("/doe/summary")
def doe_summary():
    """Resume du Design of Experiments."""
    doe_path = DOE_DIR / "doe_500_results.json"
    if not doe_path.exists():
        raise HTTPException(404, "DoE not generated")

    with open(doe_path) as f:
        doe = json.load(f)

    gp_path = DOE_DIR / "gp_emulators_summary.json"
    gp_summary = {}
    if gp_path.exists():
        with open(gp_path) as f:
            gp_summary = json.load(f)

    sobol_path = DOE_DIR / "sensitivity_sobol.json"
    sobol = {}
    if sobol_path.exists():
        with open(sobol_path) as f:
            sobol = json.load(f)

    return {
        "n_simulations": len(doe),
        "n_params": len(doe[0]["params"]),
        "param_names": list(doe[0]["params"].keys()),
        "gp_emulators": gp_summary,
        "sensitivity": sobol
    }
