"""
CDT Solver API — Phase 03
Endpoints pour lancer les simulations EP + Mécanique + Windkessel.
Port : 8001 (séparé de l'API principale :8000)
"""
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("cdt.solver_api.startup", version="0.1.0", phase="03")
    yield


app = FastAPI(
    title="CDT Solver API",
    description="Cardiac Digital Twin — Phase 03 : Solveur Multi-Physique",
    version="0.1.0",
    lifespan=lifespan,
)

_jobs: dict[str, dict] = {}


class SimulationRequest(BaseModel):
    twin_id: str = Field(..., pattern=r"^[a-f0-9]{64}$")
    mesh_pts_key: str = Field(default="", description="Clé MinIO .pts")
    mesh_elem_key: str = Field(default="", description="Clé MinIO .elem")
    mesh_lon_key: str = Field(default="", description="Clé MinIO .lon")
    sigma_l: float = Field(default=0.30, ge=0.05, le=1.0)
    sigma_t: float = Field(default=0.10, ge=0.02, le=0.5)
    T_max_kPa: float = Field(default=135.0, ge=50.0, le=300.0)
    heart_rate_bpm: float = Field(default=75.0, ge=40.0, le=150.0)
    duration_ms: float = Field(default=500.0, ge=100.0, le=2000.0)


@app.get("/health", tags=["Infrastructure"])
async def health():
    import subprocess
    opencarp_ok = subprocess.run(["which", "openCARP.par"], capture_output=True).returncode == 0
    try:
        import dolfinx
        fenicsx_version = dolfinx.__version__
    except ImportError:
        fenicsx_version = "fallback"

    return {
        "status": "healthy",
        "service": "cdt-solver",
        "phase": "03",
        "timestamp": datetime.utcnow().isoformat(),
        "solvers": {
            "opencarp": "available" if opencarp_ok else "fallback-analytique",
            "fenicsx": fenicsx_version,
            "windkessel": "available",
        }
    }


@app.post("/v1/simulate", status_code=202, tags=["Simulation"])
async def launch_simulation(request: SimulationRequest, background_tasks: BackgroundTasks):
    import threading
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "twin_id": request.twin_id,
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
    }
    # Thread separe (pas BackgroundTasks FastAPI) : _run_simulation fait un
    # subprocess.run openCARP BLOQUANT qui gelerait l'event loop mono-worker
    # et empecherait de repondre au polling. Un thread libere l'event loop.
    threading.Thread(target=_run_simulation_sync, args=(job_id, request), daemon=True).start()
    return {"status": "accepted", "job_id": job_id, "poll_url": f"/v1/jobs/{job_id}"}


@app.get("/v1/jobs/{job_id}", tags=["Monitoring"])
async def get_job_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} introuvable")
    return _jobs[job_id]


@app.post("/v1/doe", status_code=202, tags=["DoE"])
async def launch_doe(twin_id: str, n_simulations: int = 500, background_tasks: BackgroundTasks = None):
    doe_id = str(uuid.uuid4())
    _jobs[doe_id] = {
        "job_id": doe_id, "twin_id": twin_id,
        "status": "pending", "type": "doe",
        "n_simulations": n_simulations,
        "created_at": datetime.utcnow().isoformat(),
    }
    background_tasks.add_task(_run_doe, doe_id, twin_id, n_simulations)
    return {"status": "accepted", "doe_id": doe_id, "poll_url": f"/v1/jobs/{doe_id}"}


def _run_simulation_sync(job_id: str, request: SimulationRequest):
    import numpy as np
    from app.solver.coupled_solver import CoupledSolver, SimulationParameters
    _jobs[job_id]["status"] = "running"
    try:
        nodes, elements, fibers = _load_mesh_from_minio(
            request.mesh_pts_key, request.mesh_elem_key, request.mesh_lon_key)
        logger.info("solver_api.mesh_loaded", job_id=job_id,
                    n_nodes=len(nodes), n_elements=len(elements))
        params = SimulationParameters(
            sigma_l=request.sigma_l, sigma_t=request.sigma_t,
            T_max_kPa=request.T_max_kPa, heart_rate_bpm=request.heart_rate_bpm,
            duration_ms=request.duration_ms,
        )
        solver = CoupledSolver()
        result = solver.simulate(params, nodes, elements, fibers, request.twin_id, job_id)
        _jobs[job_id].update({
            "status": "done",
            "completed_at": datetime.utcnow().isoformat(),
            "ef_pct": result.wk_result.ef_pct if result.wk_result else None,
            "p_systolic_mmHg": result.wk_result.p_systolic_mmHg if result.wk_result else None,
            "cv_ms": result.ep_result.conduction_velocity_ms if result.ep_result else None,
            "benchmark_passed": result.benchmark_passed,
            "duration_s": result.duration_seconds,
        })
    except Exception as e:
        _jobs[job_id].update({"status": "failed", "error": str(e),
                               "completed_at": datetime.utcnow().isoformat()})


def _load_mesh_from_minio(pts_key: str, elem_key: str, lon_key: str):
    """Charge un maillage openCARP (.pts/.elem/.lon) depuis MinIO.

    S3_ENDPOINT est configurable : http://localhost:9000 quand le Solver API
    tourne en local (M1, openCARP natif), http://minio:9000 dans un conteneur.
    """
    import os
    import boto3
    import numpy as np

    if not pts_key:
        raise ValueError("mesh_pts_key requis (chargement MinIO)")

    client = boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT", "http://localhost:9000"),
        aws_access_key_id=os.environ.get("MINIO_USER", "cdt_admin"),
        aws_secret_access_key=os.environ.get("MINIO_PASSWORD", "cdt_minio_2024"),
    )
    bucket = os.environ.get("MESH_BUCKET", "cdt-meshes")

    def _get_text(key):
        return client.get_object(Bucket=bucket, Key=key)["Body"].read().decode()

    # .pts : 1ere ligne = nombre de noeuds, puis "x y z" par ligne
    pts_lines = _get_text(pts_key).strip().split("\n")
    nodes = np.array([list(map(float, l.split())) for l in pts_lines[1:]])

    # .elem : 1ere ligne = nombre, puis "Tt n0 n1 n2 n3 tag"
    elem_lines = _get_text(elem_key).strip().split("\n")
    elements = np.array([[int(x) for x in l.split()[1:5]] for l in elem_lines[1:]],
                        dtype=np.int32)

    # .lon : 1ere ligne = nombre de champs (1 ou 2), puis vecteurs fibre
    if lon_key:
        lon_lines = _get_text(lon_key).strip().split("\n")
        fibers = np.array([list(map(float, l.split()[:3])) for l in lon_lines[1:]])
    else:
        fibers = np.tile([1.0, 0.0, 0.0], (len(nodes), 1))

    return nodes, elements, fibers


async def _run_doe(doe_id: str, twin_id: str, n_simulations: int):
    import numpy as np, base64
    from app.solver.tasks.doe_task import run_doe_batch
    _jobs[doe_id]["status"] = "running"
    try:
        nodes = np.random.randn(100, 3) * 30
        elements = np.array([[i,i+1,i+2,i+3] for i in range(0,96,4)], dtype=np.int32)
        fibers = np.tile([1.0,0.0,0.0], (100,1))
        summary = run_doe_batch(
            twin_id=twin_id, n_simulations=n_simulations,
            nodes_b64=base64.b64encode(nodes.tobytes()).decode(),
            elements_b64=base64.b64encode(elements.tobytes()).decode(),
            nodes_shape=list(nodes.shape), elements_shape=list(elements.shape),
            fiber_b64=base64.b64encode(fibers.tobytes()).decode(),
        )
        _jobs[doe_id].update({"status": "done",
                               "completed_at": datetime.utcnow().isoformat(), **summary})
    except Exception as e:
        _jobs[doe_id].update({"status": "failed", "error": str(e),
                               "completed_at": datetime.utcnow().isoformat()})
