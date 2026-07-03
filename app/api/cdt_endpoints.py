"""
CDT API — FastAPI + WebSocket (D4.1)

Endpoints REST :
  POST /api/v1/twin/create     — creer un jumeau
  POST /api/v1/twin/simulate   — lancer une simulation
  GET  /api/v1/twin/{id}/state — etat du jumeau
  GET  /api/v1/twin/{id}/results — resultats

WebSocket :
  WS /api/v1/twin/{id}/stream  — streaming temps reel

Machine d'etat :
  CREATED → SIMULATING → COMPLETED / FAILED
"""
import asyncio
import json
import time
import uuid
from enum import Enum
from typing import Optional, Dict

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Cardiac Digital Twin API", version="1.0.0")


class TwinState(str, Enum):
    CREATED = "created"
    SIMULATING = "simulating"
    COMPLETED = "completed"
    FAILED = "failed"


class TwinSession:
    def __init__(self, twin_id: str, patient_id: str):
        self.twin_id = twin_id
        self.patient_id = patient_id
        self.state = TwinState.CREATED
        self.created_at = time.time()
        self.parameters = {}
        self.results = {}
        self.progress = 0.0
        self.error = None


sessions: Dict[str, TwinSession] = {}


class CreateTwinRequest(BaseModel):
    patient_id: str


class SimulateRequest(BaseModel):
    twin_id: str
    sigma_l: float = 0.3
    sigma_t: float = 0.1
    T_max_kPa: float = 135.0
    heart_rate_bpm: float = 75.0
    a_kPa: float = 0.496
    b: float = 7.209
    R_p: float = 1.5e8
    C_a: float = 1.0e-8


class TwinStateResponse(BaseModel):
    twin_id: str
    patient_id: str
    state: str
    progress: float
    error: Optional[str] = None


class SimulationResults(BaseModel):
    twin_id: str
    ef_pct: float = 0.0
    edv_mL: float = 0.0
    esv_mL: float = 0.0
    sv_mL: float = 0.0
    p_systolic_mmHg: float = 0.0
    p_diastolic_mmHg: float = 0.0
    p_mean_mmHg: float = 0.0
    cv_ms: float = 0.0
    apd90_ms: float = 0.0
    co_L_min: float = 0.0
    output_vector: list = []


@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "version": "1.0.0", "sessions": len(sessions)}


@app.post("/api/v1/twin/create")
async def create_twin(req: CreateTwinRequest):
    twin_id = str(uuid.uuid4())[:8]
    session = TwinSession(twin_id, req.patient_id)
    sessions[twin_id] = session
    return {"twin_id": twin_id, "state": session.state, "patient_id": req.patient_id}


@app.get("/api/v1/twin/{twin_id}/state")
async def get_state(twin_id: str):
    if twin_id not in sessions:
        raise HTTPException(status_code=404, detail="Twin not found")
    s = sessions[twin_id]
    return TwinStateResponse(
        twin_id=s.twin_id, patient_id=s.patient_id,
        state=s.state, progress=s.progress, error=s.error
    )


@app.post("/api/v1/twin/simulate")
async def simulate(req: SimulateRequest):
    if req.twin_id not in sessions:
        raise HTTPException(status_code=404, detail="Twin not found")
    session = sessions[req.twin_id]
    session.state = TwinState.SIMULATING
    session.parameters = req.model_dump()
    session.progress = 0.0

    try:
        from app.solver.coupled_solver import CoupledSolver, SimulationParameters
        import os

        mesh_dir = os.path.expanduser("~/cdt/reports/meshes_acdc/meshes")
        pid = session.patient_id

        with open(f"{mesh_dir}/{pid}.pts") as f:
            f.readline()
            nodes = np.array([list(map(float, l.split())) for l in f])
        with open(f"{mesh_dir}/{pid}.elem") as f:
            f.readline()
            elements = np.array([[int(x) for x in l.split()[1:5]] for l in f])
        with open(f"{mesh_dir}/{pid}_fibers.lon") as f:
            f.readline()
            fibers = np.array([list(map(float, l.split()))[:3] for l in f])

        params = SimulationParameters(
            sigma_l=req.sigma_l, sigma_t=req.sigma_t,
            T_max_kPa=req.T_max_kPa, heart_rate_bpm=req.heart_rate_bpm,
            a_kPa=req.a_kPa, b=req.b, R_p=req.R_p, C_a=req.C_a,
            duration_ms=500.0,
        )

        solver = CoupledSolver()
        session.progress = 10.0
        result = solver.simulate(params, nodes, elements, fibers, pid, session.twin_id)
        row = result.to_doe_row()

        session.results = row
        session.state = TwinState.COMPLETED
        session.progress = 100.0

        return {"twin_id": req.twin_id, "state": "completed", "results": row}

    except Exception as e:
        session.state = TwinState.FAILED
        session.error = str(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/twin/{twin_id}/results")
async def get_results(twin_id: str):
    if twin_id not in sessions:
        raise HTTPException(status_code=404, detail="Twin not found")
    s = sessions[twin_id]
    if s.state != TwinState.COMPLETED:
        raise HTTPException(status_code=400, detail=f"State is {s.state}, not completed")
    return SimulationResults(twin_id=twin_id, **{
        k: v for k, v in s.results.items()
        if k in SimulationResults.model_fields
    })


@app.websocket("/api/v1/twin/{twin_id}/stream")
async def stream_twin(websocket: WebSocket, twin_id: str):
    await websocket.accept()

    if twin_id not in sessions:
        await websocket.send_json({"error": "Twin not found"})
        await websocket.close()
        return

    session = sessions[twin_id]

    try:
        while True:
            msg = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
            action = msg.get("action")

            if action == "get_state":
                await websocket.send_json({
                    "type": "state",
                    "twin_id": twin_id,
                    "state": session.state,
                    "progress": session.progress,
                })

            elif action == "simulate":
                session.state = TwinState.SIMULATING
                session.progress = 0.0

                await websocket.send_json({"type": "progress", "step": "ep", "progress": 10})

                from app.solver.coupled_solver import CoupledSolver, SimulationParameters
                import os

                mesh_dir = os.path.expanduser("~/cdt/reports/meshes_acdc/meshes")
                pid = session.patient_id

                with open(f"{mesh_dir}/{pid}.pts") as f:
                    f.readline()
                    nodes = np.array([list(map(float, l.split())) for l in f])
                with open(f"{mesh_dir}/{pid}.elem") as f:
                    f.readline()
                    elements = np.array([[int(x) for x in l.split()[1:5]] for l in f])
                with open(f"{mesh_dir}/{pid}_fibers.lon") as f:
                    f.readline()
                    fibers = np.array([list(map(float, l.split()))[:3] for l in f])

                params = SimulationParameters(
                    sigma_l=msg.get("sigma_l", 0.3),
                    sigma_t=msg.get("sigma_t", 0.1),
                    T_max_kPa=msg.get("T_max_kPa", 135.0),
                    heart_rate_bpm=msg.get("heart_rate_bpm", 75.0),
                    R_p=msg.get("R_p", 1.5e8),
                    C_a=msg.get("C_a", 1.0e-8),
                    duration_ms=500.0,
                )

                await websocket.send_json({"type": "progress", "step": "mechanics", "progress": 40})

                solver = CoupledSolver()
                result = solver.simulate(params, nodes, elements, fibers, pid, twin_id)

                await websocket.send_json({"type": "progress", "step": "windkessel", "progress": 80})

                row = result.to_doe_row()
                session.results = row
                session.state = TwinState.COMPLETED
                session.progress = 100.0

                await websocket.send_json({
                    "type": "results",
                    "progress": 100,
                    "data": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                             for k, v in row.items()},
                })

            elif action == "update_param":
                param = msg.get("param")
                value = msg.get("value")
                if param and value is not None:
                    session.parameters[param] = value
                    await websocket.send_json({"type": "param_updated", "param": param, "value": value})

            elif action == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        pass
    except asyncio.TimeoutError:
        await websocket.close()
