import base64
import hashlib
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, Optional

import structlog
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config import get_settings, Settings
from app.registry.database import init_db, get_session
from app.registry.models import Twin, IngestionJob, AuditLog
from app.anonymizer.dicom_anonymizer import DicomAnonymizer
from app.storage.minio_client import CDTStorageClient, get_storage_client
from app.tasks.worker import ingest_dicom_task

logger = structlog.get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("cdt.api.startup", version="0.1.0", phase="01-ingestion")
    await init_db()
    yield
    logger.info("cdt.api.shutdown")

app = FastAPI(
    title="CDT Ingestion API",
    description="Cardiac Digital Twin — Phase 01",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

def get_anonymizer() -> DicomAnonymizer:
    return DicomAnonymizer(seed=get_settings().twin_id_seed)

def get_storage() -> CDTStorageClient:
    return get_storage_client()

@app.get("/health", tags=["Infrastructure"])
async def health_check():
    return {"status": "healthy", "service": "cdt-ingestion", "phase": "01", "timestamp": datetime.utcnow().isoformat()}

@app.post("/v1/ingest/dicom", status_code=202, tags=["Ingestion"])
async def ingest_dicom(
    request: Request,
    file: UploadFile = File(...),
    consent_id: str = Form(...),
    source_system: str = Form(...),
    clinical_context: Optional[str] = Form(None),
    session: AsyncSession = Depends(get_session),
    anonymizer: DicomAnonymizer = Depends(get_anonymizer),
):
    if not file.filename or not file.filename.lower().endswith((".dcm", ".dicom")):
        raise HTTPException(status_code=422, detail="Extension invalide — .dcm ou .dicom requis")

    dicom_bytes = await file.read()
    raw_sha256 = hashlib.sha256(dicom_bytes).hexdigest()

    try:
        import io, pydicom
        ds = pydicom.dcmread(io.BytesIO(dicom_bytes), stop_before_pixels=True, force=True)
        patient_id_raw = str(getattr(ds, "PatientID", raw_sha256)).strip() or raw_sha256
    except Exception:
        patient_id_raw = raw_sha256

    twin_id = anonymizer.compute_twin_id(patient_id_raw)
    job_id  = str(uuid.uuid4())

    existing = await session.get(Twin, twin_id)
    if existing is None:
        twin = Twin(twin_id=twin_id, consent_id=consent_id, source_system=source_system, status="ingesting", clinical_context=clinical_context)
        session.add(twin)
    else:
        existing.status = "ingesting"

    job = IngestionJob(job_id=job_id, twin_id=twin_id, data_type="dicom", status="pending", original_sha256=raw_sha256, file_size_bytes=len(dicom_bytes))
    session.add(job)

    audit = AuditLog(twin_id=twin_id, operation="INGEST_DICOM_REQUESTED", actor=source_system,
                     source_ip=request.client.host if request.client else None,
                     job_id=job_id, status="success", details=f"consent_id={consent_id} size={len(dicom_bytes)}")
    session.add(audit)
    await session.commit()

    ingest_dicom_task.apply_async(
        kwargs={"dicom_b64": base64.b64encode(dicom_bytes).decode(), "job_id": job_id,
                "twin_id": twin_id, "consent_id": consent_id, "source_system": source_system,
                "clinical_context": clinical_context, "actor": source_system},
        task_id=job_id, queue="ingestion",
    )

    logger.info("api.dicom.accepted", twin_id=twin_id[:8]+"...", job_id=job_id, size_bytes=len(dicom_bytes))
    return {"status": "accepted", "job_id": job_id, "twin_id": twin_id, "poll_url": f"/v1/jobs/{job_id}"}

@app.get("/v1/jobs/{job_id}", tags=["Monitoring"])
async def get_job_status(job_id: str, session: AsyncSession = Depends(get_session)):
    job = await session.get(IngestionJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} introuvable")
    return {"job_id": job.job_id, "twin_id": job.twin_id, "data_type": job.data_type,
            "status": job.status, "storage_key": job.storage_key,
            "created_at": job.created_at.isoformat(),
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "error_message": job.error_message}

@app.get("/v1/twins/{twin_id}", tags=["Registry"])
async def get_twin(twin_id: str, session: AsyncSession = Depends(get_session)):
    if not (len(twin_id) == 64 and all(c in "0123456789abcdef" for c in twin_id)):
        raise HTTPException(status_code=422, detail="twin_id invalide")
    twin = await session.get(Twin, twin_id)
    if twin is None:
        raise HTTPException(status_code=404, detail=f"Twin {twin_id[:8]}... introuvable")
    return {"twin_id": twin.twin_id, "status": twin.status, "has_dicom": twin.has_dicom,
            "has_ecg": twin.has_ecg, "has_eam": twin.has_eam,
            "source_system": twin.source_system,
            "created_at": twin.created_at.isoformat(), "updated_at": twin.updated_at.isoformat()}
