import asyncio
import base64
from celery import Celery
from celery.utils.log import get_task_logger
from app.config import get_settings

settings = get_settings()
logger = get_task_logger(__name__)

app = Celery("cdt-ingestion")
app.conf.update(
    broker_url=settings.redis_url,
    result_backend=settings.redis_url,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    task_acks_late=True,
    task_soft_time_limit=120,
    task_time_limit=180,
    task_default_queue="ingestion",
)

@app.task(bind=True, name="app.tasks.worker.ingest_dicom_task", max_retries=3, default_retry_delay=10)
def ingest_dicom_task(self, dicom_b64: str, job_id: str, twin_id: str, consent_id: str, source_system: str, clinical_context=None, actor="celery"):
    from app.anonymizer.dicom_anonymizer import DicomAnonymizer
    from app.ingestors.dicom_ingestor import DicomIngestor
    from app.storage.minio_client import CDTStorageClient

    dicom_bytes = base64.b64decode(dicom_b64)
    anonymizer = DicomAnonymizer(seed=settings.twin_id_seed)
    storage = CDTStorageClient()
    ingestor = DicomIngestor(anonymizer=anonymizer, storage=storage)

    result = asyncio.run(ingestor.ingest(
        dicom_bytes=dicom_bytes, job_id=job_id, consent_id=consent_id,
        source_system=source_system, clinical_context=clinical_context, actor=actor,
    ))
    logger.info(f"[{job_id}] DICOM ingéré: {result.storage_key} ({result.duration_ms:.0f}ms)")
    return {"twin_id": result.twin_id, "job_id": result.job_id, "storage_key": result.storage_key, "duration_ms": result.duration_ms}
