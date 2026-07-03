import io
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import pydicom
from pydicom.dataset import Dataset
import structlog
from app.anonymizer.dicom_anonymizer import DicomAnonymizer
from app.config import get_settings
from app.storage.minio_client import CDTStorageClient

logger = structlog.get_logger(__name__)

@dataclass
class DicomIngestionResult:
    twin_id: str
    job_id: str
    storage_key: str
    original_sha256: str
    tags_removed: int
    tags_modified: int
    num_series: int
    warnings: list
    duration_ms: float

class DicomIngestor:
    def __init__(self, anonymizer: DicomAnonymizer, storage: CDTStorageClient) -> None:
        self._anonymizer = anonymizer
        self._storage = storage

    async def ingest(self, dicom_bytes: bytes, job_id: str, consent_id: str, source_system: str, clinical_context=None, actor="api", source_ip=None) -> DicomIngestionResult:
        t_start = datetime.utcnow()
        anon_result = self._anonymizer.anonymize_bytes(dicom_bytes)
        twin_id = anon_result.twin_id

        anon_bytes = self._dataset_to_bytes(anon_result.anonymized_dataset)
        ds = anon_result.anonymized_dataset

        study_uid  = self._get_tag(ds, "StudyInstanceUID",  "unknown_study")
        series_uid = self._get_tag(ds, "SeriesInstanceUID", "unknown_series")
        sop_uid    = self._get_tag(ds, "SOPInstanceUID",    "unknown_sop")

        storage_result = self._storage.store_dicom(
            dicom_bytes=anon_bytes, twin_id=twin_id,
            study_uid=study_uid, series_uid=series_uid, sop_uid=sop_uid,
        )

        duration_ms = (datetime.utcnow() - t_start).total_seconds() * 1000
        logger.info("dicom.ingestor.complete", twin_id=twin_id, storage_key=storage_result.key, duration_ms=round(duration_ms, 1))

        return DicomIngestionResult(
            twin_id=twin_id, job_id=job_id, storage_key=storage_result.key,
            original_sha256=anon_result.original_sha256, tags_removed=anon_result.tags_removed,
            tags_modified=anon_result.tags_modified, num_series=1,
            warnings=anon_result.warnings, duration_ms=duration_ms,
        )

    def _dataset_to_bytes(self, ds: Dataset) -> bytes:
        buf = io.BytesIO()
        pydicom.dcmwrite(buf, ds)
        return buf.getvalue()

    def _get_tag(self, ds: Dataset, tag_name: str, default: str) -> str:
        try:
            return str(getattr(ds, tag_name, default)).strip() or default
        except Exception:
            return default
