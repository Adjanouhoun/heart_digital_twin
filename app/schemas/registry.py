from datetime import datetime
from enum import StrEnum
from typing import Optional
from pydantic import BaseModel, Field

class TwinStatus(StrEnum):
    CREATED   = "created"
    INGESTING = "ingesting"
    READY     = "ready"
    ERROR     = "error"

class IngestionStatus(StrEnum):
    PENDING    = "pending"
    PROCESSING = "processing"
    SUCCESS    = "success"
    FAILED     = "failed"

class TwinRecord(BaseModel):
    twin_id: str
    consent_id: str
    source_system: str
    status: TwinStatus = TwinStatus.CREATED
    created_at: datetime
    updated_at: datetime
    has_dicom: bool = False
    has_ecg: bool = False
    has_eam: bool = False
    clinical_context: Optional[str] = None

class IngestionJobRecord(BaseModel):
    job_id: str
    twin_id: str
    data_type: str
    status: IngestionStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    storage_key: Optional[str] = None
