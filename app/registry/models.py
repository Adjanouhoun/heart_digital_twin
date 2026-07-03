from datetime import datetime
from typing import Optional
from sqlmodel import Field, SQLModel, Column
from sqlalchemy import Text, DateTime, String
from sqlalchemy.sql import func

class Twin(SQLModel, table=True):
    __tablename__ = "twins"
    twin_id: str = Field(sa_column=Column(String(64), primary_key=True))
    consent_id: str = Field(max_length=128)
    source_system: str = Field(max_length=64)
    status: str = Field(default="created", max_length=32)
    has_dicom: bool = Field(default=False)
    has_ecg: bool = Field(default=False)
    has_eam: bool = Field(default=False)
    clinical_context: Optional[str] = Field(default=None, sa_column=Column(Text))
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
    )

class IngestionJob(SQLModel, table=True):
    __tablename__ = "ingestion_jobs"
    job_id: str = Field(max_length=64, primary_key=True)
    twin_id: str = Field(max_length=64, foreign_key="twins.twin_id", index=True)
    data_type: str = Field(max_length=16)
    status: str = Field(default="pending", max_length=16)
    error_message: Optional[str] = Field(default=None, sa_column=Column(Text))
    storage_key: Optional[str] = Field(default=None, max_length=512)
    file_size_bytes: Optional[int] = Field(default=None)
    original_sha256: Optional[str] = Field(default=None, max_length=64)
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )

class AuditLog(SQLModel, table=True):
    __tablename__ = "audit_log"
    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
    )
    twin_id: Optional[str] = Field(default=None, max_length=64, index=True)
    operation: str = Field(max_length=64)
    actor: str = Field(max_length=128)
    source_ip: Optional[str] = Field(default=None, max_length=45)
    job_id: Optional[str] = Field(default=None, max_length=64)
    status: str = Field(max_length=16)
    details: Optional[str] = Field(default=None, sa_column=Column(Text))
