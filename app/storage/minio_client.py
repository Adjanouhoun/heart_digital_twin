import gzip
import hashlib
import io
from dataclasses import dataclass
from typing import Optional
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
import structlog
from app.config import get_settings

logger = structlog.get_logger(__name__)

@dataclass
class StorageResult:
    bucket: str
    key: str
    etag: str
    size_bytes: int
    content_sha256: str

class CDTStorageClient:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = boto3.client(
            "s3",
            endpoint_url=f"http{'s' if settings.minio_use_ssl else ''}://{settings.minio_endpoint}",
            aws_access_key_id=settings.minio_access_key,
            aws_secret_access_key=settings.minio_secret_key,
            config=Config(signature_version="s3v4", retries={"max_attempts": 3, "mode": "adaptive"}),
            region_name="us-east-1",
        )
        self._settings = settings

    def store_dicom(self, dicom_bytes: bytes, twin_id: str, study_uid: str, series_uid: str, sop_uid: str) -> StorageResult:
        key = f"{twin_id}/{study_uid}/{series_uid}/{sop_uid}.dcm"
        return self._put_object(bucket=self._settings.bucket_dicom, key=key, data=dicom_bytes, content_type="application/dicom")

    def store_ecg_raw(self, raw_bytes: bytes, twin_id: str, job_id: str, filename: str) -> StorageResult:
        key = f"{twin_id}/{job_id}/raw/{filename}"
        return self._put_object(bucket=self._settings.bucket_ecg, key=key, data=raw_bytes)

    def store_ecg_signal(self, signal_array_bytes: bytes, twin_id: str, job_id: str, lead_name: str) -> StorageResult:
        compressed = gzip.compress(signal_array_bytes, compresslevel=6)
        key = f"{twin_id}/{job_id}/{lead_name}.npy.gz"
        return self._put_object(bucket=self._settings.bucket_ecg, key=key, data=compressed)

    def object_exists(self, bucket: str, key: str) -> bool:
        try:
            self._client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def _put_object(self, bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream", metadata: Optional[dict] = None) -> StorageResult:
        content_sha256 = hashlib.sha256(data).hexdigest()
        extra_args: dict = {"ContentType": content_type}
        if metadata:
            extra_args["Metadata"] = metadata
        response = self._client.put_object(Bucket=bucket, Key=key, Body=io.BytesIO(data), ContentLength=len(data), **extra_args)
        etag = response.get("ETag", "").strip('"')
        logger.info("storage.stored", bucket=bucket, key=key, size_bytes=len(data))
        return StorageResult(bucket=bucket, key=key, etag=etag, size_bytes=len(data), content_sha256=content_sha256)

_storage_client: Optional[CDTStorageClient] = None

def get_storage_client() -> CDTStorageClient:
    global _storage_client
    if _storage_client is None:
        _storage_client = CDTStorageClient()
    return _storage_client
