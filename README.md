# CDT Phase 01 — Ingestion, Anonymisation & Registre

## Architecture

```
cdt-ingestion/
├── app/
│   ├── main.py               # FastAPI — endpoints d'ingestion
│   ├── config.py             # Settings Pydantic v2 (env vars)
│   ├── anonymizer/
│   │   └── dicom_anonymizer.py   # DICOM PS3.15 + twin_id HMAC-SHA256
│   ├── ingestors/
│   │   ├── dicom_ingestor.py     # Pipeline DICOM complet
│   │   └── ecg_ingestor.py       # Pipeline ECG MIT-BIH (wfdb)
│   ├── storage/
│   │   └── minio_client.py       # Client MinIO S3
│   ├── registry/
│   │   ├── models.py             # SQLModel ORM (Twin, IngestionJob, AuditLog)
│   │   └── database.py           # Pool PostgreSQL async
│   ├── tasks/
│   │   └── worker.py             # Celery tasks (ingest_dicom, ingest_ecg)
│   └── schemas/                  # Pydantic v2 (DICOM, ECG, EAM, Registry)
├── tests/
│   ├── test_dicom_anonymizer.py  # 16 tests unitaires ✅
│   └── test_api_integration.py   # Tests FastAPI (mock DB/Storage)
├── alembic/                      # Migrations DB versionnées
├── docker-compose.yml            # Stack complète (DB, MinIO, Redis, MLflow)
└── .env.example                  # Template variables d'environnement
```

## Stack technique

| Composant       | Technologie                        |
|-----------------|------------------------------------|
| API             | FastAPI 0.111 + uvicorn            |
| BDD             | PostgreSQL 16 + TimescaleDB (ECG)  |
| Object store    | MinIO (S3-compatible)              |
| Task queue      | Celery 5.4 + Redis 7               |
| ML tracking     | MLflow 2.13                        |
| DICOM           | pydicom 2.4                        |
| ECG             | wfdb 4.1                           |
| Validation      | Pydantic v2                        |
| ORM             | SQLModel + Alembic                 |

## Démarrage rapide

### 1. Variables d'environnement
```bash
cp .env.example .env
# Éditer .env — OBLIGATOIRE : changer TWIN_ID_SEED
python -c "import secrets; print(secrets.token_hex(32))"  # → générer le seed
```

### 2. Lancer l'infrastructure
```bash
docker-compose up -d db minio redis
# Attendre que les healthchecks passent (~15s)
docker-compose up -d mlflow api worker flower
```

### 3. Migrations DB
```bash
docker-compose exec api alembic upgrade head
```

### 4. Vérification
```bash
# Health check
curl http://localhost:8000/health

# Documentation interactive
open http://localhost:8000/docs

# Ingestion DICOM
curl -X POST http://localhost:8000/v1/ingest/dicom \
  -F "file=@scan_cardiaque.dcm" \
  -F "consent_id=CONSENT_001" \
  -F "source_system=PACS_CHU_PARIS"

# Suivi du job
curl http://localhost:8000/v1/jobs/{job_id}
```

### 5. UIs de monitoring
| Service       | URL                          | Identifiants par défaut |
|---------------|------------------------------|-------------------------|
| API Docs      | http://localhost:8000/docs   | —                       |
| MinIO Console | http://localhost:9001        | cdt_admin / voir .env   |
| MLflow UI     | http://localhost:5000        | —                       |
| Flower        | http://localhost:5555        | —                       |
| Metrics       | http://localhost:8000/metrics| —                       |

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v --cov=app --cov-report=term-missing
```

## Architecture d'anonymisation (DICOM PS3.15)

```
Fichier DICOM brut
        │
        ▼
Calcul SHA-256 (audit trail)
        │
        ▼
Extraction PatientID
        │
        ▼
twin_id = HMAC-SHA256(PatientID, SEED_INSTITUTIONNEL)
        │
        ▼
Suppression tags nominatifs (0010,xxxx + privés)
        │
        ▼
Régénération UIDs (2.25.hash_deterministe)
        │
        ▼
PatientID ← twin_id
PatientIdentityRemoved ← YES
        │
        ▼
Stockage MinIO : cdt-dicom/{twin_id}/{study_uid}/...
Registre Postgres : twins + ingestion_jobs + audit_log
```

## Clés de stockage MinIO

```
cdt-dicom/{twin_id}/{study_uid}/{series_uid}/{sop_uid}.dcm
cdt-ecg/{twin_id}/{job_id}/{lead_name}.npy.gz
cdt-eam/{twin_id}/{job_id}/eam_openep.zip
```

## Critères de succès Phase 01 (SLOs)

- ✅ twin_id HMAC-SHA256 — irréversible, déterministe, inter-institutionnel isolé
- ✅ 0 tag nominatif (0010,xxxx) en sortie de l'anonymiseur
- ✅ Round-trip DICOM : original → anonymisé → re-lisible sans erreur
- ✅ SHA-256 original conservé pour audit trail
- ✅ UIDs re-générés (format 2.25.xxx), déterministes
- ✅ TimescaleDB hypertable pour séries ECG
- ✅ Audit log immuable (RGPD Art. 30)
- ⏳ SLO ingestion < 2s (mesuré à la création du job Celery)
- ⏳ Tests de charge 1000 patients (Phase 09)

## Prochaine étape : Phase 02 — Segmentation cardiaque
```
Input  : fichiers DICOM anonymisés (MinIO cdt-dicom/)
Output : masques de segmentation (LV, RV, myocarde, cicatrices LGE)
Stack  : TotalSegmentator + nnU-Net v2 + MONAI
```
