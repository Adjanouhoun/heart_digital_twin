"""
Tâche Celery — Orchestration du DoE Phase 03.
Lance 500+ simulations en parallèle et archive les résultats dans MinIO.
"""
import base64
import json
import uuid
from datetime import datetime
from pathlib import Path
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


def run_doe_batch(
    twin_id: str,
    n_simulations: int = 500,
    nodes_b64: str = "",
    elements_b64: str = "",
    nodes_shape: list = None,
    elements_shape: list = None,
    fiber_b64: str = "",
    seed: int = 42,
) -> dict:
    """
    Lance le DoE complet : LHS → simulations couplées → archivage MinIO.
    Retourne un résumé du DoE.
    """
    from app.solver.doe.latin_hypercube import LatinHypercubeSampler
    from app.solver.coupled_solver import CoupledSolver

    logger.info("doe.start", twin_id=twin_id, n_simulations=n_simulations)

    # Reconstruire les arrays numpy
    nodes = np.frombuffer(
        base64.b64decode(nodes_b64), dtype=np.float64
    ).reshape(nodes_shape) if nodes_b64 else np.zeros((8, 3))

    elements = np.frombuffer(
        base64.b64decode(elements_b64), dtype=np.int32
    ).reshape(elements_shape) if elements_b64 else np.zeros((6, 4), dtype=np.int32)

    fiber_vectors = np.frombuffer(
        base64.b64decode(fiber_b64), dtype=np.float64
    ).reshape(-1, 3) if fiber_b64 else np.tile([1, 0, 0], (len(nodes), 1))

    # Générer le plan LHS
    sampler = LatinHypercubeSampler(seed=seed)
    params_list = sampler.sample(n_simulations)

    # Lancer les simulations
    solver = CoupledSolver()
    results = []
    doe_rows = []
    n_success = 0
    n_failed = 0

    for i, params in enumerate(params_list):
        job_id = str(uuid.uuid4())
        try:
            result = solver.simulate(
                params=params,
                nodes=nodes,
                elements=elements,
                fiber_vectors=fiber_vectors,
                twin_id=twin_id,
                job_id=job_id,
            )
            row = result.to_doe_row()
            doe_rows.append(row)
            n_success += 1

            if (i + 1) % 50 == 0:
                logger.info("doe.progress",
                           completed=i+1, total=n_simulations,
                           success=n_success, failed=n_failed)

        except Exception as e:
            n_failed += 1
            logger.warning("doe.simulation_failed",
                          index=i, error=str(e))

    # Archiver dans MinIO
    doe_id = str(uuid.uuid4())
    _archive_doe_results(twin_id, doe_id, doe_rows)

    summary = {
        "doe_id": doe_id,
        "twin_id": twin_id,
        "n_total": n_simulations,
        "n_success": n_success,
        "n_failed": n_failed,
        "success_rate": round(n_success / n_simulations * 100, 1),
        "slo_passed": n_success >= 500,
        "timestamp": datetime.utcnow().isoformat(),
    }

    logger.info("doe.complete", **summary)
    return summary


def _archive_doe_results(twin_id: str, doe_id: str, doe_rows: list) -> None:
    """Archive les résultats DoE dans MinIO."""
    try:
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "s3",
            endpoint_url="http://minio:9000",
            aws_access_key_id="cdt_admin",
            aws_secret_access_key="cdt_minio_2024",
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )

        # Sauvegarder en JSON
        doe_json = json.dumps(doe_rows, indent=2).encode()
        key = f"{twin_id}/doe/{doe_id}/doe_results.json"

        client.put_object(
            Bucket="cdt-meshes",
            Key=key,
            Body=doe_json,
            ContentType="application/json",
        )

        logger.info("doe.archived", key=key, n_rows=len(doe_rows))

    except Exception as e:
        logger.error("doe.archive_failed", error=str(e))
        # Fallback : sauvegarder localement
        Path(f"/tmp/doe_{doe_id}.json").write_bytes(
            json.dumps(doe_rows, indent=2).encode()
        )
