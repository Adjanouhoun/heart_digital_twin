import asyncio
import base64
import uuid
import tempfile
import pathlib
from celery import Celery, chain
from celery.utils.log import get_task_logger

REDIS_URL = "redis://:cdt_redis_2024@redis:6379/0"
logger = get_task_logger(__name__)

app = Celery("cdt-phase02")
app.conf.update(
    broker_url=REDIS_URL,
    result_backend=REDIS_URL,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    task_acks_late=True,
    task_soft_time_limit=600,
    task_time_limit=900,
    task_default_queue="reconstruction",
)


@app.task(bind=True, name="app.tasks.reconstruction.preprocess_task", max_retries=2)
def preprocess_task(self, dicom_b64, job_id, twin_id, source_format="nifti"):
    from app.segmentation.preprocessor import CardiacMRIPreprocessor
    logger.info(f"[{job_id}] Prétraitement démarré")
    raw_bytes = base64.b64decode(dicom_b64)
    proc = CardiacMRIPreprocessor()
    result = proc.preprocess_nifti_bytes(raw_bytes)
    return {
        "job_id": job_id, "twin_id": twin_id,
        "nifti_b64": base64.b64encode(result.nifti_bytes).decode(),
        "spacing_mm": list(result.spacing_mm),
        "volume_shape": list(result.volume.shape),
        "preprocessing_log": result.preprocessing_log,
    }


@app.task(bind=True, name="app.tasks.reconstruction.segment_task", max_retries=2)
def segment_task(self, preprocess_result):
    import numpy as np, SimpleITK as sitk
    from app.segmentation.nnunet_wrapper import get_segmenter
    job_id  = preprocess_result["job_id"]
    twin_id = preprocess_result["twin_id"]
    spacing_mm = tuple(preprocess_result["spacing_mm"])
    logger.info(f"[{job_id}] Segmentation démarrée")
    nifti_bytes = base64.b64decode(preprocess_result["nifti_b64"])
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        f.write(nifti_bytes)
        tmp = f.name
    image = sitk.ReadImage(tmp)
    volume = sitk.GetArrayFromImage(image).astype(np.float32)
    pathlib.Path(tmp).unlink(missing_ok=True)
    seg = get_segmenter()
    result = seg.predict(volume, spacing_mm)
    logger.info(f"[{job_id}] Segmentation terminée: LV={result.volume_lv_ml}mL SCAR={result.scar_burden_pct}%")
    return {
        "job_id": job_id, "twin_id": twin_id,
        "mask_b64": base64.b64encode(result.mask.tobytes()).decode(),
        "mask_shape": list(result.mask.shape),
        "spacing_mm": list(spacing_mm),
        "volume_lv_ml": result.volume_lv_ml, "volume_rv_ml": result.volume_rv_ml,
        "volume_myo_ml": result.volume_myo_ml, "volume_scar_ml": result.volume_scar_ml,
        "scar_burden_pct": result.scar_burden_pct, "model_name": result.model_name,
    }


@app.task(bind=True, name="app.tasks.reconstruction.mesh_task", max_retries=1)
def mesh_task(self, segment_result):
    import numpy as np
    from app.meshing.gmsh_mesher import CardiacMesher
    job_id  = segment_result["job_id"]
    twin_id = segment_result["twin_id"]
    spacing_mm = tuple(segment_result["spacing_mm"])
    mask = np.frombuffer(base64.b64decode(segment_result["mask_b64"]), dtype=np.uint8).reshape(segment_result["mask_shape"])
    logger.info(f"[{job_id}] Maillage démarré")
    mesher = CardiacMesher()
    result = mesher.mesh_from_segmentation(mask, spacing_mm)
    mesh_id = str(uuid.uuid4())
    logger.info(f"[{job_id}] Maillage terminé: {result.num_nodes} noeuds, QC={'OK' if result.qc_passed else 'FAIL'}")
    return {
        "job_id": job_id, "twin_id": twin_id, "mesh_id": mesh_id,
        "nodes_b64": base64.b64encode(result.nodes.tobytes()).decode(),
        "nodes_shape": list(result.nodes.shape),
        "elements_b64": base64.b64encode(result.elements.tobytes()).decode(),
        "elements_shape": list(result.elements.shape),
        "elem_tags_b64": base64.b64encode(result.element_tags.tobytes()).decode(),
        "num_nodes": result.num_nodes, "num_elements": result.num_elements,
        "min_jacobian": result.min_jacobian, "num_degenerate": result.num_degenerate,
        "qc_passed": result.qc_passed, "duration_seconds": result.duration_seconds,
    }


@app.task(bind=True, name="app.tasks.reconstruction.fibers_task", max_retries=1)
def fibers_task(self, mesh_result):
    import numpy as np
    from app.fibers.ldrb import LDRBFiberGenerator
    job_id  = mesh_result["job_id"]
    twin_id = mesh_result["twin_id"]
    nodes    = np.frombuffer(base64.b64decode(mesh_result["nodes_b64"]),    dtype=np.float64).reshape(mesh_result["nodes_shape"])
    elements = np.frombuffer(base64.b64decode(mesh_result["elements_b64"]), dtype=np.int32).reshape(mesh_result["elements_shape"])
    elem_tags = np.frombuffer(base64.b64decode(mesh_result["elem_tags_b64"]), dtype=np.int32)
    logger.info(f"[{job_id}] Fibres LDRB démarrées")
    ldrb = LDRBFiberGenerator()
    result = ldrb.generate(nodes, elements, elem_tags)
    logger.info(f"[{job_id}] Fibres terminées ({len(nodes)} noeuds)")
    return {
        "job_id": job_id, "twin_id": twin_id, "mesh_id": mesh_result["mesh_id"],
        "num_nodes": mesh_result["num_nodes"], "num_elements": mesh_result["num_elements"],
        "min_jacobian": mesh_result["min_jacobian"], "num_degenerate": mesh_result["num_degenerate"],
        "qc_passed": mesh_result["qc_passed"], "fiber_algorithm": result.algorithm,
        "status": "complete",
    }


def run_reconstruction_pipeline(nifti_b64, job_id, twin_id):
    pipeline = chain(
        preprocess_task.s(dicom_b64=nifti_b64, job_id=job_id, twin_id=twin_id),
        segment_task.s(),
        mesh_task.s(),
        fibers_task.s(),
    )
    result = pipeline.apply_async(queue="reconstruction")
    return result.id
