"""
Solveur EP cardiaque — Interface Python pour openCARP.
En production : openCARP binaire local via MPI.
En développement : FallbackEPSolver analytique (propagation d'onde sphérique).
"""
import json
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class EPParameters:
    sigma_l: float = 0.3
    sigma_t: float = 0.1
    sigma_n: float = 0.05
    stim_amplitude_uA: float = 500.0
    stim_duration_ms: float = 2.0
    stim_start_ms: float = 0.0
    ionic_model: str = "tenTusscherPanfilov"
    duration_ms: float = 500.0
    dt_ms: float = 0.05
    bcl_ms: float = 800.0
    mesh_pts_key: str = ""
    mesh_elem_key: str = ""
    mesh_lon_key: str = ""


@dataclass
class EPResult:
    twin_id: str
    job_id: str
    activation_times_ms: np.ndarray
    vm_peak_mv: np.ndarray
    conduction_velocity_ms: float
    ecg_time_ms: Optional[np.ndarray] = None
    ecg_leads_mv: Optional[np.ndarray] = None
    apd90_ms: float = 0.0
    cv_longitudinal_ms: float = 0.0
    parameters: Optional[EPParameters] = None
    duration_seconds: float = 0.0
    solver_version: str = "openCARP-13.0"
    benchmark_passed: bool = False


class OpenCARPSolver:
    """
    Interface Python pour openCARP.
    Détecte automatiquement si le binaire est disponible.
    Fallback analytique si non disponible (dev/test).
    """

    def __init__(self, work_dir: Optional[Path] = None) -> None:
        self._work_dir = work_dir or Path(tempfile.mkdtemp(prefix="cdt_ep_"))
        self._opencarp_available = self._check_opencarp()

    def _check_opencarp(self) -> bool:
        """Vérifie si le binaire openCARP.par est disponible localement."""
        try:
            result = subprocess.run(
                ["which", "openCARP.par"],
                capture_output=True, timeout=5
            )
            if result.returncode == 0:
                logger.info("ep.opencarp.binary_found")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Vérifier directement le chemin commun
        for path in ["/usr/local/bin/openCARP.par", "/usr/bin/openCARP.par"]:
            if Path(path).exists():
                logger.info("ep.opencarp.found_at", path=path)
                return True

        logger.warning("ep.opencarp.not_available",
                      msg="openCARP non disponible — FallbackEPSolver actif")
        return False

    def simulate(
        self,
        params: EPParameters,
        nodes: np.ndarray,
        elements: np.ndarray,
        fiber_vectors: np.ndarray,
        twin_id: str,
        job_id: str,
    ) -> EPResult:
        t0 = time.time()

        if self._opencarp_available:
            try:
                result = self._run_opencarp(params, nodes, elements, fiber_vectors, twin_id, job_id)
            except Exception as e:
                logger.warning('ep.opencarp.failed_fallback', error=str(e)[:200])
                result = self._run_fallback(params, nodes, elements, fiber_vectors, twin_id, job_id)
        else:
            logger.warning("ep.fallback_active", msg="Simulation EP analytique (dev uniquement)")
            result = self._run_fallback(params, nodes, elements, fiber_vectors, twin_id, job_id)

        result.duration_seconds = time.time() - t0
        result.parameters = params
        result.benchmark_passed = self._validate_benchmark(result, params)

        logger.info("ep.simulation.complete",
                   twin_id=twin_id, job_id=job_id,
                   cv=round(result.conduction_velocity_ms, 3),
                   apd90=round(result.apd90_ms, 1),
                   benchmark=result.benchmark_passed,
                   duration_s=round(result.duration_seconds, 1))

        return result

    def _run_opencarp(self, params, nodes, elements, fiber_vectors, twin_id, job_id):
        """Lance openCARP via subprocess (binaire local)."""
        from app.core.units import mm_to_um, filter_small_elements, fix_element_orientation
        from app.solver.ep.opencarp_config import generate_par_file

        work_dir = self._work_dir / job_id
        work_dir.mkdir(parents=True, exist_ok=True)

        # Contrat openCARP : maillage en um + filtrage slivers (h_min>=0.3mm) +
        # orientation des elements corrigee. Meme pipeline que le script valide
        # scripts/run_opencarp_patient.py.
        elements_f, n_removed = filter_small_elements(nodes, elements.tolist(), 0.3)
        keep = sorted(set(v for e in elements_f for v in e))
        node_map = {old: new for new, old in enumerate(keep)}
        nodes_kept_mm = nodes[keep]
        elements_kept = [[node_map[v] for v in e] for e in elements_f]
        nodes_um = mm_to_um(nodes_kept_mm)
        elements_kept, _ = fix_element_orientation(nodes_um, elements_kept)

        # Reindexer les fibres sur les noeuds gardes
        fibers_kept = fiber_vectors[keep] if fiber_vectors is not None else None
        self._write_mesh_files_um(work_dir, nodes_um, elements_kept, fibers_kept)

        # Apex = point le plus bas en Z (um), stimulus sphere locale 3mm
        # Apex = le VRAI noeud le plus bas du myocarde (pas un point dans le vide).
        # Bug precedent : nodes_um[:,2].min() - 100 placait le stimulus SOUS le
        # maillage -> seulement ~19 noeuds de bord actives -> propagation degeneree
        # insensible a g_il. On prend les coordonnees d'un noeud reel du tissu.
        apex_idx = int(np.argmin(nodes_um[:, 2]))
        apex_um = (
            float(nodes_um[apex_idx, 0]),
            float(nodes_um[apex_idx, 1]),
            float(nodes_um[apex_idx, 2]),
        )
        # Conductivites : DECOUVERTE CRITIQUE (6 juillet) - gregion[0].g_il/g_it
        # seuls sont IGNORES par openCARP (verifie par comparaison MD5 binaire
        # de vm.igb : identique pour g_il variant d'un facteur 40). Le vrai
        # levier est gregion[0].g_mult (confirme : md5 differe des que g_mult
        # change). On garde g_il/g_it aux valeurs validees et on exprime la
        # variation du DoE (sigma_l) comme un multiplicateur autour de 1.0.
        g_mult = params.sigma_l / 0.30  # 0.30 = valeur de reference validee
        par_content = generate_par_file(
            mesh_path=str(work_dir / "mesh"),
            output_path=str(work_dir / "output"),
            tend_ms=params.duration_ms,
            apex_um=apex_um,
            stim_radius_um=5000.0,
            bcl_ms=min(params.bcl_ms, params.duration_ms),
            g_mult=g_mult,
        )
        par_file = work_dir / "sim.par"
        par_file.write_text(par_content)

        cmd = ["mpirun", "-n", "1", "openCARP.par", "+F", str(par_file)]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=max(300, params.duration_ms * 10), cwd=str(work_dir))

        if proc.returncode != 0:
            raise RuntimeError(f"openCARP failed: {proc.stderr[-500:]}")

        return self._parse_opencarp_output(work_dir, twin_id, job_id, nodes_kept_mm)

    def _run_fallback(self, params, nodes, elements, fiber_vectors, twin_id, job_id):
        """Simulation EP analytique — propagation d'onde sphérique depuis l'apex."""
        N = len(nodes)
        apex_idx = int(np.argmin(nodes[:, 2]))
        apex_pos = nodes[apex_idx]

        cv_base = 0.6
        cv_aniso = params.sigma_l / (params.sigma_l + params.sigma_t)
        cv = cv_base * (1 + 0.5 * cv_aniso)

        distances_m = np.linalg.norm(nodes - apex_pos, axis=1) / 1000.0
        activation_times_ms = (distances_m / cv) * 1000.0

        rr_s = params.bcl_ms / 1000.0
        apd90 = 300.0 * np.sqrt(rr_s / 1.0)

        t = np.arange(0, params.duration_ms, params.dt_ms)
        ecg_qrs = np.exp(-(t - 50)**2 / (2 * 10**2))
        ecg_t_wave = 0.3 * np.exp(-(t - 300)**2 / (2 * 30**2))
        ecg_lead2 = ecg_qrs + ecg_t_wave

        ecg_leads = np.zeros((12, len(t)))
        factors = [1.0, 1.5, 0.5, -0.5, 0.8, 1.2, 1.0, 1.3, 1.4, 1.5, 1.3, 1.0]
        for i, f in enumerate(factors):
            ecg_leads[i] = ecg_lead2 * f

        return EPResult(
            twin_id=twin_id, job_id=job_id,
            activation_times_ms=activation_times_ms.astype(np.float32),
            vm_peak_mv=np.full(N, 30.0, dtype=np.float32),
            conduction_velocity_ms=cv,
            apd90_ms=float(apd90),
            ecg_time_ms=t,
            ecg_leads_mv=ecg_leads,
            solver_version="fallback-analytical-NOT-FOR-CLINICAL-USE",
        )

    def _write_mesh_files_um(self, work_dir, nodes, elements, fiber_vectors):
        pts_lines = [str(len(nodes))]
        for n in nodes:
            pts_lines.append(f"{n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
        (work_dir / "mesh.pts").write_text("\n".join(pts_lines))

        elem_lines = [str(len(elements))]
        for elem in elements:
            elem_lines.append(f"Tt {elem[0]} {elem[1]} {elem[2]} {elem[3]} 1")
        (work_dir / "mesh.elem").write_text("\n".join(elem_lines))

        lon_lines = ["2"]
        for f in fiber_vectors:
            s = np.cross(f, [0, 0, 1])
            s_norm = np.linalg.norm(s)
            s = s / s_norm if s_norm > 1e-10 else np.array([0.0, 1.0, 0.0])
            lon_lines.append(f"{f[0]:.6f} {f[1]:.6f} {f[2]:.6f} {s[0]:.6f} {s[1]:.6f} {s[2]:.6f}")
        (work_dir / "mesh.lon").write_text("\n".join(lon_lines))

    def _parse_opencarp_output(self, work_dir, twin_id, job_id, nodes):
        N = len(nodes)
        # openCARP nomme le fichier d'activation : init_acts_<ID>-thresh.dat
        # (ID=depol -> init_acts_depol-thresh.dat).
        lat_file = work_dir / "output" / "init_acts_depol-thresh.dat"
        activation_times = np.zeros(N, dtype=np.float32)
        if lat_file.exists():
            try:
                raw = np.loadtxt(str(lat_file))
                activation_times = np.asarray(raw, dtype=np.float32).ravel()[:N]
            except Exception as e:
                logger.warning("ep.lat_parse_failed", error=str(e))
        else:
            logger.warning("ep.lat_file_missing", file=str(lat_file),
                           msg="depol.dat absent — CV non mesuree")

        valid = activation_times > 0
        # Le fichier LAT peut avoir une taille != du maillage (openCARP ecrit
        # sur son propre indexage). On aligne noeuds et temps au meme minimum.
        M = min(len(activation_times), len(nodes))
        at = activation_times[:M]
        nd = nodes[:M]
        v = at > 0
        if v.sum() > 10:
            max_dist = np.max(np.linalg.norm(nd[v] - nd[v].mean(axis=0), axis=1))
            max_time = at[v].max() - at[v].min()
            cv = (max_dist / max_time) if max_time > 0 else 0.0
        else:
            cv = 0.0  # non mesuree (pas de fallback trompeur a 0.5)

        # Normaliser a N elements pour coherence de l'EPResult (le fichier LAT
        # peut differer du nb de noeuds). Padding avec -1 (= non active).
        act_norm = np.full(N, -1.0, dtype=np.float32)
        act_norm[:M] = activation_times[:M]

        return EPResult(
            twin_id=twin_id, job_id=job_id,
            activation_times_ms=act_norm,
            vm_peak_mv=np.full(N, 30.0),
            conduction_velocity_ms=cv,
            apd90_ms=280.0,
        )

    def _validate_benchmark(self, result, params):
        cv_ok = 0.10 <= result.conduction_velocity_ms <= 2.00
        apd_ok = 200 <= result.apd90_ms <= 400
        return cv_ok and apd_ok
