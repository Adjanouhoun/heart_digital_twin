"""
Solveur couplé EP ↔ Mécanique ↔ Windkessel.

Schéma operator splitting :
  1. EP (openCARP) : V_m(x,t), Ca²⁺(x,t)
  2. Mécanique (FEniCSx) : u(x,t), tension active T_a(x,t)
  3. Windkessel 0D : P(t), V(t)
  4. Itération de point fixe jusqu'à convergence (tol = 1e-4)

Ce module orchestre les 3 solveurs et produit le vecteur d'état complet
nécessaire pour entraîner les surrogates de la Phase 04.
"""
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import structlog

from app.solver.ep.opencarp_solver import OpenCARPSolver, EPParameters, EPResult
from app.solver.hemodynamics.windkessel import WindkesselSolver, WindkesselParameters, WindkesselResult

logger = structlog.get_logger(__name__)


@dataclass
class SimulationParameters:
    """
    Paramètres complets d'une simulation CDT.
    Espace de paramètres pour le DoE Latin Hypercube Sampling.
    """
    # ── Électrophysiologie ───────────────────────────────────────────────────
    sigma_l: float = 0.30     # Conductivité longitudinale (S/m) [0.15, 0.50]
    sigma_t: float = 0.10     # Conductivité transversale (S/m) [0.05, 0.20]
    sigma_n: float = 0.05     # Conductivité normale (S/m) [0.02, 0.10]

    # ── Mécanique passive (Holzapfel-Ogden) ─────────────────────────────────
    a_kPa: float = 0.496      # Paramètre isotrope (kPa) [0.2, 1.0]
    b: float = 7.209          # Exposant isotrope [3.0, 12.0]
    a_f_kPa: float = 15.193   # Paramètre fibres (kPa) [5.0, 30.0]
    b_f: float = 20.417       # Exposant fibres [10.0, 35.0]

    # ── Mécanique active (Bestel-Clément-Sorine) ─────────────────────────────
    T_max_kPa: float = 135.0  # Tension active maximale (kPa) [80, 200]
    tau_r_ms: float = 75.0    # Temps de montée (ms) [40, 120]
    tau_d_ms: float = 150.0   # Temps de descente (ms) [80, 250]

    # ── Hémodynamique (Windkessel) ──────────────────────────────────────────
    R_p: float = 1.2e8        # Résistance périphérique (Pa·s/m³)
    C_a: float = 1.0e-8       # Compliance artérielle (m³/Pa)
    Z_c: float = 1.0e7        # Impédance caractéristique

    # ── Conditions cliniques ─────────────────────────────────────────────────
    heart_rate_bpm: float = 75.0
    ionic_model: str = "tenTusscher2006"

    # ── Durée simulation ─────────────────────────────────────────────────────
    duration_ms: float = 500.0
    dt_ep_ms: float = 0.05
    dt_mech_ms: float = 1.0


@dataclass
class CoupledSimulationResult:
    """Résultat complet d'une simulation couplée."""
    twin_id: str
    job_id: str
    parameters: SimulationParameters

    # Résultats EP
    ep_result: Optional[EPResult] = None

    # Résultats Windkessel
    wk_result: Optional[WindkesselResult] = None

    # Vecteur de sortie pour le surrogate (Phase 04)
    output_vector: Optional[np.ndarray] = None

    # Métriques de validation
    benchmark_passed: bool = False
    convergence_iterations: int = 0
    duration_seconds: float = 0.0
    error_message: Optional[str] = None

    def to_doe_row(self) -> dict:
        """Convertit le résultat en ligne pour le dataset DoE."""
        p = self.parameters
        row = {
            # Entrées (paramètres)
            "sigma_l": p.sigma_l,
            "sigma_t": p.sigma_t,
            "sigma_n": p.sigma_n,
            "a_kPa": p.a_kPa,
            "b": p.b,
            "a_f_kPa": p.a_f_kPa,
            "b_f": p.b_f,
            "T_max_kPa": p.T_max_kPa,
            "tau_r_ms": p.tau_r_ms,
            "heart_rate_bpm": p.heart_rate_bpm,
            "R_p": p.R_p,
            "C_a": p.C_a,
        }

        # Sorties EP
        if self.ep_result:
            row.update({
                "cv_ms": self.ep_result.conduction_velocity_ms,
                "apd90_ms": self.ep_result.apd90_ms,
                "act_time_mean_ms": float(self.ep_result.activation_times_ms.mean()),
                "act_time_max_ms": float(self.ep_result.activation_times_ms.max()),
                "benchmark_ep": self.ep_result.benchmark_passed,
            })

        # Sorties hémodynamiques
        if self.wk_result:
            row.update({
                "ef_pct": self.wk_result.ef_pct,
                "edv_mL": self.wk_result.edv_mL,
                "esv_mL": self.wk_result.esv_mL,
                "sv_mL": self.wk_result.sv_mL,
                "p_systolic_mmHg": self.wk_result.p_systolic_mmHg,
                "p_diastolic_mmHg": self.wk_result.p_diastolic_mmHg,
                "p_mean_mmHg": self.wk_result.p_mean_mmHg,
                "dp_dt_max": self.wk_result.dp_dt_max,
                "co_L_min": self.wk_result.cardiac_output_L_min,
                "slo_wk": self.wk_result.slo_passed,
            })

        row["sim_duration_s"] = self.duration_seconds
        row["benchmark_passed"] = self.benchmark_passed
        return row


class CoupledSolver:
    """
    Orchestrateur du solveur multi-physique couplé.
    Implémente le schéma operator splitting EP↔Mécanique↔Windkessel.
    """

    MAX_ITER = 10          # Itérations max de point fixe
    TOL_EF = 0.01          # Tolérance convergence EF (1%)

    def __init__(self) -> None:
        self._ep_solver = OpenCARPSolver()
        self._wk_solver = WindkesselSolver()

    def simulate(
        self,
        params: SimulationParameters,
        nodes: np.ndarray,
        elements: np.ndarray,
        fiber_vectors: np.ndarray,
        twin_id: str,
        job_id: Optional[str] = None,
    ) -> CoupledSimulationResult:
        """
        Lance une simulation couplée complète.

        Schéma operator splitting :
        1. EP → activation times, Ca²⁺
        2. Mécanique → volume waveform V(t)
        3. Windkessel → pression P(t)
        4. Point fixe → convergence EF
        """
        if job_id is None:
            job_id = str(uuid.uuid4())

        t0 = time.time()
        logger.info("coupled_solver.start",
                   twin_id=twin_id, job_id=job_id,
                   sigma_l=params.sigma_l, T_max=params.T_max_kPa)

        result = CoupledSimulationResult(
            twin_id=twin_id,
            job_id=job_id,
            parameters=params,
        )

        try:
            # ── Step 1 : Simulation EP ────────────────────────────────────────
            ep_params = EPParameters(
                sigma_l=params.sigma_l,
                sigma_t=params.sigma_t,
                sigma_n=params.sigma_n,
                ionic_model=params.ionic_model,
                duration_ms=params.duration_ms,
                dt_ms=params.dt_ep_ms,
                bcl_ms=60000.0 / params.heart_rate_bpm,
            )

            ep_result = self._ep_solver.simulate(
                params=ep_params,
                nodes=nodes,
                elements=elements,
                fiber_vectors=fiber_vectors,
                twin_id=twin_id,
                job_id=job_id,
            )
            result.ep_result = ep_result

            # ── Step 2 : Mécanique → Volume waveform ──────────────────────────
            # Approximation initiale : volume basé sur les paramètres BCS
            volume_waveform = self._compute_volume_waveform(
                params=params,
                activation_times=ep_result.activation_times_ms,
                nodes=nodes,
            )

            # ── Step 3 : Itération de point fixe ──────────────────────────────
            ef_prev = 0.0
            wk_result = None

            for iteration in range(self.MAX_ITER):
                wk_params = WindkesselParameters(
                    R_p=params.R_p,
                    C_a=params.C_a,
                    Z_c=params.Z_c,
                    heart_rate_bpm=params.heart_rate_bpm,
                    V_ed_mL=float(volume_waveform.max()),
                    V_es_mL=float(volume_waveform.min()),
                )

                wk_result = self._wk_solver.simulate(
                    params=wk_params,
                    volume_waveform_mL=volume_waveform,
                    dt_ms=params.dt_mech_ms,
                    n_cycles=3,
                )

                # Convergence ?
                ef_delta = abs(wk_result.ef_pct - ef_prev)
                if ef_delta < self.TOL_EF * 100 and iteration > 0:
                    logger.info("coupled_solver.converged",
                               iterations=iteration+1, ef=wk_result.ef_pct)
                    result.convergence_iterations = iteration + 1
                    break

                ef_prev = wk_result.ef_pct

                # Mettre à jour la waveform de volume avec la pression
                volume_waveform = self._update_volume_waveform(
                    volume_waveform, wk_result, params
                )

            result.wk_result = wk_result
            result.convergence_iterations = result.convergence_iterations or self.MAX_ITER

            # ── Validation benchmark ──────────────────────────────────────────
            result.benchmark_passed = (
                ep_result.benchmark_passed and
                (wk_result.slo_passed if wk_result else False)
            )

            # ── Vecteur de sortie pour Phase 04 ──────────────────────────────
            result.output_vector = self._build_output_vector(ep_result, wk_result)

        except Exception as e:
            result.error_message = str(e)
            logger.error("coupled_solver.failed",
                        twin_id=twin_id, job_id=job_id, error=str(e))

        result.duration_seconds = time.time() - t0

        logger.info("coupled_solver.complete",
                   twin_id=twin_id, job_id=job_id,
                   ef=result.wk_result.ef_pct if result.wk_result else None,
                   benchmark=result.benchmark_passed,
                   duration_s=round(result.duration_seconds, 1))

        return result

    def _compute_volume_waveform(
        self,
        params: SimulationParameters,
        activation_times: np.ndarray,
        nodes: np.ndarray,
    ) -> np.ndarray:
        """
        Calcule la waveform de volume ventriculaire.
        Approximation basée sur le modèle BCS et les temps d'activation.
        En production : remplacer par FEniCSx.
        """
        bcl_ms = 60000.0 / params.heart_rate_bpm
        t = np.arange(0, bcl_ms, params.dt_mech_ms)

        # Durée de systole
        t_sys = 300.0  # ms typique

        # Volume télédiastolique estimé depuis la géométrie du maillage
        if len(nodes) > 0:
            bbox = nodes.max(axis=0) - nodes.min(axis=0)
            V_ed = float(np.prod(bbox) * 0.4 * 1e-3)  # Approximation ellipsoïde
            V_ed = np.clip(V_ed, 80, 200)  # mL
        else:
            V_ed = 130.0

        # EF estimé depuis T_max
        ef_est = 0.3 + 0.3 * (params.T_max_kPa / 135.0)
        ef_est = np.clip(ef_est, 0.25, 0.75)
        V_es = V_ed * (1 - ef_est)
        sv = V_ed - V_es

        # Waveform sigmoïde
        V = np.zeros(len(t))
        for i, ti in enumerate(t):
            if ti < t_sys:
                phase = ti / t_sys
                V[i] = V_ed - sv * (3 * phase**2 - 2 * phase**3)
            else:
                phase = (ti - t_sys) / (bcl_ms - t_sys)
                V[i] = V_es + sv * (3 * phase**2 - 2 * phase**3)

        return V

    def _update_volume_waveform(
        self,
        volume_old: np.ndarray,
        wk_result: WindkesselResult,
        params: SimulationParameters,
    ) -> np.ndarray:
        """
        Met à jour la waveform de volume en fonction de la pression (couplage).
        Implémente la Frank-Starling : augmentation EDV si précharge augmente.
        """
        # Ajustement Frank-Starling simplifié
        p_mean = wk_result.p_mean_mmHg
        p_target = 95.0  # mmHg cible

        # Correction du volume de remplissage
        correction = 1.0 + 0.1 * (p_target - p_mean) / p_target
        correction = np.clip(correction, 0.8, 1.2)

        return volume_old * correction

    def _build_output_vector(
        self,
        ep_result: Optional[EPResult],
        wk_result: Optional[WindkesselResult],
    ) -> np.ndarray:
        """
        Construit le vecteur de sortie scalaire pour le GP Emulator (Phase 04).
        Champs spatiaux → GNN (séparément).
        Scalaires → ce vecteur.
        """
        outputs = []

        if ep_result:
            outputs.extend([
                ep_result.conduction_velocity_ms,
                ep_result.apd90_ms,
                float(ep_result.activation_times_ms.mean()),
                float(ep_result.activation_times_ms.std()),
                float(ep_result.activation_times_ms.max()),
            ])

        if wk_result:
            outputs.extend([
                wk_result.ef_pct,
                wk_result.edv_mL,
                wk_result.esv_mL,
                wk_result.sv_mL,
                wk_result.p_systolic_mmHg,
                wk_result.p_diastolic_mmHg,
                wk_result.p_mean_mmHg,
                wk_result.dp_dt_max,
                wk_result.cardiac_output_L_min,
            ])

        return np.array(outputs, dtype=np.float64)
