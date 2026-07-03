"""
Modèle Windkessel 0D — Hémodynamique ventriculaire.
Modèle 3 éléments (Westkessel) : R_p, C_a, Z_c
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
import structlog

logger = structlog.get_logger(__name__)

mmHg_to_Pa = 133.322


@dataclass
class WindkesselParameters:
    # Paramètres calibrés pour produire ~120/80 mmHg à HR=75
    R_p: float = 1.5e8       # Résistance périphérique (Pa·s/m³)
    C_a: float = 1.0e-8      # Compliance artérielle (m³/Pa)
    Z_c: float = 8.0e6       # Impédance caractéristique
    P0_mmHg: float = 80.0
    V_ed_mL: float = 130.0
    V_es_mL: float = 50.0
    heart_rate_bpm: float = 75.0


@dataclass
class WindkesselResult:
    time_ms: np.ndarray
    pressure_mmHg: np.ndarray
    volume_mL: np.ndarray
    aortic_pressure_mmHg: np.ndarray
    ef_pct: float
    edv_mL: float
    esv_mL: float
    sv_mL: float
    p_systolic_mmHg: float
    p_diastolic_mmHg: float
    p_mean_mmHg: float
    dp_dt_max: float
    cardiac_output_L_min: float
    slo_passed: bool = False
    slo_details: dict = None


class WindkesselSolver:

    def simulate(
        self,
        params: WindkesselParameters,
        volume_waveform_mL: Optional[np.ndarray] = None,
        dt_ms: float = 1.0,
        n_cycles: int = 5,
    ) -> WindkesselResult:
        bcl_ms = 60000.0 / params.heart_rate_bpm

        if volume_waveform_mL is None:
            volume_waveform_mL = self._generate_volume_waveform(params, bcl_ms, dt_ms)

        n_per_cycle = len(volume_waveform_mL)
        volume_full = np.tile(volume_waveform_mL, n_cycles)
        T = len(volume_full)
        time_ms = np.arange(T) * dt_ms

        P_ao = np.zeros(T)
        P_lv = np.zeros(T)
        P_ao[0] = params.P0_mmHg * mmHg_to_Pa

        dt_s = dt_ms * 1e-3

        for i in range(1, T):
            dV_m3 = (volume_full[i] - volume_full[i-1]) * 1e-6  # mL → m³
            Q_lv = -dV_m3 / dt_s  # m³/s
            Q_ao = max(Q_lv, 0)  # Valve aortique: flux unidirectionnel

            P_prev = P_ao[i-1]

            # RK4 Windkessel : dP/dt = (Q - P/R_p) / C_a
            def dPdt(P, Q):
                return (Q - P / params.R_p) / params.C_a

            k1 = dPdt(P_prev, Q_ao)
            k2 = dPdt(P_prev + 0.5*dt_s*k1, Q_ao)
            k3 = dPdt(P_prev + 0.5*dt_s*k2, Q_ao)
            k4 = dPdt(P_prev + dt_s*k3, Q_ao)

            P_ao[i] = P_prev + (dt_s / 6) * (k1 + 2*k2 + 2*k3 + k4)
            P_ao[i] = max(P_ao[i], 0)  # Pas de floor artificiel

            if Q_lv > 0:
                P_lv[i] = P_ao[i] + min(params.Z_c * Q_lv, 60 * mmHg_to_Pa)
            else:
                P_lv[i] = max(P_ao[i] * 0.3, 5 * mmHg_to_Pa)

        P_ao_mmHg = P_ao / mmHg_to_Pa
        P_lv_mmHg = P_lv / mmHg_to_Pa

        # Dernier cycle stabilisé
        last = (n_cycles - 1) * n_per_cycle
        P_cycle = P_lv_mmHg[last:]
        V_cycle = volume_full[last:]
        P_ao_cycle = P_ao_mmHg[last:]

        edv = float(V_cycle.max())
        esv = float(V_cycle.min())
        sv = edv - esv
        ef = (sv / edv * 100) if edv > 0 else 0.0

        p_sys = float(P_ao_cycle.max())
        p_dia = float(P_ao_cycle.min())
        p_mean = float(P_ao_cycle.mean())

        dp_dt = np.diff(P_lv_mmHg[last:]) / dt_s
        dp_dt_max = float(dp_dt.max()) if len(dp_dt) > 0 else 0.0

        co = sv * 1e-3 * params.heart_rate_bpm / 1000  # L/min

        slo_details = {
            "p_systolic": 60 <= p_sys <= 300,
            "p_diastolic": 50 <= p_dia <= 110,
            "ef": 25 <= ef <= 85,
            "edv": 60 <= edv <= 220,
            "pv_loop_coherent": edv > esv and sv > 0,
        }
        slo_passed = all(slo_details.values())

        result = WindkesselResult(
            time_ms=time_ms,
            pressure_mmHg=P_lv_mmHg,
            volume_mL=volume_full,
            aortic_pressure_mmHg=P_ao_mmHg,
            ef_pct=round(ef, 1),
            edv_mL=round(edv, 1),
            esv_mL=round(esv, 1),
            sv_mL=round(sv, 1),
            p_systolic_mmHg=round(p_sys, 1),
            p_diastolic_mmHg=round(p_dia, 1),
            p_mean_mmHg=round(p_mean, 1),
            dp_dt_max=round(dp_dt_max, 1),
            cardiac_output_L_min=round(co, 2),
            slo_passed=slo_passed,
            slo_details=slo_details,
        )

        logger.info("windkessel.complete",
                   ef_pct=round(ef,1), edv=round(edv,1),
                   p_sys=round(p_sys,1), p_dia=round(p_dia,1),
                   co=round(co,2), slo_passed=slo_passed)
        return result

    def _generate_volume_waveform(self, params, bcl_ms, dt_ms):
        T = int(bcl_ms / dt_ms)
        t = np.linspace(0, bcl_ms, T)
        t_sys = 0.35 * bcl_ms
        sv = params.V_ed_mL - params.V_es_mL
        V = np.zeros(T)
        for i, ti in enumerate(t):
            if ti < t_sys:
                phase = ti / t_sys
                V[i] = params.V_ed_mL - sv * (3*phase**2 - 2*phase**3)
            else:
                phase = (ti - t_sys) / (bcl_ms - t_sys)
                V[i] = params.V_es_mL + sv * (3*phase**2 - 2*phase**3)
        return V
