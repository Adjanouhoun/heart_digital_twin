"""
DoE — Design of Experiments par Latin Hypercube Sampling (LHS).
Génère 500+ simulations paramétriques pour entraîner les surrogates Phase 04.

Espace de paramètres : 10 dimensions
  {σ_l, σ_t, T_max, a, b, a_f, b_f, HR, R_p, C_a}

SLO Phase 03 : ≥ 500 simulations complètes archivées MinIO.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import uuid
import numpy as np
import structlog

from app.solver.coupled_solver import SimulationParameters

logger = structlog.get_logger(__name__)


@dataclass
class ParameterBounds:
    """Bornes de l'espace de paramètres pour le LHS."""
    # Format : (min, max, description)
    sigma_l:       tuple = (0.15, 0.50, "Conductivité longitudinale (S/m)")
    sigma_t:       tuple = (0.05, 0.20, "Conductivité transversale (S/m)")
    sigma_n:       tuple = (0.02, 0.10, "Conductivité normale (S/m)")
    a_kPa:         tuple = (0.20, 1.00, "Holzapfel a (kPa)")
    b:             tuple = (3.00, 12.0, "Holzapfel b")
    a_f_kPa:       tuple = (5.00, 30.0, "Holzapfel a_f (kPa)")
    b_f:           tuple = (10.0, 35.0, "Holzapfel b_f")
    T_max_kPa:     tuple = (80.0, 200., "Tension active max BCS (kPa)")
    heart_rate_bpm:tuple = (50.0, 100., "Fréquence cardiaque (bpm)")
    R_p:           tuple = (0.8e8, 2.0e8, "Résistance périphérique")


class LatinHypercubeSampler:
    """
    Échantillonneur LHS pour le DoE CDT.
    Garantit une couverture uniforme de l'espace paramétrique.
    """

    # Dimensions du LHS
    PARAM_NAMES = [
        "sigma_l", "sigma_t", "sigma_n",
        "a_kPa", "b", "a_f_kPa", "b_f",
        "T_max_kPa", "heart_rate_bpm", "R_p",
    ]

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.RandomState(seed)
        self._bounds = ParameterBounds()

    def sample(self, n_samples: int) -> list[SimulationParameters]:
        """
        Génère n_samples points LHS dans l'espace de paramètres.

        Latin Hypercube : divise chaque dimension en n_samples intervalles
        égaux et tire un point par intervalle → couverture uniforme garantie.
        """
        n_dims = len(self.PARAM_NAMES)

        # Générer la matrice LHS normalisée [0, 1]
        lhs_matrix = self._generate_lhs(n_samples, n_dims)

        # Mapper vers l'espace physique
        params_list = []
        for i in range(n_samples):
            params = self._map_to_physical(lhs_matrix[i])
            params_list.append(params)

        logger.info("lhs.sampled",
                   n_samples=n_samples,
                   n_dims=n_dims,
                   seed=self._rng.randint(0, 1000))

        return params_list

    def _generate_lhs(self, n: int, d: int) -> np.ndarray:
        """
        Génère une matrice LHS n×d dans [0,1].
        Algorithme : permutation aléatoire par dimension.
        """
        result = np.zeros((n, d))
        for j in range(d):
            # Intervalles réguliers
            intervals = (np.arange(n) + self._rng.uniform(size=n)) / n
            # Permutation aléatoire
            result[:, j] = self._rng.permutation(intervals)
        return result

    def _map_to_physical(self, x_normalized: np.ndarray) -> SimulationParameters:
        """Mappe un point [0,1]^d vers l'espace physique."""
        bounds_list = [
            self._bounds.sigma_l,
            self._bounds.sigma_t,
            self._bounds.sigma_n,
            self._bounds.a_kPa,
            self._bounds.b,
            self._bounds.a_f_kPa,
            self._bounds.b_f,
            self._bounds.T_max_kPa,
            self._bounds.heart_rate_bpm,
            self._bounds.R_p,
        ]

        values = {}
        for name, (lo, hi, _), x in zip(self.PARAM_NAMES, bounds_list, x_normalized):
            values[name] = lo + x * (hi - lo)

        return SimulationParameters(
            sigma_l=values["sigma_l"],
            sigma_t=values["sigma_t"],
            sigma_n=values["sigma_n"],
            a_kPa=values["a_kPa"],
            b=values["b"],
            a_f_kPa=values["a_f_kPa"],
            b_f=values["b_f"],
            T_max_kPa=values["T_max_kPa"],
            heart_rate_bpm=values["heart_rate_bpm"],
            R_p=values["R_p"],
        )

    def save_design(self, params_list: list[SimulationParameters],
                    output_path: Path) -> None:
        """Sauvegarde le plan d'expérience en JSON."""
        design = []
        for p in params_list:
            design.append({
                "sigma_l": p.sigma_l, "sigma_t": p.sigma_t,
                "sigma_n": p.sigma_n, "a_kPa": p.a_kPa,
                "b": p.b, "a_f_kPa": p.a_f_kPa, "b_f": p.b_f,
                "T_max_kPa": p.T_max_kPa,
                "heart_rate_bpm": p.heart_rate_bpm,
                "R_p": p.R_p,
            })
        output_path.write_text(json.dumps(design, indent=2))
        logger.info("lhs.design_saved", path=str(output_path), n=len(design))
