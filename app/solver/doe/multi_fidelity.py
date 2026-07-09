"""
DoE MULTI-FIDELITE — extension de LatinHypercubeSampler existant.

Contexte (voir CDT_JOURNAL_TECHNIQUE.md, sessions P15) : une simulation
mecanique complete sur maillage FIN (production) coute significativement
plus cher qu'sur maillage GROSSIER (dev), et le vrai chiffre par run est
en cours de mesure (continuation_fine_mesh_full.py, run en fond sur
plusieurs jours). Un DoE de 500 simulations toutes en haute-fidelite
n'est pas praticable en sequentiel (bug MPI ferme, cf. journal).

Strategie multi-fidelite (litterature : co-kriging, Kennedy & O'Hagan) :
  - Population BASSE FIDELITE (LF) : maillage grossier, N_LF grand
    (ex. 300-500), fiable pour les metriques GLOBALES (volume, EF
    approx, pression) -- PAS fiable pour les metriques TRANSMURALES
    (deja demontre : ~3-4 elements dans l'epaisseur pariétale sur le
    maillage grossier, cf. diagnostic geometrique du 2026-07-09).
  - Population HAUTE FIDELITE (HF) : maillage fin, N_HF petit (a
    determiner selon le cout reel/run, mesure en cours), utilisee pour
    calibrer la correction LF->HF (discrepancy function) sur TOUTES les
    metriques, avec un accent particulier sur les metriques transmurales
    que LF seul ne peut pas fournir correctement.

Les points HF sont un SOUS-ENSEMBLE EMBOITE des points LF (meme design,
pas un tirage independant) -- necessaire pour mesurer l'ecart LF/HF au
meme point de l'espace des parametres. Selection par distance MAXIMIN
(les points les plus disperses possible) pour une bonne couverture avec
peu de points HF.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import numpy as np
import structlog

from app.solver.doe.latin_hypercube import LatinHypercubeSampler, ParameterBounds
from app.solver.coupled_solver import SimulationParameters

logger = structlog.get_logger(__name__)


@dataclass
class MultiFidelityPoint:
    """Un point du DoE multi-fidelite, avec son niveau de fidelite assigne."""
    index: int
    params: SimulationParameters
    fidelity: str          # "LF" (maillage grossier) ou "HF" (maillage fin)
    mesh_variant: str       # ex. "patient001_coarse5" ou "patient001"


class MultiFidelityDoE:
    """
    Construit un plan d'experience a deux niveaux de fidelite, emboites,
    a partir du meme espace de parametres LHS que celui deja valide
    (app/solver/doe/latin_hypercube.py -- non modifie).
    """

    def __init__(self, seed: int = 42,
                 coarse_mesh_name: str = "patient001_coarse5",
                 fine_mesh_name: str = "patient001") -> None:
        self._sampler = LatinHypercubeSampler(seed=seed)
        self._coarse_mesh_name = coarse_mesh_name
        self._fine_mesh_name = fine_mesh_name

    def generate(self, n_lf: int, n_hf: int) -> list[MultiFidelityPoint]:
        """
        Genere le plan complet.

        Args:
            n_lf: nombre de points basse-fidelite (maillage grossier).
                Peu couteux (~43min/run mesure) -> peut etre eleve
                (ex. 300-500).
            n_hf: nombre de points haute-fidelite (maillage fin), choisi
                comme SOUS-ENSEMBLE emboite de n_lf via distance maximin.
                DOIT etre fixe une fois le cout reel par run connu
                (cf. continuation_fine_mesh_full.py en cours) --
                valeur par defaut fournie ici est un PLACEHOLDER a ajuster.

        Raises:
            ValueError: si n_hf > n_lf (HF doit etre un sous-ensemble de LF).
        """
        if n_hf > n_lf:
            raise ValueError(
                f"n_hf ({n_hf}) doit etre <= n_lf ({n_lf}) : "
                f"les points HF sont un sous-ensemble emboite des points LF."
            )

        all_params = self._sampler.sample(n_lf)
        lhs_matrix = self._params_to_matrix(all_params)

        hf_indices = self._select_maximin_subset(lhs_matrix, n_hf)
        hf_indices_set = set(hf_indices)

        points = []
        for i, params in enumerate(all_params):
            if i in hf_indices_set:
                points.append(MultiFidelityPoint(
                    index=i, params=params, fidelity="HF",
                    mesh_variant=self._fine_mesh_name,
                ))
            else:
                points.append(MultiFidelityPoint(
                    index=i, params=params, fidelity="LF",
                    mesh_variant=self._coarse_mesh_name,
                ))

        logger.info("multi_fidelity_doe.generated",
                    n_lf=n_lf, n_hf=len(hf_indices),
                    n_lf_only=n_lf - len(hf_indices))
        return points

    def _params_to_matrix(self, params_list: list) -> np.ndarray:
        """Reconvertit la liste de SimulationParameters en matrice normalisee
        [0,1]^d pour le calcul de distance (necessaire pour comparer des
        parametres d'echelles tres differentes, ex. R_p ~1e8 vs a_kPa ~1)."""
        bounds = ParameterBounds()
        bounds_list = [
            bounds.sigma_l, bounds.sigma_t, bounds.sigma_n,
            bounds.a_kPa, bounds.b, bounds.a_f_kPa, bounds.b_f,
            bounds.T_max_kPa, bounds.heart_rate_bpm, bounds.R_p,
        ]
        names = LatinHypercubeSampler.PARAM_NAMES
        matrix = np.zeros((len(params_list), len(names)))
        for i, p in enumerate(params_list):
            for j, (name, (lo, hi, _)) in enumerate(zip(names, bounds_list)):
                val = getattr(p, name)
                matrix[i, j] = (val - lo) / (hi - lo)
        return matrix

    def _select_maximin_subset(self, matrix: np.ndarray, n_hf: int) -> list:
        """
        Selectionne n_hf points parmi matrix (normalisee [0,1]^d) en
        maximisant la distance minimale entre points selectionnes
        (algorithme glouton "farthest point sampling"). Garantit une
        bonne couverture de l'espace meme avec peu de points HF.
        """
        n = len(matrix)
        if n_hf >= n:
            return list(range(n))

        selected = [0]  # point de depart arbitraire (le premier du LHS)
        remaining = set(range(1, n))

        while len(selected) < n_hf:
            best_idx, best_min_dist = None, -1.0
            for idx in remaining:
                dists = [np.linalg.norm(matrix[idx] - matrix[s]) for s in selected]
                min_dist = min(dists)
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = idx
            selected.append(best_idx)
            remaining.discard(best_idx)

        return sorted(selected)

    def save_design(self, points: list[MultiFidelityPoint],
                     output_path: Path) -> None:
        """Sauvegarde le plan multi-fidelite complet en JSON, avec le
        niveau de fidelite et le maillage assigne pour chaque point."""
        design = []
        for pt in points:
            p = pt.params
            design.append({
                "index": pt.index,
                "fidelity": pt.fidelity,
                "mesh_variant": pt.mesh_variant,
                "sigma_l": p.sigma_l, "sigma_t": p.sigma_t, "sigma_n": p.sigma_n,
                "a_kPa": p.a_kPa, "b": p.b,
                "a_f_kPa": p.a_f_kPa, "b_f": p.b_f,
                "T_max_kPa": p.T_max_kPa,
                "heart_rate_bpm": p.heart_rate_bpm, "R_p": p.R_p,
            })
        output_path.write_text(json.dumps(design, indent=2))
        n_hf = sum(1 for pt in points if pt.fidelity == "HF")
        n_lf = len(points) - n_hf
        logger.info("multi_fidelity_doe.design_saved", path=str(output_path),
                    n_total=len(points), n_lf=n_lf, n_hf=n_hf)
