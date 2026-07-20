# Consolidation du dépôt — classement des changements

Date : 20 juillet 2026.

Statut : inventaire préparatoire uniquement. Aucun fichier n'a été supprimé,
aucun commit n'a été créé et aucun push n'a été effectué.

## Santé vérifiée

- Pytest est désormais limité au répertoire `tests/`; les scripts de diagnostic
  nommés `test_*.py` sous `scripts/` ne sont plus collectés comme tests unitaires.
- Suite officielle : **140 réussis, 10 ignorés, 0 échec**, 29 avertissements.
- Les tests ignorés dépendent de sorties openCARP, DoE, GP ou Sobol qui ne sont
  pas disponibles ou ne sont pas scientifiquement validées.

## Lot A — Mécanique et microstructure

Fichiers produit principaux :

- `app/fibers/ldrb.py` ;
- `app/solver/mechanics/fenicsx_solver.py` ;
- `docker/Dockerfile.solver` ;
- `scripts/generate_ldrb_fibers.py` ;
- `scripts/generate_bayer2012_microstructure.py` ;
- audits Bayer, microstructure, Jacobien et continuation sous `scripts/` ;
- tests Phase 02 et manifeste.

Ce lot rassemble la loi orthotrope, la projection dyadique fibre/feuillet, les
conditions aux limites et les garde-fous de continuation. Il doit être revu
comme un ensemble avant commit.

## Lot B — EP, Windkessel et orchestration

Fichiers produit principaux :

- `app/solver/ep/opencarp_solver.py` ;
- `app/solver/hemodynamics/windkessel.py` ;
- `app/solver/coupled_solver.py` ;
- `app/solver/tasks/doe_task.py` ;
- `tests/test_phase03_solver.py`, `tests/test_api.py` et
  `tests/test_opencarp_config.py`.

Ce lot impose openCARP strict, le contrat waveform, le calcul correct du débit,
la chaîne unidirectionnelle et le blocage de l'export DoE.

## Lot C — Qualité du maillage

Outils reproductibles :

- `scripts/audit_rest_mesh_quality.py` ;
- `scripts/compare_tetra_mesh_geometry.py` ;
- `scripts/convert_mmg_mesh_candidate.py` ;
- `scripts/generate_quality_mesh_candidate.py` ;
- `docker/Dockerfile.mmg`.

`docker/Dockerfile.mesher` et la variante Gmsh 4.12.2 appartiennent à un essai
rejeté; ils ne doivent pas être confondus avec la chaîne Gmsh 4.15.2 de
référence.

Les JSON de qualité et de géométrie sont des preuves légères à conserver. Les
maillages candidats `.pts`, `.elem`, `.mesh`, `.sol` et la surface `.stl` sont
des sorties dérivées reproductibles; leur versionnement Git doit être décidé
séparément. Le dossier `sprint_artifacts/` pèse environ 5.7 Mo, dont 5.0 Mo pour
le Sprint 2.

## Lot D — Documentation et traçabilité

- `SPRINTS.md` : journal opérationnel détaillé ;
- `CDT_ETAT_AVANCEMENT_2026-07-20.md` : statut consolidé en vigueur ;
- `cdt_status_strict.md` : historique versionné marqué obsolète ;
- `reports/phase02_status.md` : historique local marqué obsolète mais situé
  sous le répertoire `reports/` ignoré par Git ;
- `pytest.ini` : frontière de la suite officielle ;
- preuves légères sous `sprint_artifacts/sprint1` et `sprint_artifacts/sprint2`.

## Éléments à exclure d'un commit global

- fichiers `.env`, caches, journaux et checkpoints ;
- champs `.h5/.xdmf`, surfaces et maillages dérivés non sélectionnés ;
- scripts racine `probe_*`, `diag_*`, `check_*`, `validate_*` et patchs
  temporaires dont la provenance n'est pas encore attribuée à un jalon ;
- résultats DoE historiques fallback ;
- archives ou poids lourds déjà couverts par `.gitignore`.

Ces éléments sont conservés sur disque. L'exclusion proposée ne constitue pas
une autorisation de suppression.
