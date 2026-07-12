# Scripts — catalogue

Ce dossier accumule les scripts de tout le projet (juin-juillet 2026).
Pas de sous-dossiers volontairement : de nombreuses commandes du
`CDT_JOURNAL_TECHNIQUE.md` referencent ces chemins exacts.

## Segmentation / donnees (WP1)
- `prepare_acdc_nnunet.py`, `prepare_mnms_nnunet.py`, `prepare_mnms_v2.py`,
  `prepare_mmwhs_nnunet.py`, `prepare_emidec_nnunet.py` : conversion des
  datasets bruts vers le format nnU-Net.
- `train_mps.py`, `train_mps_v2.py`, `train_mnms_transfer.py` : entrainements
  locaux (M1, MPS backend -- abandonne au profit de Kaggle/Colab).
- `kaggle_acdc_session3.py`, `kaggle_mnms_v3.py` : scripts d'entrainement
  Kaggle/Colab.
- `validate_acdc.py`, `watch_and_validate.py` : validation post-entrainement.
- `mlflow_nnunet_tracker.py` : tracking MLflow des runs nnU-Net.
- `generate_meshes_acdc.py`, `mesh_quality_dihedral.py` : generation et QC
  des maillages Gmsh (production, D1.2).

## Fibres
- `generate_ldrb_fibers.py` : fibres LDRB (Bayer 2012, via app/fibers/ldrb.py).
- `generate_simple_fibers.py` : champ tangentiel simplifie (dev/test uniquement,
  PAS physiologique).

## openCARP (EP)
- `run_opencarp_patient.py` : simulation EP sur un patient.
- `validate_opencarp_benchmark.py` : validation formelle vs Niederer et al. 2011
  (benchmark coin 20x7x3mm).

## FEniCSx (mecanique, P15) -- le gros du diagnostic
- `diag_fenicsx.py`, `diag_p1_vs_p2.py`, `diag_dof_coordinates.py`,
  `diag_mesh_worst_tets.py` : diagnostics precoces (formulation, locking P1,
  bug permutation DOF, qualite maillage) -- sessions 6-7 juillet.
- `diag_snes_monitor.py` : isolation du solveur SNES seul (1 palier), utilise
  pour tout debogage fin (line search, kappa, etc.).
- `diag_facet_topology.py` : verification topologie maillage (a disculpe le
  maillage du bug MPI de partitionnement SCOTCH).
- `diag_mpi_ghostmode.py` : test GhostMode.shared_facet pour le bug MPI
  (n'a pas resolu -- chantier MPI ferme).
- `diag_kappa_sensitivity.py` : sensibilite au parametre kappa_vol.
- `diag_continuation_full.py`, `diag_continuation_coarse.py` : continuation
  adaptative complete, respectivement maillage fin (bug p_hat, obsolete)
  et maillage grossier (valide, 2 convergences reussies lam=1.0).
- `continuation_fine_mesh_full.py` : version PRODUCTION de la continuation
  sur maillage fin (p-direct + garde-fou domaine corrige + checkpoint).
  Script de reference actuel pour tout run FEniCSx long.
- `remesh_coarse.py` : regeneration d'un maillage grossier depuis le masque
  de segmentation (target_mm ajustable).
- `postprocess_coarse_result.py` : extraction resultat physiologique
  (volume, deplacement endo/epi) depuis un checkpoint convergent.

## DoE
- `test_doe_real.py` : test isole du pipeline DoE (verification fallback
  vs solveurs reels).

## Divers
- `run_all_patients.sh` : lance le pipeline complet sur tous les patients.
