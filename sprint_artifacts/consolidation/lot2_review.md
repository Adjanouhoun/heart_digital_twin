# Revue du lot 2 — mécanique orthotrope et microstructure Bayer

Date : 20 juillet 2026.

Verdict : **code et preuves prêts pour un commit atomique après autorisation**.

## Noyau produit

- `app/fibers/ldrb.py` ;
- `app/solver/mechanics/fenicsx_solver.py` ;
- `app/solver/mechanics/run_manifest.py` ;
- `docker/Dockerfile.solver` ;
- `scripts/generate_ldrb_fibers.py` ;
- `tests/test_phase02_pipeline.py` ;
- `tests/test_mechanics_contracts.py` ;
- `tests/test_run_manifest.py`.

## Outils reproductibles de sprint

- génération Bayer 2012 ;
- audits des marqueurs, de la microstructure nodale et élémentaire et du
  Jacobien spatial ;
- diagnostics de continuation et de conditions basales ;
- génération du manifeste, exécution et post-traitement du Sprint 1 ;
- validation du volume cavitaire et smoke test orthotrope.

Fichiers concernés sous `scripts/` :

- `generate_bayer2012_microstructure.py` ;
- `audit_bayer_markers.py` ;
- `audit_element_microstructure.py` ;
- `audit_patient_microstructure.py` ;
- `audit_patient_jacobian_spatial.py` ;
- `diagnose_patient_minimal_bc.py` ;
- `diagnose_sprint1_continuation.py` ;
- `generate_sprint1_manifest.py` ;
- `postprocess_sprint1.py` ;
- `run_sprint1_orthotropic_full.py` ;
- `validate_endo_cavity_cohort.py` ;
- `validate_orthotropic_smoke.py`.

## Preuves légères proposées

- Sprint 1 : `result.json`, `manifest.json`, `endo_cavity_cohort.json` ;
- Sprint 2 : rapports JSON Bayer, microstructure, conditions basales,
  Jacobien spatial et paliers diagnostiques 1–2 kPa.

Les champs et checkpoints NPZ restent ignorés. Les rapports de qualité de
maillage et candidats Mmg/Gmsh appartiennent au lot 5, pas à ce commit.

## Arbitrage pression

Le terme de pression endocardique expérimental et son marquage géométrique ont
été retirés du noyau. `p_endo_kPa` demeure une interface réservée, mais toute
valeur non nulle lève une erreur expliquant que la porte qualité du maillage
n'est pas franchie. Le commit ne revendique donc aucune validation de pression.

## Contrôles

- Tests mécaniques ciblés : **26 réussis** avant l'arbitrage final.
- Suite globale après arbitrage : **141 réussis, 10 ignorés, 0 échec**.
- Vérification syntaxique des modules et douze scripts : réussie.
- `git diff --check` : aucune erreur sur le noyau suivi.

## Exclusions explicites

- `scripts/generate_meshes_acdc.py` et outils de remeshing : lot maillage ;
- `diag_continuation_pressure.py`, `diag_continuation_2phase.py` et patchs
  racine : diagnostics de pression bloqués ;
- fichiers XDMF/HDF5, maillages, checkpoints et champs ;
- modifications EP, Windkessel, couplage et DoE : lots 3 et 4.

Aucun staging, commit ou push du lot 2 n'a été réalisé pendant cette revue.
