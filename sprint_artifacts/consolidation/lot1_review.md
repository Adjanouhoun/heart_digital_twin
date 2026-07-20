# Revue du lot 1 — documentation et frontière de tests

Date : 20 juillet 2026.

Verdict : **prêt pour commit après autorisation explicite**.

## Périmètre proposé

- `CDT_ETAT_AVANCEMENT_2026-07-20.md` ;
- `SPRINTS.md` ;
- `cdt_status_strict.md` ;
- `pytest.ini` ;
- `tests/test_api.py` ;
- `tests/test_opencarp_config.py` ;
- `sprint_artifacts/consolidation/repository_classification.md` ;
- `sprint_artifacts/consolidation/commit_plan.md` ;
- `sprint_artifacts/consolidation/lot1_review.md`.

`reports/phase02_status.md` est ignoré par Git et reste hors commit. Aucun
fichier d'environnement, maillage, checkpoint ou résultat binaire n'appartient
à ce lot.

## Contrôles

- Recherche de motifs de secrets dans le périmètre : aucun fichier signalé.
- `git diff --check` sur les fichiers suivis modifiés : aucune erreur.
- Tests ciblés API et configuration openCARP : **27 réussis**, 1 avertissement
  de dépréciation Starlette/httpx.
- Suite globale exécutée avant la revue : **140 réussis, 10 ignorés, 0 échec**.

## Sens des modifications

- Le nouveau statut remplace les anciens pourcentages comme référence.
- Pytest ne collecte plus les scripts diagnostiques dépendants de Docker.
- Les tests API isolent explicitement le contrat HTTP de la physique.
- Les tests openCARP vérifient les unités et mots-clés actuels du journal.

Aucun staging, commit ou push n'a été réalisé pendant cette revue.
