# Plan de commits atomiques — proposition

Date : 20 juillet 2026.

Ce plan ne crée aucun commit et n'autorise aucun push.

## Commit 1 — Documentation de référence et frontière de tests

Inclure :

- `CDT_ETAT_AVANCEMENT_2026-07-20.md` ;
- avertissement dans `cdt_status_strict.md` ;
- `SPRINTS.md` ;
- `pytest.ini` ;
- les documents de consolidation sous `sprint_artifacts/consolidation/` ;
- ajustements déterministes de `tests/test_api.py` et
  `tests/test_opencarp_config.py` si ceux-ci ne sont pas placés avec le lot B.

Validation : `pytest -q` doit conserver 140 réussites, 10 tests ignorés et
zéro échec.

`reports/phase02_status.md` reste un fichier historique local sous un répertoire
ignoré par Git. Il n'est pas forcé dans le commit; le nouveau statut officiel
le remplace comme référence versionnée.

## Commit 2 — Mécanique orthotrope et microstructure Bayer

Inclure le lot A, les scripts reproductibles associés, le manifeste et les
rapports JSON/Markdown du Sprint 1. Ne pas inclure les checkpoints ou champs
NPZ dans Git sans décision explicite.

Validation minimale : tests Phase 02, tests du manifeste, smoke tests courts et
cohérence des hashes du manifeste. Le run complet de 1 h 57 n'a pas à être
réexécuté si son manifeste et ses artefacts restent vérifiables.

## Commit 3 — Contrats EP et Windkessel

Inclure `opencarp_solver.py`, `windkessel.py` et leurs tests. Ce commit doit
être autonome : backend strict par défaut, fallback explicite test/dev, unités
du débit et contrat d'un cycle.

Validation : tests EP/Windkessel ciblés puis suite Phase 03.

## Commit 4 — Orchestration honnête et verrou DoE

Inclure `coupled_solver.py`, `doe_task.py` et les tests de couplage. Le message
doit indiquer que le chemin est unidirectionnel et que l'export DoE est bloqué,
pas que D2.1 est validé.

Validation : un seul appel Windkessel, aucun retour de pression revendiqué,
aucun export DoE sans benchmark global.

## Commit 5 — Audit reproductible du maillage

Inclure les scripts d'audit, `docker/Dockerfile.mmg` et les rapports JSON/MD
légers. Exclure par défaut les maillages candidats dérivés et la variante
Gmsh 4.12.2 rejetée, sauf décision explicite de les archiver.

Validation : régénération des rapports de qualité de la référence et du
candidat Mmg, sans adoption du candidat.

## Conditions avant exécution du plan

1. Revue du diff de chaque lot.
2. Décision explicite sur les artefacts binaires/dérivés.
3. Aucun secret ni `.env` indexé.
4. Suite globale verte.
5. Autorisation utilisateur avant le premier commit, puis autorisation séparée
   avant tout push GitHub.
