# CDT — Etat d'avancement strict vs Projet de base
## Mise a jour : 8 juillet 2026

---

## Livrables du projet (11 deliverables, 6 WP)

### WP1 — Donnees & Reconstruction

| Livrable | Description | Statut | Reste a faire |
|----------|-------------|--------|---------------|
| D1.1 | nnU-Net v2 entraine et valide (ACDC / M&Ms / EMIDEC) | 85% | ACDC Dice MYO=0.910 (SLO 0.90 ATTEINT). M&Ms Dice=0.908 (validation generalisation). EMIDEC : scar=0.458, PMO=0.206 (ameliore via oversample_foreground_percent=0.66, +25%/+56% relatif, mais toujours insuffisant pour usage clinique -- branche parallele au chemin critique, pas bloquant, chantier en pause). |
| D1.1 | Pipeline Airflow DAG : DICOM -> .mesh | 90% | DAG a 9 taches connecte bout-en-bout et valide. |
| D1.2 | Fichiers .mesh + fibres pour 10 patients ACDC | 100% | 10/10 maillages valides (Gmsh reconstruit, sliver+dihedral filtering, resampling 1.5mm). Bug Z-spacing (9x) corrige. |
| D1.2 | Rapport qualite maillage automatise | 80% | mesh_quality_report() existe. Pas integre dans Airflow. |

**SLOs Phase 02 :**

| SLO | Cible | Actuel | Statut |
|-----|-------|--------|--------|
| Dice myocarde ACDC | >= 0.90 | 0.910 | **ATTEINT** |
| Dice cicatrices EMIDEC | >= 0.80 | 0.365 (MI) / 0.132 (PMO) | NON ATTEINT — retraining planifie |
| Jacobian min > 0 | Oui | Oui | ATTEINT |
| Elements degeneres | < 0.1% | 0% | ATTEINT |
| Pipeline < 10 min/patient | < 600s | ~12s | ATTEINT |

---

### WP2 — Simulation & Personnalisation

| Livrable | Description | Statut | Reste a faire |
|----------|-------------|--------|---------------|
| D2.1 | Solveur EP openCARP valide (benchmarks) | 75% | Root cause resolue (g_mult confirme comme levier de conductivite via comparaison MD5 binaire vm.igb). CV varie physiologiquement. Image Docker officielle arm64 validee. Full DAG 9/9 taches, EF=60%, CV=0.5 m/s. Reste : validation formelle vs benchmarks publies (activation +/-5ms). |
| D2.1 | Solveur mecanique FEniCSx (Holzapfel-Ogden + BCS) | 65% | Formulation quasi-incompressible mixte P2-P1 (Taylor-Hood) validee physiquement : isochore + kappa=1e6 + p direct + garde-fou domaine (SNESSetFunctionDomainError) + continuation adaptative. Sur maillage reduit (34K tets) : progression stable sans NaN ni inversion sur >14 paliers de charge (min_J decroit proprement 0.97->0.78). Point ouvert : temps de calcul par palier (450-2500s) trop eleve/irregulier pour boucler a pleine charge en l'etat — praticabilite a l'etude (parallelisation MPI, allegement maillage). Code de production porte (fenicsx_solver.py) mais run de validation complete pas encore aboutit a lam=1.0. |
| D2.1 | Couplage EP-Meca-Windkessel | 35% | Couplage fallback fonctionne, couplage avec vrais solveurs demontre sur maillages patients (EF=57%, P=114/80mmHg) MAIS ne peut pas encore s'appuyer sur la mecanique FEniCSx a pleine charge (cf. ci-dessus). A refaire une fois D2.1 mecanique praticable. |
| D2.1 | Dataset DoE >= 500 sims parametriques | 20% | 500 sims fallback (invalides). Besoin DoE openCARP+FEniCSx reel, bloque par mecanique. |
| D2.1 | Rapport validation vs references | 0% | Pas commence. |
| D2.2 | Calibration inverse Bayesienne patient-specifique | 20% | Prototype MCMC sur fallback (invalide). Besoin GP valides. |

**SLOs Phase 03 :**

| SLO | Cible | Actuel | Statut |
|-----|-------|--------|--------|
| Activation dans +/-5ms benchmarks | +/-5ms | Non mesure formellement | EN COURS (CV physiologique confirme) |
| EF dans +/-3% reference clinique | +/-3% | EF=57-60% observe sur couplage partiel | EN COURS |
| Boucle PV physiologique | 70-120mmHg | P=114/80mmHg observe | PARTIEL — a confirmer avec mecanique praticable |
| DoE 500 sims completes | 500 | 500 (fallback) | INVALIDE — bloque par mecanique |

---

### WP3 — Surrogates & UQ

| Livrable | Description | Statut | Reste a faire |
|----------|-------------|--------|---------------|
| D3.1 | GNN MeshGraphNets (champs EP + meca) | 0% | Pas commence. Besoin DoE reel. |
| D3.1 | GP Emulator GPyTorch (hemodynamique + UQ) | 30% | GP entraines sur fallback (invalides). Infrastructure OK. |
| D3.1 | Triton Inference Server | 0% | Pas commence. |
| D3.2 | Rapport GSA (indices Sobol) | 30% | Sobol sur fallback (invalide). Infrastructure SALib OK. |
| D3.2 | Rapport UQ (coverage CI, calibration) | 10% | Coverage mesure sur fallback. |

---

### WP4 — Orchestration Temps Reel

| Livrable | Description | Statut | Reste a faire |
|----------|-------------|--------|---------------|
| D4.1 | API REST + delegation solver | 40% | FastAPI partiel, delegation via host.docker.internal OK. |
| D4.1 | WebSocket temps reel | 0% | Pas commence. |
| D4.1 | EnKF (Ensemble Kalman Filter) | 0% | Pas commence. |

---

### WP5 — Validation Clinique

| Livrable | Description | Statut | Reste a faire |
|----------|-------------|--------|---------------|
| D5.1 | Validation EMIDEC + cohorte CRT | 0% | Pas commence. Bloque par tout l'amont. |
| D5.2 | Rapport ASME V&V40 (AUC >= 0.75) | 0% | Pas commence. |

---

### WP6 — Demonstrateur & DevOps

| Livrable | Description | Statut | Reste a faire |
|----------|-------------|--------|---------------|
| D6.1 | Docker-compose stack (10 services) | 70% | Stack fonctionnelle (Airflow, MLflow, MinIO, Redis, Postgres, API). |
| D6.1 | CI/CD GitHub Actions | 0% | Pas commence. Tests locaux seulement. |
| D6.1 | Dashboards Grafana | 0% | Pas commence. |
| D6.2 | Application React + Three.js | 10% | Prototype viewer HTML. Pas de React/TypeScript. |
| D6.2 | Shaders GLSL (deformation + colormaps) | 5% | Colormap activation basique. Pas de GLSL custom. |
| D6.2 | Panel curseurs clinicien | 0% | Pas commence. |
| D6.2 | Graphiques temps reel (ECG + boucle PV) | 0% | Pas commence. |

---

## Chemin critique strict

```
D1.1 (nnU-Net ACDC >= 0.90) <- ATTEINT (0.910)
  |
D1.2 (10 meshes + fibres) <- ATTEINT (10/10)
  |
D2.1 (openCARP valide sur benchmarks) <- PROCHE (root cause resolue, validation formelle restante)
  |
D2.1 (FEniCSx mecanique) <- BLOQUANT ACTUEL (formulation validee, praticabilite temps de calcul a l'etude)
  |
D2.1 (Couplage reel EP+Meca+WK) <- EN ATTENTE (demontre partiellement, a refaire avec mecanique praticable)
  |
D2.1 (DoE 500 sims reelles) <- BLOQUE par mecanique
  |
D3.1 (MeshGraphNets + GP valides) <- BLOQUE par D2.1
  |
D3.2 (Sobol + UQ reels) <- BLOQUE par D3.1
  |
D2.2 (Calibration MCMC valide) <- BLOQUE par D3.1
  |
D4.1 (WebSocket + EnKF) <- BLOQUE par D3.1
  |
D6.2 (Frontend React+Three.js) <- BLOQUE par D4.1
  |
D5.1/D5.2 (Validation clinique) <- BLOQUE par tout
```

## Actions immediates (par ordre de priorite)

1. Determiner la praticabilite du temps de calcul FEniCSx mecanique (run de validation en cours ; options : mpirun parallele, maillage plus grossier pour le DoE, ajustement kappa_vol)
2. Une fois mecanique praticable : porter la formulation validee en production (fait, fenicsx_solver.py mis a jour — a valider en execution complete puis committer)
3. Valider openCARP contre benchmarks publies (temps activation +/-5ms)
4. Reconstruire le couplage reel EP+Meca+Windkessel avec la mecanique praticable
5. Generer DoE 500 sims reelles (openCARP+FEniCSx)
6. Entrainer MeshGraphNets + GP sur donnees reelles
7. MCMC calibration avec GP valides
8. Frontend React + Three.js + WebSocket
9. Validation clinique ASME V&V40
10. (Parallele, hors chemin critique) EMIDEC retraining oversampling 0.66 des que quota Kaggle T4 disponible
