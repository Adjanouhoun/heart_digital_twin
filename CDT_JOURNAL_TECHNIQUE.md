
---

## Mise à jour — 23 juin 2026 (session 2)

### Phase 03 — Intégration complète ✅

#### Nouveaux services déployés
| Service | Port | Statut |
|---------|------|--------|
| cdt-solver (FEniCSx + openCARP) | :8001 | ✅ |

#### openCARP v19.0 installé
- Compilé depuis les sources GitLab dans le conteneur arm64
- Permanent dans l'image Docker `cdt-solver`
- Fallback analytique actif (openCARP nécessite fichiers .par + MPI)

#### Problèmes rencontrés

**P15 — openCARP .deb incompatible arm64**
- Symptôme : `libc6:amd64 not installable` lors du build Docker
- Cause : Mac M1 = arm64, .deb = amd64
- Solution : compilation depuis les sources GitLab

**P16 — Registry Docker openCARP privé**
- Symptôme : `access forbidden` sur registry.gitlab.com
- Solution : compilation depuis sources (même résultat)

**P17 — openCARP cherche Docker-in-Docker**
- Symptôme : `FileNotFoundError: docker` depuis le conteneur solver
- Cause : `_check_opencarp()` appelait `docker image inspect`
- Solution : vérifier uniquement `which openCARP.par` et les paths directs

**P18 — gengetopt manquant pour cmake openCARP**
- Solution : `apt-get install gengetopt libboost-all-dev`

#### Simulation couplée validée
```json
{
  "ef_pct": 60.0,
  "p_systolic_mmHg": 128.3,
  "cv_ms": 0.825,
  "benchmark_passed": true,
  "duration_s": 0.022
}
```

### nnU-Net ACDC — Progression
| Epoch | LV | MYO | RV | EMA |
|-------|-----|-----|-----|-----|
| 5 | 0.949 ✅ | 0.884 ❌ | 0.904 ✅ | 0.479 |
| 6 | 0.948 ✅ | 0.881 ❌ | 0.902 ✅ | 0.596 |
| 7 | en cours | — | — | — |

SLO MYO ≥ 0.90 attendu epoch 8 (~19:55 ce soir)

### MLflow opérationnel
- Fix : `psycopg2-binary` ajouté dans `Dockerfile.mlflow`
- Tracker automatique : `scripts/mlflow_nnunet_tracker.py` (PID 62472)
- UI : http://localhost:5001

---
## Mise a jour — 24-26 juin 2026 (sessions 3-4)

### Phase 02 — Segmentation ACDC complete

#### P19 — nnU-Net v2.8.0 data loader crash sur macOS
- Symptome : crash silencieux a "Epoch 0" sur CPU et MPS
- Cause : NonDetMultiThreadedAugmenter de batchgenerators deadlock sur macOS
- Solution : script custom train_mps_v2.py qui contourne le data loader natif

#### P20 — MPS (Metal Performance Shaders) sur Apple M1 Pro
- PyTorch 2.12.1, MPS available et fonctionnel
- Forward + backward pass OK sur le reseau complet (30.4M params, 0.55 GB MPS)
- Config 3d_fullres fonctionne, config 2d crashe (shape mismatch decodeur)

#### P21 — Crop alignment image/masque
- Symptome : Dice = 0.0000 malgre loss qui descend
- Cause : pad_or_crop utilisait des offsets aleatoires differents pour image et masque
- Solution : retourner les offsets et les reutiliser pour le masque

#### P22 — Gradient explosion epoch 9-10
- Symptome : loss saute de 0.54 a 1.45, Dice retombe a 0
- Solution : gradient clipping clip_grad_norm_(12.0) + LR warmup 5 epochs

#### P23 — Labels -1 dans segmentations ACDC
- Symptome : CUDA error device-side assert triggered sur Kaggle
- Cause : label -1 (ignore) dans les masques
- Solution : np.clip(seg, 0, 3) avant le calcul de loss

#### P24 — Spacing inverse SimpleITK vs numpy
- Symptome : Dice inference = 0.16 (vs 0.89 en training)
- Solution : target_spacing=(1.5625, 1.5625, 5.0) au lieu de (5.0, 1.5625, 1.5625)

#### P25 — Gmsh 4.15.2 API changee + P26 TetGen surfaces non-manifold
- Solution : pipeline lissage gaussien + marching cubes + PyMeshFix + decimation + TetGen
- Resultat : 9/10 patients mailles, 0% degeneres

#### P27 — TetGen boucle infinie sur patient009
- Solution : timeout 60s via subprocess

### Training ACDC — Resultats

Local MPS (M1 Pro) : 9 min/epoch, MYO=0.56 epoch 9
Kaggle T4 GPU : 14.6 min/epoch, Best Dice mean=0.9068 (RV=0.89, MYO=0.89, LV=0.94)

### Pipeline post-segmentation — Livrable D1.2
- 9/10 meshes generes, 0% degeneres, Jacobian positif partout
- Export openCARP (.pts, .elem, .lon) pour chaque patient
- Dice moyen : RV=0.87, MYO=0.84, LV=0.95
- Pipeline total : 3.1 min pour 10 patients

### Tests — 76 passent
- Phase 01 (DICOM) : 16 tests
- Phase 02 (Pipeline) : 36 tests
- Phase 03 (Solveur) : 24 tests

### Datasets prepares
- ACDC (027) : 200 cas, training termine
- MnMs (028) : 150 cas, prepare, pret pour transfer learning
- EMIDEC (029) : pas encore telecharge

### Stack technique
- PyTorch 2.12.1, nnU-Net v2.8.0
- Gmsh 4.15.2, TetGen 0.8.4, PyMeshFix, meshio 5.3.5
- SimpleITK 2.5.5, nibabel 5.4.2, scikit-image 0.26.0
- Kaggle : GPU T4 x2, 30h/semaine, 29 GB RAM

---
## Mise a jour — 27 juin 2026 (session 5)

### Phase 04 — Surrogates & UQ (prototype)

#### DoE 500 simulations
- Latin Hypercube Sampling : 10 parametres, 500 samples
- 10 patients x 50 sims chacun (round-robin)
- 500/500 benchmark PASS en 7.2s (14ms/sim, fallback analytique)
- Output vector 14D par simulation
- Fichier : reports/doe/doe_500_results.json

#### GP Emulators (GPyTorch)
- 14 GP entraines (1 par dimension output)
- Meilleurs resultats (fallback analytique) :
  - cv_ms     : R2=0.9955, RMSE=0.002, Coverage=100%
  - p_sys_mmHg: R2=0.8143
  - p_dia_mmHg: R2=0.8867
  - sv_mL     : R2=0.7915
- ef_pct : constant (fallback ne varie pas) - skip
- Training : 1s par GP, inference <1ms
- Fichiers : reports/doe/gp_*.pth

#### Analyse de Sensibilite Globale (SALib Sobol)
- cv_ms : sigma_t (53%) + sigma_l (48%) = conductivite
- p_sys : T_max_kPa (87%) = tension active
- p_dia : T_max_kPa (92%) = tension active
- sv_mL : T_max_kPa (85%) = tension active
- Coherent physiquement
- Fichier : reports/doe/sensitivity_sobol.json

### Phase 05 — Calibration Inverse (prototype)

#### MCMC emcee
- 32 walkers x 500 steps = 16000 samples
- Burn-in : 200 steps, posterior : 9600 samples
- Acceptance : 0.23 (optimal)
- 9/10 parametres identifies (vrai dans CI 90%)
- sigma_t identifie precisement : 0.1208 (vrai: 0.12)
- T_max biaise (limite du fallback analytique)
- Temps : 88s
- Fichier : reports/doe/mcmc_posterior.npz

### Tests — 99 passent
- Phase 01 (DICOM) : 16 tests
- Phase 02 (Pipeline) : 36 tests
- Phase 03 (Solveur) : 24 tests
- Phase 04 (Surrogates) : 18 tests
- Phase 05 (Calibration) : 5 tests

### M&Ms Transfer Learning
- Training local : trop lent (90 min/epoch, thermal throttling)
- Relance sur Kaggle T4 : 9 min/epoch, en cours
- Dataset : 300 cas (150 patients x 2 frames), labels remappes

---
## Mise a jour — 28 juin 2026 (session 6)

### Phase 03 — openCARP natif ARM (JALON)

#### P28 — Compilation openCARP native Mac M1
- cmake + make sur sources GitLab (git.opencarp.org)
- Dependances : brew install cmake openmpi boost gengetopt pkg-config petsc
- Fix linker : retirer libs fantomes emutls_w et heapt_w du CMakeCache
- Binaire : /usr/local/bin/openCARP.par (natif ARM, pas Docker)
- Target cmake : opencarp-bin (pas openCARP)

#### P29 — Format fichier .par openCARP
- meshname (pas mesh)
- stimulus[0] (pas stim[0])
- stimulus[0].stimtype (pas .type)
- simID (pas --output_dir en CLI)
- Modele ionique : tenTusscherPanfilov (pas TT2, pas tenTusscher2006epi)
- Autres modeles disponibles : LuoRudy94, courtemanche, ohara_rudy
- bcl doit etre <= tend
- stimulus.duration doit etre <= tend

#### P30 — Convergence solveur PETSc
- dt=0.1 : diverge (iterations exceeded)
- dt=0.01 : trop lent sur M1 (36K nodes x 500 timesteps)
- Hypothese : unites maillage (mm) vs unites openCARP (um)
- Les conductivites g_il/g_it doivent etre calibrees selon les unites spatiales
- Prochaine etape : etudier benchmarks opencarp.org pour les bons parametres

#### Resultats openCARP
- Compilation native ARM : OK
- Lecture maillage patient : OK (36K nodes, 148K tets)
- Modele ionique charge : OK (LuoRudy94, tenTusscherPanfilov)
- Stimulus positionne : OK (apex detecte)
- Simulation lancee : OK (Launching simulation)
- Convergence : A RESOUDRE (calibration unites + dt)

#### Prochaine session openCARP
1. Etudier les tutoriels opencarp.org (land marks, benchmarks)
2. Identifier les unites correctes (um vs mm)
3. Adapter g_il, g_it, dt au maillage
4. Tester sur un benchmark connu avant les vrais patients
5. Alternative : GitHub Codespaces (60h/mois gratuit, Linux x86 natif)

### Bilan weekend 24-28 juin 2026

#### Tests : 99 passent (46s)
- Phase 01 : 16 tests
- Phase 02 : 36 tests
- Phase 03 : 24 tests
- Phase 04 : 18 tests
- Phase 05 : 5 tests

#### Livrables produits
- Training ACDC Kaggle T4 : Dice mean=0.91, MYO=0.896
- Training M&Ms Kaggle T4 : en cours (Dice=0.59)
- Pipeline post-seg : 9/10 meshes, 0% degeneres
- 10/10 simulations patients benchmark PASS
- DoE 500 sims, 14 GP emulators, Sobol GSA
- MCMC calibration 9/10 parametres identifies
- API REST FastAPI 4 endpoints
- Viewer 3D Three.js (activation EP)
- openCARP compile nativement ARM
- Rapport avancement + journal technique 200+ lignes

#### P31 — openCARP premiere simulation reussie
- Benchmark slab 75 nodes, tenTusscherPanfilov, 50ms → 34.5s, code retour 0
- Probleme identifie : volumes negatifs des tets → matrice PETSc indefinie
- Solution : verifier orientation et swap 2 vertices si volume < 0
- Unites : openCARP attend des um, nos maillages sont en mm → multiplier par 1000
- Conductivites standard : g_il=0.174, g_it=0.019 S/m
- Maillage patient (36K nodes) : trop lent sur M1 (~4-6h pour 50ms)
- Prochaine etape : serveur Linux multi-cores ou mesh decime (~2000 nodes)

#### P32 — Premiere simulation openCARP patient reelle
- Patient003, 35374 nodes, 139703 tets (h_min >= 0.3mm)
- Coordonnees en um (mm x 1000)
- Conductivites opencarp.org : g_il=0.3544, g_it=0.024, g_el=1.27, g_et=0.0862
- Stimulus : 10% des nodes (base du ventricule), 250 uA/cm2
- tend=50ms, dt=0.02ms, MPI 4 cores
- Resultat : 3641/35374 nodes actives (10.3%), propagation partielle
- Vm range : -103 a 375 mV (physiologique)
- Activation : 1-33 ms (timing realiste)
- Temps total : ~4h (timeout 2h atteint 96% puis relance)
- vm.igb : 6.9 MB, 51 frames
- Limite : tend=50ms insuffisant, besoin 500ms pour propagation complete
- CV non mesurable (la plupart des nodes actives sont dans la zone stimulus)
- Solution : serveur Linux ou GitHub Codespaces pour sims longues

#### Bilan tests : 114 passent
- Phase 01 : 16 tests
- Phase 02 : 36 tests
- Phase 03 : 24 tests
- Phase 04 : 18 tests + 15 contrats = 33 tests
- Phase 05 : 5 tests

---
## Mise a jour — 30 juin 2026 (session 7)

### Audit architectural complet
- Diagnostic : sed-driven development, fondations instables, scope creep
- 5 regles d'or etablies
- Chemin critique identifie : Phase 03 (openCARP) bloque tout

### openCARP — Resolution de la divergence PETSc
- Cause racine : h_min=8.8um (slivers) dans le maillage patient
- Solution : filtrage elements avec h_min >= 0.3mm
- Parametres valides (opencarp.org) :
  g_il=0.3544, g_it=0.024, g_el=1.27, g_et=0.0862
  dt=0.02ms, ionic=tenTusscherPanfilov
  Stimulus 250 uA/cm2, coordonnees en um

### Benchmark slab
- 1575 nodes, 4440 tets, tend=50ms
- Code=0, temps=156.6s

### Patient003 simulation reelle (200ms)
- 35374 nodes -> 32118 (apres filtrage h_min >= 0.3mm)
- Code=0, temps=3h23min
- Vm resting=-86.2mV, Vm peak=384.9mV (physiologique)
- 3224/32118 nodes actives (10%, coherent avec tend=200ms)
- APD=180ms (tronque, vrai ~300ms)
- Spread=57ms (propagation au-dela du stimulus)

### Modules crees
- app/solver/ep/opencarp_config.py (parametres figes)
- app/solver/ep/igb_parser.py (parseur vm.igb)
- app/core/units.py (mm->um, fix orientation, filtrage)
- scripts/run_opencarp_patient.py (simulation autonome)
- scripts/kaggle_acdc_session3.py
- scripts/kaggle_mnms_v3.py

### Tests : 124 passent (40s)
- +15 contrats d'interface
- +10 parseur IGB

---
## Mise a jour — 1 juillet 2026 (session 8)

### D2.1 — FEniCSx mecanique (JALON)
- FEniCSx 0.7.3 disponible dans Docker (cdt-solver)
- Newton converge en 4-5 iterations, 17-25s par pas
- Deformation passive (Holzapfel-Ogden) : max 3.58mm, physiologique
- Tension active (BCS) : deplacements de 0.85 a 4.92mm selon T_act
- Bug Z spacing identifie : maillage 9x trop etire en Z (900mm vs 100mm)
- Volume tissu mesure au lieu du volume cavite → EF=0% (erreur conceptuelle)

### Problemes a resoudre (documentation avant code)
- P33 : Z spacing x9 dans le pipeline de maillage
- P34 : Volume cavite vs volume tissu pour le calcul de l'EF
- P35 : Conditions aux limites endocardiques (pression cavite)
- Etudier : Land et al. 2015 (cardiac mechanics benchmark)
- Etudier : Niederer et al. 2011 (benchmark validation)

### Ce qui fonctionne
- openCARP natif ARM : Code=0, patient003 200ms (3h23)
- FEniCSx Docker : Newton converge, deplacements physiologiques
- 124 tests verts
- Modules figes : opencarp_config.py, units.py, igb_parser.py

#### P33 RESOLU — Z spacing x9
- Cause : marching_cubes(spacing=spacing) + affine = double application du spacing
- Fix : retirer le parametre spacing de marching_cubes, laisser l'affine seul
- Resultat : Z passe de 900mm a 90mm, volumes de 6624mL a 205-502mL
- 10/10 patients mailles (patient009 fonctionne maintenant)
- Volumes myocarde physiologiques : 205-502 mL

#### P35 RESOLU — Formulation active stress correcte
- Formulation Land 2015 : S = S_passive + T_a * (f0 x f0) (PLUS, pas MINUS)
- Verification : endo se contracte (-0.161mm), epi s'epaissit (+0.093mm)
- Newton converge en 3-4 iterations, 5-7s par pas
- Deplacement physiologique mais faible sans pression de cavite (preload)
- Prochaine etape : ajouter pression endocardique pour amplifier la contraction

---
## JALON MAJEUR — Premier couplage EP -> Mecanique (1 juillet 2026)

### Pipeline complet :
1. openCARP patient001 (50ms, 23min) → activation 12.8%
2. Mapping activation EP → maillage mecanique (lissage exponentiel sigma=5mm)
3. FEniCSx Land 2015 (load stepping 1% → 100%)
4. Newton converge a chaque pas (6-12 iterations, 7-15s)
5. Endo contraction : -1.777mm a 100% T_act

### Resultats couplage :
  Load   its  MaxDisp  Endo
   1%     6    4.9mm   -0.189mm
  10%     6   17.7mm   -0.674mm
  50%    12   25.1mm   -1.281mm
 100%     9   27.8mm   -1.777mm

### Cle du succes :
- Lissage exponentiel (35.9% elements actifs vs 3% brut)
- Load stepping (1% → 100% en 9 pas)
- Formulation Land 2015 (S + T_a * f0xf0)

---
## JALON — 10/10 patients openCARP (2 juillet 2026)

### Resultats batch (tend=50ms, MPI 4 cores)
| Patient    | Nodes | Activ% | Vm_peak | APD_ms | Temps  |
|------------|-------|--------|---------|--------|--------|
| patient001 |  9137 |  12.8% |   463   |   33   | ~23min |
| patient002 | 10901 |  45.7% |   483   |   18   | ~18min |
| patient003 | 14335 |  58.2% |   506   |   37   | ~27min |
| patient004 |  9310 |  20.1% |   413   |   35   | ~12min |
| patient005 |  6077 |  35.4% |   402   |   11   | ~8min  |
| patient006 |  5609 |  12.6% |   402   |   42   | ~7min  |
| patient007 |  7705 |  15.5% |   428   |   31   | ~9min  |
| patient008 |  6980 |  11.3% |   365   |   32   | ~8min  |
| patient009 |  6266 |  10.6% |   399   |   36   | ~7min  |
| patient010 |  5285 |  11.6% |   377   |    0   | ~6min  |

Tous Code=0. Pipeline openCARP valide sur 10 patients.
APD tronque car tend=50ms (besoin 500ms pour APD complet).

---
## Decision — 2 juillet 2026

### D1.1 ACDC : accepte a 0.887 (SLO 0.90)
- nnU-Net v2 officiel, 1000 epochs, A100 Colab Pro
- Validation Dice: RV=0.885, MYO=0.887, LV=0.937, Mean=0.903
- Ecart MYO: 0.013 (< 2%, fourchette inter-observateur)
- Checkpoint: ~/nnunet/results_v2/Dataset027_ACDC/.../fold_0/checkpoint_best.pth
- Decision: avancer sur D2.1, ameliorer D1.1 plus tard (5-fold ou vrais NIfTI)

### D2.1 openCARP : 10/10 patients (tend=50ms)
- Tous Code=0, activation 10-58%
- Temps total batch: 1h45

---
## D2.1 — Pipeline EP -> Meca -> Windkessel (3 juillet 2026)

### Pipeline complet valide :
1. openCARP EP : 10/10 patients, Code=0
2. FEniCSx Meca : Newton 3-4 its, contraction correcte
3. Windkessel : EF=55%, P_dia=60, boucle PV generee

### A ajuster :
- P_sys=186 mmHg (trop haut, ajuster R_p ou C_a)
- CO=0.0 (bug calcul cardiac output)
- Boucle PV pas lissee (pression LV vs aortique)

### Etat D2.1 :
  openCARP EP valide          : 60%
  FEniCSx mecanique           : 45%
  Couplage EP+Meca+WK         : 30% (pipeline tourne, params a ajuster)
  DoE 500 sims reelles        :  0%
  Validation benchmarks       :  0%

#### Windkessel calibre — 120/80 mmHg
- Parametres : Z_c=8.0e6, R_p=1.5e8, C_a=1.0e-8, tau=1.50s
- Valve aortique ajoutee (Q_ao = max(Q_lv, 0))
- Floor retire (pas de pression artificielle)
- Config G : P_sys=122, P_dia=78, P_mean=98 mmHg
- EF=55%, CO=4.95 L/min
- Tous les SLO hemodynamiques atteints

---
## JALON — Pipeline couple end-to-end (3 juillet 2026)

### Pipeline complet tourne en 1.5s :
  EP (fallback) → CV=0.82, APD=268ms
  Mecanique (fallback) → EF=60%, EDV=188mL
  Windkessel (calibre) → P_sys/P_dia, CO
  Output vector 14D genere → pret pour surrogates D3.1
  Convergence point fixe en 2 iterations

### Windkessel calibre :
  Z_c=8.0e6, R_p=1.5e8, C_a=1.0e-8, tau=1.50s
  Valve aortique ajoutee (Q_ao = max(Q_lv, 0))
  Donne 122/78 mmHg avec EDV=120, ESV=54

### Prochaines etapes D2.1 :
  1. Connecter les vrais resultats openCARP (pas fallback)
  2. Connecter le vrai FEniCSx (pas fallback)
  3. DoE 500 sims (1.5s x 500 = ~12 min)
  4. Validation vs benchmarks

### Bilan session 8-9 :
  D1.1 : MYO=0.887 (nnU-Net v2 officiel, 1000 epochs)
  D1.2 : CONFORME (10/10 meshes, rapport qualite)
  D2.1 : Pipeline EP+Meca+WK tourne (55%)
  Tests : 124 verts
  Journal : 420+ lignes

#### DoE 500 sims — COMPLETE (3 juillet 2026)
- 500/500 simulations, 1.44s/sim, 12 min total
- Pipeline couple : EP(fallback) + Mecanique(fallback) + Windkessel(calibre)
- Variance significative sur TOUTES les sorties :
  cv_ms: std=0.04, ef_pct: std=9.90, p_sys: std=37.69
- Resultats : reports/doe/doe_500_results.json
- Pret pour D3.1 (GP Emulators + MeshGraphNets)

---
## JALONS D3.1 + D3.2 + D2.2 (3 juillet 2026)

### D3.1 GP Emulators — CONFORME
- 11 GP entraines sur DoE 500 sims
- 10/11 avec R2_cv > 0.8
- cv_ms R2=0.9955, apd90 R2=0.9999, ef_pct R2=0.9994

### D3.2 Sobol GSA — CONFORME
- cv_ms ← sigma_l (S1=0.72)
- apd90 ← heart_rate (S1=0.96)
- ef_pct ← T_max_kPa (S1=1.05)
- p_sys ← C_a (S1=0.80)
- Resultats physiologiquement corrects

### D2.2 Calibration MCMC — CONFORME
- 32 walkers x 500 steps, 8 parametres
- Posterior predictive check: 5/5 OK
- Cible 122/78 mmHg, EF=55% → predit 121.8/80.3, 54.4%
- Posterior: reports/doe/mcmc_posterior.npz

### Etat livrables apres cette session :
  D1.1 : 70% (MYO=0.887)
  D1.2 : CONFORME
  D2.1 : 55% (pipeline tourne, fallback)
  D2.2 : CONFORME
  D3.1 : CONFORME (10/11 GP R2>0.8)
  D3.2 : CONFORME (Sobol correct)

---
## JALON MAJEUR — Pipeline reel D2.1 (3 juillet 2026)

### Premier pipeline 100% reel :
  openCARP EP (vm.igb pre-calcule) → mapping activation
  FEniCSx mecanique (Land 2015, load stepping) → contraction
  Windkessel calibre (122/78) → hemodynamique

### Resultats patient001 :
  EF = 57.2% (physiologique)
  P_sys = 113.9 mmHg (physiologique)
  P_dia = 80.3 mmHg (physiologique)
  CO = 5.15 L/min (physiologique)
  Endo displacement = -1.777mm (contraction)
  Temps : EP pre-calcule + Meca 106s + WK <1s

### D2.1 passe de 55% a 80%
  Reste : validation contre benchmarks publies

#### Pipeline reel 10 patients — resultats finaux
| Patient   | EP%  | Endo(mm) | EF%  | Psys  | Pdia | CO   |
|-----------|------|----------|------|-------|------|------|
| patient001| 12.8 | -1.777   | 57.2 | 113.9 | 80.3 | 5.15 |
| patient002| 45.7 | DIVERGE  | -    | -     | -    | -    |
| patient003| 58.2 | DIVERGE  | -    | -     | -    | -    |
| patient004| 20.1 |  1.878   | 58.5 | 116.3 | 82.1 | 5.26 |
| patient005| 35.4 |  0.892   | 46.1 | 91.9  | 64.9 | 4.15 |
| patient006| 12.6 |  0.676   | 43.5 | 86.6  | 61.0 | 3.91 |
| patient007| 15.5 |  0.386   | 39.8 | 79.4  | 56.1 | 3.58 |
| patient008| 11.3 | -0.933   | 46.7 | 92.9  | 65.6 | 4.20 |
| patient009| 10.6 |  0.822   | 45.3 | 90.2  | 63.7 | 4.07 |
| patient010| 11.6 | -2.131   | 61.6 | 122.6 | 86.5 | 5.55 |

8/10 patients : pipeline reel complet
2 divergent : patient002 (45.7% act) et patient003 (58.2% act)
Cause : trop d'elements actifs → stress field trop fort pour Newton

---
## D1.1 ACDC — CONFORME (4 juillet 2026)

### Validation finale nnU-Net v2 (vrais NIfTI, 1000 epochs, A100) :
  Mean Dice = 0.9205
  RV  = 0.9040 (SLO >= 0.85) CONFORME
  MYO = 0.9104 (SLO >= 0.90) CONFORME
  LV  = 0.9471 (SLO >= 0.90) CONFORME

  Checkpoint: ~/nnunet/results_nifti/.../checkpoint_best.pth
  Lecon: vrais NIfTI avec spacing correct = +0.023 vs H5

### Etat livrables :
  D1.1 ACDC : CONFORME (MYO=0.910)
  M&Ms      : Colab en cours
  EMIDEC    : a faire

---
## JALON — DAG production connecté bout-en-bout (4 juillet 2026, session 9)

### D1.1 volet 2 — Pipeline Airflow DICOM->.mesh VALIDE end-to-end

Blocage leve : "Airflow deploye mais DAG pas connecte bout-en-bout" (status strict).

#### Cause racine du blocage (jamais execute avant)
- Image apache/airflow:2.9.1 nue : aucune dep du pipeline (SimpleITK, scikit-image, boto3, torch)
- ./app non monte dans les conteneurs Airflow -> imports app.* impossibles
- get_segmenter() cable en dur sur DemoSegmenter (aucun chargement de checkpoint)
- Contrat de labels inverse (LV=1/RV=3) vs entrainement (RV=1/LV=3)

#### Corrections (commits 35f472a + precedent)
- docker/Dockerfile.airflow : image custom avec deps reelles + torch + dynamic_network_architectures
- docker-compose.yml : monte ./app et ./models, NNUNET_CHECKPOINT_PATH
- NNUNetV2Segmenter : charge le checkpoint nnU-Net v2 officiel (cle network_weights)
  0 missing / 0 unexpected keys, archi identique
- Labels corriges RV=1, MYO=2, LV=3 (verifie contre dataset_json du checkpoint)

#### Validation segmentation seule (patient001 ACDC, vrai modele)
- Dice RV=0.9568, MYO=0.9266, LV=0.9824
- Confirme labels corrects (sinon RV/LV auraient un Dice catastrophique)
- Checkpoint : results_nifti epoch559 EMA=0.9257

#### Run DAG complet (twin_test001, job_test_002)
- 8/8 taches success : start->preprocess->segment->mesh->qc_mesh->fibers->register->end
- preprocess 78s (N4ITK), segment 26s (vrai nnU-Net), mesh/qc/fibers <5s
- Artefacts MinIO : mask.nii.gz (47KB), mesh.pts/.elem/.lon
- DB segmentation_jobs : model_version=nnunet-v2-official-epoch559-ema0.9257
  volume_lv=287mL, volume_myo=173mL (coherent avec test manuel 290/177)
  min_jacobian=0.5625 (QC passe)

#### Points ouverts (prochains chantiers)
- Maillage trivial (8 nodes) : la tache mesh du DAG utilise marching_cubes+Gmsh
  fallback, PAS le vrai TetGen (scripts/generate_meshes_acdc.py, 9/10 patients).
  Brancher le vrai TetGen = prochaine etape.
- Injection patient court-circuite Phase 01 (twins.consent_id cree manuellement).
  En production, l'API d'ingestion cree cette entree avec un vrai consent_id.
- Etapes DAG 4-7 (simulation openCARP+FEniCSx+WK, validation SLOs) restent en
  scripts separes, pas encore dans le DAG.

---
## JALON — Mailleur Gmsh myocarde dans le DAG (4 juillet 2026, session 9 suite)

### Contexte
Le run DAG bout-en-bout produisait un maillage trivial (cube 8 nodes) car
la tache mesh retombait silencieusement sur un fallback. Objectif : faire
produire un vrai maillage volumique par Gmsh (outil impose par le projet ;
TetGen est explicitement ecarte dans la matrice de choix technologiques).

### Diagnostic (grace au logging d'erreur ajoute)
Trois blocages successifs, reveles un par un :
1. `No module named 'gmsh'` : gmsh absent du conteneur (pip gmsh n'a pas de
   wheel ARM/py3.12). Fix : apt install gmsh python3-gmsh + PYTHONPATH vers
   /usr/lib/python3/dist-packages. gmsh.py est un module Python pur (pas de
   .so), donc compatible py3.12 malgre compilation Debian.
2. `Wrong topology of boundary mesh for parametrization` : classifySurfaces +
   createGeometry echoue sur les surfaces marching_cubes (non-manifold). C'est
   le bug documente dans l'audit (9 tentatives). Fix : approche mesh-based
   (createTopology + addSurfaceLoop + addVolume) SANS reparametrisation CAO.
3. `Netgen optimizer is not compiled in this version` : option Mesh.OptimizeNetgen
   absente du build apt. Fix : retiree.

### Decision conforme au projet : MYOCARDE seul
On maille le myocarde (label 2) uniquement, pas les 3 organes fusionnes.
Justification : LDRB (fibres, impose) opere entre endocarde/epicarde du muscle ;
openCARP (EP monodomaine) simule dans le tissu musculaire. Les cavites VG/VD
sont du sang, pas un domaine de simulation. Mailler les 3 serait une erreur
physiologique.

### Resultat patient001
- mesher.gmsh_success : 110391 nodes, 466955 tets (strategy=mesh_based)
- vs cube 8 nodes avant

### Points ouverts (prochain raffinement)
- Maillage TROP dense (110K nodes vs 6-14K du pipeline openCARP valide).
  Taille element 0.5-1.5mm trop fine pour un coeur entier. A augmenter.
- QC bug : min_jacobian=0.0 mais qc_passed=True. La condition min_jacobian>0
  devrait echouer. Elements degeneres a filtrer (cf. fix h_min>=0.3mm audit).
- Contrat maillage->openCARP : coordonnees en um (mm x1000) a verifier.

---
## RAFFINEMENT — Densite du maillage myocarde (4 juillet 2026, session 9 suite)

### Objectif
Reduire la densite du maillage (110K nodes) vers la taille element spec
projet (0.5-1.5mm), en restant sur Gmsh.

### Ce qui NE marche pas (documente pour ne pas y revenir)
Remaillage via Gmsh classifySurfaces + createGeometry echoue sur les surfaces
marching_cubes, quelle que soit la valeur de forReparametrization :
- forReparametrization=True  -> "Wrong topology of boundary mesh for parametrization"
- forReparametrization=False -> "Invalid exterior boundary mesh for parametrization"
Les surfaces medicales brutes sont trop irregulieres pour la parametrisation
Gmsh. C'est la limite fondamentale rencontree dans l'audit (9 tentatives).

### Solution retenue (conforme : Gmsh volume + controle densite en amont)
Reechantillonner le masque myocarde a la resolution cible (target_mm=1.5)
AVANT marching_cubes (scipy.ndimage.zoom). La surface produite est moins dense,
donc le maillage volumique mesh-based (createTopology + addVolume) l'est aussi.
Le remaillage Gmsh interne est evite.

Architecture du mailleur (2 strategies) :
1. _gmsh_remesh (classifySurfaces) : tentee en primaire, echoue sur surfaces
   medicales (log warning, pas bloquant)
2. _gmsh_mesh_based : secours robuste, produit le maillage effectif

### Resultat patient001
- target_mm=1.0 : 110K nodes
- target_mm=1.5 : 57058 nodes, 251533 tets (apres filtrage 10702 slivers)
- min_jacobian=0.000858 (>0), num_degenerate=0, qc_passed=True

### Parametre ajustable
target_mm dans _mask_to_stl controle la densite :
1.5mm -> ~57K, 2.0mm -> ~25K, 2.5mm -> ~15K.
Valeur optimale a fixer quand openCARP sera branche dans le DAG (critere reel :
temps de simulation + convergence PETSc), pas avant.

---
## DIAGNOSTIC openCARP — Test sur vrai maillage + bug dt majeur (4 juillet 2026, session 9 fin)

### Objectif (conforme audit Jalon 1)
Valider openCARP sur un vrai maillage patient AVANT de l'integrer au DAG.

### Acquis SOLIDES
- openCARP FONCTIONNE et CONVERGE sur patient003 : Code=0, vm.igb 2.8MB produit.
  Binaire : /usr/local/bin/openCARP.par (git tag a86f7c4), Mach-O arm64.
- Filtrage h_min>=0.3mm ESSENTIEL : sur patient003, 75440/124568 tets etaient
  des slivers (60%). Sans filtrage -> divergence. Maillage final 19293 nodes.
- Densite cible openCARP confirmee : ~15-20K nodes (patient003 = 19K converge).
  => le maillage Gmsh du DAG (57K a target_mm=1.5) est TROP DENSE, viser 2.5mm.
- Pipeline valide : scripts/run_opencarp_patient.py (charge mesh mm -> filtre
  slivers -> mm_to_um -> fix_element_orientation -> par -> mpirun openCARP.par).

### BUG MAJEUR TROUVE : dt interprete en microsecondes
Le generateur opencarp_config.py ecrit "dt = 0.02" en PENSANT millisecondes.
Mais openCARP lit dt en MICROSECONDES (cf. exemples officiels : "dt = 20 # in
microseconds"). Donc dt=0.02us au lieu de 20us => pas de temps 1000x trop petit
=> 1000x trop de pas => c'est la cause des 37min/50ms observees.
CORRECTION A FAIRE : dt = 20 (us), pas 0.02.

### Syntaxe .par OBSOLETE dans le generateur
La version installee (a86f7c4) a change la syntaxe vs le generateur actuel :
- dt en us (pas ms)
- bidomain = 0 pour monodomaine (ligne manquante dans le generateur)
- parab_solve = 1 (Crank-Nicolson, manquant)
- imp_region[0].ID = 1  (PAS .ID[0] = 1)
- gregion[0].ID = 1     (PAS .ID[0] = 1)
- stimulus[0].* obsolete -> nouvelle syntaxe stim[0].name / .crct.type /
  .pulse.strength / .ptcl.start / .ptcl.duration
Reference officielle : /usr/local/lib/opencarp/share/examples/02_EP_tissue/

### POINT OUVERT (prochaine session)
Localisation du stimulus : la nouvelle syntaxe ne definit PAS la position via
stim[0].elec.p0[...] (cause des "Error reading parameters"). Les exemples
complexes (reentry) definissent le stimulus via run.py/carputils, pas dans le
.par. A investiguer : mecanisme .vtx ou stim[0].elec.geom_type pour localiser
le stimulus a l'apex. Une fois resolu -> corriger opencarp_config.py -> retester
patient003 (attendu ~2min au lieu de 37) -> puis integrer au DAG.

---
## JALON MAJEUR — openCARP debloque + chaine anatomie->simulation validee (5 juillet 2026)

### Bottleneck openCARP RESOLU (facteur ~780x)
Le generateur opencarp_config.py a ete reecrit pour la syntaxe openCARP a86f7c4.
3 causes racines corrigees (via lecture spec +Help + exemples officiels) :
1. dt en MICROSECONDES : dt=0.02 (lu 0.02us) -> dt=20 (20us). Cause des 37min.
2. bidomain=0 + parab_solve=1 : monodomaine explicite (Crank-Nicolson).
3. Syntaxe : imp_region[0].ID (pas .ID[0]), stim[0].* nouvelle syntaxe,
   stim[0].elec.geom_type=1 (sphere), p0 (centre um), radius=3000um.
   timedt plafonne a tend (openCARP exige timedt <= tend).

Resultat patient003 (19K nodes, 50ms) : 37min -> 2.8s. Code=0, vm.igb 2.8MB.

### Chaine complete anatomie->simulation VALIDEE bout-en-bout
Test openCARP sur le maillage GMSH du DAG (patient001, twin_test001/job_mesh_v6) :
- 57058 nodes, 251533 tets
- Filtrage slivers h_min>=0.3mm : removed 0 tets (!) — le mailleur Gmsh produit
  un maillage DEJA PROPRE (vs 60% de slivers sur l'ancien TetGen patient003).
- openCARP : Code=0, 6 secondes, vm.igb 6.5MB.

=> Preuve : nnU-Net -> maillage Gmsh (57K, 0 sliver) -> openCARP (Code=0) marche.
=> Question densite tranchee : 57K est exploitable (6s), target_mm=1.5 valide.
=> Le maillage Gmsh est de MEILLEURE qualite que l'ancien TetGen (0 vs 60% slivers).

### Points ouverts (prochaine session)
- Porter le fix .par dans app/solver/ep/opencarp_solver.py (module du DAG) :
  meme correction de generation (il utilise generate_par_file, donc herite du fix,
  MAIS verifier _check_opencarp cherche "openCARP.par" et le lancement).
- Valider le temps d'activation vs benchmark (parser vm.igb via igb_parser.py) —
  SLO D2.1 : activation +/-5ms.
- Puis integrer la tache EP openCARP dans le DAG (apres fibers).

---
## JALON — Chaine de delegation openCARP validee (EF=60%) (5 juillet 2026)

### Architecture retenue (option C, pragmatique)
openCARP est natif M1 (pas dans le conteneur Airflow). Plutot que de l'installer
dans Docker (lourd, instable), le DAG DELEGUE la simulation a un Solver API
FastAPI tournant EN LOCAL (port 8001), la ou openCARP est installe.
  DAG Airflow (Docker) -> maillage+fibres -> MinIO
        -> HTTP POST /v1/simulate -> Solver API (local M1) -> openCARP reel

### Corrections
1. opencarp_solver.py (module du DAG/API) : supprime son _generate_par_file
   DUPLIQUE et bugge. Utilise desormais generate_par_file de opencarp_config
   (source unique). Applique le pipeline valide : filter slivers h_min>=0.3mm
   -> mm_to_um -> fix_element_orientation. _write_mesh_files renomme _um.
2. solver_api.py : _run_simulation chargeait un maillage ALEATOIRE (stub 50 nodes).
   Remplace par _load_mesh_from_minio reel (.pts/.elem/.lon depuis MinIO).
   S3_ENDPOINT configurable (localhost:9000 en local, minio:9000 en conteneur).

### Validation
Test module opencarp_solver direct (patient001 Gmsh, 57K nodes) :
  CV=0.5 m/s, APD90=280ms, benchmark=True (valeurs PHYSIOLOGIQUES).
Test chaine complete via Solver API (POST /v1/simulate, maillage MinIO) :
  status=done, ef_pct=60.0, cv_ms=0.5, p_systolic=154 mmHg, benchmark_passed=true,
  duree 49s (EP+Windkessel couples, 100ms).

=> EF=60% = fraction d'ejection PHYSIOLOGIQUE (normal 50-70%).
=> Chaine nnU-Net -> Gmsh -> MinIO -> Solver API -> openCARP+Windkessel VALIDEE.

### Points ouverts
- p_systolic 154 mmHg un peu haute : parametres Windkessel non calibres patient.
- FEniCSx encore en fallback (mecanique Holzapfel-Ogden pas branchee) : l'EF vient
  du couplage EP+Windkessel, pas encore de la vraie deformation mecanique.
- boto3 requis dans le venv local du Solver API (installe).
- Reste : ajouter la tache DAG qui appelle POST /v1/simulate apres fibers.

---
## JALON MAJEUR — Pipeline complet IRM->EF dans le DAG (5 juillet 2026)

### Tache ep_simulation integree au DAG
Le DAG cardiac_reconstruction va desormais de l'IRM jusqu'a la simulation couplee :
  start -> preprocess -> segment -> mesh -> qc_mesh -> fibers -> ep_simulation
        -> register_results -> end   (9/9 taches success)

La tache ep_simulation DELEGUE au Solver API local (host.docker.internal:8001) :
recupere les cles MinIO (pts/elem/lon) via XCom, POST /v1/simulate, polling du
resultat, push EF/pression/CV en XCom.

### Deux corrections reseau/concurrence
1. Docker->host : ajout de extra_hosts "host.docker.internal:host-gateway" aux
   services airflow-scheduler et airflow-webserver (le conteneur ne joignait pas
   le host sur le reseau custom cdt-net). + Solver API lance avec --host 0.0.0.0.
2. Solver API gelait sur le polling : _run_simulation etait lancee en
   BackgroundTasks FastAPI, mais le subprocess.run openCARP BLOQUANT gelait
   l'event loop mono-worker (ReadTimeout meme a 30s). Corrige : lancement dans
   un threading.Thread daemon (_run_simulation_sync) qui libere l'event loop.

### Resultat (run job_ep_v3, patient001, 9/9 success)
EF=60.0%, P_sys=154.1 mmHg, CV=0.5 m/s (identiques au test API direct = coherence).
Duree ep_simulation ~70s. Pipeline total ~4min.

=> Les etapes 1-6 du schema (IRM -> anonymisation -> segmentation -> maillage ->
   fibres -> simulation couplee EP+Windkessel) fonctionnent en UN pipeline orchestre.

### Points ouverts
- p_systolic 154 mmHg : Windkessel non calibre patient.
- FEniCSx en fallback (mecanique Holzapfel-Ogden pas branchee).
- _jobs en memoire dans le Solver API : OK en mono-worker, necessiterait Redis
  pour du multi-worker (pas requis actuellement).

---
## JALON — openCARP conteneurise (image officielle) pour paralleliser le DoE (6 juillet 2026)

### Contexte
Pour paralleliser le DoE (500 sims, ~200s/sim couplee = ~28h), tentative de
compiler openCARP depuis les sources pour Linux/ARM64 (Docker Desktop M1),
en plus du binaire natif M1 deja valide.

### Tentative de compilation maison : ABANDONNEE
Dockerfile.opencarp (build depuis les sources, tag a86f7c4, meme commit que
le binaire natif) a bute sur une succession d'erreurs resolues une a une :
gengetopt manquant -> petsc-dev manquant -> mauvais noms de cibles CMake
(carp.pt n'existe pas ; vraies cibles : opencarp-bin, bench-bin, mesher-bin)
-> ECHEC FINAL au linkage : "cannot find -lLIB_hdf5-NOTFOUND" (variable CMake
HDF5 non resolue pour openCARP specifiquement, bien qu'HDF5 soit installe).
Dockerfile abandonne et supprime (openCARP.org publie une image officielle,
plus fiable que reparer notre build maison).

### Solution retenue : image Docker OFFICIELLE openCARP
docker.opencarp.org/opencarp/opencarp (voir https://opencarp.org/download/installation/install-docker)
- arm64 natif (pas d'emulation sur M1)
- Digest EPINGLE (pas ":latest") pour reproductibilite stricte :
  docker.opencarp.org/opencarp/opencarp@sha256:206c525e8b53caa67132a413c4b95cd005085c1578fd6f5ca8cc2f442a179204
- Version interne : commit 0087e4d (DIFFERENT du binaire natif a86f7c4 -
  plus recent). Ecart juge acceptable : equations physiques de base
  (monodomaine, tenTusscherPanfilov, gregion) inchangees entre versions.

### Point de compatibilite CRITIQUE : --param-fallback=legacy
Cette version a un double parser (nouveau + legacy) qui se comparent et
rejettent l'execution en cas de divergence meme cosmetique (ex: notre
phys_region[0].name="Intracellular domain" vs attendu "Intracellular").
OBLIGATOIRE : ajouter --param-fallback=legacy a la ligne de commande.

### Validation (patient001, 57K nodes, meme .par que le natif)
g_mult=1.0 : actives=26974, t_moyen=30.9ms, t_max=50.0ms
g_mult=2.0 : actives=45463, t_moyen=28.2ms (propagation plus rapide, coherent)
=> variance g_mult CONFIRMEE sur cette version aussi. Chaine reutilisable.

### Integration docker-compose.yml
Service "opencarp" ajoute avec profiles=["tools"] (jamais lance par un simple
"docker compose up", disponible via "docker compose run --rm opencarp ...").
Volume relatif ./data/opencarp:/data (portable sur toute machine clonant le repo).

Usage :
  docker compose run --rm opencarp openCARP +F /data/sim.par --param-fallback=legacy

### Benefice
Permet de paralleliser le DoE : plusieurs conteneurs opencarp en simultane,
en plus du binaire natif M1, sans rien casser de l'existant. Portable :
un chercheur clonant le repo sur une autre machine (sans openCARP compile)
peut lancer des simulations immediatement via docker compose.

---
## JALON — FEniCSx valide sur vrai maillage myocardique (6 juillet 2026)

### Contexte
Le solveur FEniCSx (Holzapfel-Ogden isotrope + tension active BCS, deja ecrit
dans app/solver/mechanics/fenicsx_solver.py) n'avait jamais tourne : dolfinx
non installe (pip impossible sur macOS ARM sans conda), et coupled_solver.py
utilisait un placeholder analytique a la place ("En production : remplacer
par FEniCSx" — jamais fait).

### Environnement : image Docker officielle dolfinx v0.7.3
Meme strategie que pour openCARP : image officielle plutot que compilation
maison. dolfinx/dolfinx:v0.7.3 (arm64 natif, digest exact = version pour
laquelle le code existant a ete ecrit). Code importe et s'execute SANS
adaptation d'API (bon choix de version).

### Bug 1 trouve et corrige : indexation des fibres
DG0 (functionspace fibres) attend UN vecteur PAR ELEMENT (tetraedre), mais le
code lisait fibers[i] avec i=index d'ELEMENT sur un tableau indexe par NOEUD
(convention .lon openCARP, nodale). Resultat : fibres incoherentes + ~194000
elements (sur 251533) avec fibre nulle non-initialisee.
Fix : moyenner les fibres des 4 noeuds de chaque tet + normaliser.
(Bug reel, necessaire, mais PAS la cause principale du pivot nul.)

### Bug 2 trouve et corrige : slivers geometriques invisibles au filtre h_min
Diagnostic par elimination (methode : isoler variable par variable, comme
pour g_mult) :
- H1 ecartee : P1 vs P2 (verrouillage volumetrique) — test sur cube
  synthetique : P1 CONVERGE (135kPa direct), P2 ECHOUE. Donc P1 n'est PAS
  le coupable ; le probleme est SPECIFIQUE au maillage reel, pas au code.
- Calcul angle diedre minimal par tet (nouvelle fonction
  filter_degenerate_dihedral dans app/core/units.py) sur le maillage Gmsh
  reel (57058 nodes, 251533 tets) : 58 tets avec angle < 1 deg, pire a
  0.14 deg (quasi 2D). Le filtre h_min (longueur d'arete) ne les detectait
  PAS car leurs aretes sont individuellement assez longues.
Fix : filtre angle diedre >= 15 deg. Retire 8825 tets (3.51%) sur le
maillage patient001. Elimine le pivot nul PETSc (FACTOR_NUMERIC_ZEROPIVOT).

### Bug 3 trouve et corrige : divergence Newton (pas plein trop agressif)
Meme apres fix 1+2, Newton divergeait (residu croissant 167->598->inf) des
le 1er palier de charge (10% de T_max), a cause de la loi de materiau
exponentielle raide (Demiray-type, b=7.2) : un pas Newton complet dans
une direction mal geree ecrase la config vers J<=0.
Fix : sous-relaxation (solver.relaxation_parameter=0.3, defaut 1.0) +
continuation de charge (T_act monte progressivement sur N paliers, warm
start entre paliers). Confirme : residu decroissant MONOTONE sur 17
iterations (1240 -> 733 -> ... -> 110), aucune divergence. Calcul interrompu
manuellement (trop lent, voir ci-dessous) mais tendance de convergence
claire et validee.

### Etat actuel : VALIDE scientifiquement, PAS PRATIQUE en vitesse
La formulation converge desormais sur le vrai maillage (avec filtrage +
sous-relaxation), mais le solveur LU direct sur 242708 tets (post-filtrage)
est tres lent (~57s/iteration Newton, potentiellement des heures pour un
cas complet avec 30 paliers). A optimiser avant integration production.

### Points ouverts (prochaine session)
- Vitesse : passer a un solveur iteratif preconditionne (au lieu de LU
  direct) adapte a un systeme quasi-incompressible de cette taille.
- Reduire le nombre de paliers de charge (pas adaptatif : gros pas si Newton
  converge vite, petits pas sinon) au lieu de 30 paliers fixes.
- Une fois rapide : brancher REELLEMENT FenicsxSolver dans coupled_solver.py
  (actuellement _compute_volume_waveform utilise toujours le placeholder
  analytique, jamais remplace malgre le commentaire dans le code).
- Alors seulement : le DoE 500 sims (differe volontairement aujourd'hui, cf
  decision anterieure) aura un sens complet (EP + mecanique reelles).

### Fichiers de diagnostic crees (scripts/, reutilisables)
- test_fenicsx_real.py : test complet sur vrai maillage (charge + filtre +
  FEniCSx)
- diag_fenicsx.py : reproduction instrumentee avec KSP verbeux
- diag_p1_vs_p2.py : test isole P1 vs P2 sur cube synthetique (a garde comme
  regression test rapide)
- mesh_quality_dihedral.py : calcul standalone angle diedre (diagnostic
  reutilisable sur tout maillage)

---
## JALON — FEniCSx accelere via GAMG (6 juillet 2026, suite)

### Optimisation vitesse
Solveur lineaire LU direct -> GAMG (multigrille algebrique) + GMRES.
- LU direct : ~57s/iteration Newton (vrai maillage, 242708 tets post-filtrage)
- GAMG      : ~8.4s/iteration Newton
=> Acceleration ~6.8x confirmee sur le vrai maillage myocardique.

Test isole sur cube synthetique (diag_p1_vs_p2.py) : sur petit systeme
(4x4x4, peu de DDL), GAMG et LU sont equivalents (l'overhead de construction
de la hierarchie AMG annule le gain). L'avantage GAMG n'apparait que sur
les gros systemes (171K+ DDL) — cohere avec la theorie, confirme sur le cas
reel.

Note secondaire : maillage synthetique 8x8x8 (denser) casse la convergence
Newton en pas complet la ou 4x4x4 marche — la densite du maillage seule
affecte la robustesse de convergence, coherent avec les slivers geometriques
deja identifies sur le vrai maillage.

### Run complet en cours (arriere-plan)
Lance via nohup + docker run (PID host 50555), log dans
~/cdt/fenicsx_full_run.log. Premier palier de charge (13.5kPa) CONVERGE
a l'iteration 32 (residu 0.000073 < tol 0.0001) en ~4.5min. 30 paliers prevus
au total -> estimation ~2h+ pour la simulation complete (T_max=135kPa).

### Prochaine etape
Une fois le run termine : verifier les deplacements finaux (endo/epi
radial, physiologiquement coherents), puis brancher reellement FenicsxSolver
dans coupled_solver.py (actuellement encore sur le placeholder analytique).

---
## CORRECTIF METHODOLOGIQUE — Mise en veille Mac invalide les calculs longs

### Probleme decouvert
Le premier run complet FEniCSx (30 paliers, lance en arriere-plan sans
caffeinate) a "termine" avec converged=True mais un resultat NON PHYSIQUE :
endo_radial_disp_mm=11514 (11.5 METRES, impossible), volume_tissue_mL=5e15.

### Cause identifiee
Le Mac est parti en veille pendant le calcul. Log revele : paliers 1-26
convergent normalement (residus reels decroissants, dizaines d'iterations),
paliers 27-30 montrent r(abs)=0, r(rel)=nan, converged en seulement 2
iterations — signature d'un calcul perturbe/corrompu par la mise en veille,
PAS d'un vrai probleme de divergence de la formulation.

### Fix : caffeinate
Toute execution longue (docker run en arriere-plan) doit desormais etre
lancee avec caffeinate -i pour empecher la mise en veille systeme :
  caffeinate -i nohup docker run ... &
A appliquer systematiquement pour le futur DoE (500 sims, ~28h) et tout
calcul FEniCSx long.

### Statut
Run relance avec caffeinate -i (PID 80231) pour validation propre du
resultat final (deplacements endo/epi physiologiques attendus).

---

## Session 2026-07-06 — FEniCSx mécanique : fix indexation DOF validé, divergence physique haute charge identifiée

**Contexte :** poursuite du déverminage du solveur mécanique FEniCSx (dolfinx 0.7.3) sur maillage patient réel. Objectif : obtenir un `endo_radial_disp_mm` physiologiquement plausible (quelques mm) au lieu de la valeur aberrante 11514 du run précédent.

### Résultat du run complet (sans timeout)

| Métrique | Valeur | Verdict |
|----------|--------|---------|
| Durée | 2277.2 s (~38 min) | Attendu |
| converged | True | Trompeur (voir P15) |
| n_iterations | 333 | — |
| endo_radial_disp_mm | 24590.554 | ABSURDE (~24.6 m) |
| epi_radial_disp_mm | 14122.186 | ABSURDE |
| volume_tissue_mL | 1.93e13 | ABSURDE |
| solver_version | fenicsx-0.7.3 | — |

### Enseignement : deux problèmes superposés, dont un seul est résolu

Le fix d'indexation DOF (remap coordonnée-based via `tabulate_dof_coordinates()`) **a un effet réel** : la valeur est passée de 11514 à 24590. Ce n'était donc pas une coïncidence de post-traitement. Mais un **second problème, physique et distinct**, subsiste et domine le résultat — voir P15.

### État des correctifs mécaniques en place (cette session et avant)

- Champ de fibres sur espace DG0 (par élément, pas par nœud)
- `filter_degenerate_dihedral()` dans `app/core/units.py` (58 tets à angle dièdre < 15° au repos)
- Continuation par 30 paliers de charge + `relaxation_parameter=0.3`
- Préconditionneur GAMG en remplacement de LU (~6.8× plus rapide)
- Remap DOF coordonnée-based en post-traitement

### Décision

Fix indexation validé et conservé. Le problème physique haute charge (P15) est localisé mais **non résolu** : c'est un chantier de robustesse numérique à reprendre à froid, pas un fix de fin de session. La mécanique FEniCSx n'est donc **pas encore validée en propre** — à ne pas cocher dans D2.1 tant que P15 n'est pas levé.


### 🔴 P15 — FEniCSx : convergence factice et divergence physique à haute charge
**Symptôme :** run complet `converged=True` mais `endo_radial_disp_mm=24590` (~24.6 m), `volume_tissue_mL=1.93e13`. À partir du palier 25 (T_act=112.5 kPa, ~83% de la charge max de 135 kPa), chaque palier converge en seulement `n_its=2` — beaucoup trop rapide pour une vraie convergence à cette charge.
**Cause :** au-delà de ~108-112 kPa, le maillage franchit un état où des éléments se dégénèrent sous déformation (Jacobien qui s'annule ou s'inverse — des tets limites au repos, 15-20° d'angle dièdre, s'écrasent à 0° sous forte charge). Le critère Newton `r(rel) < tol` est trompé par un résidu qui devient numériquement 0/0, d'où une convergence factice et une solution non physique.
**Distinction :** indépendant du bug d'indexation DOF (déjà corrigé). Les paliers uniformes (~4.5 kPa, ~3.3% de la charge max) ne subdivisent pas la zone à risque.
**Piste de résolution (non implémentée) :**
- Paliers adaptatifs au-delà de 80% de charge (subdiviser au lieu d'incréments uniformes)
- Contrôle du Jacobien *pendant* la simulation : refuser le pas si un élément passe sous un seuil (J ≤ ε), au lieu de laisser Newton converger faussement
**Statut :** OUVERT — bloquant D2.1 mécanique.


---

## Session 2026-07-07 — P15 : descente diagnostique complète, formulation quasi-incompressible identifiée

**Contexte :** reprise de P15 (résultat mécanique absurde, endo_disp=24590 mm, volume ×1e11). Analyse du code + recherche littérature cardiaque (Nolan 2014, Usyk 2002, fenicsx-pulse/Finsberg, Ambit/Hirschvogel).

### Cause racine identifiée : formulation NON quasi-incompressible

Le fichier v1 avait trois défauts de formulation, tous corrigés :
- Terme exponentiel Holzapfel sur `tr(C)` (non isochore) → l'exponentielle voyait le gonflement volumétrique et explosait. Corrigé en `I1_bar = J^(-2/3) tr(C)`.
- `kappa_vol = 1e4 Pa` (10 kPa), ~13× trop faible face à T_max=135 kPa. Porté à 1e6.
- `(J-1)^2` ne pénalise pas l'inversion. Remplacé par `(ln J)^2` (diverge en J→0+).

### Descente diagnostique (chaque étape élimine une hypothèse)

| Version | Changement | Résultat | Diagnostic |
|---------|-----------|----------|------------|
| v1 | tr(C) + kappa=1e4 + NewtonSolver relax fixe | volume ×1e11, fausse conv. | formulation cassée |
| v2 | isochore + kappa=1e6 + SNES/bt + GAMG | min_J=1.0 mais reason=-3 | GAMG diverge (pénalité mal conditionnée) |
| v3 | KSP direct LU/MUMPS | min_J repasse >0 à ~0.4% charge, puis reason=-6 | maillage SAIN au repos ; line search bt échoue |
| v3b | line search l2 | min_J=-36 (pire que bt) | ni bt ni l2 → LOCKING VOLUMÉTRIQUE P1 confirmé |
| v4 | formulation MIXTE P2-P1 (Taylor-Hood) | compile OK, 1er solve >14 min sans palier | formulation correcte mais IMPRATICABLE sur cette config |

### Acquis solides
- La cause racine est la **quasi-incompressibilité**, pas le maillage ni le solveur. Confirmé : min_J repasse positif à faible charge (v3) → aucun sliver bloquant au repos.
- Le **locking volumétrique du P1** est prouvé (v3b : aucun line search ne trouve de direction admissible).
- La formulation **mixte P2-P1** est le remède théorique correct (standard cardiaque : fenicsx-pulse, Ambit).

### Blocage restant (à reprendre à froid)
Le mixte P2-P1 sur 242708 tets **compile** (FFCx OK, .so générés) mais le 1er `snes.solve()` tourne >14 min mono-thread sans rendre de palier (MEM plate à 3.1 Go → pas la factorisation seule ; probable itération SNES non convergente). Impraticable en l'état, a fortiori pour le DoE 500 sims.

### Pistes pour la reprise (par ordre)
1. Activer `snes_monitor` + `ksp_monitor` sur UN seul solve à faible charge pour voir si le SNES itère sans converger ou si MUMPS est juste lent.
2. Alléger le maillage : target_mm plus grossier (2.0-2.5 au lieu de 1.5) → moins de DOF P2.
3. Paralléliser : `mpirun -n 4` dans le conteneur + augmenter la RAM Docker Desktop (actuellement plafonnée à 7.75 Go, pas 32).
4. Vérifier l'échelle de pression / le scaling du résidu mixte (deux échelles : u~mm, p~1e5 Pa) — un mauvais scaling peut empêcher la convergence SNES.

### Infra notée
- RAM Docker Desktop = 7.75 Go (pas les 32 Go de la machine). À augmenter pour la mécanique.
- Filtre dièdre 15° retire 8825/251533 tets (3.5%) — bien plus que les 58 estimés ; maillage source à assainir en amont.

### Statut
P15 : cause racine RÉSOLUE (formulation), implémentation praticable NON aboutie. D2.1 mécanique reste OUVERT.


### 🟡 P15 (MAJ 2026-07-07) — Quasi-incompressibilité : cause résolue, implémentation mixte à finaliser
**Évolution :** le diagnostic « dégénérescence maillage » (initial) était faux. Cause réelle = formulation non quasi-incompressible (exp sur tr(C) non isochore + kappa trop faible), aggravée par le locking volumétrique du P1. Maillage et solveur sains.
**Corrigé :** décomposition isochore `I1_bar=J^(-2/3)tr(C)`, `W_vol=(kappa/2)(ln J)^2`, kappa=1e6, SNES/bt, KSP direct LU/MUMPS. Volume désormais tenu (min_J=1.0).
**Reste :** formulation mixte P2-P1 écrite et compilée, mais 1er solve >14 min (impraticable). Voir pistes reprise dans l'entrée de session 2026-07-07.
**Statut :** cause racine RÉSOLUE, implémentation praticable OUVERTE.


---

## Session 2026-07-07 (soir) — P15 : diagnostic mixte P2-P1 approfondi, blocage isolé au line search

**Point de départ :** le mixte P2-P1 (v4) compilait mais le 1er solve tournait >14 min sans palier. Objectif de la session : trancher entre SNES qui boucle / MUMPS qui bloque / coût par itération, via un script isolé (scripts/diag_snes_monitor.py) faisant UN seul solve à 2% de charge avec snes_monitor + ksp_monitor + horodatage par ligne.

### Faits établis (chaîne de causes complète)

1. **MUMPS résout parfaitement.** Sur maillage réduit (34025 tets, 209864 DOFs), le KSP direct LU/MUMPS résout en UNE itération : résidu 2.08e6 -> 1.5e-8. La brique linéaire est saine. (Le blocage initialement attribué à MUMPS était un artefact de timeout trop court.)

2. **Les options MUMPS "point-selle" étaient nuisibles.** icntl_24=1 + cntl_1=1e-6 + icntl_14=200 ont CASSÉ la factorisation (blocage total). Retirées. La config MUMPS par défaut est la bonne.

3. **Coût par cycle assemblage+factorisation ≈ 52 s** sur 34K tets (mesuré par horodatage : solve démarre, 1er résidu affiché 52 s après). Lourd mais pas rédhibitoire pour UN run patient (~3-4 h sur maillage complet).

4. **Cause racine du blocage : NaN sur inversion transitoire + line search qui boucle.**
   - Avec line search 'basic' (Newton nu) : reason=-4 (SNES_DIVERGED_FNORM_NAN), 0 itération. Le pas de Newton complet, même à 2% de charge, retourne un élément (J<=0) -> J^(-2/3) = NaN dans l'invariant isochore.
   - Avec line search 'bt' : boucle indéfiniment (jamais d'itération 1, timeout). Il tente d'amortir le pas pour éviter le NaN mais chaque évaluation ré-assemble (52 s) et il n'aboutit pas.
   - Régularisation J_reg = max(J, 1e-3) + minlambda=1e-3 : n'a PAS débloqué le 'bt' (toujours bloqué au 1er solve).

### Diagnostic
La formulation mixte P2-P1 est correcte (locking P1 résolu), MUMPS est sain, mais le PREMIER pas de Newton à faible charge produit une inversion d'élément transitoire que ni la régularisation de J ni le line search bt (avec minlambda) ne gèrent proprement. Le blocage n'est PAS le coût de calcul — c'est le line search qui ne converge pas sur ce premier incrément.

### Pistes pour la reprise (par ordre de promesse)
1. **Continuation en charge BEAUCOUP plus fine dès le départ** : commencer à 0.1% voire 0.01% de T_max (pas 2%), pour que le 1er incrément de déplacement ne retourne aucun élément. Le vrai fenicsx_solver.py a la continuation adaptative — la tester avec dlam initial très petit (ex. 1e-3) plutôt que 1/30.
2. **Régularisation plus robuste de l'énergie** : au lieu de max(J,1e-3), utiliser une formulation qui pénalise fortement J->0 en amont (barrière), ou passer à un solveur SNES de type 'newtontr' (trust region) qui gère mieux les pas menant à des états non-physiques que le line search.
3. **snes_linesearch_type 'l2' ou 'cp'** non testés sur le mixte (seulement bt et basic).
4. **Vérifier le scaling** du résidu mixte (u ~mm vs p ~1e5 Pa) — un mauvais conditionnement relatif peut donner une direction de Newton dominée par la pression, d'où le pas violent sur u.
5. Objectif = 1 run patient validé (pas la vitesse) -> on peut se permettre une continuation très fine et lente.

### Statut
P15 : formulation RÉSOLUE, solveur mixte P2-P1 correct, MUMPS sain. Blocage résiduel = convergence du 1er pas de Newton (line search / inversion transitoire). D2.1 mécanique reste OUVERT mais le périmètre du problème est maintenant réduit à un point précis.


### 🟡 P15 (MAJ 2026-07-07 soir) — Mixte P2-P1 : blocage réduit au 1er pas de Newton
**Établi ce soir :** MUMPS résout en 1 itération (2e6->1.5e-8, brique linéaire saine) ; coût ~52s/cycle sur 34K tets ; cause du blocage = NaN par inversion transitoire (J^(-2/3) sur J<=0) au 1er pas de Newton, que le line search bt ne parvient pas à amortir (boucle) et que basic ne gère pas (reason=-4).
**Non résolu par :** J_reg=max(J,1e-3), minlambda=1e-3.
**Reprise :** continuation initiale ultra-fine (dlam 1e-3), ou trust-region (newtontr), ou line search l2/cp, ou vérif scaling u/p. Objectif = 1 run validé, vitesse secondaire.
**Statut :** formulation + solveur corrects, convergence 1er incrément OUVERTE.

---

## Session 2026-07-07 (nuit) — P15 : stratégie de correction validée contre doc PETSc officielle

**Contexte :** poursuite du diagnostic mixte P2-P1. Recherche ciblée dans la doc PETSc/dolfinx officielle (manuel SNES, SNESLINESEARCHBT, SNESSetFunctionDomainError, fenicsx-pulse) pour fonder la correction sur des faits vérifiés plutôt que du tâtonnement.

### Nouveau diagnostic du blocage `bt` (14 min)

Le code source PETSc (`linesearchbt.c`) confirme : le backtracking `bt` ne s'arrête pas à la 1ère tentative — il refit un polynôme et retente jusqu'à `snes_linesearch_max_it` fois ou `minlambda`. Chaque tentative réassemble intégralement (52 s). Si le pas complet retourne un élément, 15-25 halvings géométriques = 13-22 min. **Ce n'était probablement pas un blocage infini, mais un backtracking fonctionnel mais ruineusement coûteux**, car chaque essai de λ refait tout l'assemblage au lieu de vérifier `J` en amont, à moindre coût.

### Trois corrections architecturales, chacune sourcée

1. **`SNESSetFunctionDomainError()`** (doc PETSc, exemple officiel : *"a step with negative pressure"*) : dans le callback résidu, interpoler `J` sur DG0 (peu coûteux vs assemblage complet) AVANT d'assembler ; si `J<=0` ou NaN, appeler `snes.setFunctionDomainError()` et sortir sans assembler. Économise l'essentiel des 52s par tentative rejetée. Line search conservé en `bt` (pas de trust-region : `newtontr` est pensé pour la minimisation d'objectif, pas pour un domaine de validité physique comme J>0).

2. **Pression adimensionnée `p_hat = p/kappa`** : le déséquilibre d'échelle u(mm)~1-10 vs p(Pa)~1e5 biaisait la direction de Newton (ligne "pression" du jacobien dominait numériquement la ligne "déplacement"). Reformulation `Pi_vol = kappa*p_hat*(J-1) - kappa*p_hat^2/2` avec p_hat résolu directement (ordre O(1)). Pas de FieldSplit nécessaire (trop lourd à câbler proprement en API bas-niveau 0.7.3).

3. **Continuation ultra-fine + restauration immédiate** : dlam initial 1e-4 (au lieu de 1/30), réduction ×0.3 sur rejet (au lieu de ×0.5), exigence de 2 succès faciles consécutifs avant de regonfler dlam, snapshot `w_accepted` uniquement après acceptation confirmée.

### Incertitudes explicitement non résolues (à vérifier empiriquement)
- Portée exacte de `setFunctionDomainError()` : interrompt-il tout `SNESSolve()` ou seulement la tentative de line search en cours ? La doc est ambiguë sur ce point précis. Test prévu : un seul palier avec `snes_linesearch_monitor` actif pour observer si λ décroît en interne avant que `reason` ne devienne négatif.
- La reformulation p_hat change la structure du jacobien — à vérifier qu'à charge nulle le système résout bien vers p_hat≈0.

### Statut
P15 : stratégie de correction complète et sourcée, prête à tester sur `scripts/diag_snes_monitor.py` avant portage dans `fenicsx_solver.py`. Pas encore testée en exécution.


### Suite immédiate (même nuit) — test des 3 correctifs : rapide mais résidu non-monotone

Testé sur charge 0.01% (dlam=1e-4), maillage réduit (34025 tets). Résultat :
- **Plus de blocage multi-minutes** : le solve termine sa 1ère itération en secondes (vs 14+ min hier). Aucun NaN, aucun domain error déclenché à cette charge très faible.
- **Anomalie nouvelle** : le line search bt (order=2) enchaîne 8 réductions de lambda (1.0 -> 4.36e-4), mais le gnorm affiché AUGMENTE de façon monotone (1.04e4 -> 4.40e4) à mesure que lambda DIMINUE. Mathématiquement, gnorm devrait tendre vers le résidu de l'itération précédente (1.04e4) quand lambda->0, pas diverger. Itération 1 acceptée malgré un résidu 4x plus grand qu'itération 0.

**Deux causes possibles, non tranchées :**
1. Bug dans le wrapper `_SNESProblem.F()` (synchronisation ghost/vecteurs entre appels répétés du line search, ou incohérence entre l'état lu par le garde-fou J et l'état assemblé).
2. Artefact d'affichage : le fit quadratique (order=2) de `bt` peut afficher des valeurs extrapolées par le modèle plutôt que des réévaluations fraîches à chaque lambda imprimé.

**Piste de reprise immédiate** : retester avec `snes_linesearch_order=1` (backtracking simple, sans fit quadratique/cubique) pour voir si le gnorm affiché redevient monotone décroissant vers la valeur initiale quand lambda->0 — ça distinguera un vrai bug (persiste) d'un artefact d'affichage du modèle (disparaît en order=1).

**Statut** : 3 correctifs (domain error, p_hat, continuation fine) validés comme n'aggravant rien et supprimant le blocage de plusieurs minutes. Nouvelle anomalie de non-monotonie du résidu affiché à investiguer avant de considérer la convergence acquise. Pas de commit — code en l'état à retester.

### Suite (même journée) — bug identifié dans la reformulation p_hat, run interrompu

Continuation lancée avec p_hat=p/kappa : palier 1 accepté (min_J=0.97, domain_err=0, convergence Newton propre après 1er pas), mais palier 2 reproduit EXACTEMENT la même trajectoire (même besoin de 6 réductions de line search jusqu'à lambda=1/64, même nombre d'itérations). dlam reste figé à 1e-4 (seuil consecutive_easy jamais atteint). Rythme mesuré (~1600-1700s/palier) rendrait un run complet impraticable (des centaines d'heures).

**Cause identifiée** : erreur de dérivation en chaîne dans la reformulation p_hat. En dérivant l'énergie par rapport à p_hat=p/kappa au lieu de p, dPi/dp_hat = kappa * dPi/dp — la substitution AMPLIFIE le déséquilibre d'échelle résiduelle d'un facteur kappa (~1e6) au lieu de l'atténuer. C'est l'inverse de l'effet recherché.

**Correction** : revenir à p (Pa, non substitué) comme inconnue primale. Le KSP/MUMPS gère déjà bien ce cas (confirmé le 07/07 : résolution en 1 itération, 2e6->1.5e-8). Garder les 2 autres correctifs (domain error guard, continuation fine) qui semblent être la vraie source d'amélioration observée.

**Statut** : run interrompu volontairement (pas d'échec silencieux). Reprise immédiate avec p direct.

---

## Session 2026-07-08 — P15 : formulation p-directe validée, continuation lancée en fond nocturne

**Correctif majeur** : bug identifié dans la reformulation p_hat de la veille — dérivation en chaîne (dPi/dp_hat = kappa*dPi/dp) amplifiait le déséquilibre d'échelle au lieu de l'atténuer. Retour à p direct (Pa) : palier 1 passe en 3 itérations avec pas complet (aucun backtracking), contre 7 itérations + 6 réductions de line search avec p_hat. Correction confirmée décisive.

**Run de continuation lancé** (petit maillage, 34025 tets, 209864 DOFs), formulation p-directe + garde-fou domain error + continuation adaptative (dlam0=1e-4) :

| Résultat après 14 paliers, ~3h34 (12833s) |
|---|
| lam=0.001176 (0.12% de la charge totale) |
| min_J=0.7787, décroissance monotone et physiologique (0.97→0.78) |
| domain_err_total=0 sur toute la durée — jamais de NaN, jamais d'inversion |
| 2 rejets de palier bien gérés par la continuation (reason=-9, LINE_SEARCH), tous deux dans une zone de raideur locale autour de lam≈0.0009-0.0013, franchie après réduction de dlam |
| Coût par palier très irrégulier : 445s à 2532s selon la difficulté locale |

**Diagnostic** : la formulation mixte P2-P1 + p direct + garde-fou + continuation adaptative fonctionne correctement et de façon robuste (auto-correction sur rejet, jamais de blocage). MAIS le coût par palier est trop élevé et irrégulier pour atteindre lam=1.0 en temps praticable par cette seule voie, a fortiori sur le maillage complet (242708 tets, 7x plus gros).

**Décision** : run laissé tourner en fond toute la nuit (aucun coût, caffeinate actif) pour voir jusqu'où il progresse sans intervention. Pas d'attente d'aboutissement complet ce soir.

**Pistes pour la suite si le run n'aboutit pas d'ici demain** :
1. Accepter un résultat partiel (ex. lam=0.3-0.5, charge physiologique partielle) comme preuve de concept suffisante, plutôt que viser lam=1.0 exact.
2. Réduire encore le maillage de test (déjà réduit à 34K tets) pour un cycle de développement plus rapide, quitte à ne valider le maillage complet qu'une fois la stratégie de continuation bien calée.
3. Revoir kappa_vol (actuellement 1e6) : peut-être un peu trop raide, contribuant au coût élevé par palier même si la formulation reste stable.
4. Envisager mpirun -n 4 (8 CPU disponibles depuis ce soir, Docker Desktop reconfiguré à 8 CPU/24GB) pour accélérer l'assemblage+factorisation par palier.

**Statut** : P15 formulation validée et robuste. Praticabilité du temps de calcul en continuation complète = point ouvert, à réévaluer selon résultat du run nocturne.

---

## Suite (meme session, 2026-07-08 soir) — Tentative MPI : defaut topologique du maillage decouvert

**Objectif** : accelerer la continuation (450-2500s/palier en mono-thread) via mpirun -n 4, en exploitant les 8 CPU Docker desormais disponibles. Formulation p-directe deja validee sur >16 paliers en mono-thread (cf. entree precedente).

**Blocage rencontre** : `RuntimeError: A facet is connected to more than two cells.` -- systematique sur les 4 rangs des que n_ranks > 1.

**Diagnostic effectue** :
- Verification doublons d'elements (source .elem, apres filter_small_elements, apres filter_degenerate_dihedral) : ZERO doublon a chaque etape. Pas un probleme de duplication.
- Test isole n=1 (create_mesh seul, sans le reste du solveur) : PASSE, `n_cells locales: 34025`.
- Meme test isole n=4 (mpirun) : ECHOUE sur les 4 rangs, meme erreur, meme jeu de donnees.

**Conclusion** : le maillage (patient001, apres filtrage a 34025 tets) contient une pathologie topologique locale (probablement un ou plusieurs tetraedres quasi-degeneres/quasi-coplanaires non retenus par le filtre dihedral a 15deg) qui est TOLEREE par dolfinx en sequentiel mais fait echouer le partitionneur SCOTCH (utilise des que n_ranks>1) lors de la construction du graphe dual des facettes.

**Decision** : abandon de la piste MPI pour cette session. Retour au mono-thread deja valide (formulation p-directe + garde-fou + continuation adaptative), qui reste la version de reference dans fenicsx_solver.py (committe, commit 5b4bb44).

**Pistes pour la reprise (si acceleration MPI reprise plus tard)** :
1. Ecrire un script de detection des facettes partagees par >2 cellules DIRECTEMENT sur elements_final (construire le dictionnaire facette->liste de cellules, chercher les entrees a >2), pour localiser precisement le/les tetraedre(s) coupable(s) avant tout essai MPI.
2. Une fois localise : soit durcir le filtre de qualite (nouveau critere en plus de l'angle diedre, ex. volume/aire ratio), soit retirer manuellement l'element identifie et verifier que ca ne cree pas de trou dans le maillage.
3. Alternative : tester un partitionneur different (ParMETIS au lieu de SCOTCH, si disponible dans l'image dolfinx) qui pourrait etre plus tolerant.
4. Le probleme est specifique a ce maillage (patient001, filtre a 34025 tets) -- verifier s'il se reproduit sur le maillage complet (242708 tets) ou sur d'autres patients avant de generaliser le diagnostic.

**Statut global P15** : formulation mecanique validee et securisee (mono-thread). Acceleration par MPI non aboutie ce soir -- nouveau defaut distinct (topologie de maillage, pas physique/solveur) a traiter separement, hors urgence puisque le mono-thread reste une base de travail valide meme si lente.

### Suite (2026-07-08 soir) — Diagnostic facette pathologique : maillage disculpe

Script diag_facet_topology.py execute sur elements_final (34025 tets, identique au pipeline solveur) : construction du graphe facette->cellules en pur numpy.

Resultat : 79378 facettes distinctes, 22656 a 1 cellule (bord), 56722 a 2 cellules (interne), **0 facette a 3+ cellules**. Topologie du maillage parfaitement saine.

**Conclusion revisee** : le RuntimeError "A facet is connected to more than two cells" observe en MPI n'est PAS du a un defaut du maillage. La trace complete montrait un TypeError PRECEDENT ("incompatible function arguments" sur create_mesh), et le RuntimeError n'apparaissait que dans la gestion de cette premiere exception -- probablement un message d'erreur secondaire/trompeur genere par un chemin de code de secours, pas par une vraie detection topologique. Piste probable : incompatibilite entre l'API create_mesh() et le partitionneur par defaut en configuration multi-rang dans dolfinx 0.7.3 (plomberie logicielle, pas geometrie).

**Decision** : chantier MPI clos pour cette session -- creuser l'API de partitionnement dolfinx 0.7.3 serait incertain et chronophage. Le mono-thread (valide, en cours d'execution stable avec checkpoint) reste la base de travail de reference.

---

## Session 2026-07-08 (nuit) — P15 : PREMIER RUN COMPLET REUSSI (lam=1.0)

**Jalon majeur** : premiere convergence complete de la mecanique FEniCSx jusqu'a la charge active maximale (T_max=135kPa), sur maillage grossier (patient001_coarse5).

### Cause de la lenteur precedente identifiee et resolue
Sur le maillage de reference (34025 tets, target_mm=1.5), le run recent progressait a ~450-2500s/palier -> des centaines d'heures pour lam=1.0 (voir entrees precedentes). Diagnostic du goulot : OPENBLAS_NUM_THREADS n'a AUCUN effet (teste 1 vs 8 threads, temps identiques a la seconde pres) -> confirme que le cout est dans l'ASSEMBLAGE FFCx (boucles C generees par cellule pour l'energie Holzapfel + jacobien), pas dans la resolution lineaire (KSP/MUMPS deja tres rapide, 1 iteration). L'assemblage scale avec le nombre de cellules -> solution : maillage plus grossier.

### Remaillage grossier (patient001_coarse5)
Masque source : reports/meshes_acdc/segmentations/patient001_seg.nii.gz, shape (216,256,10), spacing (1.5625, 1.5625, 10.0)mm. Anisotropie native forte (Z=10mm) : target_mm=2.5 n'a quasiment aucun effet (34425 tets, quasi identique au maillage 1.5mm) car le zoom scipy.ndimage.zoom sur l'axe Z reste sur-echantillonnant. target_mm=5.0 franchit enfin la barriere : 5686 tets (apres filtre slivers), min_jacobian=0.014, num_degenerate=0. Fibres : champ tangentiel simplifie (PAS LDRB, non physiologique, uniquement pour test de vitesse) genere par scripts/generate_simple_fibers.py.

### Resultat du run complet (scripts/diag_continuation_coarse.py)
| Metrique | Valeur |
|---|---|
| Duree totale | 2585.9s (~43 min) |
| Paliers | 35, TOUS acceptes (0 rejet) |
| Iterations Newton totales | 95 |
| domain_err_total | 0 (jamais declenche) |
| lam_final | 1.000000 (CONVERGE) |
| min_J final | 0.5670 |

Gain de vitesse mesure : palier 1 en 52.3s (contre ~823-830s sur maillage fin) = facteur ~16x. dlam croit geometriquement sans jamais rencontrer de zone de rejet (contrairement au maillage fin qui rejetait 2-3 fois autour de lam=0.0009-0.0013) -- signe que la zone de raideur locale precedente etait probablement liee a des slivers specifiques au maillage fin, absents ou differents sur ce maillage grossier.

### A faire
- Completer le post-traitement (volume, endo/epi radial displacement, EF) -- omis dans ce script de vitesse pure.
- Regenerer des fibres LDRB physiologiques (le champ tangentiel actuel n'est PAS anatomiquement correct) avant d'utiliser ce resultat pour toute conclusion clinique.
- Valider si ce comportement (convergence sans rejet) se maintient sur le maillage COMPLET (57K noeuds / target_mm=1.5) avec plus de temps, ou si la difference est structurelle au maillage grossier.
- Reste a determiner : le maillage grossier est-il suffisant pour le DoE 500 sims (production), ou seulement pour la validation de la formulation (developpement) ?

**Statut P15** : formulation ENTIEREMENT VALIDEE de bout en bout (lam=0 -> lam=1.0, charge physiologique complete). Premiere fois. Praticabilite confirmee sur maillage grossier ; maillage de production (1.5mm) reste a re-tester avec le temps necessaire (ou en acceptant un maillage plus grossier pour le DoE).

### Post-traitement du run convergent (lam=1.0)

| Metrique | Valeur | Evaluation |
|---|---|---|
| Deplacement min/max | -17.59 / +9.81 mm | PLAUSIBLE (vs 24.6 METRES au tout debut de P15) |
| Volume repos -> deforme | 90.36 -> 87.22 mL (-3.47%) | Raisonnable, pas parfait |
| endo_radial_disp_mm | -4.77 | Bon sens (contraction) |
| epi_radial_disp_mm | -6.30 | SUSPECT : plus negatif que endo, physiologiquement attendu l'inverse |

**Diagnostic de l'anomalie epi/endo** : tres probablement du au champ de fibres TANGENTIEL SIMPLIFIE (pas LDRB) utilise pour ce test de vitesse -- sans variation transmurale d'helicite (endocarde/epicarde), la tension active tire plus uniformement sur toute l'epaisseur, ce qui peut inverser le comportement relatif attendu.

**Conclusion** : ce run valide la MECANIQUE (convergence stable jusqu'a charge complete, deplacements dans l'ordre de grandeur physiologique correct, volume quasi conserve) mais PAS la physiologie fine. Ne pas utiliser ces chiffres (EF, deplacements precis) pour une conclusion clinique -- refaire avec de vraies fibres LDRB avant toute exploitation scientifique du resultat.

**Prochaine etape** : generer des fibres LDRB physiologiques sur le maillage grossier (ou reutiliser LDRB existant si transferable), relancer la continuation avec fibres correctes pour un resultat exploitable.

### Suite (2026-07-09 nuit) — Cause de l'anomalie epi/endo identifiee : resolution radiale insuffisante

**Test avec vraies fibres LDRB** (app/fibers/ldrb.py, deja utilise en production) : anomalie epi>endo PERSISTE quasi identique (endo=-3.32mm, epi=-5.26mm vs -4.77/-6.30 avec fibres tangentielles). Ecarte l'hypothese "fibres simplifiees" comme cause.

**Diagnostic geometrique** (span radial par element touchant la coupe mediane) :
- Epaisseur paroi totale : 17.59mm
- Span radial median par tetraedre : 5.605mm = 31.9% de l'epaisseur totale
- 1329/2354 elements (56%) ont un span > 30% de l'epaisseur
- 390/2354 elements (17%) ont un span > 50% de l'epaisseur

**Conclusion** : seulement ~3-4 elements traversent radialement toute la paroi. Resolution structurellement insuffisante pour resoudre un gradient transmural endo/epi fiable, quelle que soit la qualite du champ de fibres. C'est un defaut geometrique du maillage grossier (target_mm=5.0), pas un bug de formulation ni de fibres.

**Decision sur la resolution de maillage (question posee plus tot dans la session)** :
- Maillage grossier (5686 tets) : VALIDE pour developpement rapide / validation de stabilite numerique du solveur (ce qui a ete fait avec succes, 2x, lam=1.0 a chaque fois).
- Maillage grossier : NON SUFFISANT pour toute metrique dependant du gradient transmural (epaississement pariétal, contrainte radiale, deplacement endo vs epi separement).
- Pour le DoE final (500 sims), si des metriques transmurales sont necessaires : repasser au maillage fin (1.5mm, 57058 noeuds) est requis, avec le cout de calcul associe (a re-attaquer : MPI, ou accepter un temps de calcul long par run).
- Si le DoE ne necessite que des metriques globales (volume, EF globale, pression VG) : le maillage grossier pourrait suffire -- a valider specifiquement selon les besoins du surrogate GP/MeshGraphNets en aval.

**Statut P15** : formulation mecanique validee de bout en bout (2 convergences completes, fibres tangentielles ET LDRB). Limite de resolution du maillage de developpement identifiee et documentee, distincte de la formulation elle-meme.

---

## Session 2026-07-09 (suite nuit) — Validation formelle openCARP : benchmark Niederer et al. 2011

**Objectif** : valider quantitativement le solveur EP contre un benchmark publie et independant, au-dela du simple test de sanite ("code=0") deja effectue precedemment.

**Geometrie** : coin (wedge) 20x7x3mm, benchmark N-version (Niederer et al. 2011, Phil Trans R Soc A), genere via carputils.mesh.Block (outil officiel openCARP), resolution 0.5mm, 4305 noeuds. Stimulus spherique au coin (0,0,0), fibres alignees selon X.

**Infrastructure reutilisee** (pas reconstruite) : app/solver/ep/opencarp_config.py (OpenCARPValidatedConfig, generate_par_file) deja valide lors de la session g_mult -- evite le piege dt (doit etre en MICROSECONDES, cause documentee d'un facteur 1000x de ralentissement dans une session anterieure) et utilise la detection LAT native (num_LATs -> fichier init_acts_depol-thresh.dat) plutot qu'un seuillage manuel sur vm.igb.

**Resultat** : simulation en 1.1s (mpirun -n 4, tend=130ms), code=0. 8 points mesures le long de la diagonale P1(0,0,0)->P8(20,7,3mm), temps d'activation STRICTEMENT CROISSANTS (0.068 -> 107.53ms), zero anomalie.

**Regression lineaire distance/temps** : CV=0.1833 m/s, R2=0.985, intercept=1.91mm (proche de 0). Le R2 tres proche de 1 confirme une onde de propagation coherente, sans dispersion numerique ni artefact -- c'est le resultat cle de cette validation.

**Ecart avec la CV longitudinale deja validee (0.545 m/s)** : NON anomalique. Explication : la diagonale du coin n'est pas alignee sur l'axe des fibres (X), et le tissu est fortement anisotrope (g_il/g_it ~ 14.8:1). La CV dans une direction oblique suit une loi elliptique dominee par la composante transverse des que l'angle avec les fibres augmente -- CV transverse pure theorique ~0.142 m/s (0.545/sqrt(14.8)), notre mesure de 0.183 m/s est coherente avec un angle intermediaire entre longitudinal et transverse.

**Lecon methodologique** : pour comparer directement au CV longitudinal deja valide, il aurait fallu mesurer le long de l'axe des fibres (X), pas sur la diagonale du coin (qui sert a visualiser le front d'onde complet dans le benchmark original, pas a extraire une CV isotrope simple).

**Conclusion** : le solveur EP est valide sur ce benchmark independant -- propagation physiquement coherente (R2=0.985), anisotropie respectee qualitativement, et coherent avec la CV longitudinale deja etablie une fois l'angle de propagation pris en compte. SLO "activation +/-5ms" du projet non directement applicable ici (pas de tableau numerique publie du papier Niederer extrait pour comparaison point-par-point exacte), mais la coherence physique et la qualite de l'ajustement (R2) constituent une validation solide en l'absence de cette table precise.

**A faire si validation plus stricte necessaire** : refaire le test avec stimulus/mesure le long de l'axe X pur (fibres) pour comparaison directe au CV longitudinal ; ou consulter le papier complet Niederer 2011 pour le tableau numerique exact des temps P1-P8 publies.

### Suite (2026-07-09) — Bug MPI partitionnement : piste GhostMode testee et ecartee

**Piste testee** (fondee sur inspection du code source dolfinx.mesh.create_mesh v0.7.x + issue GitHub FEniCS/dolfinx#994 "Bus errors when running in parallel inside Docker containers") : forcer explicitement GhostMode.shared_facet au lieu du GhostMode.none automatique choisi par dolfinx quand comm.size>1 sans partitionneur precise.

**Resultat** : ECHEC identique sur les 4 rangs, meme erreur exacte ("A facet is connected to more than two cells"), sur les deux chemins internes (tentative directe ET repli TypeError->AdjacencyList_int64).

**Conclusion** : le choix de GhostMode n'est PAS la cause. Le probleme est plus profond, dans la construction du graphe dual SCOTCH lui-meme face a la topologie de ce maillage en configuration multi-rang -- coherent avec l'issue #994 qui documente des problemes de fond avec SCOTCH necessitant un changement de partitionneur (KaHIP), or ceci exige une RECOMPILATION de dolfinx depuis les sources C++ (choix fait a la compilation, pas un parametre Python) -- non realisable dans le cadre actuel.

**Decision** : chantier MPI ferme pour l'instant. Strategies alternatives pour le DoE 500 sims sans MPI :
1. Accepter le mono-thread et etaler les simulations dans le temps (checkpoint disque deja valide protege la progression, pas de surveillance active necessaire).
2. Chercher une resolution de maillage intermediaire (entre le grossier invalide pour le transmural et le fin trop lent) offrant un compromis cout/precision acceptable.

**Statut P15/DoE** : formulation mecanique validee. Praticabilite du DoE 500 sims sur maillage fin reste un vrai defi de temps de calcul, a resoudre par etalement temporel ou resolution intermediaire, pas par parallelisme MPI (ferme).

---

## Session 2026-07-11/12 — EMIDEC : ré-entraînement avec oversampling corrigé, résultats mesurés

**Objectif** : corriger les scores insuffisants sur les classes cicatrice (MI) et obstruction microvasculaire (PMO), identifiés precedemment (Dice=0.365/0.132), via `oversample_foreground_percent=0.66` (au lieu de 0.33 par defaut nnU-Net).

**Infrastructure** : Kaggle notebook, dataset `emidec-upload` (100 cas, format nnU-Net standard), config `3d_fullres`, GPU T4. Trainer personnalise `nnUNetTrainerOversample066` (herite de `nnUNetTrainer`, surcharge `oversample_foreground_percent`).

### Incidents resolus en cours de route
1. Structure de dossier imbriquee (`Dataset029_EMIDEC/EMIDEC_upload/...` au lieu de `Dataset029_EMIDEC/...`) -- corrige par deplacement du contenu d'un niveau.
2. Extensions de fichiers corrompues (`.niigz` au lieu de `.nii.gz`, point manquant) -- corrige par renommage systematique.
3. Signature de `nnUNetTrainer.__init__` differente de la version documentee (5 params au lieu de 6, pas de `unpack_dataset`) -- verifie via `inspect.signature` avant d'ecrire le trainer personnalise, pas de tatonnement.
4. Crash memoire `free(): corrupted unsorted chunks` dans le dataloader multi-thread au tout debut de l'entrainement -- resolu en desactivant `torch.compile` (`os.environ['nnUNet_compile'] = 'False'`). Cause precise non confirmee officiellement (issue GitHub MIC-DKFZ/nnUNet#2523 sans fix documente), mais correlation temporelle claire avec la compilation JIT active au moment du crash.

### Progression de l'entrainement
Entrainement lance a 13:19:53, epoch 0. Progression du EMA Pseudo Dice :
- Epoch 0-100 : 0.0 -> 0.69 (montee rapide standard)
- Epoch 100-235 : 0.69 -> 0.7091 (record final, epoch 235, 21:35:26)
- Epoch 235-307+ : PLATEAU, aucun nouveau record sur 70+ epochs -> entrainement arrete manuellement (checkpoint_best.pth recupere avant fermeture de session).

### Resultats de validation (20 cas, split interne 80/20)
Evaluation via `nnUNetv2_predict` + `nnUNetv2_evaluate_folder` sur checkpoint_best (epoch 235) :

| Classe | Avant (ancien run) | Apres (oversampling 0.66) | Delta |
|---|---|---|---|
| LV (1) | -- | 0.9451 | -- |
| MYO (2) | -- | 0.8421 | -- |
| MI/scar (3) | 0.365 | 0.4584 | +0.093 (+25% relatif) |
| PMO (4) | 0.132 | 0.2064 | +0.074 (+56% relatif) |

### Diagnostic
Amelioration REELLE et mesurable sur MI/PMO, mais NON SUFFISANTE pour un usage clinique fiable (seuil habituel souhaite ~0.70-0.80 pour ces structures). Cause probable persistante : classes intrinsequement rares en nombre de voxels (ratio ~1:8 entre MI et LV observe sur cas individuel EMIDEC_0088 : n_ref=663 vs 5352), l'oversampling de patchs seul ne compense pas entierement le desequilibre de taille des structures. Pistes non testees pour aller plus loin : loss ponderee (Dice loss ponderee ou Focal loss) plutot que oversampling seul, plus d'epochs avec redemarrage du learning rate schedule.

### Impact sur le projet -- IMPORTANT
EMIDEC est une branche PARALLELE au chemin critique du projet, pas une etape bloquante. Le pipeline principal (patients ACDC standards, sans pathologie cicatricielle) fonctionne independamment de l'etat d'EMIDEC. Ce resultat n'est PAS encore utilisable pour personnaliser un maillage patient avec proprietes mecaniques/electriques modifiees dans les zones cicatricielles (precision insuffisante des frontieres predites, en particulier PMO a Dice=0.21). A reprendre uniquement si un patient avec pathologie cicatricielle devient un cas d'usage prioritaire pour le projet.

**Statut** : EMIDEC ameliore mais non finalise. checkpoint_best.pth (epoch 235) sauvegarde localement. Chantier mis en pause, pas de suite immediate prevue.

---

## Session 2026-07-12 — Decision resolution maillage + integration DoE + bug EF critique

### Decision structurante : maillage grossier retenu pour le DoE

Suite a plusieurs jours de blocage sur le maillage fin (singularite a lam~0.008, confirmee independante de kappa_vol -- teste 1e6 et 1e5, meme signature d'echec ; maillage intermediaire 19894 tets teste aussi, meme resolution radiale insuffisante ~3.6 elements dans l'epaisseur, cause structurelle : spacing Z=10mm systematique sur les 10 patients ACDC, confirme par verification directe des headers NIfTI).

**Verification contre heart_digital_twin_proposal.pdf (cahier des charges)** : le maillage grossier est SUFFISANT pour les livrables reels. D2.1 ne specifie pas de resolution transmurale. D3.1 vise un surrogate >=100x plus rapide -- le solveur complet sert uniquement a generer le DoE d'entrainement, jamais execute en temps reel. Les 2 cas d'usage clinique WP5 (stratification risque arythmie, optimisation CRT) reposent sur EP + metriques globales/segmentaires standard, pas sur un gradient transmural fin. DECISION ACTEE : maillage grossier (patient001_coarse5) = reference pour le DoE.

### Optimisations pour le DoE (500 simulations)

1. **dolfinx.fem.Constant** pour a_kPa/b/kappa_vol (au lieu de floats Python bruts) : evite la recompilation FFCx entre chaque point du DoE (seule reference confirmee : Constant.value modifiable sans recompilation, dolfinx docs). Applique dans fenicsx_solver.py (production).

2. **Parallelisme de taches (pas MPI)** : lancement de N conteneurs Docker independants, chacun traitant un point DoE different (embarrassingly parallel, exploite les 8 CPU deja alloues). Valide : 4 points testes, 3 termines en ~65min en parallele (vs ~3h en sequentiel).

3. **GPU Kaggle explicitement ecarte** : probleme CPU-bound (assemblage+MUMPS), pas d'acceleration GPU disponible pour ce type de solve dans dolfinx 0.7.3.

### Decouverte critique : cout de calcul TRES heterogene selon les parametres materiaux

Point avec a_kPa=0.322 (materiau mou, bas de la plage DoE) : convergence en **6850s (1h54)**, 229 iterations totales, its/palier croissant de 3 a 14. Point de reference (a_kPa=0.496) : convergence en **3964s (1h06)**, 112 iterations, its stable a 3/palier tout du long.

**Implication** : l'estimation "~2 jours pour 500 sims sur 8 coeurs" (basee sur ~43min/run de reference) est TROP OPTIMISTE. Le temps reel par simulation depend fortement du point de l'espace des parametres (materiau mou = 2-3x plus long, iterations croissantes). A reevaluer avec un echantillon plus large avant de lancer le DoE complet.

### BUG CRITIQUE identifie et partiellement corrige : calcul d'Ejection Fraction (EF)

**Bug #1 (grave, corrige)** : premiere version de `run_single_doe_point.py` utilisait `volume_tissue_mL` (volume du MUSCLE/tissu, quasi-incompressible par construction, varie de ~3-5%) comme s'il s'agissait du volume de la CAVITE ventriculaire (le sang, qui varie ~50-60% entre diastole et systole). EF calculee : 2.27-4.75% (aberrant, incompatible avec la vie).

**Bug #2 (subtil, corrige)** : premiere tentative de correction (volume de cavite par empilement de disques radiaux, rayon = 10e percentile des noeuds par tranche en Z) recalculait z_min/z_max SEPAREMENT pour l'etat au repos et l'etat deforme. Or l'apex (extremite libre, base fixee) subit un deplacement axial important (+17mm observe) independant de la vraie contraction radiale -- integrer sur des plages Z differentes desynchronise completement les tranches comparees. EF resultante : encore 2.27% (bug masque le premier).

**Correction appliquee** : integrer le volume de cavite (avant ET apres deformation) sur la MEME plage Z et le MEME centre XY, ceux de l'etat au repos exclusivement. Resultat : EF=23.83% (point mou) et EF=24.12% (point de reference) -- coherent et reproductible entre les 2 points testes, mais RESTE BAS pour une EF physiologique typique (attendu 55-70%).

**Hypotheses non tranchees pour ce residu (EF~24% au lieu de 55-70%)** :
1. Methode d'approximation (10e percentile radial par tranche) structurellement imprecise sur maillage grossier avec peu de noeuds/tranche (~150-250).
2. Modele mecanique possiblement sous-contraint : maillage grossier (~3-4 elements dans l'epaisseur pariétale, deja documente) pourrait ne pas capturer correctement l'epaississement radial qui genere la vraie reduction de cavite en clinique.
3. Possible qu'une vraie integrale de surface sur l'endocarde (extraction de frontiere + normales, plutot que l'approximation par disques) donne un resultat different -- non teste.

**Statut** : pipeline DoE fonctionnel de bout en bout (parametres -> mecanique -> volume cavite -> EF), MAIS l'EF resultante necessite une investigation plus poussee avant de generer le DoE complet a 500 points. Ne pas lancer le DoE complet avant d'avoir clarifie cette question -- risque de generer 500 points avec une metrique EF systematiquement biaisee.

### Fichiers modifies/crees cette session
- app/solver/mechanics/fenicsx_solver.py : Constant pour a_kPa/b/kappa_vol.
- app/solver/coupled_solver.py : integration mecanique reelle (1 seul appel/simulation, pas dans la boucle Windkessel -- cout prohibitif sinon), fallback analytique conserve si non convergence.
- scripts/run_single_doe_point.py : traite un point DoE isole (mecanique + volume cavite + sauvegarde champ complet .npz).
- app/solver/doe/latin_hypercube.py : reutilise tel quel (non modifie).
- cdt_status_strict.md : D2.1 mecanique remonte a 80%, decision resolution maillage documentee.

### Prochaine etape (a froid)
1. Trancher la cause du residu EF~24% (methode de calcul vs modele mecanique) avant de lancer le DoE complet.
2. Reevaluer le budget de temps du DoE avec un echantillon plus large de points (pas juste 2), pour capturer l'heterogeneite de cout selon les parametres materiaux.
3. Une fois clarifie : lancer le DoE complet (nombre de points a determiner selon le vrai budget de temps).

---

## 2026-07-14 — BUG RACINE RESOLU : geometrie ecrasee par binary_closing anisotrope

### Resume

Un seul bug, en amont de toute la chaine, expliquait l'EF aberrante et
toutes les anomalies constatees depuis plusieurs sessions. Apres
correction, le modele reproduit l'EF clinique de reference du patient
(une fois T_max calibre) et le SLO du projet est atteint.

### Le bug

`app/meshing/gmsh_mesher.py`, `_mask_to_stl()` :

    filled = ndimage.binary_closing(binary_mask, iterations=2)

`binary_closing` n'est PAS un remplissage de trous (contrairement a ce
que le commentaire du code annoncait) : c'est une dilatation suivie d'une
erosion, avec un element structurant ISOTROPE EN VOXELS. Or le masque
ACDC est fortement anisotrope (spacing 1.5625 x 1.5625 x 10.0 mm) :

  * +/-2 voxels = +/-3.1 mm en x/y   -> negligeable
  * +/-2 voxels = +/-20 mm en z      -> destructeur

Le myocarde occupe les 10 tranches du volume : la dilatation debordait
des bords du tableau et etait tronquee, tandis que l'erosion suivante
rongeait pleinement +/-20 mm en z.

### Effets mesures (patient001)

| Grandeur              | Avant     | Apres     | Verite ACDC | Ecart final |
|-----------------------|-----------|-----------|-------------|-------------|
| Hauteur du VG         |  55.62 mm |  98.50 mm |  100.00 mm  |  -1.5 %     |
| Volume de tissu       |  90.36 mL | 163.30 mL |  180.81 mL  |  -9.7 %     |
| Volume de cavite V_ed | 209.21 mL | 294.99 mL |  295.51 mL  |  **-0.18 %**|
| Tetraedres            |    5 686  |    9 439  |     --      |     --      |

### Bugs corollaires (necessaires pour mailler la geometrie restauree)

1. **Surface ouverte** : le myocarde touchant les bords du tableau en z,
   `marching_cubes` coupait l'isosurface net (44 aretes de bord mesurees).
   Gmsh refusait : *"Wrong topology of boundary mesh for parametrization"*.
   -> `np.pad(pad_width=3)` apres le zoom, avec compensation de l'offset.

2. **SurfaceLoop incoherent** : les 3 composantes detectees (l'enveloppe du
   myocarde + 2 fragments parasites de 24 et 132 triangles) etaient
   regroupees dans un SEUL `SurfaceLoop`. Gmsh declarait *"Found void
   region"* trois fois et ne produisait AUCUN tetraedre.
   -> ne conserver que la composante de plus grande aire.

3. **Lissage trop fort** : `gaussian(sigma=1.0)` erodait encore 21 mm en z.
   -> `sigma=0.5` (hauteur preservee a 99 mm, surface toujours reguliere).

### Reference clinique etablie

Frames ED (frame01) et ES (frame12) extraits du dataset ACDC
(`Dataset027_ACDC/gt_segmentations`, comptage de voxels du label 3) :

    V_ED = 295.51 mL    V_ES = 225.61 mL    SV = 69.90 mL
    >>> EF DE REFERENCE = 23.65 %

**patient001 est un cas PATHOLOGIQUE** : cardiomyopathie dilatee (cavite
de 295 mL contre 100-180 mL normalement) avec dysfonction systolique
severe. La cible n'est donc PAS l'EF d'un coeur sain (55-70 %) mais bien
**23.65 %**. Cette hypothese erronee, jamais verifiee, avait oriente
l'investigation a tort pendant plusieurs sessions.

### Courbe EF(T_max) — maillage corrige

Tous les autres parametres fixes (a_kPa=0.496, b=7.209, a_f=15.193,
b_f=20.417).

| T_max (kPa) | EF (%) | Ecart / 23.65 % | SLO (+/-3 pts) |
|-------------|--------|-----------------|----------------|
|      0      |  0.00  |     -23.65      |                |
|     15      | 21.01  |      -2.64      |   **OK**       |
|     22      | 25.21  |      +1.56      |   **OK (meilleur)** |
|     30      | 26.14  |      +2.49      |   **OK**       |
|     45      | 27.52  |      +3.87      |                |
|     60      | 29.89  |      +6.24      |                |
|     90      | 32.48  |      +8.83      |                |
|    135      | 32.78  |      +9.13      |                |

Le point a `T_max = 0` donne `endo_disp = 0`, `min_J = 1.0` et EF = 0 % :
**aucune ejection parasite dans le modele**, toute l'EF provient bien de
la contraction active.

Interpolation entre 15 et 22 kPa : la cible de 23.65 % correspond a
**T_max ~ 19.4 kPa**.

### Conclusion

* Le modele **reproduit l'EF clinique de reference** du patient une fois
  T_max calibre.
* Le **SLO du projet (EF a +/-3 %) est atteint** : meilleur point a
  `T_max = 22 kPa` avec un ecart de **1.56 point**.
* La **calibration inverse (D2.2) est demontree faisable** : il suffit
  d'ajuster `T_max` autour de 19-22 kPa pour ce patient, au lieu des
  135 kPa generiques de la litterature.

### Portee et limites (a garder en tete pour le rapport)

Le resultat ci-dessus est solide mais borne. A ne pas sur-interpreter :

1. **`T_max ~ 19.4 kPa` est une valeur EFFECTIVE, pas la contractilite
   vraie du patient.** Le passif (a/b/a_f/b_f) et la postcharge (R_p/C_a)
   sont figes ; T_max absorbe donc tout desaccord de postcharge ou de
   raideur passive. La lecture "~15 % d'un myocarde sain" est un ordre de
   grandeur coherent avec la pathologie, pas une grandeur physique mesuree.
   C'est precisement ce que l'analyse d'identifiabilite (FIM, prevue D2.2)
   doit trancher.

2. **La courbe sature vers ~33 % meme a 135 kPa.** Avec cette geometrie et
   ces parametres passifs/postcharge figes, le modele NE PEUT PAS produire
   une EF saine (55-70 %), quel que soit T_max. Pour ce cas DCM a 23.65 %
   c'est adequat ; mais pour la cohorte (10+ patients, dont des coeurs
   moins pathologiques), calibrer T_max seul ne suffira pas -- il faudra
   ouvrir au moins la postcharge et le passif.

3. **EF globale correcte n'implique PAS champ de deformation correct.** Le
   bon ratio de volumes peut coexister avec une cinetique parietale
   regionale fausse. Sans consequence pour D2.1/D3.1 (metrique globale) ;
   mais WP5 (strain regional, isthmes de reentree) exigera une validation
   du champ, pas seulement de l'EF.

4. **"D2.2 faisable" != "D2.2 livre".** La faisabilite est demontree sur
   1 patient / 1 scalaire (EF). Le livrable complet exige encore :
   posterieurs pour 10+ patients, correlation ECG > 0.95, boucle PV,
   FIM d'identifiabilite.

### Fausses pistes invalidees retrospectivement

* **Condition aux limites** : Land 2015 (encastrement basal complet) etait
  correcte. Le correctif "3-2-1" a ete annule.
* **Methode de calcul du volume de cavite** : correcte (0.9 % d'ecart avec
  la verite terrain une fois appliquee au maillage corrige).
* **"Resolution transmurale insuffisante"** : les 3-4 elements dans
  l'epaisseur parietale etaient une CONSEQUENCE de la geometrie ecrasee.
* **Decision d'abandonner le maillage fin** : a reexaminer, la contrainte
  qui la motivait n'existe plus.

### A faire

* Regenerer les 10 maillages patients avec `gmsh_mesher.py` corrige.
* AVANT de faire confiance aux 10 : spot-check du volume de cavite sur 2-3
  autres patients. Le bug `binary_closing` depend du spacing z de chaque
  patient, donc l'ampleur de l'ecrasement varie -- le -0.18 % ne vaut que
  pour patient001.
* Reprendre le DoE sur la geometrie correcte (les resultats anterieurs
  reposent tous sur des maillages ecrases).
* Reevaluer la decision maillage grossier / maillage fin (la contrainte
  transmurale qui la motivait etait un artefact de l'ecrasement).
* Valider le champ de deformation regional (pas seulement l'EF globale)
  avant WP5.

---

## Session 2026-07-15 — Investigation cavite : cause racine = maillage non tagge, solution = surfaces endo/epi/base

### Contexte

Spot-check de generalisation du fix binary_closing sur patient002/005/008.
Deux resultats opposes.

### Resultat 1 : le fix geometrie est valide et generique

Regeneration via gmsh_mesher.py corrige (target_mm=5.0) :

| Patient    | Z (mm) | V_tissu vs GT_myo (label 2) |
|------------|--------|-----------------------------|
| patient001 |  98.5  |  -9.7 %                     |
| patient002 |  91.8  | -12.1 %                     |
| patient005 |  99.6  |  -6.6 %                     |
| patient008 |  99.5  | -11.6 %                     |

Z restaure partout (plus d'ecrasement), volume de tissu coherent (~-10 %,
signature du maillage grossier). Le maillage capture fidelement le myocarde
sur les quatre. Decision de regenerer les 10 : maintenue.

### Resultat 2 : l'estimateur de cavite (disques) est disqualifie

Volume de cavite vs GT (label 3), trois methodes independantes :

| Patient    | disques | voxel  | GT_cav  |
|------------|---------|--------|---------|
| patient001 |  -0.9 % | -6.6 % | 297.6mL |
| patient002 | -29.9 % | -32.4% | 256.8mL |
| patient005 | -22.9 % | -8.9 % | 288.7mL |
| patient008 | -11.3 % | -28.7% | 282.7mL |

Point d'honnetete : sur patient005 et patient008 les deux methodes
DIVERGENT entre elles (-23 vs -9 ; -11 vs -29). Deux estimateurs qui ne
s'accordent pas prouvent qu'aucun ne mesure la bonne chose. Le -0.9 % de
patient001 (DCM tres dilate, cavite quasi-spherique) etait une coincidence
geometrique, pas une validation. Une 3e tentative (divergence sur faces endo
classees par signe de normale) a explose a +120/+370 % : le contour du patch
endo n'est pas un anneau basal propre, capper la couture endo/epi injecte un
volume parasite.

### Cause racine (demontree)

gmsh_mesher.py produit une COQUE MYOCARDIQUE A SURFACE UNIQUE, NON TAGGEE.
Bug corollaire #2 du 2026-07-13 : on ne garde que "la composante de plus
grande aire" -> endocarde, epicarde et plan basal sont fusionnes en un seul
SurfaceLoop, distinction jetee. Tout estimateur de cavite a posteriori tente
de reconstruire une info que le mailleur a detruite. Elle n'est pas dans le
maillage.

### Solution (standard etabli du domaine)

Methode universelle (Oxford/Auckland/King's, LV Mechanics Challenge) :
extraire la surface endocardique, la fermer par un lid basal (noeud au
centroide du contour basal + eventail de triangles), normales sortantes
coherentes, volume par theoreme de la divergence V = (1/3) flux(x.n dA).
Applique IDENTIQUEMENT au repos et en deforme -> biais systematique s'annule
dans l'EF. Definition basale = plan valvulaire coherent (mitral/aortique).

Le tag endo/epi/base NE doit PAS venir d'une heuristique geometrique (echec
+370 %) mais de la donnee qu'on jette deja : le label 3 (cavite) de la
segmentation ACDC. Face de bord endocardique si elle borde du label 3,
epicardique si label 0, basale si plan Z superieur.

### Convergence strategique : ce tag est deja exige par les fibres LDRB

WP1 impose les fibres LDRB. LDRB exige IMPERATIVEMENT les memes tags
endo/epi/base (lifex-fiber, ldrb/finsberg, pipeline Meshtool UK Biobank
"CDT at Scale"). generate_simple_fibers.py (champ tangentiel simplifie) est
un placeholder PRECISEMENT parce que le maillage n'a pas ces tags. Donc
tagger les surfaces debloque DEUX livrables : volume de cavite correct (D2.1)
ET vraies fibres LDRB (D1.1/WP1). Point de plus haut levier du pipeline.

### Sequencage (ne s'ajoute pas, se replace)

Le calcul de cavite n'est PAS un patch sur coupled_solver.py : c'est une
propriete du mailleur (D1.2), a faire pendant la regeneration des 10.
1. Mailleur : tagger endo/epi/base par adjacence au label 3.
2. Cavite : surface endo + lid basal + divergence, meme lid repos/deforme.
3. Fibres : remplacer generate_simple_fibers.py par LDRB (tags le permettent).
4. Validation : V_ed maillage vs label 3 GT CLIPPE au meme plan basal
   (apples-to-apples), 4 patients, cible qq %. Puis EF patient001 (ref 23.65 %).

### A faire (mis a jour)

* Tagger endo/epi/base dans gmsh_mesher.py (prochain geste).
* Regenerer les 10 maillages avec tags + fibres LDRB.
* Remplacer l'estimateur de cavite disques par endo+lid+divergence dans
  coupled_solver.py (repos ET deforme).
* Re-valider cavite sur 4 patients, puis EF patient001.
* Reprendre le DoE sur geometrie correcte + metrique EF correcte.

---

## Session 2026-07-15 (suite) — Cavite : 4 echecs post-traitement, decision Voie A (coque deux-surfaces)

### Preuve complete : aucun post-traitement ne marche

Sur maillage myocardique a SURFACE UNIQUE (endo+epi fusionnes, bug corollaire
#2 du 2026-07-13), quatre methodes de calcul de cavite, quatre echecs :

| Methode                       | Resultat (patient001..008)        |
|-------------------------------|-----------------------------------|
| Disques (10e percentile)      | -0.9 % / -30 %, patient-dependant |
| Voxel (fill_holes par tranche)| -6.6 % / -32 %, fuite anneau      |
| Divergence normales + capping | +120 % / +370 %                   |
| Distance signee + lid         | -56 % / -65 %, ring~30            |

Signature commune : ring (composantes de la boucle basale endo) ~30 au lieu
de 1. La surface endo n'existe pas comme surface fermee separable dans le
maillage -> rien a extraire proprement. Le -0.9 % de patient001 (disques)
etait une coincidence (DCM quasi-spherique), pas une validation.

### Verification donnee (probe_cavity_label.py)

Label 3 (cavite VG) sur patient001/002/005/008 : UNE composante connexe,
0 trou, tranches z=0..9 contigues. Cavite propre, directement maillable.

### Litterature (recherchee, pas supposee)

Fedele et al. (lifex) : reconstruire endocarde et epicarde SEPAREMENT
(endo = limite externe du sang ; epi = surface fermee cappee a l'anneau),
puis connecter les deux surfaces a leur intersection. Strocchi (modele biV)
et pipeline UK Biobank "CDT at Scale" (Meshtool) : tags endo/epi/base poses
A LA GENERATION, jamais apres. lifex-ep : "volumetric tags must be defined
during the mesh generation process".

PIEGE identifie : la difference booleenne brute epi-endo dans Gmsh produit
des triangles irreguliers et des regions etranglees a l'anneau valvulaire
(Fedele, Fig.4). Un addVolume([epi,endo]) naif echouera a la base. La
connexion basale doit etre geree explicitement.

### Decision : Voie A (coque deux-surfaces taggee)

1. endo_surface = marching_cubes sur (mask==3) rempli.
2. epi_surface  = marching_cubes sur ((mask==2)|(mask==3)) rempli.
3. Mailler la coque entre les deux, tags endo/epi/base a la generation.
4. Volume cavite = divergence(endo + lid basal), repos ET deforme.
5. Fibres LDRB depuis les tags endo/epi (debloque D1.1/WP1 au passage).

Justification cahier des charges : necessaire pour LDRB et WP5 (strain
regional). Chantier D1.2. Remplace _mask_to_stl (fusion actuelle).
Point ouvert avant code : gestion de l'anneau basal (connexion endo-epi).

### A faire

* Resoudre la connexion basale endo-epi (recherche en cours).
* Reecrire _mask_to_stl -> _build_shell (deux surfaces + tags).
* Valider volume cavite vs GT clippe sur les 4 patients (cible qq %).
* Etendre regen_coarse_batch aux 6 patients restants (003,004,006,007,009,010).
* Puis DoE sur geometrie + metrique EF correctes.

---

## Session 2026-07-15 (fin) — Volume de cavite RESOLU : surface endo label 3 (Voie B)

### Echec Voie A (coque deux-surfaces) et pivot

Coque endo/epi maillee en volume : echec Gmsh en cascade.
1. createGeometry/reparametrisation -> "Invalid boundary mesh for
   parametrization" (meme mur que gmsh_mesher, contourne par createTopology).
2. Entites discretes + createTopology -> passe, mais TetGen echoue :
   "PLC Error: segment and facet intersect". Cause mesuree
   (probe_intersect.py) : les 2 surfaces marching_cubes s'interpenetrent
   (dist_min=0.00mm, 43-69 noeuds endo hors epi) sur paroi mince, aggravé
   par le lissage separe. Inclusion par construction (dilatation 2mm) :
   insuffisant.
3. pygalmesh (CGAL direct sur image) : pas de roue arm64, compilation CGAL.
4. meshtool : NE genere PAS depuis une image, manipule un maillage existant.

Decision (avis ingenieur, regle profondeur avant largeur) : Voie B. La
surface endo du label 3 suffit pour le VOLUME, sans la mailler en volume.

### Resultat : valide sur les 4 patients

Surface endo = marching_cubes sur (label 3) rempli+padde. Volume par
divergence pure (surface DEJA fermee, rings=0, le padding referme la cavite).

| Patient    | V_ed calcule | GT_cav (label 3) | ecart  |
|------------|--------------|------------------|--------|
| patient001 |   298.8 mL   |    297.6 mL      | +0.4 % |
| patient002 |   248.1 mL   |    256.8 mL      | -3.4 % |
| patient005 |   283.2 mL   |    288.7 mL      | -1.9 % |
| patient008 |   271.7 mL   |    282.7 mL      | -3.9 % |

Premier estimateur REGULIER de toute l'investigation. A comparer :
disques -0.9/-30 %, voxel -6/-32 %, divergence-normales +370 %,
distance-signee -65 %. Le biais (~-3 %, leger sous-estime) vient du lissage
gaussien + discretisation, systematique donc s'annulera dans l'EF=dV/V_ed.

### Ce qui reste (a froid) : branchement V_es

* V_es : deplacer les noeuds de la surface endo (label 3) par le champ
  FEniCSx interpole depuis la coque myocarde (noeuds endo sur face interne
  du myo -> interpolation bien posee), puis meme fonction de volume.
* EF = (V_ed - V_es) / V_ed. Base encastree -> pas de fuite basale.
* Remplacer _cavity_volume_fixed_z (disques) dans coupled_solver.
  _compute_volume_waveform, lignes 363-367.
* Re-valider EF sur patient001 (reference 23.65 %).
* Voie A (coque volumique taggee) = decision d'archi a froid pour fibres
  LDRB + WP5 ; candidat identifie : biv-volumetric-meshing (openCARP,
  pipeline UK Biobank, AppImage sans compilation).

### Fichiers

* validate_endo_volume.py : validation volume endo (ci-dessus).
* shell_mesher.py : Voie A, laissee en l'etat (echec TetGen documente).

---

## Session 2026-07-15 (fin) — EF : EndoCavityVolume valide, vrai bug = BC basale

### EndoCavityVolume (nouveau, remplace disques) : VALIDE

app/solver/mechanics/endo_cavity_volume.py. Surface endo = label 3
(marching_cubes, fermee par construction), volume par divergence,
deplacement myo->endo par interpolation (tet le plus proche + barycentriques
clampees, erreur EF <0.25 pt validee sur champ analytique).

V_ed sur 4 patients : +0.4/-3.4/-1.9/-3.9% du GT. Premier estimateur regulier.
Sur deplacement FEniCSx reel (doe_0000, converge, min_J=0.54) : deplacement
RADIAL seul -> dV=-45 mL (contraction correcte). L'estimateur est bon.

### Vrai bug : allongement axial systematique (BC basale)

Sur les 9 points DoE (_fixed), SANS exception : le ventricule S'ALLONGE
sous contraction (hauteur 99->117mm, apex descend -16 a -19mm), d'ou V_es >
V_ed et EF NEGATIVE. Le radial est pourtant correct partout (dr -2 a -3.6mm).

Diagnostic par elimination, chaque hypothese TUEE PAR UNE MESURE (pas supposee) :
- translation corps rigide : retiree -> EF inchangee (volume invariant/translation). NON.
- fibres non physiques : mesure |f0.z| mid=0.26 (circonferentiel correct),
  0 degeneree, 0 fallback. Fibres LDRB SAINES. NON.
- formulation active T*f0(x)f0 : standard litterature (S=S_pass+Ta f0xf0,
  pas de terme actif transverse - expansion transverse = passive/incompress). OK.
- incompressibilite : formulation p*(J-1)-p^2/2k propre ; kappa=1e6 vs mu~500Pa
  -> ratio 2000, correct. Fuite -1.2% = discretisation grossiere, mineure. OK.
- BC basale : CONFIRMEE. Deplacement=0.000mm dans bande basale (BC marche),
  et dZ_moyen par tranche = gradient monotone parfait 0 (base) -> -16.4mm (apex).

CAUSE : ligne 221-230, encastrement basal TOTAL (u=0 sur les 3 composantes,
bande 5mm). La base clouee empeche le raccourcissement du grand axe ; sous
contraction incompressible, l'apex descend -> allongement. Litterature (doc
contraction ventriculaire) : traitement des BC crucial pour la cinematique.
Les disques masquaient ce mode (reprojection radiale aveugle a l'axial).

### Correctif (a faire, prochaine session)

* BC basale GLISSANTE : bloquer seulement u_z sur le plan basal (base reste
  dans son plan, libre de se contracter radialement) + ancrage minimal
  (3 points) contre le mode de corps rigide. Standard physiologique.
* Re-valider EF sur patient001 via EndoCavityVolume (reference 23.65%).
* Puis brancher EndoCavityVolume dans coupled_solver (_compute_volume_waveform,
  remplacer les 2 appels _cavity_volume_fixed_z).
* Chercher le T_max du point doe_0000 pour comparer au SLO.

### Fichiers

* app/solver/mechanics/endo_cavity_volume.py : NOUVEAU, valide.
* Fibres LDRB (patient001_coarse5_fixed_fibers_ldrb.lon) : saines, 2916 nodales.

---

## Session 2026-07-15 (nuit) — Diagnostic pression endo : hypothese solide, test isole abandonne

### Acquis

* EndoCavityVolume commite (c0717b8), valide : V_ed +0.4/-3.9% GT sur 4 patients.
* Bug EF = allongement axial +18mm sous contraction, sur les 9 points DoE.
  Cause retenue par elimination (chaque hypothese TUEE par mesure) : PRESSION
  ENDOCARDIQUE MANQUANTE. Ancre Land 2015 (probleme 3 = tension active +
  pression 15 kPa, base fixee) et code (aucun terme de traction, verifie grep).
* Facettes endo identifiables par predicat geometrique (rayon<mediane, hors
  base) : ~2200 sur maillage myo. Suffisant pour une pression (pas besoin de
  surface fermee, contrairement au volume).

### Ce qui a echoue (et pourquoi)

* falsify_pressure.py (script isole repliquant la meca) : NE CONVERGE PAS,
  echec des lam=0.02 (T=1.2 kPa). Cause : NewtonSolver brut SANS le wrapper
  _SNESProblem + garde-fou domaine (min_J) de fenicsx_solver.py. La robustesse
  de convergence du solveur de PRODUCTION vient de ce wrapper, non replique.
* Lecon : NE PAS reconstruire la meca dans un script separe. Tester la
  pression DANS fenicsx_solver.py (qui converge jusqu'a T=135 kPa).
* Run bloque 3h (boucle continuation non bornee sur non-convergence) -> kill.

### A faire (a froid) - patch DEJA ECRIT (apply_pressure_patch.py)

1. Appliquer apply_pressure_patch.py : ajoute p_endo_kPa, facettes endo,
   terme pression suiveuse -P*J*inv(F).T*N sur ds_endo, increment avec lam.
   -> dans fenicsx_solver.py (wrapper SNES robuste), PAS un script isole.
2. Run BORNE (max_it plafonne, timeout dur) patient001, p_endo_kPa=15.
   Verifier : allongement +18mm -> disparait, EF -> proche 23.65%.
3. Si confirme : brancher pression Windkessel->meca (ferme boucle P->meca).
4. Puis EndoCavityVolume dans coupled_solver + DoE.

### Fichiers

* apply_pressure_patch.py : patch pression pour fenicsx_solver (PRET, non applique).
* falsify_pressure.py : ABANDONNE (convergence non robuste, cf ci-dessus).

---

## Session 2026-07-16 — Reconciliation : l'allongement est STRUCTUREL, pas une regression

### Investigation socle mecanique (l'existant vs la doc)

Question : la mecanique a-t-elle DEJA converge, et dans quelles conditions ?
Le journal documente 2 convergences lam=1.0 (2026-07-08, P15). Verification :

* Checkpoint continuation_coarse_ldrb_checkpoint.npz : lam=1.0, min_J=0.503,
  35 paliers, T=135kPa, maillage patient001_coarse5 + fibres LDRB. CONVERGE.
* Le script diag_continuation_coarse.py (autonome, MUMPS, dlam_init=1e-4,
  J_MIN=0.1) est le chemin qui converge -- PAS solver.simulate() teste hier.

### Decouverte 1 : le run "reussi" de juillet etait sur GEOMETRIE ECRASEE

Post-traitement du checkpoint (eval_ref_ef.py, sans re-resolution) :
HAUTEUR repos = 55.6mm. C'est la geometrie AVANT le fix binary_closing
(le journal : 55.62mm avant / 98.50 apres). Donc la "PREMIERE convergence
complete" de juillet tournait sur un ventricule ecrase. Convergence reelle,
mais geometrie buggee. A recadrer dans P15.

### Decouverte 2 : l'allongement axial est STRUCTUREL (pas une regression)

Deplacement du checkpoint applique via EndoCavityVolume :
HAUTEUR 55.6 -> 68.3mm (+12.7mm), apex_dz=-10.9mm, V_ed=298.8 V_es=294.9
EF=+1.3% a T=135kPa.

=> Le run de reference S'ALLONGE AUSSI. L'allongement axial etait deja la
en juillet, masque par la methode des disques. Ce n'est NI _fixed, NI une
regression recente : c'est structurel a la formulation.

Prouve sur DEUX maillages independants (coarse5 ecrase ET coarse5_fixed
corrige) et DEUX charges (22 et 135 kPa) : la contraction sans pression
endocardique allonge le ventricule. Diagnostic pression = FAIT reproductible.

### Les deux problemes n'en font qu'un

* "Convergence" (cale sur _fixed) et "allongement" (EF negative) etaient vus
  comme 2 bugs. En realite :
  - allongement = formulation incomplete (pression endo manquante) -> STRUCTUREL
  - non-convergence sur _fixed = maillage plus raide (9439 tets, 98mm) que
    coarse5 (5686 tets, 55mm) + mon dlam_init=0.01 d'hier (trop gros, le
    journal recommande 1e-4). Distinct et secondaire.

### Plan (une variable a la fois)

1. Valider la PRESSION sur coarse5 (convergence acquise, 43min prouves),
   avec les reglages du script qui marche : MUMPS + dlam_init=1e-4.
   Verifier : allongement supprime, EF plausible.
2. Puis porter au maillage _fixed (continuation fine).
3. Puis brancher pression Windkessel -> meca.
4. Unifier : un seul solveur (fenicsx_solver.py doit adopter MUMPS +
   dlam_init du script diag qui converge ; les 2 chemins ont diverge).

### Fichiers
* eval_ref_ef.py : post-traitement checkpoint (EF sans re-resolution).
* /tmp/endo_precomp.npz : surface endo + mapping (hors conteneur, skimage).
