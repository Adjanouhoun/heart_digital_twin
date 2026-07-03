
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
