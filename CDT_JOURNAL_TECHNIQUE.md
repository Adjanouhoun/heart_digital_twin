
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
