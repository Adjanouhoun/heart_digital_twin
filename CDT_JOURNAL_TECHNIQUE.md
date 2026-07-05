
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
