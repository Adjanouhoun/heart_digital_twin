# CDT — Audit Architectural
## Weekend 24-28 juin 2026 — Analyse post-mortem

---

## 1. DIAGNOSTIC CRITIQUE — Pourquoi on tourne en rond

### 1.1 Le pattern destructeur : "sed-driven development"

Le code du mailleur (`generate_meshes_acdc.py`) a été modifié **15 fois** par des remplacements inline (`sed -i`, `code.replace()`) pendant cette session. Chaque modification cassait la précédente. Séquence réelle :

```
Gmsh classifySurfaces → erreur API 4.15.2
→ retirer forceParametrizablePatches → surfaces non-manifold
→ Delaunay scipy → 52% dégénérés (convex hull)
→ TetGen → surfaces non-manifold
→ TetGen + PyMeshFix → verbose argument inconnu
→ retirer verbose → TetGen timeout patient009
→ signal.alarm → ne peut pas interrompre du C
→ multiprocessing → can't pickle local object
→ subprocess timeout → finalement stable (9/10)
```

Neuf tentatives pour un mailleur. Le même pattern s'est reproduit pour openCARP (8 tentatives sur le format .par).

### 1.2 Fondations instables propagées vers le haut

Les GP Emulators (Phase 04) sont entraînés sur les sorties du **fallback analytique**. Ce fallback retourne des valeurs quasi-constantes :

```
ef_pct  : std = 0.0 → toujours 60%
co_Lmin : std = 0.0006 → quasi-constant
```

Conséquence : les GP "fonctionnent" (R² élevé sur cv_ms qui varie) mais sont **scientifiquement inutiles**. La calibration MCMC Phase 05 est biaisée (T_max non identifiable). Pourtant on a 99 tests verts — les tests valident l'infrastructure, pas la physique.

### 1.3 Absence de contrats d'interface

Les intégrations inter-phases échouent systématiquement à cause de conventions non documentées :

| Interface | Problème découvert à l'exécution |
|-----------|----------------------------------|
| Maillage → openCARP | mm vs µm (facteur 1000) |
| .par paramètres | mesh vs meshname, stim vs stimulus |
| Modèle ionique | TT2, tenTusscher2006epi, tenTusscherPanfilov — 3 essais |
| Output vector | noms d'attributs incohérents (ecg_12leads vs ecg_leads_mv) |
| Stimulus | stimtype vs type — 2 essais |
| bcl vs tend | bcl doit être ≤ tend — découvert 3 fois |

### 1.4 Scope creep horizontal

En un weekend, on a touché 7 phases (01→07) sans en compléter une seule à 100%. Phase 02 est à ~90% (M&Ms stagne à Dice 0.59), Phase 03 est à ~65% (openCARP diverge), Phase 04 est à ~40% (GP sur données factices). On a de la largeur mais pas de profondeur.

### 1.5 Friction technique récurrente

`zsh: command not found: #` apparaît **23 fois** dans le transcript. Les commentaires bash dans les heredocs cassent systématiquement. C'est un micro-problème qui consomme du temps cumulé significatif.

---

## 2. CARTOGRAPHIE DES DÉPENDANCES ET PRÉREQUIS STRICTS

### 2.1 Graphe de dépendances réel

```
Phase 01 (DICOM) ✅
    │
    ▼
Phase 02 (Segmentation) ←── BLOQUANT pour tout le reste
    │   SLO: Dice MYO ≥ 0.90 (ACDC=0.896 ✓, M&Ms=0.48 ✗)
    │   Prérequis: checkpoint validé, inférence reproductible
    │
    ▼
Phase 02b (Maillage) ←── BLOQUANT pour Phase 03
    │   SLO: 0% dégénérés, Jacobian > 0, ≥ 9/10 patients
    │   Prérequis: TetGen pipeline stable ✓
    │   CONTRAT: .pts (mm), .elem (Tt), .lon (2 vecteurs)
    │
    ▼
Phase 03 (Solveur couplé) ←── BLOQUANT pour Phase 04
    │   SLO: openCARP converge sur ≥ 1 patient
    │   Prérequis: unités µm, volumes positifs, g_il/g_it calibrés
    │   CONTRAT: output_vector 14D avec variance significative
    │   ÉTAT: benchmark slab 75 nodes OK, patient 36K nodes BLOQUÉ
    │
    ▼
Phase 04 (Surrogates) ←── BLOQUÉ par Phase 03
    │   Prérequis: ≥ 500 sims avec openCARP réel
    │   GP actuels entraînés sur fallback → invalides
    │
    ▼
Phase 05 (Calibration) ←── BLOQUÉ par Phase 04
    │   Prérequis: GP valides avec outputs variables
    │
    ▼
Phase 06 (Orchestration) + Phase 07 (Frontend)
    │   Peuvent avancer en parallèle (infrastructure)
    │
    ▼
Phase 08 (DevOps) → Phase 09 (Validation clinique)
```

### 2.2 Contrats d'interface manquants

**C1 — Contrat Maillage → Solveur EP**
```python
@dataclass
class MeshContract:
    nodes: np.ndarray      # shape (N, 3), unités: µm (PAS mm)
    elements: np.ndarray   # shape (M, 4), indices 0-based, volumes > 0
    fibers: np.ndarray     # shape (M, 6), [fx,fy,fz, sx,sy,sz] normalisés
    patient_id: str
    units: str = "um"      # TOUJOURS µm pour openCARP
```

**C2 — Contrat openCARP .par**
```python
@dataclass
class OpenCARPConfig:
    meshname: str           # PAS "mesh"
    ionic_model: str        # "tenTusscherPanfilov" (vérifié par strings)
    stimulus_prefix: str    # "stimulus[0]" (PAS "stim[0]")
    stimulus_type_key: str  # "stimtype" (PAS "type")
    output_key: str         # "simID" (PAS "--output_dir")
    bcl_ms: float           # DOIT être ≤ tend
    duration_ms: float      # stimulus.duration DOIT être ≤ tend
    dt_ms: float            # 0.02 pour stabilité PETSc
    g_il: float             # 0.174 S/m (standard, unités µm)
    g_it: float             # 0.019 S/m (standard, unités µm)
```

**C3 — Contrat Output Vector**
```python
OUTPUT_NAMES = [
    "cv_ms", "apd90_ms", "act_max_ms", "act_range_ms", "vm_peak_mv",
    "p_sys_mmHg", "edv_mL", "p_dia_mmHg", "sv_mL", "map_mmHg",
    "ef_pct", "esv_mL", "lv_mass_g", "co_Lmin"
]
# 14 dimensions, TOUTES doivent avoir std > 0.01
```

### 2.3 Ce qui est "gravé dans le marbre" vs ce qui ne l'est pas

| Composant | Statut | Confiance |
|-----------|--------|-----------|
| Phase 01 (DICOM pipeline) | Stable | Haute — 16 tests, pas touché |
| nnU-Net architecture | Stable | Haute — PlainConvUNet 3d_fullres validé |
| Checkpoint ACDC | Stable | Moyenne — MYO=0.896 (proche du SLO) |
| Pipeline maillage (TetGen) | Stable | Moyenne — 9/10 patients, subprocess timeout |
| Connexion Phase 02→03 | Stable | Haute — 10/10 sims, fallback analytique |
| GP Emulators | INVALIDE | Basse — entraînés sur données constantes |
| MCMC Calibration | INVALIDE | Basse — biaisé par GP invalides |
| openCARP intégration | INSTABLE | Basse — diverge sur vrais patients |
| API REST | Prototype | Moyenne — fonctionne mais pas connecté |
| Viewer Three.js | Prototype | Basse — bouton Animer cassé |
| M&Ms training | INSUFFISANT | Basse — Dice 0.59, stagne |

---

## 3. PLAN DE SAUVETAGE ET FEUILLE DE ROUTE IMMÉDIATE

### 3.1 Principe : profondeur avant largeur

On arrête de toucher à 7 phases en parallèle. On verrouille chaque phase avant de passer à la suivante. La Phase 03 (openCARP réel) est le bottleneck de TOUT le projet.

### 3.2 Feuille de route séquentielle stricte

**JALON 1 — openCARP benchmark réussi sur vrai maillage (1-2 sessions)**

Problème identifié : PETSc diverge ("indefinite system matrix" ou "iterations exceeded").
Causes probables restantes :
- Éléments avec volumes négatifs dans le maillage original (swap vertices manquant)
- Conductivités mal calibrées pour les unités du maillage
- dt trop grand pour la résolution spatiale

Actions :
1. Lire les tutoriels opencarp.org/documentation/examples (benchmark N1-N5)
2. Reproduire le benchmark N1 (slab propagation) EXACTEMENT comme documenté
3. Vérifier les unités et conductivités sur le benchmark
4. Appliquer les mêmes paramètres au maillage patient (converti en µm)
5. Si 36K nodes est trop lent → décimer le maillage à 5K nodes avec le VRAI pipeline TetGen (pas Delaunay)

**JALON 2 — DoE avec openCARP réel (1-2 sessions)**

Prérequis : Jalon 1 validé.
Actions :
1. Convertir les 9 maillages patients en µm avec correction des volumes
2. Lancer 10 sims openCARP (1 par patient, paramètres standards)
3. Vérifier que les 14 outputs varient significativement
4. Si les sims sont trop lentes (>1h chacune) : utiliser GitHub Codespaces (gratuit, Linux x86)
5. Lancer le DoE 500 sims (probablement sur Codespaces si M1 trop lent)

**JALON 3 — Surrogates valides (1 session)**

Prérequis : Jalon 2 validé.
Actions :
1. Ré-entraîner les 14 GP sur les vraies données openCARP
2. Vérifier que R² > 0.8 sur TOUTES les dimensions (pas juste cv_ms)
3. Sobol GSA sur les vrais GP
4. Tests de validation croisée

**JALON 4 — Calibration MCMC valide (1 session)**

Prérequis : Jalon 3 validé.
Actions :
1. MCMC avec les GP valides
2. Vérifier 10/10 paramètres dans CI 90%
3. Posterior predictive checks

**JALON 5 — Frontend connecté (1-2 sessions)**

Prérequis : Jalons 1-4 validés.
Actions :
1. Connecter l'API REST aux vrais résultats
2. Viewer Three.js avec données openCARP réelles
3. Panel clinicien avec curseurs

### 3.3 Choix architecturaux pour corriger le tir

**openCARP** : Créer un fichier `app/solver/ep/opencarp_config.py` qui centralise TOUS les paramètres .par validés. Plus jamais de sed inline sur le format .par.

**Unités** : Créer un module `app/core/units.py` avec des fonctions explicites :
```python
def mm_to_um(nodes_mm: np.ndarray) -> np.ndarray:
    return nodes_mm * 1000.0

def fix_element_orientation(nodes, elements) -> np.ndarray:
    """Swap vertices if volume is negative."""
```

**Tests de contrat** : Chaque interface a un test qui vérifie le format AVANT de passer au composant suivant.

---

## 4. CHARTE DE DÉVELOPPEMENT — 5 RÈGLES D'OR

### Règle 1 : Documentation avant code

Avant de coder quoi que ce soit avec un outil externe (openCARP, TetGen, Gmsh), on lit sa documentation officielle et on reproduit un exemple connu. On ne devine JAMAIS le format des paramètres par essai-erreur.

### Règle 2 : Un fichier, un commit, un test

On ne modifie jamais un fichier par `sed -i` ou `code.replace()` en inline. On écrit le fichier complet avec `cat >` ou on le crée via le terminal. Chaque modification est suivie d'un test qui vérifie que rien n'est cassé.

### Règle 3 : Profondeur avant largeur

On ne passe à la phase N+1 que quand la phase N a ses SLOs atteints ET ses tests verts. Exception : les prototypes frontend (Phase 07) qui peuvent avancer en parallèle car ils ne dépendent que de l'API.

### Règle 4 : Contrats d'interface explicites

Chaque composant qui passe des données à un autre a un contrat documenté : types, unités, dimensions, noms d'attributs. On crée un fichier `contracts.py` par phase. Les tests vérifient les contrats, pas juste les fonctions.

### Règle 5 : Pas de commentaires bash dans les heredocs

Les commentaires `# ...` dans les blocs `cat <<` ou inline bash causent des erreurs `zsh: command not found`. On les supprime systématiquement ou on utilise Python pour créer les fichiers.

---

## RÉSUMÉ EXÉCUTIF

Le weekend a produit une infrastructure impressionnante (99 tests, 7 phases touchées) mais les fondations scientifiques sont fragiles. Les GP et la calibration MCMC sont invalides car construits sur le fallback analytique. openCARP fonctionne en benchmark (75 nodes) mais diverge sur les vrais maillages.

Le chemin critique est clair : **résoudre openCARP sur vrais patients**. Tout le reste en dépend. La prochaine session doit être 100% focalisée sur ce jalon unique, en commençant par la lecture des benchmarks officiels opencarp.org.
