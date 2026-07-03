# CDT — Etat d'avancement strict
## Mise a jour : 3 juillet 2026

### LIVRABLES CONFORMES (7/9 hors validation clinique)

| Livrable | Description | Statut | Detail |
|----------|-------------|--------|--------|
| D1.2 | 10 meshes + rapport qualite | CONFORME | 10/10, Z corrige, volumes 205-502mL |
| D2.2 | Calibration MCMC | CONFORME | 5/5 predictive check OK |
| D3.1 | GP Emulators | CONFORME | 10/11 R2>0.8, cv_ms R2=0.9955 |
| D3.2 | Sobol GSA | CONFORME | Physiologiquement correct |
| D4.1 | API WebSocket | CONFORME | REST + WS, 12 tests |
| D6.1 | CI/CD + GitHub | CONFORME | GitHub Actions, 133 tests |
| D6.2 | Frontend React | CONFORME | Dashboard Three.js + curseurs |

### LIVRABLES EN COURS (2/9)

| Livrable | Description | Statut | Reste |
|----------|-------------|--------|-------|
| D1.1 | nnU-Net segmentation | 80% | ACDC MYO=0.887 (SLO 0.90), Colab en cours. M&Ms et EMIDEC a faire |
| D2.1 | Solveur couple | 80% | Pipeline reel 8/10 patients. 2 divergent (patient002/003) |

### LIVRABLES EXCLUS (validation clinique)

| Livrable | Description | Statut | Blocage |
|----------|-------------|--------|---------|
| D5.1 | Validation retrospective | 0% | Besoin Dr. Namoano + cliniciens |
| D5.2 | Pilote prospectif + ASME V&V40 | 0% | Besoin comite ethique |

### PIPELINE REEL VALIDE

8/10 patients avec openCARP EP + FEniCSx Land 2015 + Windkessel :
  EF  = 49.8 +/- 7.5% (range 39.8-61.6%)
  Psys = 99.2 +/- 14.9 mmHg
  Pdia = 70.0 +/- 10.5 mmHg
  CO  = 4.23 +/- 0.63 L/min

### ACTIONS IMMEDIATES

1. D1.1 : Attendre Colab ACDC vrais NIfTI (~8h)
2. D1.1 : Lancer M&Ms sur Colab apres ACDC
3. D2.1 : Fixer patient002/003 (Newton diverge)
4. D2.1 : DoE 500 sims avec pipeline reel
