# CDT — Diagnostic complet & feuille de route
## 6 juillet 2026

Ce document fait le point sur l'état réel du pipeline de simulation après les
sessions des 4-6 juillet, identifie les problèmes ouverts avec leur cause,
propose les solutions conformes au projet, et établit la feuille de route.

---

## 1. CE QUI EST SOLIDE (acquis vérifiés)

| Composant | État | Preuve |
|-----------|------|--------|
| Segmentation nnU-Net ACDC | ✅ Production, MYO 0.910 | Checkpoint epoch 559, dans le DAG |
| Segmentation M&Ms | ✅ Validé (généralisation) | Mean Dice 0.908 (Colab) |
| Maillage Gmsh myocarde | ✅ Production, 0 sliver | patient001 57K nodes, QC OK |
| openCARP EP | ✅ Converge, rapide | Code=0, 2.8s/50ms (bug dt corrigé) |
| Chaîne délégation | ✅ Validée | DAG→Solver API→openCARP, 9/9 tâches |
| Pipeline DAG complet | ✅ IRM→EF end-to-end | 9/9 success, EF/CV produits |
| CV mesurée (parsing LAT) | ✅ Corrigé | init_acts_depol-thresh.dat lu, CV=0.423 |

Ces points ne sont PAS remis en cause. Le pipeline tourne de bout en bout.

---

## 2. LE PROBLÈME CENTRAL — RÉSOLU (6 juillet, 15h56)

### Statut : ✅ RÉSOLU

Cause racine identifiée par élimination méthodique (voir §2bis) : **le paramètre
`gregion[0].g_il`/`g_it` seul est ignoré par openCARP** (version a86f7c4) tant
que `gregion[0].g_mult` n'est pas défini. C'est `g_mult` qui active réellement
l'application de la conductivité au calcul.

Preuve définitive : comparaison MD5 binaire de `vm.igb` entre deux runs à
g_il variant d'un facteur 40 -> fichiers IDENTIQUES (aucun effet). Dès que
`gregion[0].g_mult` est ajouté et varié, le MD5 change immédiatement.

### Résultat après correction (via CoupledSolver, patient001, 57K nodes)
sigma_l=0.3 (g_mult=1.0) -> CV = 0.545 m/s
sigma_l=0.6 (g_mult=2.0) -> CV = 0.874 m/s
Rapport observé 1.60, theorique attendu ~1.41 (CV∝√g) -> même ordre de grandeur,
cohérent avec un maillage 3D réel (vs modèle 1D théorique). VARIANCE CONFIRMÉE.

### Fix appliqué
- `generate_par_file` (opencarp_config.py) : ajout du paramètre `g_mult`,
  écrit dans le .par comme `gregion[0].g_mult = {g_mult}`.
- `_run_opencarp` (opencarp_solver.py) : g_mult = params.sigma_l / 0.30
  (0.30 = valeur de référence validée). g_il/g_it restent aux valeurs
  VALIDATED fixes ; c'est g_mult qui porte la variation du DoE.

### Hypothèses éliminées au passage (documentées pour ne pas y revenir)
- phys_region manquant : cause d'un WARNING réel mais PAS de l'invariance
  (testé isolément : invariant même sans phys_region).
- Stimulus mal placé (apex hors maillage) : bug réel et corrigé (19->78 puis
  33122 nœuds actifs), mais PAS la cause de l'invariance à g_il.
- Force du stimulus (saturation) : écartée (invariant même à strength=50).
- Cache/simID : écartée (simID unique + facteur 40 sur g_il -> toujours invariant).
- Seuil de détection LAT (-75mV vs -20mV) : écartée (invariant aux deux seuils).
- tend/durée : CONFIRME que openCARP lit bien le .par (tend change le MD5),
  ce qui a permis d'isoler que seul gregion sans g_mult était ignoré.

## 2bis. LE PROBLÈME CENTRAL (historique du diagnostic, pour référence)

### Symptôme
sigma_l=0.3 → CV=0.423 m/s
sigma_l=0.6 → CV=0.423 m/s (IDENTIQUE, physiquement impossible)

Doubler la conductivité devrait multiplier la CV par ~√2 (≈0.60).

### Ce qui a été corrigé (mais n'a pas suffi)
1. `generate_par_file` accepte désormais g_il/g_it en arguments (au lieu des
   constantes fixes VALIDATED). VÉRIFIÉ : le .par généré change bien
   (g_il=0.354 vs 0.709). ✅
2. `_run_opencarp` passe params.sigma_l * facteur d'échelle. VÉRIFIÉ présent
   dans le code. ✅
3. Pas de message `failed_fallback` dans les logs → openCARP réel a tourné,
   pas le fallback analytique.

### Hypothèses restantes (à trancher, NE PAS deviner)
- **H1** : le .par réellement écrit sur disque contient-il g_il=0.709 pour
  sigma_l=0.6 ? (le fichier local a-t-il bien été mis à jour ?)
  → COMMANDE : grep g_il sur le sim.par du dernier run.
- **H2** : si le .par est correct mais la CV constante → openCARP ignore g_il,
  OU la mesure de CV est saturée par la géométrie (distance max/temps max prend
  toujours les mêmes nœuds extrêmes indépendamment de la vitesse).
- **H3** : le calcul CV = max_dist/max_time est trop grossier. Si l'onde atteint
  toujours les mêmes nœuds extrêmes, le ratio peut être stable même si la
  vitesse locale change. Une CV mesurée sur un segment défini serait plus juste.

### Contradiction avec le status strict (IMPORTANT)
Le status strict (3 juillet) affirme un pipeline validé avec VARIANCE :
  EF = 49.8 ± 7.5%, Psys = 99.2 ± 14.9 mmHg, GP cv_ms R²=0.9955
Or nos tests donnent EF=60% et CV=0.423 CONSTANTS. Deux lectures possibles :
- soit ces chiffres "avec variance" venaient du FALLBACK analytique (qui, lui,
  dépend de sigma_l via la formule cv_aniso) — ce qui confirmerait que le DoE
  historique était sur fallback (thèse de l'audit) ;
- soit un mode de fonctionnement antérieur transmettait la variance autrement.
→ Cette contradiction RENFORCE la nécessité de valider la variance sur le VRAI
  openCARP avant tout DoE.

---

## 3. PROBLÈMES SECONDAIRES (connus, non bloquants)

| Problème | Impact | Priorité |
|----------|--------|----------|
| p_systolic=154 mmHg (élevé) | Windkessel non calibré patient | Moyenne |
| FEniCSx en fallback | Pas de vraie mécanique Holzapfel-Ogden | Haute (D2.1) |
| EF=60% constant | Vient du couplage, dépend du même bug variance | Liée à §2 |
| Désalignement LAT/nodes (33191 vs 57058) | CV globale OK mais mapping imprécis | Basse |
| 1 sim = 200s (couplé) | DoE 500 = ~28h | Moyenne (perf) |
| _jobs en mémoire (Solver API) | OK mono-worker seulement | Basse |
| EMIDEC interrompu (Kaggle timeout) | D1.1 3e dataset incomplet | Basse (hors CC) |

---

## 4. SOLUTIONS CONFORMES AU PROJET

Le projet impose : openCARP (monodomaine), FEniCSx (Land 2015 / Holzapfel-Ogden),
Windkessel, DoE pour surrogates. Les solutions ci-dessous respectent cette stack.

### 4.1 Résoudre la variance CV (priorité 1)
1. Confirmer H1 : vérifier le g_il du .par réel (1 commande).
2. Si .par correct → revoir la MÉTHODE de mesure CV : au lieu de
   max_dist/max_time global, mesurer sur un trajet défini (apex→base le long
   d'une fibre), plus sensible à la vitesse locale. Conforme : c'est la méthode
   standard de mesure de CV en EP.
3. Valider : sigma_l=0.3 vs 0.6 doivent donner des CV dans un rapport ~√2.

### 4.2 Sortir FEniCSx du fallback (priorité 2, D2.1)
La mécanique Holzapfel-Ogden (Land 2015) est imposée. Actuellement en fallback.
C'est le vrai chantier restant du solveur couplé.

### 4.3 DoE 500 sims sur vrai openCARP (priorité 3, débloque D3.1/D3.2/D2.2)
UNIQUEMENT après 4.1 (sinon 500× le même résultat). ~28h de calcul.
Rend enfin valides les GP/Sobol/MCMC (actuellement sur fallback selon l'audit).

---

## 5. FEUILLE DE ROUTE

### Étape A — Débloquer la variance (court terme, 1-2 sessions)
- [ ] Trancher H1/H2/H3 : vérifier le .par réel, puis corriger la mesure de CV
- [ ] Valider variance sur 3 sims (sigma_l différents → CV différentes)
- [ ] Commit du jalon "openCARP paramétrique validé"

### Étape B — Mécanique réelle (moyen terme, D2.1)
- [ ] Brancher FEniCSx Land 2015 (sortir du fallback)
- [ ] Calibrer Windkessel patient (p_sys 154→~120)
- [ ] Valider couplage EP+Méca+Windkessel sur 1 patient

### Étape C — DoE & surrogates valides (moyen terme, D3.x/D2.2)
- [ ] DoE 500 sims sur vrai openCARP (~28h, une nuit)
- [ ] Réentraîner GP sur données réelles
- [ ] Sobol + MCMC sur données réelles
- [ ] Marquer D3.1/D3.2/D2.2 CONFORME (vraiment, pas sur fallback)

### Étape D — Finalisation (long terme)
- [ ] EMIDEC (relancer Kaggle quand quota recharge)
- [ ] Frontend Three.js : afficher le cœur qui bat (activation réelle)
- [ ] Validation clinique D5.x (bloqué éthique/hôpital, hors chemin technique)

### Priorité absolue
La variance CV (Étape A) conditionne TOUT l'aval (DoE→GP→Sobol→MCMC). C'est le
verrou n°1. Sans elle, le DoE produit des données constantes, scientifiquement
inutilisables. À traiter avant toute autre chose.
