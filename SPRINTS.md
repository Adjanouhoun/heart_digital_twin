# CDT — Journal des sprints

Ce journal est la trace opérationnelle du projet. Chaque sprint commence par
un état des lieux vérifiable et se termine par des résultats mesurés, les
écarts avec le document de projet et une décision explicite.

## Sprint 1 — Validation mécanique orthotrope sans pression

### État des lieux initial — 19 juillet 2026

#### Position dans le document de projet

- Lot concerné : WP2 — simulation multiphysique.
- Livrable concerné : D2.1 — solveur électromécanique-hémodynamique couplé.
- Dépendances aval bloquées : DoE physique, D3.1 substituts temps réel,
  D3.2 incertitude, D2.2 calibration inverse, WP4 assimilation et D6.2
  démonstrateur interactif.

#### Acquis vérifiés avant le sprint

- Géométrie patient001 corrigée après suppression du `binary_closing`
  anisotrope destructeur.
- Volume ED géométrique : 294,99 mL ; référence ACDC : 295,51 mL.
- EF ACDC de référence : 23,65 %.
- Ancien run à `T_max=30 kPa`, modèle passif isotrope incomplet : convergence,
  EF 26,14 %. Ce résultat sert uniquement de trace historique.
- Loi passive désormais complétée avec `W_f`, `W_s` et `W_fs`.
- Champ LDRB patient001 régénéré à six composantes fibre+sheet : 2 916 nœuds,
  normes unitaires, `max |f·s| = 1,32e-6`.
- Tests ciblés : 40 réussis.
- Smoke test FEniCSx réel sans pression : deux paliers acceptés,
  `load_fraction=0.0002`, 13 itérations, `min_J=0.9869`, aucune erreur de
  domaine, durée 750 s.

#### État du dépôt au démarrage

- Branche locale : `main`, huit commits en avance sur `origin/main`.
- Arbre de travail déjà modifié avant le sprint ; aucun nettoyage destructif.
- Scripts de diagnostic historiques non suivis conservés en l’état.
- Aucun push autorisé ou réalisé dans ce sprint.

### Objectif unique

Exécuter jusqu’à charge complète (`load_fraction=1.0`) la mécanique
Holzapfel-Ogden orthotrope du patient001 corrigé, sans pression endocardique,
avec `T_max=30 kPa`, puis mesurer la réponse mécanique et volumétrique.

### Configuration figée

- Patient : `patient001_coarse5_fixed`.
- Nœuds : 2 916 ; tétraèdres : 9 439.
- Microstructure : LDRB fibre+sheet à six composantes.
- `T_max_kPa=30.0`.
- `p_endo_kPa=0.0`.
- Paramètres passifs : valeurs présentes dans `MechanicsParameters` au début
  du sprint ; aucune calibration pendant le run.
- Solveur : FEniCSx 0.7.3, formulation mixte P2-P1, continuation adaptative.
- Checkpoint disque obligatoire.

### Critères d’acceptation

1. `converged=True` et `load_fraction=1.0`.
2. Aucun NaN/Inf et aucune inversion d’élément ; `min_J > 0.1`.
3. Checkpoint fonctionnel en cas d’interruption et supprimé uniquement après
   convergence complète.
4. EDV, ESV et EF calculés par `EndoCavityVolume`, pas par le volume du tissu.
5. Déplacements axial et radial rapportés ; absence de conclusion clinique
   si la déformation demeure non physiologique.
6. Comparaison quantitative à l’EF ACDC de 23,65 %, sans recalibration pendant
   ce sprint.

### Conditions d’arrêt contrôlé

- continuation bloquée sous `dlam_min` ;
- `min_J <= 0.1` ;
- erreur de domaine répétée ou valeurs non finies ;
- incohérence de maillage/microstructure ;
- coût observé incompatible avec une exécution raisonnable sur la machine,
  auquel cas le checkpoint et les mesures de performance sont conservés.

### Hors périmètre

- pression endocardique ;
- Windkessel et couplage complet ;
- calibration inverse ;
- DoE, GP, Sobol ou MCMC ;
- frontend et démonstrateur.

### Résultats du sprint

#### Point intermédiaire — arrêt contrôlé pour impraticabilité

- Trois paliers orthotropes acceptés sans pression.
- Checkpoint : `load_fraction=0.0003`, `dlam=0.0001`, `min_J=0.984949`.
- Itérations observées sur les deux premiers paliers mesurés : 7 puis 6.
- Coût observé : environ 6,25 minutes par palier sur Docker/Mac local.
- La règle d'augmentation du pas exige `its <= 4`; elle ne s'est donc jamais
  déclenchée et `dlam` est resté à `0.0001`.
- Projection au rythme observé : environ 10 000 paliers, soit ~43 jours.
- Décision : arrêt propre du conteneur ; checkpoint conservé. La formulation
  orthotrope compile et accepte des paliers, mais la stratégie de continuation
  est impraticable. Aucun résultat EF n'est produit à ce stade.

#### Blocage à résoudre avant reprise

Établir une stratégie de continuation plus efficace sans relâcher les critères
de convergence ou de Jacobien et sans modifier les paramètres physiques. Toute
nouvelle valeur de `dlam` ou règle adaptative devra être testée sur une séquence
courte et comparée au checkpoint actuel avant un nouveau run complet.

#### Diagnostic de continuation

- Expérience isolée depuis une copie du checkpoint `lambda=0.0003`.
- Incrément demandé : `0.0005` (cinq fois le pas historique).
- Cible `lambda=0.0008` acceptée en 6 itérations.
- `min_J=0.971680`, aucune erreur de domaine.
- Conclusion : le petit pas n'est pas imposé par la convergence locale ; la
  règle héritée `croissance si iterations <= 4` empêche artificiellement
  l'adaptation du modèle orthotrope.
- Correctif retenu : seuil rendu configurable, valeur historique 4 conservée
  par défaut, valeur 8 utilisée uniquement par le Sprint 1. Les règles de
  rejet, `dlam_min`, `dlam_max` et `min_J` restent inchangées.

#### Clôture du Sprint 1 — 19 juillet 2026

**Statut : convergence numérique obtenue, validation scientifique rejetée.**

- Charge complète atteinte : `load_fraction=1.0`, `converged=true`.
- 124 itérations cumulées, 37 paliers acceptés, aucune erreur de domaine.
- `min_J=0.570101`, donc aucune inversion et critère géométrique numérique
  satisfait.
- Durée totale solveur : `7027.98 s`, soit environ 1 h 57 min.
- Checkpoint repris avec succès puis supprimé après convergence complète.
- Déplacement maximal `27.10 mm`, médian `12.66 mm`.
- Déplacement radial endocardique `-4.59 mm` et épicardique `-4.55 mm`.
- Hauteur ventriculaire : `98.50 → 117.17 mm`, soit un allongement de
  `18.67 mm`; déplacement apical axial `-18.67 mm`.
- Volume cavitaire ED : `298.81 mL`.
- Volume cavitaire ES : `305.03 mL`.
- EF calculée : `-2.08 %`, contre référence ACDC `23.65 %`, écart absolu
  `25.73` points.

La formulation orthotrope et la continuation sont désormais praticables,
mais la réponse mécanique est non physiologique : malgré une contraction
radiale, l'allongement axial domine et augmente le volume cavitaire. Le Sprint
1 ne valide donc ni `T_max=30 kPa` comme point nominal physiologique, ni la
mécanique pour le DoE ou le couplage hémodynamique.

Artefacts finaux : `sprint_artifacts/sprint1/result.json`, `fields.npz`,
`manifest.json` et rapports de validation associés.

#### Décision de passage

Le Sprint 2 hémodynamique peut corriger ses défauts unitaires isolés, mais le
couplage pression–mécanique reste bloqué tant que la cause de l'allongement
axial n'est pas résolue. Le prochain sprint mécanique doit auditer, dans cet
ordre, les conditions aux limites basales, le signe/orientation des fibres et
feuillets, puis la cinématique de tension active. Aucun recalibrage de
`T_max` ne sera effectué avant cet audit structurel.

## Sprint 2 — Diagnostic de la mécanique active

### État des lieux initial — 19 juillet 2026

- La formulation Holzapfel-Ogden orthotrope converge à pleine charge sur
  patient001 avec `T_max=30 kPa`, sans pression et sans inversion.
- Le résultat est non physiologique : hauteur `+18.67 mm`, EDV `298.81 mL`,
  ESV `305.03 mL`, EF `-2.08 %`.
- La contraction radiale est présente (`-4.59 mm` à l'endocarde), mais elle
  est dominée par l'allongement axial.
- Les champs finaux complets sont disponibles ; aucun nouveau calcul FEniCSx
  n'est requis pour la première phase de diagnostic.
- La cause n'est pas établie. Conditions basales, orientation LDRB, signe de
  la tension active et post-traitement volumique sont des hypothèses à tester,
  pas des conclusions.

### Objectif unique

Identifier par tests discriminants la cause nécessaire de l'allongement axial
et de l'EF négative, puis corriger uniquement cette cause démontrée.

### Critères de décision

1. Tension active nulle : déplacement nul à la tolérance numérique.
2. Bloc élémentaire avec fibre imposée : raccourcissement dans la direction
   active attendue et absence d'expansion active de signe opposé.
3. Champ LDRB : normes, orthogonalité et angles transmuraux vérifiés sur les
   éléments effectivement utilisés par FEniCSx.
4. Conditions basales : degrés de liberté contraints explicitement inventoriés
   et absence de mode axial parasite.
5. Volume cavitaire : variation confirmée par au moins deux estimateurs
   indépendants avant d'accuser la mécanique.
6. Smoke test corrigé : `min_J > 0.1`, raccourcissement axial et diminution du
   volume avant toute relance complète.

### Interdictions du sprint

- aucun ajustement opportuniste de `T_max` ou des paramètres passifs ;
- aucune modification simultanée de plusieurs mécanismes ;
- aucune relance complète avant identification et smoke test ;
- aucune conclusion clinique à partir du résultat du Sprint 1.

### Diagnostic initial sur les champs sauvegardés

- La base (`z > z_max-5 mm`) représente 116 nœuds, soit `3.98 %`, et reste
  quasi fixe comme imposé.
- Le déplacement axial varie presque linéairement de la base vers l'apex :
  corrélation `z–u_z = 0.979`; l'apex se déplace d'environ `-18.3 mm`.
- Stretch fibre élémentaire médian `0.946`; 69.3 % des éléments raccourcissent
  suivant la fibre. Le signe actif inversé n'est donc pas retenu à ce stade.
- Stretch axial médian `1.231`; 98.5 % des éléments s'allongent suivant l'axe
  global. La dilatation axiale est une réponse globale à la contraction
  principalement circonférentielle.
- Les champs nodaux fibre/feuillet sont unitaires et orthogonaux à `1.4e-6`
  près. Après moyenne DG0 indépendante par élément, `|f·s|` atteint `0.0444` :
  la base structurale utilisée par FEniCSx n'est plus exactement orthogonale.
- Le contrôle `min_J` échantillonne le Jacobien P2 en un seul point DG0 par
  cellule. La projection P1 finale présente 39 tétraèdres affinement inversés,
  représentant seulement `0.031 %` du volume de référence ; cela ne prouve
  pas une inversion du champ P2 mais montre que le garde-fou et le
  post-traitement n'évaluent pas la même représentation.

### Audit du générateur dit LDRB

- `_solve_apex_base` est une normalisation de la coordonnée globale `z`, pas
  une résolution de Laplace avec frontières apex/base.
- `_solve_endo_epi` normalise la distance au centre global, pas un potentiel
  Laplacien défini sur les surfaces endocardique et épicardique.
- L'algorithme est néanmoins étiqueté `LDRB-Bayer2012` dans les artefacts.

Cette divergence est établie dans le code et invalide la qualification LDRB
stricte. Son rôle causal dans l'allongement doit encore être isolé par un
benchmark à fibres imposées avant correction du générateur.

### Benchmarks à fibres imposées

#### Fibre axiale, base supérieure entièrement fixée

- Bloc `10 × 10 × 20 mm`, `T_act=5 kPa`.
- Convergence complète, `min_J=0.9579`.
- Longueur axiale `20.00 → 16.71 mm`, soit `-3.29 mm`.
- Face libre déplacée de `+3.92 mm` vers la base.

La tension active raccourcit correctement un bloc lorsque la fibre est
alignée avec l'axe normal à la base fixée. Un signe actif globalement inversé
est écarté.

#### Fibre transverse, même clamp basal complet

- Convergence complète, `min_J=0.9589`.
- Dimensions : x `+4.60 mm`, y `+3.22 mm`, z `+4.46 mm`.
- La face libre se déplace axialement de `-3.55 mm`, donc s'éloigne de la base.

Ce cas ne teste pas une traction transverse libre : le clamp basal bloque
aussi x et y sur toute la face supérieure. Il démontre néanmoins que la
combinaison « activation majoritairement transverse + base entièrement fixée
en trois composantes » peut reproduire l'allongement axial observé sur le
ventricule. La prochaine expérience discriminante doit remplacer le clamp
complet par des contraintes minimales supprimant uniquement les modes rigides,
sans changer la loi constitutive ni les fibres.

#### Expérience discriminante — contraintes basales minimales

Un second mode de bord diagnostique a été ajouté sans modifier le mode
historique par défaut : déplacement normal nul sur la base, plus trois
contraintes ponctuelles supprimant les translations et la rotation rigides.
La loi constitutive, la tension active, le maillage bloc et les fibres restent
strictement identiques.

- Fibres axiales `z`, appui minimal, `T=5 kPa` : convergence à charge complète,
  `min_J=0.9986`, raccourcissement selon les fibres `-4.877 mm`.
- Fibres transverses `x`, appui minimal, `T=5 kPa` : convergence à charge
  complète, `min_J=0.9986`, raccourcissement selon les fibres `-1.607 mm`.
- Le même cas transverse avec encastrement complet donnait au contraire un
  allongement `x` de `+4.599 mm`.

Le changement de signe sous la seule modification des conditions de bord
démontre que l'encastrement basal complet fausse causalement la réponse du cas
transverse. Cela ne prouve pas encore que cette correction suffit sur la
géométrie patient : la prochaine porte est un smoke test patient à faible
charge, avec comparaison des déplacements et du Jacobien avant tout recalcul
long à `30 kPa`.

#### Porte patient à faible charge — résultat

Le smoke test `patient001_coarse5_fixed`, sans pression, avec appui basal
minimal et une tension volontairement faible de `1 kPa`, a atteint la charge
complète :

- convergence : oui, `45` itérations, durée `2712.05 s` ;
- `min_J=0.7747`, aucune erreur de domaine ;
- variation de hauteur : `+7.945 mm` ;
- déplacement apical selon `z` : `-7.945 mm` ;
- déplacement maximal : `10.478 mm`.

Verdict : **validation numérique, rejet scientifique**. Le retrait de
l'encastrement basal complet corrige le bloc transverse contrôlé, mais ne
suffit pas à corriger l'allongement du patient. Il est donc interdit de lancer
le recalcul long à `30 kPa` avec cette seule modification. La prochaine
expérience doit isoler le champ de microstructure patient, en particulier la
construction pseudo-LDRB déjà identifiée (axe apex-base global et distance au
centre global), sans simultanément changer la loi active ou la pression.

Artefact :
`sprint_artifacts/sprint2/patient001_minimal_bc_tmax1.json`.

### Porte microstructure — audit anatomique

Le document de projet impose pour WP1 une génération de fibres rule-based
`LDRB`. Le champ actuellement nommé `LDRB-Bayer2012` ne résout aucun problème
de Laplace : sa coordonnée apex-base est le `z` global normalisé, sa coordonnée
transmurale est la distance normalisée au centroïde global, et les tags
d'éléments transmis au générateur ne sont pas utilisés.

L'audit sur les `2916` nœuds de `patient001_coarse5_fixed` compare ce proxy à
une coordonnée de distance physique construite depuis les labels de cavité
`RV=1`, `LV=3` et le fond extérieur de la segmentation :

- normes fibre/feuillet et orthogonalité : conformes numériquement, erreur
  maximale de l'ordre de `1.3e-6` ;
- corrélation proxy/coordonnée segmentée : `0.4950` ;
- erreur absolue moyenne de coordonnée transmurale : `0.2734` ;
- erreur absolue au 95e percentile : `0.6081` ;
- `49.45 %` des nœuds dépassent `0.25` d'erreur transmurale ;
- erreur d'angle hélicoïdal moyenne induite : `32.81°` ;
- `49.45 %` des nœuds dépassent `30°` d'erreur d'angle.

Verdict : la base structurale est algébriquement valide, mais le champ n'est
pas un LDRB anatomiquement validé et ne peut pas servir de référence WP1/WP2.
Le PDF ne précise ni la variante LDRB/Bayer, ni les conditions de bord
ventriculaires, ni les angles endo/épi. Conformément à la directive zéro
tâtonnement, aucune nouvelle microstructure ni simulation patient ne doit être
produite avant arbitrage explicite de cette spécification.

Artefact :
`sprint_artifacts/sprint2/patient001_microstructure_audit.json`.

### Génération Bayer 2012 autorisée

À la suite de l'arbitrage utilisateur, Bayer et al. (2012) devient la
spécification complémentaire officielle de la mention LDRB du document.
L'intégration utilise `fenicsx-ldrb==0.1.4`, version compatible avec
FEniCSx `0.7.3`, et `adios4dolfinx==0.7.3`. Les versions plus récentes ont été
écartées après preuve d'une dépendance à FEniCSx `>=0.10`.

Les surfaces ont été reconstruites déterministement depuis la segmentation et
les facettes extérieures du maillage : `BASE=277`, `RV=655`, `LV=2141`,
`EPI=2403`. Les Laplaciens scalaires et les règles d'orientation proviennent
du paquet de référence. Pour éviter la projection nodale dégénérée de cette
ancienne combinaison FEniCSx/ldrb, les gradients élémentaires sont moyennés
aux nœuds par le volume, sans epsilon ni remplacement arbitraire.

Le champ produit est fini sur les `2916` nœuds, avec erreurs maximales de
norme et d'orthogonalité inférieures à `1.3e-15`. Il est écrit dans un nouveau
fichier et n'écrase pas le champ historique :
`patient001_coarse5_fixed_fibers_bayer2012.lon`.

Comparaison avec le champ historique : écart d'axe fibre moyen `51.92°`,
médian `53.53°`, 95e percentile `86.63°`; `78.88 %` des nœuds dépassent
`30°`. Cette différence justifie un smoke test mécanique comparatif mais ne
prouve pas encore à elle seule la causalité de l'allongement.

Rapport :
`sprint_artifacts/sprint2/patient001_bayer2012_generation.json`.

#### Smoke test mécanique Bayer à faible charge — rejet

Le test comparatif a conservé le maillage, la loi constitutive, l'appui basal
minimal, la pression nulle et `T_max=1 kPa`; seul le champ historique a été
remplacé par le champ Bayer 2012.

Le solveur n'a accepté aucun palier : charge finale `0.0`, `0` itération
acceptée, `121` détections de sortie du domaine et durée `558.13 s`. Le
garde-fou PETSc 3.20 a correctement rejeté chaque essai avec Jacobien invalide,
sans enregistrer de déformation ni produire un état corrompu.

Verdict : **rejet numérique, aucune conclusion physiologique possible**. Le
champ Bayer est algébriquement orthonormal aux nœuds, mais son utilisation par
la mécanique doit être auditée au niveau élémentaire (norme après moyenne DG0,
continuité angulaire entre éléments et gradients nuls) avant une nouvelle
tentative. Aucun run `30 kPa` n'est autorisé.

Artefact :
`sprint_artifacts/sprint2/patient001_bayer2012_minimal_bc_tmax1.json`.

#### Cause élémentaire et correction de projection

L'audit de `9439` tétraèdres et `16140` paires voisines démontre que la
moyenne vectorielle nodale utilisée pour construire le champ DG0 n'est pas
invariante au signe des axes LDRB. Pour Bayer 2012, elle produit : norme fibre
minimale `0.0116`, `12.44 %` des éléments sous `0.5`, `65.60 %` des éléments
avec `|f·s|>0.1`, et un maximum `|f·s|=0.99998`. Le champ historique lisse ne
présentait pas ce défaut, ce qui expliquait pourquoi la branche était restée
invisible.

La projection DG0 moyenne désormais les tenseurs structuraux `f⊗f` et `s⊗s`,
extrait leur axe principal puis orthogonalise le feuillet par rapport à la
fibre. Cette opération est invariante aux retournements de signe, conformément
aux invariants de Holzapfel-Ogden et à la contrainte active `f⊗f`.

- Test explicite de retournement de signe : réussi.
- Tests microstructure ciblés : `4` réussis.
- Champ Bayer projeté : `9439/9439` éléments finis et unitaires,
  `max |f·s|=1.12e-15`.
- Régressions bloc : raccourcissement transverse `-1.607 mm`, axial
  `-4.877 mm`, charges complètes et `min_J=0.9986`.

Rapport d'audit :
`sprint_artifacts/sprint2/patient001_element_microstructure_audit.json`.

#### Smoke test Bayer corrigé à `1 kPa` — validation

Le smoke test patient a été relancé avec les mêmes paramètres que le test
historique à `1 kPa`, en remplaçant uniquement la microstructure par Bayer
2012 et sa projection DG0 tensorielle :

- charge complète atteinte, `41` itérations ;
- durée `2454.49 s` ;
- `min_J=0.74035`, aucune erreur de domaine ;
- variation de hauteur `-0.677 mm` ;
- déplacement apical selon `z` `+0.984 mm` vers la base ;
- déplacement maximal `7.607 mm`.

Le test comparable avec le champ historique donnait une variation de hauteur
`+7.945 mm` et un déplacement apical `-7.945 mm`. Le changement de signe sous
la correction isolée microstructure/projection établit causalement que le
champ pseudo-LDRB historique et sa moyenne vectorielle DG0 étaient
responsables de l'allongement axial anormal.

Verdict : **validation numérique de la porte faible charge**. Ce résultat ne
constitue pas encore une validation physiologique à `30 kPa` : le déplacement
maximal de `7.607 mm` à `1 kPa` doit être localisé et contrôlé avant tout run
long. Le prochain jalon est un post-traitement spatial du déplacement puis, si
aucune singularité locale n'est détectée, une montée de charge intermédiaire.

Artefact :
`sprint_artifacts/sprint2/patient001_bayer2012_dyadic_minimal_bc_tmax1.json`.

#### Localisation spatiale du déplacement à `1 kPa`

La répétition du cas a sauvegardé les champs nodaux complets. Le maximum de
`7.607 mm` se trouve au nœud `750`, près de l'apex, et est dominé par la
composante latérale `uy=-7.342 mm` (`ux=-0.837`, `uz=+1.806 mm`). Il n'est pas
isolé : les sept nœuds de sa couronne élémentaire sont compris entre `6.357`
et `7.607 mm`; six nœuds du maillage dépassent `7 mm`.

Les conditions de bord sont satisfaites à la précision machine : déplacement
de l'ancre A nul, `uy` de l'ancre B nul et `max |uz|` basal `7.7e-17 mm`.
L'amplitude ne provient donc pas d'un mode rigide non contraint. Un ajustement
global translation+rotation explique néanmoins une part importante de la
cinématique de flexion/torsion (`RMS 3.980 mm`), avec un résidu déformable
`RMS 1.258 mm`, maximum `3.772 mm`.

Verdict : pas de singularité nodale ni de violation des appuis, mais une
flexion/torsion globale importante à faible charge. La montée directe à
`30 kPa` reste refusée ; il faut d'abord arbitrer et valider la symétrie
anatomique des marqueurs LV/RV/EPI et des angles Bayer, puis effectuer une
charge intermédiaire bornée.

Champs :
`sprint_artifacts/sprint2/patient001_bayer2012_dyadic_tmax1_spatial.npz`.

#### Audit des marqueurs Bayer 2012

- Les 5 476 facettes de frontière ont été classées dans les quatre marqueurs
  requis : base 277, RV 655, LV 2 141 et épicarde 2 403.
- Chaque marqueur possède une composante principale très dominante. Les
  petites composantes isolées des surfaces RV, LV et épicardique sont
  conservées et documentées : les supprimer sans critère anatomique validé
  introduirait une correction arbitraire.
- Les distances médianes à la segmentation sont de 1,618 mm (RV), 1,407 mm
  (LV) et 1,088 mm (épicarde). Les 95e percentiles restent sous 4 mm.
- Les marges d'ambiguïté inférieures à 2,5 mm concernent 15,11 % des facettes
  RV, 13,50 % des facettes LV et 8,99 % des facettes épicardiques. Cette
  incertitude interdit une validation anatomique définitive, mais reste
  compatible avec un palier mécanique diagnostique borné.

Rapport :
`sprint_artifacts/sprint2/patient001_bayer_marker_audit.json`.

#### Décision de passage au palier 2 kPa

Le palier 2 kPa est autorisé uniquement comme diagnostic de progression. Il
doit sauvegarder les champs complets et sera arrêté comme non concluant en cas
de perte de convergence, de dégradation excessive de `min_J` ou de réponse
cinématique non proportionnelle au cas 1 kPa. Il ne constitue pas une charge
physiologique et n'autorise pas, à lui seul, le retour direct à 30 kPa.

#### Résultat du palier diagnostique 2 kPa

- Charge complète atteinte (`load_fraction=1`) en 53 itérations et
  3 139,56 s, sans erreur de domaine.
- Après deux rejets initiaux, la continuation a convergé jusqu'à 2 kPa ;
  `min_J=0,69247`, contre `0,74035` à 1 kPa.
- La hauteur diminue de 0,716 mm et l'apex remonte de 1,267 mm. Le déplacement
  maximal atteint 9,895 mm, contre 7,607 mm à 1 kPa.
- Le champ 2 kPa reste spatialement très aligné sur le champ 1 kPa : similarité
  cosinus `0,99976`, facteur global optimal `1,3294` et résidu relatif `2,20 %`.
  La réponse est donc cohérente en forme mais sous-linéaire en amplitude : le
  doublement de la tension ne double pas les déplacements.
- La baisse absolue de `min_J` est de 0,04788 (6,47 % relativement au cas
  1 kPa). Aucun seuil quantitatif de dégradation acceptable n'étant défini
  dans le document de projet, aucun palier supérieur n'est automatiquement
  autorisé sur ce seul résultat.

Rapport et champs :
`sprint_artifacts/sprint2/patient001_bayer2012_dyadic_tmax2_spatial.json` et
`sprint_artifacts/sprint2/patient001_bayer2012_dyadic_tmax2_spatial.npz`.

#### Audit spatial du jacobien — porte bloquante

- L'archive ne conserve que le déplacement aux sommets, alors que le solveur
  utilise un champ P2. Le jacobien reconstruit par tétraèdre est donc un
  diagnostic affine P1 et ne reproduit pas exactement le `J` P2 déclaré par
  le solveur.
- Cette reconstruction détecte néanmoins 11 tétraèdres non positifs à 1 kPa
  et 16 à 2 kPa. Les zones critiques sont très stables entre les deux charges :
  corrélation des `J` `0,99697` et recouvrement de 93,68 % du percentile le
  plus faible.
- L'anomalie n'est pas principalement localisée aux ancrages : pour le 1 %
  le plus faible à 2 kPa, la distance médiane au plus proche ancrage est de
  49,49 mm et seulement 13,68 % des éléments sont dans la couche basale.
- Le maillage de référence contient des tétraèdres très aplatis : volume
  minimal `0,0121 mm³`, conditionnement maximal `737,62`. Parmi les 16
  éléments P1 non positifs à 2 kPa, le conditionnement médian vaut `97,42` et
  81,25 % dépassent 50, contre 0,67 % sur le maillage complet.
- Le garde-fou actuel interpole `J` dans un espace DG0, donc en un seul point
  par élément. Il peut déclarer un pas admissible sans contrôler les extrema
  du champ P2 dans l'élément. Le `min_J=0,69247` du run n'est donc pas une
  preuve suffisante d'absence d'inversion locale.

Décision : le palier 3 kPa est bloqué. La prochaine action doit renforcer la
qualité du maillage et/ou l'évaluation de `J` à plusieurs points dans chaque
élément, puis répéter les diagnostics 1 et 2 kPa. Il est interdit d'interpréter
les inversions P1 comme une mesure clinique ; elles constituent une alerte de
validité numérique.

Rapport :
`sprint_artifacts/sprint2/patient001_bayer2012_jacobian_spatial_audit.json`.

#### Renforcement du garde-fou de jacobien

- Le contrôle DG0 à un seul barycentre a été remplacé par une grille
  tétraédrique équidistante d'ordre 3, soit 20 évaluations de `J` par élément,
  incluant les quatre sommets, les arêtes et l'intérieur.
- Ce choix correspond au degré polynomial maximal de `det(F)` lorsque le
  déplacement est P2 (`grad(u)` P1, déterminant cubique). Il améliore fortement
  la détection, sans constituer à lui seul une preuve analytique de positivité
  entre les points échantillonnés.
- Le même minimum multipoint est désormais utilisé pendant les évaluations
  SNES et lors de l'acceptation de chaque palier de continuation.
- Trois tests unitaires ciblés passent. Les deux benchmarks FEniCSx passent
  également : bloc fibres z, `min_J=0,63571`, raccourcissement `-3,288 mm` ;
  bloc fibres x avec appui minimal, `min_J=0,99864`, raccourcissement
  `-1,607 mm`.

Porte suivante : répéter d'abord le patient à 1 kPa avec ce nouveau garde-fou.
Un échec est une information de validation attendue et interdit le cas 2 kPa ;
une convergence devra encore être confrontée à l'audit spatial.

#### Revalidation patient 1 kPa avec garde-fou multipoint

- Le patient atteint la charge complète en 41 itérations et 2 461,23 s.
- Le champ mécanique final est identique au run précédent : hauteur
  `-0,6772 mm`, apex `+0,9839 mm`, déplacement maximal `7,6068 mm`.
- Le minimum multipoint vaut `0,44637`, nettement inférieur au minimum DG0
  historique `0,74035`, mais reste positif et au-dessus du seuil solveur
  existant `j_min_accept=0,1`.
- Onze erreurs de domaine ont été détectées pendant les essais Newton rejetés.
  Le solveur a réduit les pas, restauré le dernier état accepté, puis convergé.
- La reconstruction P1 depuis les seuls sommets reste inchangée et ne doit pas
  être confondue avec le champ P2 : la dérivée P2 aux sommets dépend aussi des
  degrés de liberté d'arête, absents des archives. Les valeurs P1 négatives ne
  prouvent donc pas une inversion du champ P2 final.

Décision : le garde-fou renforcé remplit sa fonction à 1 kPa. La répétition du
palier 2 kPa avec ce même contrôle est techniquement autorisée comme diagnostic
borné ; le remaillage reste un chantier nécessaire en raison des tétraèdres
très mal conditionnés, indépendamment de cette autorisation.

Rapport et champs :
`sprint_artifacts/sprint2/patient001_bayer2012_multipoint_tmax1.json` et
`sprint_artifacts/sprint2/patient001_bayer2012_multipoint_tmax1.npz`.

#### Revalidation patient 2 kPa avec garde-fou multipoint

- Charge complète atteinte en 53 itérations et 3 066,04 s.
- Le minimum multipoint final vaut `0,38302`, contre `0,44637` à 1 kPa et
  `0,69247` avec l'ancien contrôle DG0 à 2 kPa.
- Douze erreurs de domaine ont été détectées pendant les essais Newton rejetés.
  Un essai initial a atteint `J=-2,7588` et un autre `J=0,0789` ; aucun de ces
  états n'a été accepté. La continuation réduite a ensuite convergé.
- Le champ final est bit à bit identique à l'ancien résultat 2 kPa
  (`max_abs_field_difference_mm=0`) : hauteur `-0,7156 mm`, apex
  `+1,2668 mm`, déplacement maximal `9,8950 mm`.
- Entre 1 et 2 kPa, le minimum multipoint baisse de `0,06335`, soit 14,19 %
  relativement au cas 1 kPa, tout en restant positif et supérieur au seuil
  solveur existant `0,1`.

Décision : les cas bornés 1 et 2 kPa sont revalidés avec le garde-fou renforcé.
Le palier 3 kPa reste suspendu jusqu'au traitement du chantier de qualité du
maillage ; la convergence seule ne justifie pas d'extrapoler vers les charges
physiologiques.

Rapport et champs :
`sprint_artifacts/sprint2/patient001_bayer2012_multipoint_tmax2.json` et
`sprint_artifacts/sprint2/patient001_bayer2012_multipoint_tmax2.npz`.

### Chantier qualité du maillage — état des lieux

- Le maillage `patient001_coarse5_fixed` contient 2 916 nœuds et 9 439
  tétraèdres, tous orientés positivement au repos.
- Le volume minimal vaut `0,01213 mm³`, l'angle dièdre minimal `0,1365°` et
  le conditionnement maximal `737,62`.
- Le critère de 15° déjà implémenté dans `app/core/units.py` est violé par
  1 203 éléments (12,74 %). L'union avec le diagnostic de conditionnement
  supérieur à 50 contient 1 205 éléments : 664 de frontière et 541 intérieurs.
- Un filtrage direct est interdit : supprimer les éléments intérieurs créerait
  des cavités et supprimer les éléments de frontière modifierait l'anatomie.
- Une surface candidate a été reconstruite séparément depuis la segmentation.
  Aucun fichier du maillage de référence n'a été modifié.
- La régénération candidate a échoué avant production de tétraèdres. Le dépôt
  déclare Gmsh 4.12 dans `MeshingParameters`, tandis que l'image Airflow locale
  expose Gmsh 4.8. Cette version échoue sur la reparamétrisation de la surface
  puis sur la stratégie discrète multicoque (`No tetrahedra in region`).

Décision : construire une image de maillage dédiée et figée sur Gmsh 4.12,
puis régénérer une variante séparée optimisée Netgen. L'adoption restera
conditionnée à une comparaison géométrique et aux métriques de qualité.

Rapport :
`sprint_artifacts/sprint2/patient001_coarse5_fixed_rest_mesh_quality.json`.

#### Variante Gmsh 4.12.2 + Netgen — rejetée

- Une image dédiée reproductible `cdt-mesher:4.12.2-amd64` a été construite.
  L'architecture amd64 est requise car la roue Gmsh 4.12.2 n'existe pas en
  ARM64 sur l'index utilisé.
- Le masque contient quatre composantes myocardiques de 7 183, 209, 13 et
  1 voxels. Conformément à la règle annoncée dans les scripts de régénération,
  seule la composante principale est conservée pour la variante ; les fichiers
  de référence restent inchangés.
- La reparamétrisation échoue encore (`Wrong topology ...`). La stratégie
  discrète de repli réussit et produit 3 162 nœuds et 10 638 tétraèdres, puis
  l'optimisation Netgen élimine ses éléments déclarés illégaux.
- La géométrie globale est préservée : mêmes bornes, volume `163255,80 mm³`
  contre `163299,99 mm³` (-0,027 %). Les distances nodales de frontière au
  95e percentile sont inférieures à `0,00013 mm`; l'écart maximal de 5,45 mm
  correspond aux fragments exclus.
- La qualité reste insuffisante. Le taux sous 15° baisse de 12,74 % à 9,97 %,
  mais le conditionnement maximal augmente de 737,62 à 974,07, les éléments
  au-dessus de 50 passent de 63 à 141 et l'arête minimale chute de 0,301 mm à
  0,0167 mm. Le candidat est donc rejeté et ne recevra ni fibres ni simulation.

Une voie historique présente dans `scripts/generate_meshes_acdc.py` utilise
PyMeshFix puis TetGen avec `mindihedral=5`, `minratio=2` et `quality=True`.
Après relecture du document de projet, elle n'est pas retenue pour ce sprint :
la chaîne de référence indiquée pour « Meshing & fibres » est Gmsh, meshtool et
LDRB. Le document exige également des environnements conteneurisés et figés,
un maillage simulation-ready et un contrôle qualité. TetGen/PyMeshFix ne seront
donc pas introduits pour contourner le défaut courant.

La suite reste dans la chaîne documentaire : Gmsh 4.12.2 figé, correction de
la surface d'entrée, régénération volumique séparée, puis contrôle qualité de
type meshtool avant toute adoption. L'image openCARP locale ne contient pas le
binaire `meshtool`; son ajout devra lui aussi être versionné explicitement.

Rapports :
`sprint_artifacts/sprint2/mesh_candidate/patient001_coarse5_netgen_quality.json`
et
`sprint_artifacts/sprint2/mesh_candidate/patient001_mesh_geometry_comparison.json`.

### Chantier parallèle 1A — Validation et reproductibilité

#### État des lieux initial

- Le run mécanique sauvegarde un checkpoint et des champs finaux, mais ne
  produit pas encore un manifeste cryptographique complet des entrées.
- `EndoCavityVolume` est présent et son historique annonce une validation sur
  quatre patients, mais cette preuve n'est pas générée automatiquement dans
  les artefacts du sprint.
- Le post-traitement Sprint 1 existe, mais son rapport ne contient pas encore
  les hashes du maillage, de la microstructure et du code solveur.
- Ce chantier ne modifie ni `fenicsx_solver.py`, ni le checkpoint actif, ni les
  paramètres physiques du run.

#### Objectif

Rendre le résultat du Sprint 1 traçable et reproductible, puis régénérer la
preuve quantitative de `EndoCavityVolume` sur les quatre patients disponibles.

#### Livrables et critères d'acceptation

1. Manifeste JSON contenant commit Git, état dirty, paramètres figés, versions
   et SHA-256 des entrées scientifiques et du solveur.
2. Validation cohorte de `EndoCavityVolume` sur patient001, patient002,
   patient005 et patient008, avec volumes maillage/GT et erreurs relatives.
3. Rapport final Sprint 1 référençant le manifeste et les résultats de volume.
4. Tests automatisés des fonctions de hash/manifeste.
5. Aucun accès en écriture aux checkpoints du calcul actif.

#### Résultats du chantier parallèle 1A

- Manifeste généré : `sprint_artifacts/sprint1/manifest.json`.
- Le manifeste contient le commit Git, l'état dirty, les paramètres figés,
  les versions hôte et les SHA-256 du maillage, de la microstructure LDRB, de
  la segmentation, du solveur et des scripts d'exécution/post-traitement.
- Validation `EndoCavityVolume` régénérée sur quatre patients :
  - patient001 : erreur EDV `+0.414 %` ;
  - patient002 : erreur EDV `-3.401 %` ;
  - patient005 : erreur EDV `-1.885 %` ;
  - patient008 : erreur EDV `-3.912 %`.
- Erreur relative absolue moyenne : `2.403 %` ; maximum : `3.912 %`.
- Rapport : `sprint_artifacts/sprint1/endo_cavity_cohort.json`.
- Tests automatisés après ajout de la traçabilité : 43 réussis.
- Checkpoint mécanique actif non modifié par ce chantier.

### Chantier parallèle 1B — Hygiène et inventaire du dépôt

#### État des lieux initial

- La branche locale `main` est en avance de huit commits sur `origin/main`.
- L'arbre contient simultanément des modifications produit, l'infrastructure
  du Sprint 1, des artefacts scientifiques et de nombreux diagnostics
  historiques non suivis.
- L'origine de chaque fichier historique n'est pas suffisamment établie pour
  autoriser un déplacement, une suppression ou un commit global.
- Aucun nettoyage destructif, commit ou push n'est autorisé dans ce chantier.

#### Objectif et décision

Établir un inventaire classé permettant de préparer plus tard des commits
atomiques et vérifiables, tout en conservant intégralement l'état de travail.
L'inventaire est publié dans
`sprint_artifacts/sprint1/repository_inventory.md`. Le classement ne vaut ni
validation scientifique des diagnostics historiques, ni autorisation de les
versionner.

### Chantier parallèle 1C — Contrat scientifique du DoE

#### État des lieux initial

- Le DoE est une dépendance aval du Sprint 1 et ne doit pas être lancé avant
  validation de la mécanique orthotrope.
- Le code et sa documentation ne décrivent pas la même dixième variable :
  `C_a` dans le texte, `sigma_n` dans `PARAM_NAMES`.
- La plage codée de `T_max`, `[80, 200] kPa`, exclut la référence du Sprint 1
  à `30 kPa`. Le document de projet ne fournit pas de bornes numériques
  permettant d'arbitrer cette incohérence.
- La stratégie dite imbriquée marque certains points haute fidélité sans
  garantir leur évaluation appariée en basse fidélité au même point, alors
  que l'estimation d'un écart multi-fidélité exige cette paire.
- La tâche DoE peut substituer des tableaux synthétiques lorsque des entrées
  physiques manquent, ce qui est incompatible avec un jeu d'entraînement
  scientifique traçable.

#### Objectif et décision

Formaliser les prérequis, entrées, sorties, exclusions et décisions encore
requises avant toute exécution. Le contrat est publié dans
`sprint_artifacts/sprint1/doe_contract.md`. Les bornes et la dixième variable
restent explicitement non validées : aucune correction de code n'est faite
par supposition.

### Chantier parallèle 1D — Consolidation des entraînements WP1

#### État des lieux initial

- Le dépôt contient les scripts Kaggle/Colab ACDC et M&Ms ainsi que plusieurs
  checkpoints locaux nommés ACDC.
- `cdt_status_strict.md` annonce Dice MYO ACDC `0.910` et Dice M&Ms `0.908`,
  mais le seul rapport ACDC local est une exécution du `DemoSegmenter` et ne
  constitue pas la preuve du modèle réel.
- Aucun notebook, journal complet ou checkpoint explicitement M&Ms n'était
  initialement archivé dans le dépôt.
- Ce chantier audite les preuves existantes sans relancer d'entraînement et
  sans modifier les poids.

#### Preuve M&Ms récupérée

- Notebook source : `Untitled1.ipynb`, fourni le 19 juillet 2026 ; SHA-256
  `99d4c13a5f23ec67e31523862edb4f3f06f43ec4e24a651e4b015d65f273127f`.
- Dataset nnU-Net : `Dataset028_MnMs`, 300 cas vérifiés par nnU-Net.
- Configuration : nnU-Net v2 officiel, `3d_fullres`, fold 0, batch 3,
  patch `[14, 320, 320]`, spacing `[8.0, 1.25, 1.25]`, GPU A100.
- Entraînement terminé à l'epoch 999.
- Meilleur EMA pseudo-Dice consigné : `0.9186000`.
- Validation finale terminée : Dice moyen `0.9082657921`.
- L'archive `/content/nnunet_mnms_results.zip` a été créée et téléchargée
  depuis Colab, mais elle n'est pas présente dans le dépôt local audité.

#### Limites et décision

- La preuve porte sur le fold 0, pas sur une validation croisée cinq folds.
- Le notebook installe `nnunetv2` sans version figée et signale un conflit de
  dépendance NumPy/Numba ; l'environnement n'est donc pas encore reproductible.
- Les métriques finales par classe et le checkpoint M&Ms doivent être extraits
  de l'archive de résultats.
- Le score M&Ms `0.908` est désormais confirmé comme Dice moyen de validation
  interne du fold 0 ; il ne doit pas être qualifié de généralisation externe.
- ACDC a été entraîné sur Kaggle à partir du dataset publié
  `amadouadjanouhoun/acdc-preprocessed`. L'archive corrigée
  `nnunet_acdc_nifti_results.zip` contient les deux checkpoints, le journal,
  les prédictions et le résumé de validation sur 40 cas : Dice moyen
  `0.920472`, RV `0.903975`, MYO `0.910375`, LV `0.947065`. Le SLO MYO 0.90
  est donc prouvé sur la validation du fold 0.
- Rapport détaillé : `sprint_artifacts/sprint1/wp1_training_audit.md`.

## Préparation du Sprint 2 — Validation hémodynamique et couplage

### État des lieux initial

- Le document de projet exige un solveur couplé
  `EP ↔ mécanique ↔ hémodynamique` et un Windkessel fermé 0D–3D.
- `WindkesselSolver` implémente l'ODE
  `dP/dt = (Q - P/R_p) / C_a` avec RK4, valve aortique unidirectionnelle et
  cinq cycles par défaut.
- Les unités sont explicites : `R_p` en `Pa·s/m³`, `C_a` en `m³/Pa`, `Z_c`
  en impédance caractéristique, pression convertie par `133.322 Pa/mmHg`.
- Les valeurs nominales sont déclarées comme calibrées pour environ 120/80
  mmHg, mais leur provenance et leur calibration ne sont pas documentées.
- Le débit cardiaque est calculé avec un facteur `÷1000` surnuméraire : pour
  80 mL et 75 bpm, le code retourne `0.006 L/min` au lieu de `6 L/min`.
- Le test accepte actuellement `0.001–15 L/min` et ne détecte donc pas cette
  erreur d'échelle.
- La mécanique est appelée une seule fois sans pression. La boucle de point
  fixe Windkessel multiplie ensuite toute la waveform de volume par un facteur
  empirique Frank-Starling ; la pression n'est pas appliquée comme charge à
  la mécanique. Le couplage bidirectionnel D2.1 n'est donc pas établi.
- L'EF est invariant sous la multiplication uniforme EDV/ESV, ce qui peut
  provoquer une convergence apparente de la boucle sans retour mécanique.
- Les tests unitaires vérifient principalement de larges plages et peuvent
  utiliser les solveurs de repli ; ils ne constituent pas une validation
  physiologique du couplage réel.

### Décision avant implémentation

Le Sprint 2 ne commencera qu'après clôture du Sprint 1. Il devra d'abord
corriger et tester les unités, figer le contrat de waveform, puis démontrer un
retour de pression réel vers la mécanique sur un cas contrôlé. Aucun DoE
hémodynamique ne sera lancé avant ces portes de validation.

Rapport détaillé : `sprint_artifacts/sprint2/initial_state.md`.

### Préparation du Sprint 2 — Audit contrôlé du remaillage Mmg

#### Référence et porte d'acceptation

- La métrique appliquée est exactement celle consignée dans
  `CDT_JOURNAL_TECHNIQUE.md` et dans `diag_stiffness.py` :
  `abs(volume)/(mean_edge_length**3/6)`.
- La porte imposée est : qualité minimale strictement supérieure à `0.1` et
  aucun tétraèdre sous `0.05`.
- Le maillage de référence `patient001_coarse5_fixed` échoue : minimum
  `0.002390676`, 119 tétraèdres sous `0.05` et 434 inférieurs ou égaux à
  `0.1`.

#### Candidat Mmg reproductible

- Mmg 5.8.0 a été construit depuis le tag officiel `v5.8.0`, commit
  `4d8232c8aebfed877935d75d4d4a67e850962422`, au moyen de
  `docker/Dockerfile.mmg`.
- La commande contrôlée utilise l'optimisation d'un maillage existant et
  interdit toute adaptation de surface : `mmg3d_O3 -optim -nosurf`.
- Le candidat comporte 3 783 nœuds et 14 041 tétraèdres. Il conserve les
  2 625 nœuds de frontière, les 5 476 faces de frontière, les bornes et le
  volume total de `163299.99156069595 mm³`; les distances entre nœuds de
  frontière correspondants sont nulles.
- La qualité progresse : minimum `0.018897953`, percentile 1 `0.090398432`,
  percentile 5 `0.202824572` et médiane `0.514939516`.
- Il reste 29 tétraèdres sous `0.05` et 175 inférieurs ou égaux à `0.1`.
  Parmi eux, respectivement 24 sur 29 et 146 sur 175 touchent la frontière.

#### Décision

Le candidat Mmg est **rejeté** comme maillage de production car il ne franchit
pas la porte d'acceptation du journal. Il est conservé uniquement comme preuve
diagnostique d'une amélioration importante sans modification de frontière.
Aucune fibre n'est transférée et aucune simulation mécanique n'est lancée sur
ce candidat. La majorité des défauts résiduels étant adjacente à la frontière,
la prochaine variante nécessiterait une adaptation de surface et donc une
tolérance géométrique explicite, absente du document et du journal; elle ne
sera pas inventée.

Rapports :

- `sprint_artifacts/sprint2/patient001_coarse5_fixed_rest_mesh_quality.json` ;
- `sprint_artifacts/sprint2/mesh_candidate/patient001_coarse5_mmg_optim_quality.json` ;
- `sprint_artifacts/sprint2/mesh_candidate/patient001_coarse5_mmg_optim_geometry.json`.

### Arbitrage de la voie « filtrage sliver plus agressif »

Le journal mentionne cette voie comme alternative au remaillage, mais ses
propres diagnostics établissent que le filtre historique `h_min >= 0.3 mm`
ne détecte pas les tétraèdres plats. Le dépôt confirme que
`CardiacMesher._filter_slivers` supprime uniquement les éléments possédant une
arête sous ce seuil. Un filtrage direct selon la qualité normalisée a donc été
évalué **sans produire de maillage filtré** :

- sur la référence, supprimer les 119 éléments sous `0.05` ferait disparaître
  99 tétraèdres de frontière, exposerait 283 faces auparavant intérieures et
  ferait perdre 149 faces de la frontière originale ;
- sur le candidat Mmg, supprimer les 29 éléments sous `0.05` ferait disparaître
  24 tétraèdres de frontière, exposerait 52 faces auparavant intérieures et
  ferait perdre 24 faces de la frontière originale ;
- le seuil `<= 0.1` détériore davantage la frontière dans les deux cas.

Décision : la suppression directe de slivers est rejetée, car elle crée des
trous ou recule localement la frontière anatomique. L'expression « filtrage
plus agressif » du journal ne constitue pas une autorisation de modifier la
topologie. Aucun élément n'a été supprimé.

### Alignement de version Gmsh

La variante exploratoire Gmsh 4.12.2 reste archivée comme essai rejeté, mais
elle n'est pas la chaîne de référence. Le journal fixe Gmsh `4.15.2`, version
présente dans `.venv`, et documente déjà `Mesh.Optimize=1` dans le mailleur
mesh-based. Le maillage `patient001_coarse5_fixed` provient précisément du
chemin `target_mm=5.0`, Gmsh mesh-based, optimisation Gmsh et filtre historique
`h_min=0.3 mm`. Réexécuter cette même chaîne ne constitue donc pas une nouvelle
méthode de correction de qualité.

En conséquence, les deux branches explicitement citées par le plan du journal
ont été évaluées : optimisation Mmg sans adaptation de surface et filtrage
renforcé sans remaillage. Aucune ne franchit la porte tout en préservant la
géométrie et la topologie. Une nouvelle variante ne sera pas inventée sans
spécification documentaire supplémentaire.

### Sprint 2 — Jalon 2A : contrat d'unité du débit cardiaque

#### État des lieux

`WindkesselSolver` calculait le débit à partir du volume d'éjection en mL et
de la fréquence en battements par minute avec deux divisions par 1 000. Le
résultat était donc mille fois trop faible. Le test annoncé comme plage
physiologique `4–8 L/min` acceptait en réalité toute valeur entre `0.001` et
`15 L/min` et ne détectait pas l'erreur.

#### Correction et validation

- Le contrat dimensionnel est désormais explicite :
  `CO[L/min] = SV[mL/battement] × HR[battements/min] / 1000`.
- La seconde division par 1 000 a été supprimée.
- Le test exige effectivement une valeur entre `4` et `8 L/min` et vérifie
  l'égalité avec `SV × HR / 1000`, avec une tolérance cohérente avec
  l'arrondi de la sortie à deux décimales.
- Tests Windkessel ciblés : **6 réussis**, 23 non sélectionnés.

Ce jalon corrige uniquement le contrat d'unité. Il ne valide ni les paramètres
Windkessel nominaux, ni la physiologie de la boucle pression-volume, ni le
couplage bidirectionnel avec la mécanique.

### Sprint 2 — Jalon 2B : contrat de waveform ventriculaire

#### État des lieux

- `WindkesselSolver.simulate` acceptait toute longueur de tableau, des valeurs
  non finies ou non positives et une waveform sans éjection.
- En l'absence de waveform, le solveur autonome produit un cycle synthétique
  à partir de `V_ed_mL` et `V_es_mL`; ce mode reste limité aux tests isolés du
  modèle 0D.
- Dans le chemin couplé, `_compute_volume_waveform` produisait également une
  EDV et une EF synthétiques lorsque la mécanique était absente ou en échec.
- Un résultat mécanique pouvait être traité comme réel sur le seul booléen
  `converged`, même avec `load_fraction=0` et `min_J=0`.
- La mise à jour dite Frank-Starling multiplie uniformément toute la waveform.
  Elle ne transmet aucune pression au solveur mécanique et ne constitue donc
  pas le couplage bidirectionnel requis.

#### Contrat implémenté

- Une waveform fournie au Windkessel représente exactement un cycle cardiaque,
  avec un nombre d'échantillons déterminé par `BCL=60000/HR` et `dt_ms`.
- `heart_rate_bpm` et `dt_ms` doivent être finis et strictement positifs;
  `n_cycles` doit être un entier strictement positif.
- Le volume est un vecteur 1D d'au moins trois points, fini, strictement
  positif et vérifiant `EDV > ESV`.
- Le générateur autonome utilise désormais la même grille temporelle
  semi-ouverte `arange(0, BCL, dt_ms)` que le contrat.
- Le chemin couplé refuse désormais de fabriquer une waveform si la mécanique
  ne vérifie pas simultanément `converged=True`, `load_fraction=1.0`,
  `min_J>0.1` et un déplacement nodal fini de la bonne dimension.
- Le repli synthétique mécanique/cavité a été retiré de ce chemin. Aucun
  résultat hémodynamique ni vecteur DoE n'est publié en cas de refus.

#### Validation

- Tests ciblés Windkessel et couplage : **15 réussis**.
- Suite Phase 03 complète : **32 réussis, 3 échoués**.
- Les trois échecs restants sont tous dans `TestEPSolver` : l'environnement
  détecte le vrai binaire openCARP sur le cube minimal, tandis que ces tests
  étaient écrits pour le fallback analytique. Le cas n'active qu'un nœud,
  conserve sept temps d'activation à `-1`, donne `CV=0` et ne passe pas le
  benchmark. Ces échecs sont indépendants du contrat waveform et ne sont pas
  masqués.

#### Porte suivante

Le Windkessel possède maintenant une entrée contrôlée, mais la boucle reste
unidirectionnelle `mécanique -> volume -> pression`. La prochaine étape du
journal demeure bloquée par la porte maillage : appliquer la pression au
solveur mécanique exigerait d'abord un maillage conforme au seuil qualité.
La multiplication uniforme Frank-Starling ne sera pas présentée comme une
solution de remplacement.

### Sprint 2 — Jalon 2C : contrat du backend EP

#### État des lieux

`OpenCARPSolver` sélectionnait automatiquement le vrai binaire lorsqu'il était
présent et le fallback analytique sinon. Les tests intitulés « fallback » ne
figeaient pas ce choix : leur résultat dépendait donc de la machine. De plus,
une erreur d'exécution du vrai binaire était interceptée puis transformée en
résultat analytique, y compris dans le chemin couplé.

#### Correction

Trois modes sont maintenant explicites :

- `fallback` : modèle analytique réservé aux tests et au développement ;
- `opencarp` : backend scientifique strict, binaire obligatoire et propagation
  de toute erreur d'exécution ;
- `auto` : comportement historique conservé uniquement pour compatibilité.

Le mode par défaut est `opencarp`; `auto` doit être demandé explicitement.

Les tests unitaires EP utilisent explicitement `fallback` et ne dépendent plus
de la présence locale du binaire. `CoupledSolver` construit désormais le
backend `opencarp` strict : aucun résultat EP analytique ne peut alimenter la
mécanique, le Windkessel ou le DoE par bascule silencieuse.

#### Validation

- Le backend inconnu est refusé.
- Un échec openCARP contrôlé en mode strict est bien propagé.
- Les anciens contrôles fallback redeviennent déterministes.
- Suite Phase 03 complète : **37 tests réussis, 0 échec**.

Ce jalon valide la séparation des backends, pas une nouvelle validation
scientifique openCARP sur le patient. La preuve patient réelle reste celle déjà
consignée dans le journal; le cube minimal n'est pas utilisé comme benchmark
d'intégration openCARP.

### Sprint 2 — Jalon 2D : retrait de la fausse boucle de point fixe

#### État des lieux

La mécanique était exécutée une seule fois sans pression. Le Windkessel
multipliait ensuite uniformément la waveform par un facteur empirique dépendant
de la pression moyenne. Cette transformation conserve l'EF, n'applique aucune
traction endocardique et pouvait donc annoncer une convergence en deux
itérations sans retour de pression vers FEniCSx.

#### Correction

- La boucle `MAX_ITER/TOL_EF` et `_update_volume_waveform` ont été supprimées.
- Le Windkessel est exécuté exactement une fois sur la waveform mécanique
  acceptée.
- Le résultat déclare explicitement
  `coupling_mode=unidirectional_ep_mechanics_windkessel` et
  `pressure_feedback_applied=false`.
- `convergence_iterations` reste à zéro : aucune convergence de couplage n'est
  revendiquée.
- Le benchmark global reste faux tant qu'un vrai retour de pression n'est pas
  appliqué et validé, même si les sous-composants EP et Windkessel passent leurs
  contrôles propres.
- `to_doe_row` refuse désormais tout export si le retour de pression ou les
  benchmarks globaux manquent. Une exécution diagnostique unidirectionnelle ne
  peut donc pas alimenter silencieusement un entraînement.

#### Validation

- Un test instrumenté confirme un seul appel Windkessel.
- Il confirme l'absence de retour de pression, l'absence d'itération de
  convergence et le refus de l'export DoE.
- Suite Phase 03 complète : **38 tests réussis, 0 échec**.

Le code reflète maintenant l'état scientifique réel du projet : chaîne
unidirectionnelle opérationnelle, couplage bidirectionnel D2.1 non validé et
DoE bloqué jusqu'à franchissement de la porte maillage puis validation de la
pression endocardique.

## Sprint de consolidation — 20 juillet 2026

### État des lieux

- Les anciens fichiers `cdt_status_strict.md` et `reports/phase02_status.md`
  contenaient des pourcentages et conclusions antérieurs aux portes
  scientifiques des Sprints 1 et 2.
- Pytest collectait aussi trois scripts diagnostiques sous `scripts/`, dont un
  dépendait du chemin Docker `/cdt`; la suite s'arrêtait donc avant les tests.
- Après limitation de la collecte à `tests/`, quatre attentes obsolètes sont
  apparues : deux anciennes propriétés de configuration openCARP et deux tests
  API sans maillage ni isolation du solveur physique.
- Le dépôt reste volontairement non nettoyé, avec de nombreux diagnostics et
  artefacts dont la provenance doit être conservée.

### Actions et résultats

- Statut officiel créé : `CDT_ETAT_AVANCEMENT_2026-07-20.md`.
- Anciens statuts marqués comme documents historiques obsolètes.
- `pytest.ini` limite la suite officielle au répertoire `tests/` sans supprimer
  les diagnostics.
- Les tests openCARP vérifient désormais `dt_us=20` et la syntaxe actuelle
  `stim[0]`, conformes au journal.
- Les tests API utilisent un maillage temporaire et un solveur simulé
  explicitement; ils vérifient le contrat HTTP sans prétendre valider la
  physique.
- Suite globale : **140 tests réussis, 10 ignorés, 0 échec**, 29 avertissements
  de dépréciation.
- Les tests ignorés dépendent d'artefacts openCARP, DoE, GP ou Sobol absents ou
  non validés; ils ne sont pas comptés comme preuves réussies.
- Classement du dépôt et plan de commits publiés sous
  `sprint_artifacts/consolidation/`.

### Décision

Aucun commit ni push n'est effectué automatiquement. Les changements doivent
être revus par lot atomique. Les maillages candidats et diagnostics historiques
restent sur disque; aucune suppression n'est autorisée par ce sprint.
