# CDT — État d'avancement consolidé

Date de référence : 20 juillet 2026.

Sources d'arbitrage : `heart_digital_twin_proposal.pdf`,
`CDT_JOURNAL_TECHNIQUE.md`, preuves sous `sprint_artifacts/` et résultats de
tests reproductibles. En cas de conflit, un ancien pourcentage d'avancement ne
prime pas sur une porte scientifique mesurée plus récente.

## Verdict exécutif

Le dépôt contient une chaîne technique avancée, mais le jumeau numérique
cardiaque complet n'est pas scientifiquement validé. Le chemin actuellement
établi est unidirectionnel : `EP -> mécanique -> Windkessel`. Le retour de
pression vers la mécanique, le DoE physique et les livrables aval restent
bloqués par la qualité du maillage et par le rejet physiologique de la réponse
mécanique à 30 kPa.

## État par lot

| Lot | Preuve actuelle | Verdict |
|---|---|---|
| WP1 / ACDC | Fold 0, 40 cas, Dice moyen 0.920472, MYO 0.910375 | SLO MYO atteint sur validation interne fold 0 |
| WP1 / M&Ms | Entraînement fold 0 terminé, Dice moyen 0.908266 | Preuve notebook; archive, checkpoint et métriques par classe manquants localement |
| WP1 / EMIDEC | Résultats historiques insuffisants dans l'ancien statut | Non revalidé dans les sprints actuels |
| D1.2 maillage | Référence : qualité min 0.002391, 119 tets sous 0.05 | Porte du journal échouée |
| D1.2 Mmg | Qualité min 0.018898, 29 tets sous 0.05, géométrie conservée | Candidat amélioré mais rejeté |
| D2.1 EP | openCARP strict par défaut; fallback explicite test/dev | Contrat logiciel validé; benchmark patient formel restant |
| D2.1 mécanique | Charge 1.0, min_J 0.570101, aucune erreur de domaine | Convergence numérique, rejet physiologique |
| D2.1 Windkessel | Unités de débit et contrat waveform corrigés | Fonctionnel isolément; validation physique complète restante |
| D2.1 couplage | Un passage EP -> mécanique -> Windkessel, sans retour pression | Bidirectionnel non validé |
| DoE physique | Export refusé sans retour de pression et benchmarks | Bloqué; anciens runs fallback invalides |
| WP3 à WP5 | Prototypes ou infrastructures dépendant du DoE réel | Non validés ou bloqués |
| WP6 | Stack et prototypes partiels | Non audité à nouveau dans les sprints actuels |

## Résultat mécanique de référence

Patient `patient001_coarse5_fixed`, Holzapfel-Ogden orthotrope, sans pression,
`T_max=30 kPa` :

- `converged=true`, `load_fraction=1.0` ;
- 124 itérations, 37 paliers, 0 erreur de domaine ;
- `min_J=0.570101` ;
- EDV `298.81 mL`, ESV `305.03 mL` ;
- EF `-2.08 %` contre `23.65 %` pour ACDC ;
- allongement axial `+18.67 mm`.

La convergence ne vaut pas validation scientifique. Ce résultat ne peut pas
alimenter le DoE, un surrogate ou une conclusion clinique.

## Contrats désormais imposés par le code

- EP scientifique : vrai backend openCARP, sans bascule analytique silencieuse.
- Mécanique acceptée : convergence, charge complète, `min_J>0.1` et champ de
  déplacement fini.
- Waveform : exactement un cycle, pas de NaN/Inf, volumes positifs et
  `EDV>ESV`.
- Couplage déclaré unidirectionnel, sans itération de convergence fictive.
- Export DoE interdit sans retour de pression et benchmark global validé.

## Portes bloquantes

1. Maillage : qualité minimale strictement supérieure à 0.1 et aucun
   tétraèdre sous 0.05.
2. Pression : aucune relance mécanique endocardique avant franchissement de la
   porte maillage.
3. D2.1 : retour réel de la pression Windkessel vers FEniCSx puis convergence
   sur pression, volume et déplacement.
4. DoE : seulement après validation D2.1; les 500 simulations fallback ne sont
   pas des données physiques acceptables.
5. WP1 M&Ms : récupérer l'archive de résultats et figer l'environnement.

## Traçabilité dépôt

Au 20 juillet 2026, `main` est en avance de huit commits sur `origin/main` et
le répertoire de travail contient de nombreuses modifications et preuves non
commitées. Aucun nettoyage destructif, commit global ou push ne doit être fait
avant classement fichier par fichier et validation utilisateur.

Le détail chronologique et les artefacts de preuve sont consignés dans
`SPRINTS.md` et `sprint_artifacts/`.
