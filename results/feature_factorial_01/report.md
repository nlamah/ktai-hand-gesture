# Acht varianten van handfeatures en gegroepeerde permutatie

D = 190 afstanden; S = 630 Shin-richtingen; E = 70 categorische waarden voor 19 eigen controles; J = 15 wereldgewrichtsmetingen.
Alle acht varianten zijn opnieuw getraind met SVM en LightGBM op dezelfde 60.251 geaccepteerde oorspronkelijke beelden en vijf vaste filename-folds. Geen 26 werkboekovereenkomstscores, spiegelingen of handnormalisatie in deze hoofdvergelijking.

| Variant | Invoer | SVM precisie (%) | SVM F1 | LightGBM precisie (%) | LightGBM F1 |
|---|---:|---:|---:|---:|---:|
| DS: Basis: afstanden en richtingen | 820 | 97.38815 | 0.97386235 | 96.80215 | 0.96784341 |
| D: Alleen afstanden | 190 | 94.42535 | 0.94376340 | 93.63347 | 0.93550772 |
| DSE: Basis + eigen categorieën | 890 | 97.35679 | 0.97347720 | 96.78134 | 0.96763271 |
| DE: Afstanden + eigen categorieën | 260 | 94.63529 | 0.94597400 | 93.71895 | 0.93639522 |
| DSJ: Basis + wereldgewrichtshoeken | 835 | 97.53902 | 0.97524391 | 96.71271 | 0.96691985 |
| DJ: Afstanden + wereldgewrichtshoeken | 205 | 95.05396 | 0.95010689 | 93.83213 | 0.93765908 |
| DSEJ: Basis + categorieën + wereldhoeken | 905 | 97.38902 | 0.97376260 | 96.77617 | 0.96752529 |
| DEJ: Afstanden + categorieën + wereldhoeken | 275 | 94.97488 | 0.94926247 | 93.86942 | 0.93791397 |

## Bijdrage hangt af van de overige invoer

`factor_effects.csv` geeft de toevoeging van S, E of J in alle vier bijbehorende contexten. `paired_changes.csv` telt herstel en verlies tegenover DS. De oude 931-invoerproef bevatte ook werkboekscores en is daarom niet dezelfde variant als de nieuwe DSEJ (905 invoerwaarden).

## Permutatie na het trainen

De volledige classificatiescores hierboven gebruiken alle 60.251 beelden. Voor permutatie zijn vooraf 100 toetsbeelden per letter per fold gekozen: 13.000 unieke beelden. Iedere variant en leermethode krijgt exact dezelfde ID’s. Vijf vaste husselingen per groep/fold; de labels blijven staan. Het model wordt niet opnieuw getraind. Alle kolommen van een groep gebruiken dezelfde rijpermutatie.
D/S/E/J worden in elke variant getest voor zover aanwezig. De 19 afzonderlijke categorieën worden in DSE en DE onderzocht. In DSEJ en DEJ worden vijf categorische families en vijf gewrichtsfamilies onderzocht. Dit voorkomt dat alle detailtests uitsluitend onder dubbel aanwezige hoekinformatie plaatsvinden.
De CSV bevat dalingen van macro-precisie en macro-F1. De figuren tonen gemiddelde macro-F1-daling in procentpunten. SD over folds/herhalingen is beschrijvend; het is geen onafhankelijk betrouwbaarheidsinterval. Negatieve dalingen worden niet weggefilterd.

## Interpretatie en beperkingen

Permutatie meet afhankelijkheid van een getraind model onder verstoring; sterk samenhangende features kunnen elkaar vervangen. Het doorbreken van samenhang tussen groepen kan onnatuurlijke combinaties veroorzaken. De acht trainingsvergelijkingen beantwoorden daarom een andere, aanvullende vraag dan de permutatietests.
D/S gebruiken beeldhandpunten; E/J komen uit wereldhandpunten. De bestaande controles met J uit dezelfde beeldpunten blijven apart gerapporteerd. De coördinatenbron, schaling en vaste modelprofielen begrenzen de conclusies. Geen nieuwe sensor of onafhankelijk bewezen anatomische informatie.
De filename-folds zijn tijdens ontwikkeling herhaald gebruikt en zijn geen geverifieerde personen. Scores gelden uitsluitend bij geaccepteerde detectie. Geen onafhankelijke eindtest of aangetoonde webcamverbetering. De eerdere augmentatie- en livevarianten behouden hun eigen protocol.

## Bestanden

`protocol.json`, `input_verification.json`, `metrics.csv`, `per_letter.csv`, `paired_changes.csv`, `predictions.csv`, `confusions.csv`, `factor_effects.csv`, `same_coordinate_controls.csv`, `permutation_selection.csv`, `permutation.csv`, `permutation_summary.csv`, `verification.json`. Foldmodellen en tussenresultaten staan onder `runs/feature_factorial_01/`.
