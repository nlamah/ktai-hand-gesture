# SVM, LightGBM en de bijdrage van extra features

Dezelfde 60.251 bruikbare Kaggle ASL Alphabet-beelden en vijf gegroepeerde buitenfolds als de eerdere experimenten. Shin gebruikt 820 waarden; gecombineerd gebruikt Shin plus 70 categorieën, 26 Excel-overeenkomstscores en 15 gewrichtshoeken. Extra features zijn hier ongewogen. De 17.749 ontbrekende detecties vallen buiten de classifierscores hieronder.

## Modelvergelijking

| Model | Shin accuracy | Met extra features | Verschil | Shin macro-F1 | Met extra macro-F1 |
|---|---:|---:|---:|---:|---:|
| svm | 97.637% | 97.628% | -0.008 pp | 0.97386 | 0.97372 |
| lightgbm | 97.097% | 97.164% | +0.066 pp | 0.96784 | 0.96860 |

SVM: RBF, C=10, gamma=scale, balanced. LightGBM: 100 boostingrondes, 31 bladeren, learning rate 0,1, minimaal 20 voorbeelden per blad, seed 42; overige projectinstellingen vast. LightGBM gebruikt vier rekenthreads bij de resterende fits; deterministic en force_col_wise staan aan. Er is geen nieuwe hyperparameterzoekactie gedaan. Imputatie en schaling van hoeken worden uitsluitend op training gefit.

## Permutation importance

Vijf herhalingen per groep en fold, maximaal 100 validatiebeelden per letter per fold. Onderstaande waarden zijn gemiddelde macro-F1-dalingen in procentpunten. Positief betekent slechter na husselen; negatief betekent beter na husselen. Kolommen binnen een groep worden gezamenlijk verwisseld; dezelfde permutaties en steekproeven worden voor beide modellen gebruikt. Spreiding staat in de CSV.

| Gehusselde groep | SVM: macro-F1-daling | LightGBM: macro-F1-daling |
|---|---:|---:|
| all_extra | +1.263 pp | +0.290 pp |
| categories | +0.049 pp | +0.055 pp |
| angles | +0.570 pp | +0.031 pp |
| excel_scores | +0.008 pp | +0.200 pp |
| thumb | +0.129 pp | +0.014 pp |
| finger_flexion | +0.499 pp | +0.017 pp |
| spread | -0.002 pp | +0.055 pp |
| crossing | +0.008 pp | +0.000 pp |
| contact | +0.000 pp | +0.000 pp |

`all_extra` husselt alle extra features samen. De andere groepen overlappen: hun importances zijn niet optelbaar. `thumb` bevat duimcategorieën en duimhoeken; `finger_flexion` bevat overige buigcategorieën en hoeken. Excel-scores zijn afgeleid van categorieën. Redundantie kan importance drukken; onmogelijke combinaties door husselen kunnen haar vergroten. Importance meet modelgebruik, geen zelfstandig bewijs van extra informatie.

## Ablatie: opnieuw trainen zonder featuregroep

Alle validatiebeelden worden gebruikt. Hieronder staat de samengevoegde accuracy na verwijderen. De delta is ten opzichte van het volledige gecombineerde model: positief betekent verbetering na weglaten.

| Weggelaten groep | SVM accuracy | SVM delta | LightGBM accuracy | LightGBM delta |
|---|---:|---:|---:|---:|
| angles | 97.593% | -0.035 pp | 97.094% | -0.070 pp |
| categories | 97.744% | +0.116 pp | 97.129% | -0.035 pp |
| excel_scores | 97.632% | +0.003 pp | 97.069% | -0.095 pp |

De drie groepen worden afzonderlijk weggelaten; alle overige input blijft aanwezig. Een groep kan dus weinig ablatiewinst opleveren omdat een andere groep dezelfde informatie levert. Macro-F1 per fold en recallveranderingen per letter staan in de detailbestanden.

## Per letter: recall

| Letter | Beelden | SVM Shin | SVM extra | LightGBM Shin | LightGBM extra | LightGBM winst door extra |
|---|---:|---:|---:|---:|---:|---:|
| A | 2188 | 98.67% | 98.67% | 98.03% | 97.90% | -0.14 pp |
| B | 2204 | 99.77% | 99.73% | 99.18% | 99.64% | +0.45 pp |
| C | 1912 | 99.84% | 99.84% | 98.74% | 98.95% | +0.21 pp |
| D | 2456 | 97.68% | 97.64% | 97.19% | 97.15% | -0.04 pp |
| E | 2308 | 99.00% | 99.22% | 98.53% | 98.48% | -0.04 pp |
| F | 2859 | 98.60% | 98.57% | 98.36% | 98.67% | +0.31 pp |
| G | 2430 | 99.42% | 99.47% | 98.89% | 98.89% | +0.00 pp |
| H | 2371 | 98.14% | 98.23% | 97.34% | 97.26% | -0.08 pp |
| I | 2380 | 96.68% | 96.72% | 96.34% | 96.43% | +0.08 pp |
| J | 2576 | 96.78% | 96.93% | 96.97% | 96.93% | -0.04 pp |
| K | 2701 | 98.26% | 98.30% | 97.89% | 97.85% | -0.04 pp |
| L | 2516 | 99.64% | 99.56% | 98.45% | 98.17% | -0.28 pp |
| M | 1602 | 90.01% | 90.76% | 88.45% | 88.70% | +0.25 pp |
| N | 1280 | 88.12% | 86.80% | 85.16% | 85.70% | +0.55 pp |
| O | 2264 | 98.19% | 97.48% | 93.90% | 94.39% | +0.49 pp |
| P | 2041 | 98.63% | 98.82% | 98.48% | 98.68% | +0.20 pp |
| Q | 2106 | 97.72% | 97.72% | 98.53% | 98.34% | -0.19 pp |
| R | 2543 | 93.43% | 92.96% | 95.24% | 95.44% | +0.20 pp |
| S | 2559 | 98.09% | 98.40% | 97.73% | 97.69% | -0.04 pp |
| T | 2354 | 98.43% | 98.43% | 98.05% | 98.13% | +0.08 pp |
| U | 2513 | 96.70% | 96.54% | 96.74% | 96.74% | +0.00 pp |
| V | 2552 | 98.24% | 98.24% | 97.73% | 97.81% | +0.08 pp |
| W | 2456 | 98.25% | 98.37% | 98.37% | 98.57% | +0.20 pp |
| X | 2157 | 97.03% | 97.36% | 96.89% | 97.13% | +0.23 pp |
| Y | 2580 | 99.46% | 99.38% | 99.42% | 99.30% | -0.12 pp |
| Z | 2343 | 97.44% | 97.57% | 95.69% | 95.43% | -0.26 pp |

## Interpretatie en grenzen

Combineer drie soorten bewijs: winst tegenover Shin, daling bij husselen, en daling na opnieuw trainen zonder de groep. Alleen hoge permutation importance is onvoldoende om te zeggen dat jouw features werkelijk iets toevoegen boven Shin. De evaluatie gebruikt zes filenameblokken, niet geverifieerde personen; het is ontwikkelonderzoek. De referenties en drempels zijn eerder op de aparte Roboflow-set gekozen. Er is geen onafhankelijke webcam- of persoonstest. De standaardafwijkingen over folds en herhalingen zijn geen onafhankelijke betrouwbaarheidsintervallen.

Ruwe resultaten: `model_metrics.csv`, `per_letter.csv`, `permutation_summary_r5_cap100.csv`, `permutation_letters_r5_cap100.csv`, `ablation.csv`, `ablation_letters.csv`, `oof_predictions.csv`. De modelspecifieke bestanden blijven behouden. Lokaal herhalen: `docs/KAGGLE_LIGHTGBM_IMPORTANCE.md`.

## Vervolgtkandidaat: SVM zonder losse categorieën

Shin + 15 continue hoeken + 26 Excel-scores (861 invoerwaarden) haalt 58.892/60.251 correct: **97,744%**. Dat zijn 65 extra juiste beelden tegenover Shin/project (+0,108 procentpunt), en 70 tegenover de volledige ongewogen combinatie. De 70 expliciete one-hotcategorieën worden weggelaten; categorische informatie blijft indirect aanwezig in de Excel-scores.

| Letter | Shin recall | Kandidaat recall | Extra correcte beelden |
|---|---:|---:|---:|
| M | 90.01% | 91.64% | +26 |
| R | 93.43% | 93.87% | +11 |
| S | 98.09% | 98.40% | +8 |
| Z | 97.44% | 97.78% | +8 |
| O | 98.19% | 97.92% | -6 |
| N | 88.12% | 87.89% | -3 |
| U | 96.70% | 96.62% | -2 |

Deze kandidaat is achteraf tussen de ablatievarianten gekozen en moet op nieuwe onafhankelijke beelden bevestigd worden. Dit is een accuracyresultaat; de opgeslagen ablaties bevatten macro-F1 per fold, geen gepoolde macro-F1. Alle letters staan in `candidate_without_categories_per_letter.csv`.
