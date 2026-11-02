# Kaggle ASL Alphabet: geneste tuning van featuregewichten

Datasetbron uit Shin: [grassknoted/asl-alphabet](https://www.kaggle.com/grassknoted/asl-alphabet), bevestigd in [het artikel](https://pmc.ncbi.nlm.nih.gov/articles/PMC8434249/). 78.000 letterbeelden, 60.251 bruikbare detecties; alle 26 letters. De datasetbron komt overeen, maar detector en splitsing zijn geen exacte reproductie van het artikel.

5 outer and 3 inner StratifiedGroupKFold; session filename blocks, seed 42; one coordinate pass over 0.25/0.5/1/2. Bound internal search to 150 images/class in each fitting and validation partition; outer training and evaluation use all usable images. Choose inner mean macro-F1 then accuracy. Outer validation never used to select weights. Angle imputation/scaling fitted on inner training partitions and refitted on outer training partition.

Same public dataset source as Shin ASL Alphabet experiment, not the exact published split, detector or a replication of published scores. Filename blocks are not verified subjects. Fixed categorical thresholds/Excel proposals came from separate Roboflow development data; no Kaggle-fold threshold tuning. Only letter classes A-Z. The provided 26 test photos duplicated training pixels and were excluded; no independent final test. Coordinate search is bounded and not exhaustive. Current fixed SVM C/gamma not jointly tuned.

## Belangrijkste uitkomsten

- De datasetbron is dezelfde Kaggle ASL Alphabet-bron die Shin noemt. Het gaat niet om de kleine Roboflow-set uit de vorige vergelijkingen.
- De getunede combinatie is marginaal nauwkeuriger: 58.848 correcte beelden tegenover 58.827 voor Shin/project (+21) en 58.842 voor Shin/artikelprofiel (+6).
- Ongewogen extra features leveren 58.822 correcte beelden op: vijf minder dan de projectbaseline. Gewichten helpen dus tegenover die ongewogen combinatie (+26).
- De hoogste macro-F1 blijft bij de pure Shin-baseline met artikelprofiel (0,97469); getuned gecombineerd haalt 0,97434. Er is geen duidelijke winst op alle prestatiematen.
- M-recall verbetert van 90,01% naar 91,01% (+16 beelden); S van 98,09% naar 98,36% (+7 beelden).
- R daalt van 93,43% naar 93,08% (−9 beelden), N van 88,13% naar 87,58% (−7), O van 98,19% naar 97,88% (−7).
- De geselecteerde categoriegewichten zijn 1; 0,25; 0,25; 0,25; 0,5. Meer gewicht blijkt hier dus niet automatisch beter.

Dit verschil is klein en geen aangetoonde statistisch betrouwbare algemene verbetering. De groepering bestaat uit zes filenameblokken, niet uit bekende personen. De zoekstap is begrensd tot maximaal 150 beelden per letter per interne deelset, maar de buitenste training en evaluatie gebruiken alle 60.251 bruikbare beelden. De 17.749 detectiefouten verklaren waarom de score over alle 78.000 beelden lager ligt.

| Variant | Accuracy bruikbaar | Macro-F1 | Accuracy alle beelden |
|---|---:|---:|---:|
| shin_project | 97.637% | 0.97386 | 75.419% |
| shin_paper_profile | 97.661% | 0.97469 | 75.438% |
| combined_unweighted | 97.628% | 0.97372 | 75.413% |
| combined_tuned | 97.671% | 0.97434 | 75.446% |

`shin_project`: C=10, balanced. `shin_paper_profile`: C=1000, geen class weighting. Beide RBF, gamma=scale. De gecombineerde modellen gebruiken C=10 en balanced om alleen featuregewicht te vergelijken. Excel-scores worden voor alle 26 kandidaatletters berekend, zonder het echte label als input.

## Gekozen gewichten per buitenfold

| Fold | Hoeken | Categorieën | Excel-scores |
|---|---:|---:|---:|
| 1 | 0.5 | 1.0 | 2.0 |
| 2 | 1.0 | 0.25 | 0.25 |
| 3 | 0.5 | 0.25 | 2.0 |
| 4 | 1.0 | 0.25 | 1.0 |
| 5 | 1.0 | 0.5 | 0.25 |

## Recall per letter

| Letter | Beelden | Shin project | Shin artikelprofiel | Ongewogen | Getuned | Verschil getuned − Shin |
|---|---:|---:|---:|---:|---:|---:|
| A | 2188 | 98.67% | 98.58% | 98.67% | 98.72% | +0.05 pp |
| B | 2204 | 99.77% | 99.77% | 99.73% | 99.73% | -0.05 pp |
| C | 1912 | 99.84% | 99.84% | 99.84% | 99.84% | +0.00 pp |
| D | 2456 | 97.68% | 97.88% | 97.64% | 97.72% | +0.04 pp |
| E | 2308 | 99.00% | 98.83% | 99.22% | 99.18% | +0.17 pp |
| F | 2859 | 98.60% | 98.53% | 98.57% | 98.57% | -0.03 pp |
| G | 2430 | 99.42% | 99.30% | 99.47% | 99.42% | +0.00 pp |
| H | 2371 | 98.14% | 98.06% | 98.23% | 98.23% | +0.08 pp |
| I | 2380 | 96.68% | 96.85% | 96.72% | 96.68% | +0.00 pp |
| J | 2576 | 96.78% | 96.93% | 96.93% | 96.74% | -0.04 pp |
| K | 2701 | 98.26% | 98.37% | 98.30% | 98.37% | +0.11 pp |
| L | 2516 | 99.64% | 99.64% | 99.56% | 99.64% | +0.00 pp |
| M | 1602 | 90.01% | 93.38% | 90.76% | 91.01% | +1.00 pp |
| N | 1280 | 88.12% | 88.12% | 86.80% | 87.58% | -0.55 pp |
| O | 2264 | 98.19% | 97.97% | 97.48% | 97.88% | -0.31 pp |
| P | 2041 | 98.63% | 98.78% | 98.82% | 98.82% | +0.20 pp |
| Q | 2106 | 97.72% | 97.67% | 97.72% | 97.63% | -0.09 pp |
| R | 2543 | 93.43% | 94.06% | 92.96% | 93.08% | -0.35 pp |
| S | 2559 | 98.09% | 98.12% | 98.40% | 98.36% | +0.27 pp |
| T | 2354 | 98.43% | 98.22% | 98.43% | 98.43% | +0.00 pp |
| U | 2513 | 96.70% | 95.66% | 96.54% | 96.62% | -0.08 pp |
| V | 2552 | 98.24% | 97.92% | 98.24% | 98.39% | +0.16 pp |
| W | 2456 | 98.25% | 98.53% | 98.37% | 98.37% | +0.12 pp |
| X | 2157 | 97.03% | 95.92% | 97.36% | 97.22% | +0.19 pp |
| Y | 2580 | 99.46% | 99.38% | 99.38% | 99.38% | -0.08 pp |
| Z | 2343 | 97.44% | 97.52% | 97.57% | 97.61% | +0.17 pp |

Alle detailresultaten: `per_letter.csv`, `fold_metrics.csv`, `inner_search.csv`, `oof_predictions.csv`, `summary.json`. De historische 97,43% gebruikte drie folds; vergelijk die niet rechtstreeks met deze nieuwe vijf folds. De correcte vergelijking is tussen de modellen op de gedeelde huidige folds.

## Reproduceren

```bash
python -m scripts.tune_kaggle_feature_weights \
  --out experiments/kaggle_nested_feature_weights_02
```

De inputcache bewaart hashes van manifest, Excel en drempels. Iedere buitenfold heeft een checkpoint met alle voorspellingen en de interne zoekresultaten. Gecontroleerd: 60.251 unieke out-of-foldvoorspellingen, vijf disjuncte buitenfolds, vijftien disjuncte binnenfolds, geen interne selectiebeelden uit de buitenvalidatie, en herberekende accuracy/macro-F1 gelijk aan de opgeslagen cijfers. Alle zeven gerichte preprocessing-, wegings- en invoertests slagen.
