# Shin: geïsoleerde vergelijking van gewrichtshoeken en Excel-scores

SVM en LightGBM zijn vergeleken op dezelfde 60.251 bruikbare beelden van Kaggle ASL Alphabet, met exact dezelfde vijf gegroepeerde folds als de voorgaande experimenten. De eerder opgeslagen Shin-baselinevoorspellingen zijn hergebruikt en op inputhashes gecontroleerd. Alle drie toevoegingsvarianten zijn opnieuw per fold getraind; de 70 losse one-hotcategorieën zijn afwezig.

## Totaalscores

| Model | Invoer | Aantal features | Accuracy | Macro-F1 | Verschil accuracy t.o.v. Shin |
|---|---|---:|---:|---:|---:|
| svm | Shin alleen | 820 | 97.637% | 0.97386 | +0.000 pp |
| svm | Shin + hoeken | 835 | 97.748% | 0.97524 | +0.111 pp |
| svm | Shin + Excel-scores | 846 | 97.648% | 0.97400 | +0.012 pp |
| svm | Shin + hoeken + Excel-scores | 861 | 97.744% | 0.97515 | +0.108 pp |
| lightgbm | Shin alleen | 820 | 97.097% | 0.96784 | +0.000 pp |
| lightgbm | Shin + hoeken | 835 | 97.017% | 0.96692 | -0.080 pp |
| lightgbm | Shin + Excel-scores | 846 | 97.119% | 0.96815 | +0.022 pp |
| lightgbm | Shin + hoeken + Excel-scores | 861 | 97.129% | 0.96824 | +0.032 pp |

## Wat voegen alleen de hoeken toe?

Bij SVM: 58827 → 58894 juiste beelden (+67). Accuracy 97.637% → 97.748%; macro-F1 0.97386 → 0.97524.

Bij LightGBM: 58502 → 58454 juiste beelden (-48). Accuracy 97.097% → 97.017%.

## Per fold: verschil accuracy van hoeken t.o.v. Shin

| Fold | SVM | LightGBM |
|---|---:|---:|
| 1 | +0.068 pp | -0.062 pp |
| 2 | +0.230 pp | -0.010 pp |
| 3 | +0.237 pp | -0.035 pp |
| 4 | +0.038 pp | -0.181 pp |
| 5 | +0.017 pp | -0.116 pp |

## Recall per letter: svm

| Letter | Beelden | Shin | + hoeken | + Excel | + beide | Hoeken − Shin |
|---|---:|---:|---:|---:|---:|---:|
| A | 2188 | 98.67% | 98.72% | 98.72% | 98.72% | +0.05 pp |
| B | 2204 | 99.77% | 99.73% | 99.77% | 99.73% | -0.05 pp |
| C | 1912 | 99.84% | 99.84% | 99.84% | 99.84% | +0.00 pp |
| D | 2456 | 97.68% | 97.68% | 97.68% | 97.72% | +0.00 pp |
| E | 2308 | 99.00% | 99.22% | 99.05% | 99.22% | +0.22 pp |
| F | 2859 | 98.60% | 98.57% | 98.57% | 98.57% | -0.03 pp |
| G | 2430 | 99.42% | 99.42% | 99.42% | 99.42% | +0.00 pp |
| H | 2371 | 98.14% | 98.19% | 98.14% | 98.23% | +0.04 pp |
| I | 2380 | 96.68% | 96.76% | 96.68% | 96.72% | +0.08 pp |
| J | 2576 | 96.78% | 96.78% | 96.82% | 96.82% | +0.00 pp |
| K | 2701 | 98.26% | 98.37% | 98.26% | 98.37% | +0.11 pp |
| L | 2516 | 99.64% | 99.60% | 99.64% | 99.60% | -0.04 pp |
| M | 1602 | 90.01% | 91.95% | 90.45% | 91.64% | +1.94 pp |
| N | 1280 | 88.12% | 87.97% | 87.97% | 87.89% | -0.16 pp |
| O | 2264 | 98.19% | 97.88% | 98.23% | 97.92% | -0.31 pp |
| P | 2041 | 98.63% | 98.78% | 98.63% | 98.78% | +0.15 pp |
| Q | 2106 | 97.72% | 97.72% | 97.72% | 97.72% | +0.00 pp |
| R | 2543 | 93.43% | 93.83% | 93.39% | 93.87% | +0.39 pp |
| S | 2559 | 98.09% | 98.40% | 98.09% | 98.40% | +0.31 pp |
| T | 2354 | 98.43% | 98.47% | 98.43% | 98.47% | +0.04 pp |
| U | 2513 | 96.70% | 96.58% | 96.74% | 96.62% | -0.12 pp |
| V | 2552 | 98.24% | 98.43% | 98.24% | 98.39% | +0.20 pp |
| W | 2456 | 98.25% | 98.33% | 98.21% | 98.29% | +0.08 pp |
| X | 2157 | 97.03% | 97.22% | 97.03% | 97.22% | +0.19 pp |
| Y | 2580 | 99.46% | 99.38% | 99.46% | 99.42% | -0.08 pp |
| Z | 2343 | 97.44% | 97.78% | 97.44% | 97.78% | +0.34 pp |

## Recall per letter: lightgbm

| Letter | Beelden | Shin | + hoeken | + Excel | + beide | Hoeken − Shin |
|---|---:|---:|---:|---:|---:|---:|
| A | 2188 | 98.03% | 97.94% | 98.08% | 98.03% | -0.09 pp |
| B | 2204 | 99.18% | 98.91% | 99.59% | 99.59% | -0.27 pp |
| C | 1912 | 98.74% | 98.80% | 98.80% | 98.90% | +0.05 pp |
| D | 2456 | 97.19% | 97.23% | 97.39% | 97.35% | +0.04 pp |
| E | 2308 | 98.53% | 98.35% | 98.40% | 98.44% | -0.17 pp |
| F | 2859 | 98.36% | 98.36% | 98.29% | 98.29% | +0.00 pp |
| G | 2430 | 98.89% | 99.05% | 99.05% | 98.97% | +0.16 pp |
| H | 2371 | 97.34% | 97.22% | 97.43% | 97.26% | -0.13 pp |
| I | 2380 | 96.34% | 96.34% | 96.30% | 96.51% | +0.00 pp |
| J | 2576 | 96.97% | 96.82% | 97.01% | 96.78% | -0.16 pp |
| K | 2701 | 97.89% | 97.89% | 97.85% | 97.82% | +0.00 pp |
| L | 2516 | 98.45% | 98.29% | 98.41% | 98.45% | -0.16 pp |
| M | 1602 | 88.45% | 87.39% | 88.89% | 88.14% | -1.06 pp |
| N | 1280 | 85.16% | 85.08% | 84.84% | 85.70% | -0.08 pp |
| O | 2264 | 93.90% | 93.46% | 94.17% | 94.26% | -0.44 pp |
| P | 2041 | 98.48% | 98.53% | 98.53% | 98.58% | +0.05 pp |
| Q | 2106 | 98.53% | 98.29% | 98.39% | 98.58% | -0.24 pp |
| R | 2543 | 95.24% | 95.36% | 95.01% | 95.32% | +0.12 pp |
| S | 2559 | 97.73% | 97.77% | 97.69% | 97.58% | +0.04 pp |
| T | 2354 | 98.05% | 98.17% | 98.13% | 98.13% | +0.13 pp |
| U | 2513 | 96.74% | 96.74% | 96.66% | 96.90% | +0.00 pp |
| V | 2552 | 97.73% | 97.77% | 97.61% | 97.61% | +0.04 pp |
| W | 2456 | 98.37% | 98.49% | 98.41% | 98.53% | +0.12 pp |
| X | 2157 | 96.89% | 96.66% | 96.71% | 96.80% | -0.23 pp |
| Y | 2580 | 99.42% | 99.30% | 99.38% | 99.22% | -0.12 pp |
| Z | 2343 | 95.69% | 95.60% | 95.90% | 95.65% | -0.09 pp |

## Protocol en interpretatie

- De 15 continue metingen zijn 14 binnenhoeken van de gewrichten plus de getekende duim-MCP-afwijking, berekend uit wereldlandmarks. Imputatie en standaardisatie worden alleen op de trainingsfold geleerd; Shin blijft ongewijzigd.
- Excel-scores zijn 26 overeenkomstfracties: één per mogelijke letter. Ze worden voor elk beeld berekend zonder het echte letterlabel als invoer. Zij behouden indirect categorische informatie; de variant met alleen hoeken gebruikt geen categoriedrempels of Excel-referenties.
- SVM: RBF, C=10, gamma=scale, balanced. LightGBM: vaste projectinstellingen (100 rondes, 31 bladeren, learning rate 0,1). Geen nieuwe hyperparameter- of gewichtstuning.
- De scores hierboven gelden voor de 60.251 bruikbare detecties. De overige 17.749 van de 78.000 letterbeelden kunnen niet door een landmarkclassifier worden verwerkt. All-images-scores staan in summary.json.
- Filenameblokken zijn geen bekende personen. Dit blijft herhaald ontwikkelonderzoek op dezelfde folds; kleine verschillen zijn geen bewijs van algemene of onafhankelijke winst. Een nieuwe persoon-/webcamtest blijft nodig.

## Lokaal herhalen

```bash
conda activate ktai-hand-gesture
python -m scripts.compare_kaggle_angles_scores --models svm lightgbm
python -m scripts.report_kaggle_angles_scores
```

Modellen en validatievoorspellingen worden per fold opgeslagen, zodat hervatten bestaande checkpoints gebruikt. De vorige resultaten worden niet overschreven. Details: metrics.csv, per_letter.csv, angle_effects_per_letter.csv en de modelspecifieke out-of-foldvoorspellingen.
