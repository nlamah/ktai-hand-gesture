# Rechtsnormalisatie vóór de featureberekening

Verkennende vergelijking op dezelfde bronbeelden, vaste folds en apart gedetecteerde RGB-presentaties als de spiegelproef.
Detector-Left wordt naar detector-Right gespiegeld. Het is geen geverifieerd label van de menselijke hand of voorkeurshand.
Primaire vergelijking: oorspronkelijke beelden met geldige handdetectie, tegen de oorspronkelijke 835-inputreferentie. Mislukte detecties zijn uitgesloten. Het aparte synthetische mengsel is een andere toetsverdeling.

## Alleen geaccepteerde detecties (primair)

| Model | Toets | Training | N geldig | Macro-precisie (%) | Macro-F1 | Correct |
|---|---|---|---:|---:|---:|---:|
| svm | Origineel | Zonder aanpassing | 60251 | 97.53902 | 0.97524391 | 58894 |
| svm | Origineel | Spiegeltraining | 60251 | 97.49538 | 0.97492776 | 58868 |
| svm | Origineel | Rechtsnormalisatie | 60251 | 97.32223 | 0.97319602 | 58758 |
| svm | Gespiegeld | Zonder aanpassing | 60201 | 26.00735 | 0.19248891 | 14027 |
| svm | Gespiegeld | Spiegeltraining | 60201 | 97.29327 | 0.97250277 | 58676 |
| svm | Gespiegeld | Rechtsnormalisatie | 60201 | 94.88493 | 0.94802260 | 57239 |
| svm | Vast mengsel | Zonder aanpassing | 60225 | 73.56587 | 0.62649875 | 36452 |
| svm | Vast mengsel | Spiegeltraining | 60225 | 97.45079 | 0.97420633 | 58801 |
| svm | Vast mengsel | Rechtsnormalisatie | 60225 | 96.07574 | 0.96047248 | 57987 |
| lightgbm | Origineel | Zonder aanpassing | 60251 | 96.71271 | 0.96691985 | 58454 |
| lightgbm | Origineel | Spiegeltraining | 60251 | 96.99001 | 0.96961927 | 58581 |
| lightgbm | Origineel | Rechtsnormalisatie | 60251 | 96.48978 | 0.96459280 | 58304 |
| lightgbm | Gespiegeld | Zonder aanpassing | 60201 | 58.10641 | 0.49977814 | 29779 |
| lightgbm | Gespiegeld | Spiegeltraining | 60201 | 96.99586 | 0.96940072 | 58488 |
| lightgbm | Gespiegeld | Rechtsnormalisatie | 60201 | 95.08552 | 0.94959642 | 57322 |
| lightgbm | Vast mengsel | Zonder aanpassing | 60225 | 78.73684 | 0.74139200 | 44124 |
| lightgbm | Vast mengsel | Spiegeltraining | 60225 | 97.01177 | 0.96962795 | 58540 |
| lightgbm | Vast mengsel | Rechtsnormalisatie | 60225 | 95.75919 | 0.95678864 | 57798 |

## Alle pogingen (secundair, inclusief detectiemislukkingen)

| Model | Toets | Training | N aangeboden | Macro-precisie (%) | Macro-F1 | Correct |
|---|---|---|---:|---:|---:|---:|
| svm | Origineel | Zonder aanpassing | 78000 | 97.53902 | 0.84669832 | 58894 |
| svm | Origineel | Spiegeltraining | 78000 | 97.49538 | 0.84640358 | 58868 |
| svm | Origineel | Rechtsnormalisatie | 78000 | 97.32223 | 0.84483115 | 58758 |
| svm | Gespiegeld | Zonder aanpassing | 78000 | 26.00735 | 0.16554426 | 14027 |
| svm | Gespiegeld | Spiegeltraining | 78000 | 97.29327 | 0.84401102 | 58676 |
| svm | Gespiegeld | Rechtsnormalisatie | 78000 | 94.88493 | 0.82301442 | 57239 |
| svm | Vast mengsel | Zonder aanpassing | 78000 | 73.56587 | 0.53849130 | 36452 |
| svm | Vast mengsel | Spiegeltraining | 78000 | 97.45079 | 0.84559912 | 58801 |
| svm | Vast mengsel | Rechtsnormalisatie | 78000 | 96.07574 | 0.83379477 | 57987 |
| lightgbm | Origineel | Zonder aanpassing | 78000 | 96.71271 | 0.84006268 | 58454 |
| lightgbm | Origineel | Spiegeltraining | 78000 | 96.99001 | 0.84206142 | 58581 |
| lightgbm | Origineel | Rechtsnormalisatie | 78000 | 96.48978 | 0.83795342 | 58304 |
| lightgbm | Gespiegeld | Zonder aanpassing | 78000 | 58.10641 | 0.43133157 | 29779 |
| lightgbm | Gespiegeld | Spiegeltraining | 78000 | 96.99586 | 0.84132148 | 58488 |
| lightgbm | Gespiegeld | Rechtsnormalisatie | 78000 | 95.08552 | 0.82437168 | 57322 |
| lightgbm | Vast mengsel | Zonder aanpassing | 78000 | 78.73684 | 0.64103855 | 44124 |
| lightgbm | Vast mengsel | Spiegeltraining | 78000 | 97.01177 | 0.84176955 | 58540 |
| lightgbm | Vast mengsel | Rechtsnormalisatie | 78000 | 95.75919 | 0.83093481 | 57798 |

## Paarsgewijze veranderingen

- svm, origineel, tegenover zonder aanpassing: 157 hersteld, 293 verloren, netto -136.
- svm, origineel, tegenover spiegeltraining: 186 hersteld, 296 verloren, netto -110.
- svm, vast mengsel, tegenover zonder aanpassing: 21.850 hersteld, 315 verloren, netto +21535.
- svm, vast mengsel, tegenover spiegeltraining: 253 hersteld, 1.067 verloren, netto -814.
- lightgbm, origineel, tegenover zonder aanpassing: 285 hersteld, 435 verloren, netto -150.
- lightgbm, origineel, tegenover spiegeltraining: 283 hersteld, 560 verloren, netto -277.
- lightgbm, vast mengsel, tegenover zonder aanpassing: 14.032 hersteld, 358 verloren, netto +13674.
- lightgbm, vast mengsel, tegenover spiegeltraining: 268 hersteld, 1.010 verloren, netto -742.

## Afbakening

De primaire scores sluiten mislukte detecties uit, met dezelfde geaccepteerde toetsbeelden voor iedere aanpak binnen een presentatie. De secundaire scores over alle 78.000 pogingen tellen detectiemislukkingen als false negative. Precisie blijft gelijk doordat geen letter is voorspeld; recall en F1 veranderen. De aanpassing verbetert de handdetectie zelf niet.
Training gebruikt één rij per geldige oorspronkelijke bron. Spiegeltraining gebruikt extra RGB-afgeleiden; rechtsnormalisatie niet.
MediaPipe-handschattingen worden gebruikt zonder grondwaarheidscontrole; foutieve handlabels kunnen de verkeerde normalisatie veroorzaken.
Dezelfde filename-folds zijn eerder gebruikt tijdens ontwikkeling. Spiegelbeelden van dezelfde bron zijn geen onafhankelijke deelnemers. Geen claim over ongeziene linkshandige personen.
De camera past de geometrie vóór de features alleen aan als het geselecteerde model de normalisatie in zijn bundle vermeldt. Eerdere modellen blijven ongewijzigd.

Details: `protocol.json`, `metrics.csv`, `per_letter.csv`, `handedness_metrics.csv`, `paired_changes.csv`, `predictions.csv`, `geometry_verification.json`, `verification.json`.
