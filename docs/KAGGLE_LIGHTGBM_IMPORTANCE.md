# SVM versus LightGBM en featurebijdrage

Deze vergelijking gebruikt de bewaarde 60.251 bruikbare Kaggle ASL Alphabet-
beelden en exact de vijf buitenfolds uit `kaggle_nested_feature_weights_01`.
De 17.749 detectiefouten worden niet gebruikt om landmarkmodellen te trainen.
Alle analyses zijn ontwikkelonderzoek; filenameblokken zijn geen bekende personen.

Er zijn vier modellen: SVM en LightGBM, elk met Shin alleen (820 waarden) en
Shin plus categorieën, Excel-scores en gewrichtshoeken (931 waarden). De extra
features zijn ongewogen. Er wordt in deze run geen modelhyperparameterzoekactie
gedaan: de vergelijking gebruikt de vaste projectprofielen. Voorspellingen voor
ieder beeld komen uitsluitend uit de fold waarin dat beeld validatiedata was.

```bash
conda activate ktai-hand-gesture
python -m scripts.compare_kaggle_lightgbm --stage train --models svm lightgbm
python -m scripts.compare_kaggle_lightgbm --stage permutation --models svm lightgbm --repeats 5 --max-per-class 100
python -m scripts.compare_kaggle_lightgbm --stage permutation --models lightgbm --repeats 20 --max-per-class 0
python -m scripts.compare_kaggle_lightgbm --stage ablation --models svm lightgbm --groups angles categories excel_scores
```

Resultaten: `experiments/kaggle_lightgbm_importance_01/`.

## Permutatie

Iedere featuregroep krijgt één gezamenlijke rijpermutatie voor alle kolommen.
Hierdoor blijven one-hotcategorieën geldig. Dezelfde seeds en validatiebeelden
worden gebruikt voor SVM en LightGBM. De eerste analyse beperkt validatie tot
maximaal 100 beelden per letter per fold. De uitgebreide LightGBM-analyse gebruikt
alle validatiebeelden, met twintig herhalingen. Het model wordt hierbij niet
opnieuw getraind. De gemiddelde daling en spreiding worden over folds en
herhalingen gerapporteerd; dit zijn geen onafhankelijke betrouwbaarheidsintervallen.

Groepen: alle extra features samen, categorieën, hoeken, Excel-scores, duim,
vingerbuiging, spreiding, kruising en contact. Deze groepen overlappen bewust:
het zijn afzonderlijke interventies, geen optelbare onderdelen van één totaalscore.
Excel-scores worden uit de categorieën afgeleid en Shin beschrijft dezelfde
landmarks. Redundantie kan importance verlagen; inconsistente gehusselde
combinaties kunnen importance juist verhogen. Importance toont modelgebruik,
niet automatisch extra waarde boven Shin.

## Ablatie

Voor iedere verwijderde groep wordt het model opnieuw getraind op dezelfde
trainingsfold. Imputatie en schaling van graden worden uitsluitend op training
geleerd. Validatiebeelden blijven gelijk. Accuracydaling wordt voor de totaalscore
naar foldgrootte gewogen. Macro-F1-dalingen per fold zijn beschrijvend; per-letter
recall kan met het aantal validatiebeelden per letter worden samengevoegd.

Belangrijkste bestanden:

- `model_metrics.csv`: fold- en samengevoegde scores.
- `per_letter.csv`: recall per letter en modelvariant.
- `permutation_summary_r5_cap100.csv`: eerste importancevergelijking.
- `permutation_summary_r20_cap0.csv`: uitgebreidere LightGBM-importance.
- `ablation.csv` en `ablation_letters.csv`: weglaten plus opnieuw trainen.
- `oof_predictions.csv`: individuele out-of-foldvoorspellingen.

Een positieve importance of ablatiedaling betekent dat verstoren of verwijderen
de prestatie verlaagt. Een negatieve waarde betekent verbetering na die ingreep.
De sterkste aanwijzing voor toegevoegde waarde is winst tegenover Shin én een
herhaalbare verslechtering bij weglaten, niet alleen hoge permutation importance.
