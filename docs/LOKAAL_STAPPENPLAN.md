# Lokaal stappenplan: ASL-features en modelvergelijking

Dit is een uitvoerbare workflow voor de nieuwe ASL-featureroute. Er is tijdens
de ontwikkeling van deze route **geen** echte kalibratie, training, ablation
of eindtest uitgevoerd. De tests gebruiken uitsluitend synthetische data.

## 1. Omgeving

Voer commando's uit vanuit de projectmap:

```bash
conda activate ktai-hand-gesture
```

De MediaPipe-detectordrempels uit `docs/DETECTOR_TUNING.md` zijn iets anders
dan de categoriegrenzen in `config/asl_feature_candidates.example.json`.
Verander die niet tegelijk: detectorinstellingen bepalen of een hand wordt
gevonden; ASL-categoriedrempels bepalen hoe gevonden landmarks worden ingedeeld.

## 2. Dataset voorbereiden

Gebruik de bestaande Kaggle- of COCO-importer als de dataset daarmee overeenkomt.
Voor een andere, nog niet ondersteunde dataset is er een generieke CSV-adapter.
Vul daarvoor een CSV met precies deze vereiste velden:

```csv
image_path,label,participant,session,condition,role,source_group
```

`image_path` is relatief aan een lokale afbeeldingsmap, `label` is A–Z,
`role` is `development` of `test`, en `source_group` houdt gecorreleerde of
afgeleide beelden bij elkaar. De bronnaam moet de echte dataset identificeren:

```bash
python -m scripts.import_asl_csv \
  --csv data/external/second_asl/metadata.csv \
  --images-root data/external/second_asl \
  --source-name "VUL DE ECHTE DATASETNAAM IN" \
  --project data/asl_second
python -m scripts.analyze_dataset --project data/asl_second
```

De import start geen handdetectie. Na import analyseert het tweede commando
landmarks. Gebruik hiervoor een nieuwe projectmap en behoud de bronbestanden.

## 3. Referenties en beoordelaars

Bekijk het schema en de voorlopige conceptmatrix voor A–Z:

```bash
python -m streamlit run apps/asl_feature_matrix_app.py -- --matrix config/asl_letter_references.json
```

De matrix bevat ruwe, gedeeltelijke visuele inschattingen van
`assets/asl_alphabet.jpg`; dit zijn geen gevalideerde ASL-regels. Controleer
of corrigeer de conceptwaarden met een betrouwbare ASL-bron. Een letterlabel
is geen bewijs voor een specifieke vingerstand. Laat onbekende verwachtingen
leeg; gebruik `not_relevant` alleen als dat inhoudelijk is beoordeeld. Een
patroon kan pas `verified` zijn met bron, beoordelaar en toelichting.

Maak daarna een groepsgescheiden rolbestand en verzamel onafhankelijke
beeldannotaties. Kies de groepering die de bronafhankelijkheid werkelijk
weerspiegelt; `source` vereist `source_group` voor elk bruikbaar beeld:

```bash
python -m scripts.prepare_asl_roles \
  --project data/asl_second \
  --out exports/asl_feature_roles.json \
  --grouping source --folds 5 --seed 42
python -m streamlit run apps/asl_feature_annotation_app.py -- --project data/asl_second
```

De annotatie-app verbergt letterlabels en voorspellingen. Laat waar mogelijk
twee beoordelaars dezelfde foto's beoordelen. Alleen developmentfoto's zijn
toegestaan; reserved testbeelden worden niet geannoteerd. Het rolbestand
reserveert één gegroepeerde fold als `calibration_validation`, de andere
developmentbeelden als `calibration_train`, en houdt test apart als
`final_test`.

## 4. Thresholds kalibreren

Bekijk `config/asl_feature_candidates.example.json`; dit zijn alleen
formaatvoorbeelden, geen aanbeveling. Maak een eigen kandidaatbestand met de
thresholds die je vooraf wilt vergelijken. Neem `{}` op als basisscenario en
controleer dat parameterparen geldig en in de juiste volgorde staan.

```bash
python -m scripts.calibrate_asl_features \
  --project data/asl_second \
  --role-map exports/asl_feature_roles.json \
  --candidates config/asl_feature_candidates.json \
  --out experiments/asl_feature_calibration_01
```

De selectie gebeurt op consensusannotaties uit `calibration_train`.
`calibration_validation` rapporteert daarna de gekozen configuratie en de
baseline; die rol wordt niet voor selectie gebruikt. `final_test` wordt niet
gescoord. Resultaten en de gekozen thresholds staan in
`experiments/asl_feature_calibration_01/`.

## 5. Vaste modelroutes vergelijken

SVM, MLP en geverifieerde referentieregels gebruiken dezelfde rolverdeling.
Kies per vergelijking een nieuwe uitvoermap. De routes trainen alleen op
development; validatie wordt gebruikt voor modelvergelijking en het opgeslagen
SVM/MLP-model wordt daarna op alle developmentbeelden gefit.

```bash
python -m scripts.train_asl_categorical \
  --project data/asl_second --role-map exports/asl_feature_roles.json \
  --model svm --input categorical \
  --thresholds experiments/asl_feature_calibration_01/categorical_thresholds.json \
  --out experiments/asl_categories_svm_01

python -m scripts.train_asl_categorical \
  --project data/asl_second --role-map exports/asl_feature_roles.json \
  --model mlp --input categorical \
  --thresholds experiments/asl_feature_calibration_01/categorical_thresholds.json \
  --out experiments/asl_categories_mlp_01

python -m scripts.train_asl_categorical \
  --project data/asl_second --role-map exports/asl_feature_roles.json \
  --model svm --input shin2021_combined \
  --out experiments/asl_shin_svm_01

python -m scripts.train_asl_categorical \
  --project data/asl_second --role-map exports/asl_feature_roles.json \
  --model svm --input categorical_plus_shin2021 \
  --thresholds experiments/asl_feature_calibration_01/categorical_thresholds.json \
  --out experiments/asl_categories_plus_shin_svm_01
```

De `categorical_plus_shin2021`-route behoudt de bestaande 820 Shin-inputwaarden
en voegt de categorische one-hot-features toe. SVM/MLP-paden zijn
webcam-compatibel via het opgeslagen `model.joblib`. Voor de regelsroute is een
matrix nodig met ten minste één gecontroleerd posepatroon:

```bash
python -m scripts.train_asl_categorical \
  --project data/asl_second --role-map exports/asl_feature_roles.json \
  --model rules --input categorical \
  --reference-matrix config/asl_letter_references.json \
  --thresholds experiments/asl_feature_calibration_01/categorical_thresholds.json \
  --out experiments/asl_reference_rules_01
```

De rapporten bevatten validatiescores, per-lettermetrics, confusion matrix,
inputdefinitie, featurekolommen en provenance. Een `unknown`-voorspelling telt
als abstentie; referentieregels gebruiken alleen patronen met status
`verified`.

Vergelijk opgeslagen routes alleen wanneer ze dezelfde rolmap, dataset en
validatiebeelden gebruiken. Dit leest bestaande rapporten en voorspellingen;
het traint niets opnieuw en leest geen final-testbeelden:

```bash
python -m scripts.compare_asl_models \
  --run categories=experiments/asl_categories_svm_01 \
  --run categories-plus-shin=experiments/asl_categories_plus_shin_svm_01 \
  --out exports/asl_model_comparison_01
```

De uitvoer bevat een CSV-rangschikking, gepaarde validatievoorspellingen en
een HTML-overzicht. Herhaald kiezen op dezelfde validatieset kan daarop
overfitten; de vergelijking is geen onafhankelijke eindtest.

## 6. Importance en ablations

Voer deze pas uit nadat een vaste modelroute is gekozen. Permutation importance
gebruikt de opgeslagen validation-modelversie; ablations refitten uitsluitend
op de bevroren development-trainingsrijen. Beide gebruiken dezelfde
validatieset en geen final-testbeelden:

```bash
python -m scripts.analyze_asl_feature_importance \
  --run experiments/asl_categories_plus_shin_svm_01 \
  --out exports/asl_feature_importance_01 --repeats 20 --seed 42
python -m scripts.ablate_asl_features \
  --run experiments/asl_categories_plus_shin_svm_01 \
  --out experiments/asl_feature_ablation_01 --seed 42
```

Deze analyses zijn ontwikkelbewijs, geen onafhankelijk prestatiebewijs. De
afzonderlijke final letter-recognition-evaluatie blijft nog een workflowstap
nadat model, thresholds en annotaties zijn vastgezet.
