# Static ASL landmark feature study

This research repository tests whether interpretable descriptions such as bent, extended, spread and touching add predictive value to a Shin-inspired geometric reference. Continuous joint measurements are a comparison with the categorical descriptions; a coordinate control checks how much they add when Shin already contains the underlying geometry. SVM and LightGBM use one Kaggle ASL Alphabet dataset.

- [Toegankelijke Nederlandse werkversie, ICML 2021-opmaak](paper/report.pdf)
- [Supplementary appendix](paper/appendix.pdf)
- [Editable LaTeX](paper/report.tex)

De paper is een **Nederlandse groepswerkversie voor lezers zonder machine-learningvoorkennis**, over foto's bewerken (RQ1), oefenen met meer voorbeelden (RQ2) en de handvorm beschrijven (RQ3). De eerste twee bijdragen hebben nog TODO's. De hoofdtekst legt de aanpak en resultaten in gewone taal uit; modelinstellingen en precieze scores staan in bijlagen A–B; de featuretool en oriëntatieproeven staan in C–F. Er geldt ruimte voor maximaal vijf pagina's hoofdtekst; referenties mogen afzonderlijk doorlopen. [Aanleverchecklist en invoegplekken](paper/INTEGRATION_TODO.md).
- [All setup and reproduction steps](REPRODUCE.md)
- [Start from scratch: Kaggle download, landmark detection, rejected images and threshold experiments](REPRODUCE.md#3-from-downloaded-images-to-accepted-landmarks-start-from-scratch)
- [Frozen results and verification](results/)

## Hoofdonderzoek: acht combinaties van handfeatures

De nieuwe hoofdvergelijking gebruikt vier bouwblokken: **D** (190 afstanden), **S** (630 Shin-richtingen), **E** (70 categorie-invoerwaarden voor 19 eigen controles) en **J** (15 wereldgewrichtsmetingen). Afstanden blijven altijd aanwezig; de acht varianten zijn **DS, D, DSE, DE, DSJ, DJ, DSEJ en DEJ**. De 26 werkboekovereenkomstscores horen hier niet bij. Het vroegere volledige pakket met 931 inputs is dus een andere, historische proef dan DSEJ met 905 inputs.

Alle varianten zijn opnieuw met SVM en LightGBM op dezelfde vijf folds en 60.251 geaccepteerde oorspronkelijke beelden getraind. De primaire baseline is **DS (820 inputs)**. De ondersteunende spiegel-, normalisatie- en G/H-proeven behouden hun eigen referentie met wereldgewrichten (835 inputs); die baselines mogen niet door elkaar worden gebruikt.

Boven op DS geven de eigen categorieën bij SVM netto 22 minder correcte voorspellingen en bij LightGBM één minder. Alleen wereldgewrichtsmetingen toevoegen geeft bij SVM 67 meer, bij LightGBM 48 minder; beide toevoegingen samen geven respectievelijk drie en zeventien minder. De geteste eigen beschrijvingen vervangen de Shin-richtingen niet volledig. Dit geldt voor deze vaste modelprofielen en hergebruikte folds, niet als onafhankelijke eindtest.

Na training volgen gegroepeerde permutatietests op 13.000 vaste toetsbeelden: vijf husselingen per groep/fold, zonder opnieuw trainen. Grote D/S/E/J-blokken worden in iedere variant onderzocht. De 19 eigen controles worden afzonderlijk onderzocht in DSE en DE; categorische en gewrichtsfamilies in DSEJ en DEJ. Samenhangende features kunnen elkaar vervangen, dus trainingsvergelijkingen en permutatiedalingen beantwoorden aanvullende vragen.

[Resultaten, alle varianten en beperkingen](results/feature_factorial_01/report.md) · [Reproduceerbaarheid](REPRODUCE.md#eight-variant-feature-study-and-held-out-permutations)

De controle met gewrichtshoeken uit dezelfde punten als Shin blijft apart staan: zij onderscheidt een andere beschrijving van een andere coördinatenbron. Bijlage C bewaart de feature-uitleg, echte screenshot en schema’s. De overige oriëntatieproeven staan in D–F. De modelprofielen zijn vast; er is geen nieuwe parameterzoektocht of onafhankelijke eindtest.

## Included artifacts and language

The deployed live-model `.joblib` files under `models/`, their provenance JSON files, the four G/H diagnostic models and the MediaPipe model asset are included in version control. The mirror experiment adds a matched fresh-original and mirror-trained pair. No retraining is required when those files are present after cloning. The pinned Python/scikit-learn environment is still required to load them.

Confusion matrices, OOF prediction tables and analysis reports under `results/` are versioned. Raw webcam captures under `captures/` remain local and ignored, as requested; they can contain participant information. Analysis reports may describe captures that are therefore unavailable in a fresh clone, and capture-replay commands require those local files. The Kaggle images under `data/` and new experiment runs under `runs/` are also ignored; see the download and extraction steps in `REPRODUCE.md`.

The concise paper and saved-capture feature review are Dutch working versions. The compiled appendix is also Dutch. Existing live interfaces, workbook labels and archived technical sources largely retain English. Legacy category/viewpoint IDs, session names and filenames remain unchanged for compatibility with the models and evidence.

### Live controleren en voorbeelden bewaren

De browsertool draait op `http://localhost:8501` wanneer Streamlit actief is:

```bash
python -m streamlit run app.py
```

Kies links een van zes actuele modellen; oudere kantelvarianten staan achter **Meer modellen uit eerdere proeven**. Zet **Camera aan**, vul de **Bedoelde letter** in en gebruik **Meetmoment bewaren** voor het laatst geanalyseerde beeld, inclusief model, voorspelling en landmarks. Ook een mislukte detectie kan worden bewaard. Het doelletterveld beïnvloedt de voorspelling niet. De cameraweergave is gespiegeld; de detector krijgt het oorspronkelijke beeld.

Voor G/H vergelijk je **World joints [2]** met **G/H y180 only [9]**. De eerste is de 835-inputreferentie; de tweede voegt alleen bij de training gedraaide G/H-beschrijvingen toe. De zes huidige livekeuzes zijn bestaande deploymentmodellen: ze vormen niet de volledige achtvariantenmatrix uit bijlage B. Los gekozen opnamen zijn foutdiagnostiek, geen representatieve precisie-/F1-test.

De native tool heeft dezelfde zes hoofdkeuzes, met de overige modellen achter **A**. **Spatie** bevriest een beeld; kies de bedoelde letter en bevestig met **Enter**. **N** begint een nieuwe ronde, **L** toont/verbergt handpunten en **Q/Esc** sluit de camera:

```bash
python -m research.live_camera
```

Beide tools bewaren lokaal onder `captures/confusions/`. Voor een betere evaluatie is een geplande ronde met goede én foute pogingen nuttiger dan alleen confusions. Zie [wat nog nodig is en het testplan](docs/LIVE_TESTPLAN_NL.md).

### Opgeslagen vingerkenmerken bekijken

```bash
python -m streamlit run app.py -- --capture captures/confusions/20261031T152116143793Z_11756e66
```

Dit opent een bestaand lokaal handbeeld zonder camera of training. Je kunt de vergelijkingsletter wijzigen; de voorspelling blijft daarvan onafhankelijk. Bijlage C toont de eigen screenshot van de tool, twee scherp heropgemaakte detailkaartjes en zes schema’s van de handkenmerken, naast de G/H- en spiegelvoorbeelden in de andere bijlagen. [Invoegplek en herkomst](docs/APPENDIX_VISUALS.md). De genoemde opname moet lokaal aanwezig zijn.

## Direction features versus explicit joint angles

A separate controlled comparison uses the same normalized image landmarks for Shin directions and joint measurements. It tests the 820-input reference, a 205-input replacement (distances + joints), and the 835-input combination with both SVM and LightGBM on the frozen five folds. The previous world-angle model is a contextual fourth arm. The main paper reports the near-zero addition result and Appendix B gives the relevant comparison. See [results and all-letter effects](results/direction_joint_01/report.md) and [reproduction commands](REPRODUCE.md#same-coordinate-direction-versus-joint-angle-experiment). No live model is replaced by this study.

## Simple live camera (recommended)

```bash
conda activate ktai-hand-gesture
python -m research.live_camera
```

A single fixed-size window displays the camera image and current letter. Select a trained model using the buttons below the image or keys **1-9** (or **0** for Right normalized), without restarting:

| Button | Model |
|---|---|
| **Reference [1]** | Shin reference: distances and directions (820 inputs) |
| **World joints [2]** | Reference plus estimated world-joint measurements (835 inputs) |
| **All-letter tilt [3]** | Joint-angle model with all-letter ±15° tilt training |
| **C-only tilt [4]** | Joint-angle model with ±15° tilt training for C only |
| **G/H rotation [5]** | Joint-angle model with larger direction rotations for G/H only |
| **Image + depth [6]** | Reference plus 3D and projected 2D joint angles (849 inputs; no augmentation) |
| **Fresh original [7]** | Newly detected original-only training, matched baseline for the mirror experiment (835 inputs) |
| **Mirror trained [8]** | Same fitting sources plus accepted horizontal RGB mirrors (835 inputs) |
| **G/H y180 only [9]** | Standalone G/H vertical-axis 180° training turns; original data only, without mirrors or camera calibration |
| **Right normalized [0]** | Canonical detector-Right landmarks before features, in training and inference; one original row per source (835 inputs) |

The default is World joints (called Joint angles in earlier reports), the earlier **world-coordinate** model. It is not the same-coordinate +4 variant; the two new control variants remain research-only. The augmented models are experimental comparisons, not generally better models. Model selection is locked during a guided round. Snapshot exports preserve the actual model hash and its 820/835/849-input schema.

The camera window keeps a fixed layout while models switch. No workbook, feature panels or browser reruns are involved. The snapshot button opens a frozen-frame letter annotation only when requested. Hand landmarks are shown by default; press **L** to toggle them. Overlays never change the detector input. A dash means no accepted hand detection. Press **Q** or **Esc** to close. Close other camera apps first. For a different camera, use `--camera 1`; on macOS allow camera access for your terminal/Python if requested.

### Save a confusion for analysis

Click **Bewaar verwarring** below the camera or press **Space** when you see a mistake. The currently displayed frame freezes. Type the intended letter (A-Z), then press **Enter** to save it; **Esc** cancels. The last intended letter is prefilled for the next capture, and Backspace clears it if unknown. Q is a valid letter while annotating; it only quits in live mode.

Each tool launch starts a new round with a unique ID. **New round / N** starts another round without restarting the camera. Switching models stays within the current round, allowing comparisons. The round label/number appears below the buttons and is saved as `capture_round`, alongside `session_id` and the exact active model in each `capture.json`. Previous captures remain untouched; older ones already contain a `session_id` but no friendly round label.

Optionally name the round when starting:

```bash
python -m research.live_camera --round-name "GH-back-of-hand"
```

Each snapshot creates a new folder under `captures/confusions/` containing:

- `frame.png`: original unmirrored camera photo.
- `model_input.png`: exact unmirrored image passed to the landmark detector.
- `preview.png`: mirrored preview with the predicted letter and visible landmarks.
- `capture.json`: intended/predicted letters, timestamps, landmarks and detection status, 820, 835 or 849 raw and preprocessed features matching the chosen model, joint angles when used, all class decision scores, image checksums and model SHA256/provenance.

Nothing is recorded until you request and confirm a snapshot. Captures stay local and are ignored by Git. Select another destination with `--output /absolute/path/to/confusions`. These manually chosen examples are for error analysis; they do not estimate live accuracy and are not automatically added to training.

The model recipe scored **97.748% accuracy** and **0.97524 macro-F1** on the shared five-fold accepted-image evaluation. The deployment model was then fitted on all 60,251 accepted images; it has no separate final-test or webcam accuracy measurement.

The model is included under `models/`. If it is missing, generate it with `python -m research.live_model`. The tool accepts `--model /absolute/path/to/model.joblib`.

## Guided tilt experiment: C, D and O

You do not need to manually edit or add training images. First collect planned webcam comparisons with the current model, separately from training:

```bash
python -m research.live_camera --guided --role development
```

The window guides 27 attempts: C/D/O × wrist neutral/up/down × three repetitions. Keep your finger shape and camera position fixed; tilt at the wrist only, and relax/reform the hand between repetitions. The tilt names are qualitative instructions, not measured angle targets. Press Space or click the button for each planned attempt, inspect the frozen pose and press Enter. Esc retries the current pose; Q exits in live mode. Predictions are hidden during collection to reduce selection of only correct examples. Save every planned attempt, including incorrect predictions and detection failures.

At a separate time, collect an evaluation session:

```bash
python -m research.live_camera --guided --role evaluation
```

The terminal prints the new session directory under `captures/guided/`. Resume an interrupted round with:

```bash
python -m research.live_camera --guided --resume captures/guided/YOUR_SESSION_DIRECTORY
```

The app resumes from finalized capture records, locks the baseline model version, and keeps development/evaluation roles separate. Both roles remain outside training. Use the development session to choose augmentation of the existing Kaggle training data, then compare the fixed baseline and chosen variant on the reserved evaluation session. Each saved record contains the planned letter, qualitative tilt, repetition, session role and the usual image/landmark/feature data. Collection does not automatically retrain a model or change its reported scores. A second session from the same signer tests session transfer, not signer-independent performance.

## Try the trained tilt-augmentation variant

```bash
python -m research.live_camera --model models/asl_shin_angles_tilt15.joblib
```

The original default model remains `models/asl_shin_angles.joblib`. The new experiment augments the 630 Shin direction features with -15°/+15° camera-x rotations, preserving distances and joint angles. The candidate has already been trained locally; fresh clones can reproduce it with `python -m research.studies.tilt_augmentation --stage train`.

Results are mixed: development captures improve from 17/27 to 18/27 (one additional O; C unchanged). On the same five Kaggle folds, accuracy changes from 97.74776% to 97.70294% and macro-F1 from .975244 to .974686. This does not establish a general improvement. [Full results and reproduction](results/tilt_augmentation_01/report.md).

## C-only tilt experiment

```bash
python -m research.live_camera --model models/asl_shin_angles_C_tilt15.joblib
```

This separate candidate augments only the 1,912 C training images with -15°/+15° direction rotations; the other 58,339 originals remain single examples. It uses 64,075 fitting rows, with original total weight per source image preserved. On the 27 development captures its predictions are unchanged from baseline (17/27 correct; C 3/9). Kaggle five-fold accuracy is also unchanged at 97.74776%; macro-F1 differs by less than 0.000001. [Details and reproduction](results/tilt_C_only_01/report.md). The default model remains the original baseline.

## Feature inspection app (optional)

```bash
conda activate ktai-hand-gesture
python -m streamlit run app.py
```

Enable **Webcam on**; the app displays **Recognized letter** using the full-data SVM with 820 Shin descriptors and 15 joint-angle measurements. A typed target letter is optional and only controls the Excel-feature comparison. Predictions use each current frame; no letter persists when no hand is detected. Alternatives are SVM rankings, not calibrated probabilities.

The trained artifact is `models/asl_shin_angles.joblib`, included in version control. If it is missing, train it once before starting the app:

```bash
python -m research.live_model
```

Use `--force` to retrain or `python -m streamlit run app.py -- --model /absolute/path/to/model.joblib` to select another compatible local bundle. The Excel reference remains the sole original workbook in `assets/`.

The world-joint comparison changes SVM accuracy from 97.637% to 97.748% on 60,251 accepted detections. The complete interpretable package does not improve SVM, and same-coordinate joints yield only four net extra correct images. These exploratory comparisons do not establish a generally improved recognizer. Superseded work is in the reversible sibling archive recorded in `CLEANUP.json`.

## Experimental 2D + 3D angle model

```bash
conda activate ktai-hand-gesture
python -m research.live_camera --model models/asl_shin_angles_2d.joblib --round-name "2D-3D-test"
```

The new **Image + depth [6]** model adds 14 aspect-correct projected joint angles to World joints. The unchanged five-fold comparison scores 97.693% accuracy and .97465 macro-F1, slightly below World joints (97.748%, .97524). It corrects T in the latest 11 selected webcam errors, while G remains wrong; this is an experimental comparison, not a demonstrated general improvement. See `results/angles_2d_01/report.md` for all-letter results, fold variation and development replays.

## Live G/H-only diagnosis

```bash
conda activate ktai-hand-gesture
python -m research.live_gh --variant middle
```

This separate native window always chooses G or H; other letters are outside its scope. Buttons/keys 1-4 select all 3D angles, all 2D angles, both, or middle-PIP-only. L toggles the landmark overlay; Q/Esc closes. Close the ordinary camera window before starting this one. The selected development examples motivated the middle-only model; recognizing them does not establish independent live accuracy. The ordinary nine-model tool remains `python -m research.live_camera`.

## Horizontal RGB mirroring

The [mirror experiment](results/mirror_01/report.md) compares original-only and mirror-augmented training with SVM and LightGBM. All 78,000 originals and 78,000 reflected images are detected anew with the same fixed settings. Evaluation includes originals, mirrors and a fixed random mix selecting half the sources per letter/fold for reflection. Failed detections remain in all-attempt scores; accepted-only and paired-source results are also available.

On the fixed balanced test mix, SVM macro-precision changes **73.566% -> 97.451%** and macro-F1 **0.53849 -> 0.84560**, with **22,349 net additional correct attempts**. LightGBM macro-F1 changes **0.64104 -> 0.84177**. Both improve on this mix in all five folds. On original-only attempts, SVM loses 26 correct predictions net (F1 0.84670 -> 0.84640). On accepted balanced-mix images alone, its F1 changes 0.62650 -> 0.97421. These all-attempt and conditional scores have different denominators; neither should be confused with the earlier 97.748% conditional accuracy.

Historically, MediaPipe labels **95.31%** of accepted original images Right and **4.69%** Left. The selected test mix has **49.12% detector-labelled Left and 50.88% Right** among accepted detections. These are detector labels, not verified signer handedness. Synthetic mirroring tests a controlled shift; it does not establish real left-handed signer performance. Use **Fresh original [7]** versus **Mirror trained [8]** for a matched live comparison. See [protocol and reproduction instructions](docs/MIRROR_EXPERIMENT.md).

## Standalone G/H rotation

The [standalone G/H experiment](results/rotation_GH_y180_01/report.md) compares original-only training with extra 180° turns around the vertical camera y axis for G/H only. It uses the same 60,251 accepted original images, five folds, preprocessing, resolved SVM gamma and class weights. No mirrors, webcam calibration or routing rule enter either arm. Validation images are unchanged originals. Compare **World joints [2]** with **G/H y180 only [9]** live; [protocol and commands](docs/GH_ROTATION_STANDALONE.md) describe the experiment.

Macro-precision changes **97.53902% -> 97.53874%** and macro-F1 **0.97524391 -> 0.97523305**: 4 errors corrected, 5 correct predictions lost, net **-1** of 60,251. G/H recall is unchanged and both letters' F1 decreases slightly. A known back-view H is recovered but two known back-view G examples remain wrong. These selected cases are diagnostic, not an independent test. The paper reports this negative result separately in Appendix E.

**Aanvullende controle:** de oorspronkelijke toets hierboven meet behoud op de oude dataset, inclusief palmgerichte G/H-labels. De training voegt een tweede geometrie toe; de toets wordt daar niet gedraaid. Als uitsluitend de G/H-toetslandmarks wél worden gedraaid, stijgt macro-F1 door rotatietraining van **0,869284 naar 0,975715** (macro-precisie 85,0689% naar 97,5861%). Het model leert de toegevoegde geometrie dus wel. Dit is geen toets met echte handrugfoto's of een bevestiging van de ASL-correctheid van datasetlabels.

Een verkenning met grove palm-/vingervolumes geeft **0/128 geaccepteerde renders**, tegenover 64/64 oorspronkelijke foto's. Een afzonderlijke proxy voegt 2% of 5% puntverstoring toe aan vermoedelijk afgedekte vingerpunten na rotatie. Tien nieuwe foldmodellen geven op beide bovenstaande toetsen geen extra winst boven alleen rotatie. Deze aannames zijn geen gemeten MediaPipe-afdekkingsfouten; er wordt geen nieuw live-model gekozen. [Resultaten en protocollen](results/gh_occlusion_01/), [commando's](REPRODUCE.md#g-h-afdekking-verkennende-volumemodellen-en-puntverstoring) en bijlage E bevatten de uitwerking.

Vier nieuwe vrije live-opnamen tonen de resterende beperking: beide G's met zichtbare handrug blijven na rotatietraining H; een H wordt wel hersteld. De geslaagde G toont de palm. De geschatte wereldgewrichtshoeken verschillen van de oorspronkelijke G-training, terwijl een zuivere rotatie die hoeken behoudt. De hoge kunstmatige score bewijst dus geen oplossing voor live-herkenning. Deze geselecteerde voorbeelden zijn foutanalyse, geen nauwkeurigheidsmeting; ze worden niet voor training gebruikt. Zie [opgeslagen analyse](results/gh_live_diagnostics_20261107/) en bijlage E.4.

## Earlier combined G/H adaptation

The earlier [combined G/H adaptation](docs/GH_VERTICAL_ADAPTATION.md) adds vertical turns, three camera calibration images and a routing gate on top of mirror training. It remains available with `--model models/asl_gh_vertical_adapter.joblib`, for comparison with **Mirror trained [8]**. Button **9** now selects the standalone experiment above. Known calibration examples are training data, so their recognition does not establish independent live accuracy. [Results and regressions](results/gh_vertical_01/report.md) are also separate from the earlier [index-axis roll](docs/GH_BACK_ADAPTATION.md), which preserves pointing direction.

The balanced source-data check corrects 11 errors and loses 5 correct predictions (net +6); all-attempt macro-F1 changes from 0.845599 to 0.845682. Rotation alone fixes the known back-facing H but leaves the two G examples wrong. Adding their calibration data fits all three. This is a small development result, not evidence that the live viewpoint problem is solved.

## Next G/H rotation experiments

The existing **G/H rotation [5]** candidate rotates landmark-direction features, not training photos. Rotating RGB images and rerunning MediaPipe is a different experiment, as is capturing the back of a real hand. See [the G/H rotation experiment plan](docs/GH_ROTATION.md) before making another candidate. Existing models and reported results remain the fixed reference for that next experiment.

## Guided G/H viewpoint collection

```bash
conda activate ktai-hand-gesture
python -m research.gh_view_capture
```

Three participants, G/H, four views (palm, oblique palm, oblique back, back), five repetitions: **120 planned photos, 40 per person**. This supersedes the earlier 60/90-photo three-view proposal. Predictions are hidden; participant transitions require Enter. Use `--participants 2` for 80 photos. Photos are separate from training. P01 is development; P02/P03 are reserved evaluation. Full instructions, the four-view illustration and resume/status commands are in [docs/GH_CAPTURE.md](docs/GH_CAPTURE.md).

## Named alphabet test with an example for every letter

```bash
conda activate ktai-hand-gesture
python -m research.alphabet_capture
```

The participant form records name/code, development/new-person status, ASL fingerspelling experience and hand. Five A-Z rounds give 130 photos per person. A fixed dataset example appears beside the camera; **Space saves directly**, **H** marks extra guidance, and Q/Esc pauses. Predictions are hidden. The same stored images can later be scored by both frozen models, separately for new and development participants. See [the complete instructions](docs/ALPHABET_CAPTURE.md), including resume commands and guidance rules. Dataset examples are not independently validated ASL instructions; this is an example-guided static reproduction task.

## Links/rechts normaliseren vóór de features

De afzonderlijke variant **Right normalized [0]** spiegelt detector-Left-handpunten naar detector-Right vóór de featureberekening, zowel in training als bij voorspellen. Er worden geen extra trainingsbeelden toegevoegd. `Left`/`Right` zijn detectorschattingen, geen onafhankelijk gecontroleerde handlabels. De webcamweergave blijft het oorspronkelijke camerabeeld tonen; de modelinvoer wordt intern aangepast.

```bash
python -m research.live_camera --model models/asl_shin_angles_right_hand.joblib
```

In de bestaande camera-app kun je ook op **Right normalized [0]** klikken of **0** indrukken. Voor de featuretool:

```bash
python -m streamlit run app.py -- --model models/asl_shin_angles_right_hand.joblib
```

De vergelijking gebruikt dezelfde bronnen, vijf folds en vaste synthetische toetsmix als de eerdere spiegelproef. Resultaten en beperkingen staan in [het experimentverslag](results/handedness_01/report.md) en bijlage F. De bestaande standaardmodellen worden niet vervangen.

De primaire baseline gebruikt **alleen de 60.251 oorspronkelijke beelden met geldige handdetectie**, met dezelfde 835 inputs en dezelfde toetsbeelden voor alle drie aanpakken:

| Aanpak | Macro-precisie | Macro-F1 | Correct / 60.251 |
|---|---:|---:|---:|
| Zonder aanpassing | 97,539% | 0,97524 | 58.894 |
| Rechtsnormalisatie | 97,322% | 0,97320 | 58.758 |
| Spiegeltraining | 97,495% | 0,97493 | 58.868 |

Normalisatie geeft hier **136 minder** correcte voorspellingen dan de oorspronkelijke baseline, en **110 minder dan spiegeltraining**. De oorspronkelijke baseline heeft 97,75% nauwkeurigheid; dat is een andere maat dan de 97,54% macro-precisie.

Het aparte synthetische mengsel is een andere toetsverdeling. Na uitsluiting van mislukte detecties blijven 60.225 beelden over:

| Aanpak | Macro-precisie | Macro-F1 | Correct / 60.225 |
|---|---:|---:|---:|
| Zonder aanpassing | 73,566% | 0,62650 | 36.452 |
| Rechtsnormalisatie | 96,076% | 0,96047 | 57.987 |
| Spiegeltraining | 97,451% | 0,97421 | 58.801 |

Op dit mengsel levert normalisatie 21.535 extra correcte voorspellingen op tegenover niets aanpassen, maar 814 minder dan spiegeltraining. Dit vervangt de oorspronkelijke baseline niet en is geen onafhankelijke test met echte linkshandige deelnemers. Het uitsluiten van detectiemislukkingen verandert hier de precisie niet (er was geen letter voorspeld), maar wel recall en F1. Scores over alle aangeboden beelden staan afzonderlijk in het experimentverslag en de CSV-bestanden. Er is niet opnieuw getraind.
