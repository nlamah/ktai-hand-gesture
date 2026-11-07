# Beelden en uitleg in de bijlage

De nieuwe Nederlandstalige bijlagen bevatten:

- **E – Geroteerde trainingslandmarks:** G1128 en H1198 met hun opgeslagen trainingshandpunten, daarnaast het oorspronkelijke en het 180° om de verticale camera-as gedraaide skelet. Pols 0 is het draaipunt; x en z wisselen van teken, y blijft gelijk. De uit de rotatie herberekende features zijn gecontroleerd tegen de daadwerkelijke trainingsaugmentatie. Dit genereert geen nieuwe handrugfoto. Bronnen en coördinaten staan in `results/gh_training_rotation_01/`.
- **C – Featurecontrole (twee pagina’s):** zes schema’s voor buiging, vingers naast/over elkaar, spreiding van wijs–middel en middel–ring, duim over/naast de palm en duimtopcontact. Daarna de echte aangeleverde C-screenshot met twee scherp heropgemaakte detailkaartjes, plus meetregels. De screenshot vervangt de placeholder.
- **D – Spiegeling:** A1208.jpg en H1198.jpg in hun oorspronkelijke en gespiegelde presentatie. De voorspellingen komen uit de spiegelstudie, fold 5. Zonder spiegeltraining worden de spiegelbeelden M en U; met spiegeltraining A en H. Dit zijn geselecteerde synthetische voorbeelden, geen gecontroleerde reeks echte linkshandige deelnemers.

De screenshot is op 7 november 2026 door de gebruiker aangeleverd; het bestand is ongewijzigd gekopieerd. De recente G/H-cameraopnamen zijn op verzoek ingetrokken wegens een onjuiste uitvoering; hun figuur en conclusies zijn uit de appendix verwijderd. Er zijn bij de overige figuurexports geen nieuwe detectorresultaten gemaakt. Uitsneden en handpuntlijnen dienen de leesbaarheid; modellen ontvangen de opgeslagen oorspronkelijke geometrie. Bronlabels, beelden, voorspellingen en parameters zijn niet aangepast om de illustraties te laten kloppen. De bronbeelden en opnameregistraties blijven lokaal.

## De controle zelf openen

```bash
conda activate ktai-hand-gesture
python -m streamlit run app.py -- --capture captures/confusions/20261031T152116143793Z_11756e66
```

Je kunt ook het pad naar `capture.json` doorgeven. Deze weergave start geen camera en traint geen model. Standaard is de bestaande wereldgewrichtreferentie geselecteerd; `--model` kan een ander vertrouwd lokaal model kiezen. De cameraweergave zonder `--capture` blijft beschikbaar.

## De featurebeelden beheren

De echte screenshot staat in **`paper/figures/feature_tool_screenshot.png`** en blijft ongewijzigd. De twee oorspronkelijke detailuitsneden waren slechts circa 190 pixels breed en werden onscherp bij vergroting. Daarom zet `paper/appendix_tool_nl.tex` dezelfde verwachte/gemeten categorieën opnieuw als Nederlandse vectortekst, met een rood of groen kader. Het bijschrift benoemt deze heropmaak. De kaartjes zijn geen nieuwe meetresultaten. De oorspronkelijke tekst, pixelvakken, vertaling en bronhash staan in `results/feature_illustrations_01/detail_cards.json`.

De vectorillustraties staan in `paper/figures/feature_concepts_nl.pdf` en zijn uit tekeninstructies gemaakt, niet uit gemeten handpunten. Ze zijn uitleg, geen aanvullende experimentele resultaten. Ze maken twee verschillen expliciet: niet gespreid betekent niet automatisch aanraken; duim over/naast de palm is het bedoelde concept, terwijl de huidige software dit benadert met de getekende duim-MCP-afwijking (B-vormig of niet B-vormig). Kruising blijft een vermoeden op basis van geometrie.

De eerdere H/G-rapportweergaven blijven beschikbaar als bronmateriaal, maar staan niet in de PDF. Een overeenkomst met het werkboek bevestigt niet automatisch correcte anatomie of ASL-uitvoering. Bij een andere screenshot moeten het bijschrift en de overgenomen kaartinhoud in `paper/appendix_tool_nl.tex` worden gecontroleerd.

## Figuren opnieuw maken

```bash
MPLCONFIGDIR=/tmp/ktai-mpl python -m research.studies.feature_illustrations
MPLCONFIGDIR=/tmp/ktai-mpl python -m research.studies.appendix_visuals
```

Nodig zijn de oorspronkelijke lokale opnamemappen, de Kaggle-dataset, `runs/mirror_01/detections/` en de bestaande referentie-/G/H-y180-modellen. De generator controleert opnamehashes, beide spiegelpixelhashes en modelvoorspellingen. Het selectiebeleid kiest per A/H het eerste passende bron-ID; het is een illustratieve selectie, geen schatting van een succespercentage.

De controleerbare brongegevens staan in `results/appendix_visuals_01/provenance.json`, `feature_comparisons.json` en `mirror_selection.json`. Afgeleide PDF-figuren staan onder `paper/figures/`; leesbare PNG's staan naast de brongegevens.

De schematische featurefiguur en de ongewijzigde screenshot hebben aparte herkomstregistratie in `results/feature_illustrations_01/provenance.json`. De bestaande camerabeeld- en spiegelregistraties blijven ongewijzigd.

De ingetrokken cameraopnamen en afgeleide figuur staan uitsluitend in een herstelbaar, genegeerd archief onder `runs/discarded_gh_20261107/`. De registratie staat in `results/gh_capture_reset_20261107/withdrawal.json`. De oude export wordt geblokkeerd; gebruik de opnamen niet als onderzoeksbewijs.

De figuur van de trainingsrotatie gebruikt het ongewijzigde historische manifest en de bijbehorende featurecache:

```bash
MPLCONFIGDIR=/tmp/ktai-mpl python -m research.studies.gh_training_rotation_figure
```

Dit schrijft `paper/figures/gh_training_rotation.pdf`. Er worden geen modellen getraind of beelden opnieuw gedetecteerd.
