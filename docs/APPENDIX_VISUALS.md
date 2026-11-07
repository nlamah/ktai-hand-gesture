# Beelden en uitleg in de bijlage

De nieuwe Nederlandstalige bijlagen bevatten:

- **C – Featurecontrole:** uitleg van MCP/PIP/DIP, buiging, spreiding, kruising, duimvorm en contact, met een zichtbare placeholder voor een eigen foto of screenshot van de tool. De eerdere rapportweergaven zijn uit de PDF vervangen door deze invoegplek.
- **E – G/H-verwarringen:** twee datasetvoorbeelden en vier bekende camerabeelden, met handpunten en voorspellingen van referentie [2] en de losse rotatieproef [9]. Dit zijn eerder bekeken ontwikkelvoorbeelden. Ze zijn niet gebruikt voor het trainen van deze twee modellen, maar wel in andere kalibratieproeven.
- **D – Spiegeling:** A1208.jpg en H1198.jpg in hun oorspronkelijke en gespiegelde presentatie. De voorspellingen komen uit de spiegelstudie, fold 5. Zonder spiegeltraining worden de spiegelbeelden M en U; met spiegeltraining A en H. Dit zijn geselecteerde synthetische voorbeelden, geen gecontroleerde reeks echte linkshandige deelnemers.

Er zijn geen nieuwe foto's of detectorresultaten gemaakt. Uitsneden en handpuntlijnen dienen de leesbaarheid; modellen ontvangen de opgeslagen oorspronkelijke geometrie. Bronlabels, beelden, voorspellingen en parameters zijn niet aangepast om de illustraties te laten kloppen. De bronbeelden en opnameregistraties blijven lokaal.

## De controle zelf openen

```bash
conda activate ktai-hand-gesture
python -m streamlit run app.py -- --capture captures/confusions/20261031T152116143793Z_11756e66
```

Je kunt ook het pad naar `capture.json` doorgeven. Deze weergave start geen camera en traint geen model. Standaard is de bestaande wereldgewrichtreferentie geselecteerd; `--model` kan een ander vertrouwd lokaal model kiezen. De cameraweergave zonder `--capture` blijft beschikbaar.

## De eigen foto of screenshot toevoegen

Sla het beeld op als **`paper/figures/feature_tool_screenshot.png`**. Toon bij voorkeur de hand met handpunten, de gekozen letter en de verwachte/gemeten features. Bij opnieuw compileren wordt de placeholder automatisch vervangen door dit bestand. Pas daarna het TODO-bijschrift in `paper/appendix_tool_nl.tex` aan de werkelijke opname aan.

De eerdere rapportweergaven van de featurecontrole blijven beschikbaar in de resultaatmap, maar staan niet meer in de PDF. De opgeslagen-beeldenweergave van de app is getest met Streamlits AppTest. Een overeenkomst met het werkboek bevestigt niet automatisch correcte anatomie of ASL-uitvoering.

## Figuren opnieuw maken

```bash
MPLCONFIGDIR=/tmp/ktai-mpl python -m research.studies.appendix_visuals
```

Nodig zijn de oorspronkelijke lokale opnamemappen, de Kaggle-dataset, `runs/mirror_01/detections/` en de bestaande referentie-/G/H-y180-modellen. De generator controleert opnamehashes, beide spiegelpixelhashes en modelvoorspellingen. Het selectiebeleid kiest per A/H het eerste passende bron-ID; het is een illustratieve selectie, geen schatting van een succespercentage.

De controleerbare brongegevens staan in `results/appendix_visuals_01/provenance.json`, `feature_comparisons.json` en `mirror_selection.json`. Afgeleide PDF-figuren staan onder `paper/figures/`; leesbare PNG's staan naast de brongegevens.
