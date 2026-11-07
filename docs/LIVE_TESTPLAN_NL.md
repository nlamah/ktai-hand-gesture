# Wat nog nodig is voor de afronding

Status 7 november 2026: de twee recente G/H-sessies (22 opnamen) zijn op verzoek ingetrokken wegens een onjuiste uitvoering. Ze staan niet meer in de actieve opnamen, resultaten of appendix. We gebruiken nu op jouw voorkeur de vrije live-tool: zelf het model kiezen, de voorspelling bekijken en interessante verwarringen bewaren. De begeleide ronde hieronder blijft beschikbaar voor een eventuele latere effectmeting.

## Nu: zelf modellen vergelijken en confusions bewaren

```bash
conda activate ktai-hand-gesture
python -m research.live_camera --model models/asl_shin_angles_GH_y180.joblib
```

Dit opent de vrije camerawindow met de G/H-rotatievariant geselecteerd. Gebruik de modelknoppen of **2** voor **World joints** (referentie) en **9** voor **G/H y180 only** (rotatietraining). De voorspelling blijft zichtbaar en je kunt tussendoor van model wisselen.

Bij een interessante verwarring: **Spatie** bevriest het beeld, typ de bedoelde letter en druk **Enter**. De tool bewaart onder `captures/confusions/` een `preview.png`, het oorspronkelijke beeld, de exacte detectorinvoer en `capture.json` met model, voorspelling en landmarks. Het terminalvenster toont de opslagmap. Je hoeft dus geen losse schermafbeelding te maken. **L** schakelt de getekende handpunten aan/uit, **N** begint een nieuwe ronde en **Q/Esc** sluit de tool.

Voor G/H toon je de gewenste uitvoering met handrug of zijkant naar de camera. Bewaar vooral de gevallen die je wilt bespreken; enkele geslaagde voorbeelden helpen bij het vergelijken. Deze vrije voorbeelden dienen voor foutanalyse. Voor precisie/F1 is een vaste reeks nodig, zoals het optionele protocol hieronder.

De eerste vier nieuwe voorbeelden zijn opnieuw beoordeeld met beide bestaande modellen op exact dezelfde opgeslagen landmarks. Twee G's met zichtbare handrug worden na rotatietraining H; de H wordt wel hersteld en de herkende G toont de palm. De geschatte gewrichtshoeken verschillen van de oorspronkelijke G-training en dus van een zuiver gedraaid trainingsskelet. Dat ondersteunt de beperking van de kunstmatige toets: die leert de aangeboden geometrie, maar bootst geen nieuwe detectoruitvoer na. Het stelt niet vast of de verschillen door uitvoering of MediaPipe ontstaan. De vier geselecteerde beelden zijn geen nauwkeurigheidsmeting en worden niet gebruikt voor training. Analyse: `results/gh_live_diagnostics_20261107/`; voorbeelden: bijlage E.

De grootste inhoudelijke meerwaarde komt nu van nieuwe, vooraf afgesproken waarnemingen en een complete gezamenlijke paper.

## 1. Eerst de G/H-herstelhypothese toetsen

De huidige toetsfoto’s komen uit dezelfde dataset als de oefenbeelden. De bijna gelijke score na rotatie zegt vooral dat de herkenning op die dataset behouden blijft. Of de rotatie de gewenste uitvoering met zichtbare handrug beter herkent, vraagt nieuwe opnamen.

**Aanvullende controle uitgevoerd:** als uitsluitend de G/H-toetslandmarks kunstmatig worden gedraaid, stijgt macro-F1 door rotatietraining van 0,8693 naar 0,9757. Het model leert de tweede geometrie dus wel. Dit is geen echte handrugtoets: er zijn geen nieuwe camerabeelden of detectorfouten. Palmgerichte G/H-datasetlabels bevestigen ook geen correcte ASL-uitvoering; het behouden daarvan is een controle op het oude aanzicht, niet het gebruiksdoel.

Ook de voorgestelde afdekking is verkend met bestaande landmarks. Een grof palm-/vingervolume verbergt daadwerkelijk delen via een dieptebuffer, maar MediaPipe accepteert geen van 128 renders (64 bronnen × twee aanzichten); alle 64 echte bronfoto’s worden wel geaccepteerd. Een afzonderlijke training met alleen verstoring van vermoedelijk afgedekte punten (2% of 5% van de palmlengte per coördinaat) geeft in de oorspronkelijke en kunstmatig gedraaide toets geen extra winst boven alleen rotatie. Dit zijn verkennende aannames, geen gevalideerde simulatie van echte afdekkingsfouten. Resultaten: `results/gh_occlusion_01/`; uitleg en beelden: bijlage E. De live-modellen blijven de bestaande varianten.

**Optioneel, voor een latere effectmeting:** een vaste ronde van je eigen hand, liefst aangevuld met twee mensen van wie de handen nog niet tijdens de ontwikkeling bekeken zijn. Start daarvoor de bestaande begeleide G/H-opname:

```bash
conda activate ktai-hand-gesture
python -m research.gh_view_capture --participants 1
```

Met `--participants 1` plan je **40 foto’s van alleen jezelf**: G en H, vier aanzichten en vijf afzonderlijke pogingen. De grote pijl toont de palmzijde, niet de vingerrichting. Voor twee personen in totaal: `python -m research.gh_view_capture --participants 2`. Jij bent P01, de ontwikkelpersoon; P02/P03 blijven voor evaluatie. Je kunt na jouw ronde pauzeren met Q/Esc en dezelfde sessie later hervatten; de terminal toont het sessiepad. Details: [G/H-opnameprotocol](GH_CAPTURE.md).

- Vorm het gebaar bij iedere poging opnieuw; sla ook correcte herkenningen en mislukte detecties op.
- Houd hand, vingerstand, camera, afstand en licht gelijk. De wijsrichting op het scherm mag met de draaiing veranderen; forceer geen vaste links/rechts-richting. De palmbeelden zijn een experimentele controle, geen ASL-instructie.
- Laat de uitvoering zo mogelijk controleren door iemand met ASL-kennis. [ASL University](https://www.lifeprint.com/asl101/topics/feedback.htm) en [Shield e.a. (2022)](https://doi.org/10.3389/fpsyg.2022.953019) onderbouwen de palmoriëntatie. Alleen links/rechts in een gespiegeld videobeeld beoordelen is onvoldoende.
- De voorspelling blijft verborgen. Dezelfde opgeslagen beelden worden later door de vaste referentie **[2]** en rotatievariant **[9]** beoordeeld. Je hoeft geen foto’s per model te maken.
- Pas modellen niet aan op uitkomsten van P02/P03. Doe je dat wel, dan zijn die beelden ontwikkeldata geworden.

Rapporteer per persoon en aanzicht: detectiedekking, aantallen goed/fout, precisie/F1 voor G/H binnen de 26-lettertaak en gepaard herstel/verlies. Een G/H-selectie geeft geen volledige A–Z-evaluatie: fouten op de andere 24 echte letters ontbreken. Ook veel foto’s van twee nieuwe personen blijven een kleine personensteekproef.

## 2. Eventueel een bredere controle

Voor de featurevraag helpt een nieuwe A–Z-ronde meer dan nog meer geselecteerde confusions. De bestaande tool verzamelt vijf alfabetten: **130 beelden per persoon**.

```bash
python -m research.alphabet_capture
```

Deze bestaande ronde en evaluator vergelijken **Shin-basis [1]** en **wereldgewrichten [2]**. Ze toetsen niet automatisch alle acht varianten of de eigen categorieën. De getoonde voorbeelden komen uit de dataset: behandel dit als een begeleide reproductietaak en laat vooral G/H vooraf controleren. Zie [alfabetprotocol en evaluatie](ALPHABET_CAPTURE.md).

Test licht, een andere hand of camerahoek in een afzonderlijke reeks met vaste aantallen en één gewijzigde factor. De vier G/H-aanzichten zijn kwalitatief, geen gemeten hoeken. Eén gedraaide hand is geen onafhankelijke toets van linkshandige gebruikers.

## 3. Losse confusions voor foutanalyse

Bewaar opvallende gevallen met **Meetmoment bewaren** in de browser of **Spatie → doelletter → Enter** in de native tool. De opnamemap bevat het beeld, de landmarks, de bedoelde letter en het model. Noteer daarnaast hand, aanzicht, licht en wat je ziet misgaan. Geef het lokale mappad door; alleen een screenshot van de voorspelling is minder bruikbaar.

Gebruik deze gevallen om fouten te verklaren, niet als noemer voor precisie/F1. Bewaar ook succesvolle pogingen. Dezelfde beelden kunnen later met meerdere modellen worden vergeleken.

## 4. Voor de gezamenlijke paper

- Definitieve methode, code en detectieresultaten van RQ1, inclusief controle van de teruggevonden handpunten.
- Leercurve en precisie/F1-waarden van RQ2, met vaste toetsbeelden, aantallen unieke trainingsbeelden en herhalingen.
- Namen/studentnummers en inhoudelijke controle van bronnen, featurebetekenissen en conclusies door alle auteurs.

De rubricbeoordeling in de repository is historisch en indicatief. Dit plan geeft prioriteiten voor sterkere onderbouwing en volledigheid; het is geen nieuwe cijferinschatting.
