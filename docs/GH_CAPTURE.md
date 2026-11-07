# G en H opnemen met duidelijke handposities

De twee recente sessies van 7 november zijn ingetrokken omdat de uitvoering niet goed ging. Hun opnamen en afgeleide resultaten zijn uit de actieve verzameling en appendix gehaald. Een herstelbaar archief staat onder `runs/discarded_gh_20261107/`; details en hashes staan in `results/gh_capture_reset_20261107/withdrawal.json`. Een ingetrokken sessie kun je niet hervatten.

## Opnieuw beginnen

Sluit het oude cameravenster. Start vanuit de repository een nieuwe ronde voor alleen je eigen hand:

```bash
conda activate ktai-hand-gesture
python -m research.gh_view_capture --participants 1
```

Er worden 40 pogingen gepland: twee letters, vier aanzichten en vijf herhalingen. Je kunt tussendoor pauzeren. Voor twee of drie personen gebruik je `--participants 2` of `--participants 3`; zonder argument is de standaard drie. Iedere nieuwe start krijgt een nieuw sessiepad.

## Wat moet je doen?

Rechts van het camerabeeld staan nu grote Nederlandse aanwijzingen:

1. **Hand:** gebruik de getoonde fysieke rechter- of linkerhand. De cameraweergave kan gespiegeld zijn; verwissel je hand daarom niet op basis van de schermrichting.
2. **Letter:** bij **G** strek je wijsvinger en duim ongeveer evenwijdig en buig je de andere drie vingers. Bij **H** strek je wijs- en middelvinger naast elkaar; ringvinger en pink zijn gebogen en de duim is ingevouwen.
3. **Aanzicht:** maak eerst de letter en draai vervolgens je hele hand naar het gevraagde aanzicht. Houd de vingerstand vast. De wijsrichting op het scherm mag daarbij veranderen.

| Aanwijzing | Wat ziet de camera? |
|---|---|
| HANDPALM NAAR CAMERA | De palm; de knokkels wijzen van de camera af. Dit is een experimentele palmcontrole. |
| SCHUIN: PALM ZICHTBAAR | Begin bij de palmcontrole en draai ongeveer 45 graden; een deel van de palm blijft zichtbaar. |
| SCHUIN: RUG ZICHTBAAR | Begin met de handrug naar de camera en draai ongeveer 45 graden; een deel van de handrug blijft zichtbaar. |
| HANDRUG NAAR CAMERA | De knokkels en handrug; de palm wijst van de camera af. |

Het kleine schema is een **bovenaanzicht**: oranje is de palmzijde, blauw de handrug en de camera staat onderaan. **De pijl wijst vanaf de palm, niet langs de vingers.** Het schema geeft alleen het aanzicht aan; het is geen foto of anatomisch handmodel. De genoemde hoeken zijn kwalitatieve aanwijzingen, geen gekalibreerde metingen.

Voor de gebruikelijke ASL-uitvoering van G/H beschrijft [ASL University](https://www.lifeprint.com/asl101/fingerspelling/fingerspelling3.htm) een naar binnen of opzij gerichte palm. De palmcontrole is een onderzoeksconditie. Bekijk de uitvoering vooraf en laat die zo mogelijk controleren door iemand met ASL-kennis.

## Bediening

- **Enter bij de start:** bevestig persoon en hand en begin de ronde.
- **Spatie of fotoknop:** bevries het gekozen camerabeeld.
- **Enter na de foto:** bewaar de poging; de bedoelde letter staat vast.
- **Esc bij de bevestiging:** opnieuw proberen bij een procedurefout, zoals een verkeerde letter of onbedoelde toetsdruk.
- **Q/Esc in het livebeeld:** pauzeer en sluit de camera.

De voorspelling blijft verborgen. Bewaar bij een geplande ronde ook fouten en mislukte detecties. Ontspan de hand en vorm het gebaar opnieuw na iedere foto. Houd camera, afstand, licht en de gebruikte hand gelijk. Zorg dat hand en pols volledig in beeld zijn.

## Pauzeren en hervatten

De terminal toont het nieuwe sessiepad. Gebruik datzelfde pad om later verder te gaan:

```bash
python -m research.gh_view_capture --resume captures/gh_viewpoint/GH_views_DATE_ID
python -m research.gh_view_capture --resume captures/gh_viewpoint/GH_views_DATE_ID --status
```

`GH_views_DATE_ID` is een voorbeeld: vul het werkelijke nieuwe mapje in. Ingetrokken sessies zijn geblokkeerd. Nieuwe sessies leggen instructieversie `explicit-gh-pose-v3` en weergaveversie `large-nl-gh-guidance-v1` vast; meng instructieversies niet ongemerkt.

## Opslag en interpretatie

Elke foto bevat het oorspronkelijke beeld, de exacte detectorinvoer, een preview zonder voorspelling en `capture.json`. Daarin staan doelletter, persoon, hand, aanzicht, detectiestatus, landmarks en modelgegevens. De vergelijking gebruikt later dezelfde beelden voor beide vooraf vastgelegde modellen. Er wordt bij de opname niet getraind.

P01 is ontwikkeling. Alleen als je meer personen toevoegt, zijn P02/P03 bedoeld voor een vooraf afgesproken controle; pas de modellen niet op die resultaten aan. De opnamen blijven lokaal onder `captures/gh_viewpoint/` en worden niet automatisch trainingsdata.

## Afdekking van vingers nabootsen

Alleen landmarks draaien genereert geen afdekking. Voor een realistischer synthetisch experiment kun je een handmodel met oppervlakken en dikte vanuit de rugzijde renderen en MediaPipe opnieuw op die beelden toepassen. Daarmee kan een vinger achter een andere vinger of de palm verdwijnen. Een eenvoudig model met volumes voor vingers en palm is een grove benadering; pasvorm, huidskleur, belichting en materiaal blijven onzekerheden.

Alleen punten verwijderen is een andere proef: MediaPipe schat normaal een volledige handpose, ook bij gedeeltelijke zichtbaarheid. De [officiële documentatie](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md#hand-landmark-model) beschrijft zowel zelfafdekking als training met gerenderde handbeelden. Puntverstoring of maskering kan een robuustheidstest zijn, maar meet niet automatisch echte detectiefouten bij een handrug.

Deze benadering is inmiddels verkend: een grof handvolume rond bestaande landmarks levert na rendering geen geaccepteerde handdetecties op (0/128, tegenover 64/64 echte bronfoto's). Een afzonderlijke proxy verstoort alleen vermoedelijk afgedekte vingerpunten na rotatie, met twee vaste sterktes. Daarvoor zijn tien foldmodellen getraind. Ze geven op de oorspronkelijke en kunstmatig gedraaide toets geen extra winst boven alleen rotatie. Dit zijn geen gevalideerde nieuwe handrugbeelden. De live-modellen zijn niet vervangen. Zie bijlage E en [de opgeslagen resultaten](../results/gh_occlusion_01/); reproduceercommando's staan in [REPRODUCE.md](../REPRODUCE.md).
