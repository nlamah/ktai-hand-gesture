# Alfabetopnamen voor de vergelijking Shin versus gewrichtshoeken

## Starten

Sluit andere camera-apps. Vanuit de repository:

```bash
conda activate ktai-hand-gesture
python -m research.alphabet_capture
```

Er verschijnt eerst een venster voor naam/deelnemerscode, eerdere betrokkenheid bij ontwikkeling, ervaring met het ASL-vingeralfabet en gebruikte hand. De camera start pas na **Start opnamen**. Kies voor de onderzoeker zelf **Mijn hand is al gebruikt bij ontwikkeling**, ook als de eigen foto's niet in de trainingsdata zitten. Kies **Nieuwe persoon** alleen als diens beelden/uitkomsten de aanpak niet eerder hebben beïnvloed.

Elke persoon krijgt een eigen sessie. Standaard zijn er vijf rondes A-Z: **130 foto's per persoon**, dus 390 voor drie personen. Ontspan en vorm het volgende gebaar opnieuw; het zijn geen vijf videoframes van dezelfde aangehouden hand.

## Voorbeelden en de betekenis van de test

Rechts van het camerabeeld staat een vast voorbeeld van de gevraagde letter. [Bekijk vooraf alle voorbeelden](../assets/alphabet_capture_examples.jpg). Dit zijn bestaande geaccepteerde beelden uit de Kaggle-dataset, onder hun oorspronkelijke datasetlabels. Het zijn geen onafhankelijk gecontroleerde officiële ASL-instructies. Laat de voorbeelden bij voorkeur vóór het definitieve experiment controleren door iemand die het ASL-vingeralfabet kent; wijzig tijdens de test niet stilzwijgend de voorbeelden. De configuratie staat in `config/alphabet_capture_examples.json` en de sessie bewaart de gebruikte voorbeeldbestanden en checksums.

In nieuwe sessies tonen alle voorbeelden een rechterhand. Per voorbeeld is vastgelegd of de oorspronkelijke foto daarvoor horizontaal wordt gespiegeld. Het camerabeeld is nu ongespiegeld, zodat een rechterhand op dezelfde manier te vergelijken is met het voorbeeld. De modelinvoer blijft ongewijzigd en bevat nooit het voorbeeldplaatje. Oude sessies behouden hun eerder vastgelegde weergave; start een nieuwe sessie voor deze rechterhandvoorbeelden. Het voorbeeld is geen voorspelling. J en Z worden als statische datasetvorm vastgelegd; de beweging wordt niet geëvalueerd.

Dit is een **voorbeeldgestuurde reproductietaak**. Bij onervaren deelnemers meet je zowel het nadoen van de voorbeelden als de herkenning daarvan; je meet geen spontane, deskundige ASL-productie. Een ervaren deelnemer kent het gebaar waarschijnlijk beter, maar dat bewijst nog geen volledige taalkundige juistheid van iedere foto.

## Bediening

1. Vul de deelnemersgegevens in en klik Start opnamen.
2. Controleer de naam, gebruikte hand en rol; druk Enter om de ronde te beginnen.
3. Vorm de gevraagde letter met dezelfde hand. Houd camera, verlichting, afstand en opstelling gelijk. Breng de hele hand en pols in beeld, bij voorkeur zonder gezicht.
4. Druk **kort op Spatie** of klik op de fotoknop. De foto wordt direct bewaard; Enter per foto is niet nodig. Een korte blokkering voorkomt snel dubbel opslaan.
5. Ontspan de hand en volg de volgende letter. De tool loopt automatisch door alle vijf rondes.
6. Gebruik **H vóór de foto** om aan te geven dat je extra hulp of gebaarcorrectie hebt gegeven. De markering geldt voor die ene foto en wordt daarna uitgezet.
7. **Q/Esc** in het livevenster pauzeert en geeft de camera vrij.

Voorspellingen blijven verborgen. Bewaar ook mislukte detecties; doe een poging niet opnieuw omdat de classifier het fout heeft. De tool sluit na de laatste foto. Vraag deelnemers vooraf toestemming voor het lokaal bewaren van foto's en gebruik desgewenst een code in plaats van een echte naam.

## Hoeveel mag je begeleiden?

- **Voor de echte opnamen:** geef iedereen dezelfde uitleg en een oefenmogelijkheid zonder modeluitkomsten. Geef dan ook feedback op een verkeerd uitgevoerd gebaar.
- **Tijdens de opnamen:** standaard aanwijzingen voor opstelling mogen, zoals 'hele hand in beeld' of de vooraf afgesproken afstand.
- **'Meer handpalm naar voren':** alleen als die kijkrichting vooraf onderdeel van de gemeenschappelijke opdracht is; niet omdat een bepaalde letter anders slecht herkend wordt.
- **'Buig je pink meer':** dit verandert het gebaar zelf. Geef zo'n correctie liefst tijdens oefenen. Als het tijdens de test nodig is, gebruik dezelfde inhoudelijke maatstaf voor iedereen en markeer hulp met H. De uitkomst betreft dan begeleide uitvoering.
- **Niet doen:** de testpersoon laten schuiven/draaien tot de verborgen of elders bekeken voorspelling correct wordt, alleen geslaagde foto's bewaren, of modellen aanpassen op basis van deze eindtest.

Het ontbreken van training met een deelnemer maakt die persoon niet automatisch onafhankelijk: eigen handen en eerder bekeken mislukkingen kunnen de onderzoekskeuzes hebben beïnvloed. Ervaring en begeleiding blijven daarom in de metadata staan. Een deskundige of ervaren ASL-vingerspeller kan de labelgeldigheid versterken. Een novicegroep is ook mogelijk als dat je bedoelde gebruikssituatie is; benoem die scope en meng ervaring niet ongemerkt in één algemene prestatieclaim.

## Pauzeren en hervatten

De terminal toont bij het starten de exacte sessiemap onder `captures/alphabet_test/`. Hervat die map in plaats van voor dezelfde persoon een nieuwe sessie te beginnen:

```bash
python -m research.alphabet_capture --resume captures/alphabet_test/alphabet_DATUM_ID
```

Alleen voortgang bekijken, zonder camera:

```bash
python -m research.alphabet_capture --resume captures/alphabet_test/alphabet_DATUM_ID --status
```

Zonder invulvenster starten kan ook:

```bash
python -m research.alphabet_capture --name "Ton" --kind development --experience familiar
python -m research.alphabet_capture --name "Deelnemer 2" --kind new --experience beginner
```

Naam/rol zijn voorbeelden, geen al gemaakte deelnemersregistratie. `--repeats 1` maakt een kortere ronde van 26 foto's; gebruik voor een gezamenlijke evaluatie voor iedereen hetzelfde aantal herhalingen en dezelfde voorbeelden.

## Evalueren na afloop

Geef de werkelijk gemaakte, voltooide sessiemappen op:

```bash
python -m research.evaluate_alphabet captures/alphabet_test/SESSIE_1 captures/alphabet_test/SESSIE_2 captures/alphabet_test/SESSIE_3
```

Deze stap hergebruikt de opgeslagen beelden/landmarks en de twee vooraf vastgelegde modelbestanden. Niemand hoeft voor een tweede model opnieuw foto's te maken. Er wordt niets getraind.

De uitkomst staat onder `results/alphabet_evaluation/`: rapport, voorspellingen, precision/recall/F1 per letter, confusion matrices en samenvatting. Er zijn scores per persoon, voor alle personen en apart voor nieuwe personen. Bekende ontwikkelpersonen worden niet als onafhankelijke nieuwe gebruikers behandeld. De tabel telt ontbrekende detecties als onjuiste pogingen; daarnaast staan scores voor alleen geaccepteerde detecties en de detectiedekking in `summary.json`. Hulpmarkeringen blijven bewaard, zonder achteraf ongunstige foto's stilzwijgend weg te laten.

Dit is een kleine nieuwe-personencontrole, geen representatieve bevolkingsstudie. Zodra je deze uitkomsten gebruikt om features of modellen te kiezen, is voor een volgende onafhankelijke eindbeoordeling nieuw materiaal nodig.
