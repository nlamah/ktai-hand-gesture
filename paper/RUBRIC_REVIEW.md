# Herbeoordeling volgens Assignment 3

Beoordeeld: de herziene Engelse `report.pdf` en de afzonderlijke `appendix.pdf`. De hoofdvraag is nu: **In hoeverre verbeteren expliciete handvormkenmerken de ASL-letterherkenning ten opzichte van een Shin-gebaseerde referentie?** Webcam-, augmentatie- en latere G/H-proeven zijn ondersteunende diagnostiek, geen extra voorwaarden om die hoofdvraag te beantwoorden.

Dit is een AI-gegenereerde, indicatieve beoordeling op basis van de door de gebruiker gedeelde rubric. Het is geen officiële docentbeoordeling of garantie; opleidingsniveau, docentinterpretatie en het meetellen van de appendix kunnen de uitkomst veranderen. Er zijn geen nieuwe empirische gegevens verzameld voor deze revisie. De verbetering tegenover de vorige beoordeling zit in afbakening, bewijsvoering en rapportage, niet in sterkere onafhankelijke validatie.

## Indicatieve score

| Rubriconderdeel | Maximaal | Inschatting | Reden en belangrijkste beperking |
|---|---:|---:|---|
| Abstract | 5 | 4 | Motivatie, vraag, methode, aantallen, accuracy/F1 en begrensde conclusie aanwezig. De samenvatting blijft vrij informatiedicht. |
| Introduction | 10 | 8 | Expliciete hoofdvraag, drie toetsbare deelvragen, concrete motivatie en afgebakende empirische bijdrage. De bijdrage is geen nieuw algoritme. |
| Related work | 10 | 7 | Zes primaire werken worden met de eigen aanpak verbonden: geometrische baseline, landmarks, dieptesensoren, boosting, feature reliance en selectiebias. Het is geen brede of recente ASL-literatuurstudie. |
| Methods | 20 | 15 | Features, klassescores, algoritmen, doelstellingen, preprocessing en variaties zijn verklaard; technische formalisering staat in de appendix. De methodologische innovatie is beperkt en de oorspronkelijke detectorinstellingen zijn niet volledig te reconstrueren. |
| Experiments and Analysis | 40 | 28 | Gelijke beelden/folds, twee algoritmen, toevoegings- en verwijderingsproeven, permutation importance, geneste gewichtstuning, alle letters en gepaarde foutanalyse. Geen onaangeroerde eindtest of bewezen persoonsgescheiden splits; veel verkennende keuzes op dezelfde evaluatie. |
| Discussion | 10 | 8 | Beantwoordt de hoofdvraag, bespreekt klein effect, regressies, selectiebias, detectiedekking, webcambeperkingen, ethiek en AI-gebruik. Sommige verklaringen blijven hypotheses. |
| Conclusions | 5 | 4 | Bondig antwoord met grootte en scope van het effect; gericht vervolgonderzoek. De generalisatie buiten de dataset blijft open. |
| **Totaal** | **100** | **74** | **Grove cijferindicatie: ongeveer 7,4; een ruime bandbreedte rond 7–8 is realistischer dan een exact cijfer.** |

## 1. Zijn ML-technieken juist toegepast?

De gedeelde rubric vraagt letterlijk:

> Is the evaluation correctly performed from a data analytics/machine learning perspective (depending on the field)

Onder hetzelfde onderdeel staan onder meer:

> Does the evaluation contain proper baselines?
>
> Are there ablation experiments?
>
> Were there statistical analysis?

**Oordeel: grotendeels voldaan, met expliciete grenzen.**

Sterke punten:

- Hoofdtabel vergelijkt de baseline en featurevarianten op dezelfde 60.251 geaccepteerde beelden en vijf vastgelegde folds, met accuracy én macro-F1 voor SVM en LightGBM.
- Hoekimputatie en schaling worden op de trainingspartitie geleerd. De geneste gewichtstuning houdt binnenste selectie en buitenste evaluatie gescheiden binnen die zoekproef.
- Toepassingsverschillen tussen toevoegen, verwijderen/hertrainen en permutation importance worden uitgelegd. Permutation importance wordt niet als causale of additieve featurebijdrage gepresenteerd.
- De 67 extra correcte voorspellingen worden uitgesplitst in 143 herstelde en 76 verloren correcte voorspellingen. Alle letters en volledige confusion matrices blijven beschikbaar.
- De statistische stabiliteitsanalyse heet nadrukkelijk geen onafhankelijke betrouwbaarheidsinterval of bevestigende significantietoets.
- De vergelijking met Shin noemt de protocolverschillen en claimt geen exacte reproductie of superioriteit tegenover de gepubliceerde 87,60%.

Beperkingen die punten blijven kosten:

1. Bestandsnummerblokken zijn geen bewezen personen/sessies. Vergelijkbare beelden kunnen over groepen heen voorkomen.
2. Dezelfde evaluatie is herhaaldelijk bekeken bij het kiezen van representaties en vervolgexperimenten. Geneste tuning binnen één proef corrigeert dit grotere selectieproces niet volledig.
3. De scores zijn conditioneel op succesvolle detectie (77,25% dekking); het is geen end-to-end accuracy over alle bronbeelden.
4. Er is geen onaangeroerde eindtest. Geselecteerde webcamfouten zijn diagnostisch materiaal, geen representatieve testset.
5. De referentietabel/relaxaties zijn verkennend ontwikkeld, en een categorie-alleen-zonder-workbook-vergelijking ontbreekt. Het rapport trekt daar geen sterkere conclusie uit dan de uitgevoerde proeven toelaten.

Voor de afgebakende vraag zijn dit beperkingen van het bewijs, geen reden dat alle bestaande experimenten opnieuw moeten worden uitgevoerd. Voor een sterke claim over nieuwe gebruikers is aanvullend prospectief bewijs nodig.

## 2. Zijn de onderzoeksvragen inhoudelijk voldoende?

Er staat in de aangeleverde rubric geen letterlijk criterium dat onderzoeksvragen “niet te eenvoudig” mogen zijn. De meest relevante formuleringen zijn:

> Is the method complex/innovative?
>
> Is the contribution well argued?
>
> Is the method well formalized/explained?

**Oordeel: voldoende voor een afgebakende empirische ML-studie; geen sterke algoritmische innovatie.**

De hoofdvraag gaat niet meer over het bouwen van een app of het oplossen van elke webcamfout. RQ1 toetst het brede pakket en classifierafhankelijkheid, RQ2 isoleert continue metingen en RQ3 onderzoekt verdeling en stabiliteit van het effect. Daarmee zijn de representatiekeuze, de waargenomen winst en de beperkingen expliciet toetsbaar.

Het is inhoudelijk waardevol dat het volledige pakket niet consequent helpt, continue hoeken een kleine SVM-winst geven, LightGBM anders reageert en sommige letters verslechteren. Meer modellen of features zijn daarvoor niet automatisch beter onderzoek. De bijdrage moet worden beoordeeld als een gecontroleerde empirische vergelijking met foutanalyse, niet als een nieuw algoritme.

De huidige data beantwoorden geen causale vraag over palm-/handrugzicht, geen claim van klinisch correcte gewrichtsmetingen en geen claim dat de webcamproblemen opgelost zijn. Die claims zijn uit de kernconclusie gehouden.

## Wat is door deze revisie verbeterd?

- Abstract, onderzoeksvragen, resultatensecties en conclusie beantwoorden nu dezelfde hoofdvraag.
- De numerieke kernvergelijking en alle-letter-effecten staan in de hoofdtekst, niet uitsluitend in de appendix.
- RQ1 maakt onderscheid tussen het volledige pakket, het verwijderen van categorieën en ontbrekende isolatieproeven.
- De hoofdtekst bevat een beknopte uitleg van modelinstellingen, train-only preprocessing, selectiebias en de statistische grenzen.
- Latere experimenten blijven transparant bewaard in Appendix I–J, zonder het hoofdonderzoek onnodig onvoltooid te laten lijken.
- AI-generatie en de beperkte betekenis van code-inspectie/unit tests zijn expliciet vermeld.

## Wat zou het cijfer vooral verder verhogen?

1. Eén vooraf vastgelegde vergelijking op een nieuwe, ongeziene persoon/sessie. Dat versterkt het zwaarst gewogen onderdeel; 120 foto's zijn geen expliciete rubricverplichting.
2. Een iets bredere, gerichte vergelijking met bestaande handvorm-/oriëntatierobuuste featuremethoden, met betrouwbare primaire bronnen. Dat kan ook de beperkte nieuwheid scherper positioneren.
3. Onafhankelijke inhoudelijke controle van de workbookbeschrijvingen en een gerichte controle van de kerncode. Unit tests alleen bewijzen geen taalkundige of anatomische geldigheid.

Voor inlevering moeten auteursnaam/studentnummer nog worden ingevuld en moet de auteur de bronnen en interpretatie zelf controleren. Deze administratieve punten zijn niet als extra experimentele eisen behandeld.
