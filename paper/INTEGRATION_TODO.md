# Gezamenlijke paper: in te voegen bijdragen

De hoofdtekst is een toegankelijke Nederlandse werkversie voor lezers zonder ML-voorkennis, met ruimte voor drie deelonderzoeken. RQ1 en RQ2 blijven zichtbaar als **TODO**; hun uitkomsten zijn nog niet ingevuld. Leg ook nieuwe bijdragen uit met gewone woorden en concrete voorbeelden. De eerdere preprocessing-PDF is alleen gebruikt om het onderwerp af te bakenen. De technische bijlage bevat alleen de onderbouwing waarnaar de hoofdtekst verwijst, in het Nederlands. Instellingen staan in A, resultaten in B, de featuretool in C en de twee augmentatieproeven in D–E en rechtsnormalisatie in F.

| Onderdeel | Vraag | Status / invoegplek |
|---|---|---|
| RQ1 – beeldvoorbewerking | Kunnen oorspronkelijk gemiste handdetecties worden teruggewonnen, en zijn de landmarks visueel plausibel? | Bijdrage medestudent verwacht; `team_methods_pending.tex`, `team_results_pending.tex`, bijlage A |
| RQ2 – hoeveelheid trainingsdata | Hoe veranderen macro-precisie en macro-F1 bij meer unieke trainingsbeelden, bij een vaste evaluatie? | Bijdrage medestudent verwacht; dezelfde invoegbestanden en bijlage A |
| RQ3 – handfeatures | Wat voegen eigen categorieën en wereldgewrichtshoeken toe, met en zonder Shin-richtingen? | Acht varianten plus permutatietests; methode A, resultaten B, feature-uitleg C |
| Ondersteunende oriëntatieproeven | Spiegeling, afzonderlijke G/H-rotatie en normalisatie naar één handoriëntatie | Afzonderlijke resultaten, bijlagen D–F |

De drie oude vragen uit de featurestudie vallen nu samen onder **RQ3**. De langere werkversie gebruikte hiervoor RQ3a/RQ3b/RQ3c; de compacte hoofdtekst heeft alleen RQ3. Het eerdere preprocessing-template gebruikte eveneens een andere nummering: gebruik in nieuwe bijdragen de indeling hierboven.

## PREP: aanlevering over beeldvoorbewerking

- Definitieve methode en bronnen: dilatatie uitleggen; kernel, iteraties, kleur/resolutie, erosie en Canny-overlay vastleggen. De eerdere PDF noemt een 3×3-kernel; dat is nog als voorlopig beschreven.
- Datasetmanifest, bron-ID’s, detector/modelversie, instellingen en codeversie.
- Oorspronkelijke detectiestatussen, herstel per variant, unieke unie en niet-herstelde beelden, met aantallen/noemers en resultaten per letter.
- Een herstelfiguur en een visuele kwaliteitscontrole, of expliciet de beperking dat die controle ontbreekt. `ok` is technische acceptatie, geen bewezen correcte handanatomie.
- Eerdere ontwikkelvoorbeelden apart benoemen. Herstelpercentages zijn geen classificatieprecisie of F1. Een winst in herkenning na gebruik van herstelde beelden vereist een afzonderlijke vergelijking.

## SIZE: aanlevering over trainingsomvang

- Omvang per klasse, aantal unieke bronbeelden, selectiebeleid en geaccepteerde landmarks. Vermeld of herstelde beelden worden gebruikt.
- Vaste test-ID’s of folds, bron-groepen, subset-ID’s, seeds en herhalingen. Testbeelden blijven gelijk bij alle groottes; subsets komen alleen uit het trainingsdeel.
- Detector, features, classifierinstellingen en binnen elke trainingssubset geleerde preprocessing. Houd die keuzes vast om het effect van hoeveelheid data te meten.
- Een leercurve met macro-precisie én macro-F1, onderliggende waarden, variatie tussen herhalingen/folds en relevante letterverschillen. Augmentatiekopieën tellen niet als nieuwe bronbeelden.
- Geen resultaten voorspellen: ook een plateau, achteruitgang of onduidelijke trend rapporteren als dat uit de metingen blijkt.

## TEAM: samenvoegen

- Gereed: eigen screenshot van 7 november 2026 toegevoegd als `paper/figures/feature_tool_screenshot.png`, met passende uitleg in bijlage C. Zes schema’s verduidelijken de 19 kenmerken. Bij een nieuwe screenshot ook de heropgemaakte detailkaartjes en het bijschrift controleren.

- Namen/studentnummers en definitieve hoofd-/deelvragen afstemmen.
- Hoofdtekst: leg uit wat het programma doet en wat de uitkomst betekent. Definieer nieuwe termen bij het eerste gebruik. Zet instellingen, formules, seeds en uitgebreide tabellen in de bijlage; vermijd lange reeksen decimalen in de hoofdtekst.
- Dataset, splits en baseline afstemmen of verschillen transparant beschrijven. De historische featurecache bewaart geen detectiedrempels; gelijkheid niet veronderstellen op basis van gelijke aantallen.
- Hoofdtekst: kernresultaten met tabel-/figuurnummers. Appendix: methode, controles en detailresultaten. Supplement: code, versie, CSV/JSON met stabiele paden en checksums. De applicatie kan de werking demonstreren, maar vervangt de resultaatbestanden niet.
- Abstract, discussie en conclusie na ontvangst van beide bijdragen afronden. Detectieherstel, visuele kwaliteit en classifierprestatie hebben verschillende noemers; resultaten niet optellen of rechtstreeks rangschikken.
- In het definitieve verslag maximaal vijf pagina’s hoofdtekst exclusief referenties behouden. De huidige plekken zijn redactionele ruimte; bij invoegen van figuren de balans tussen de drie bijdragen opnieuw bewaken.
- Zoek open plekken in de actieve bronnen met `rg -n 'drafttodo|TODO' paper/report.tex paper/team_*pending.tex paper/appendix_*nl.tex`. Verwijder ze pas wanneer de betreffende tekst en resultaten zijn gecontroleerd.

De huidige PDF’s zijn een groepsconcept. De acht featurevarianten zijn opnieuw getraind op dezelfde vaste folds; de aanvullende oriëntatieproeven behouden hun eigen opgeslagen resultaten. Historische rubricbeoordelingen en PDF-verificaties van eerdere revisies zijn geen beoordeling of paginacontrole van dit concept; de actuele opmaakcontrole staat in `team_draft_verification.json`. Het praktische aanlever- en testplan staat in `docs/LIVE_TESTPLAN_NL.md`.

## Afbakening van de opgeschoonde bijlage

| Bijlage | Verwijzing vanuit hoofdtekst | Actieve bron |
|---|---|---|
| A – Data, evaluatie en modelinstellingen | §2, §3.1 en §3.2 | `appendix_methods_nl.tex` |
| B – Acht varianten en featurebijdragen | §4.3–4.4, scores, controles en permutatietests | `appendix_features_nl.tex` |
| C – Handkenmerken en controle in de tool | §3.4, meetregels en toolfoto | `appendix_tool_nl.tex` |
| D – Spiegeltraining en links/rechts-variatie | §4.5, spiegelproef | `appendix_mirroring_nl.tex` |
| E – Afzonderlijke rotatieproef voor G en H | §4.5, losse rotatieproef | `appendix_gh_nl.tex` |
| F – Normaliseren naar dezelfde handoriëntatie | §4.5, vergelijking met niets aanpassen en spiegeltraining | `appendix_handedness_nl.tex` |

De rubricbeoordeling, aanleverchecklist, oude verkennende proeven, volledige verwarringsmatrices en dubbele uitleg worden niet meer in de PDF opgenomen. Historische bronbestanden en alle experimentele gegevens blijven beschikbaar in de repository. Voeg alleen nieuwe bijlagestof toe als de hoofdtekst er voor een concrete onderbouwing naar verwijst.

De huidige RQ3-opzet gebruikt DS, D, DSE, DE, DSJ, DJ, DSEJ en DEJ. E zijn alleen de eigen categorieën; J zijn wereldgewrichtsmetingen. De 26 werkboekovereenkomstscores zijn uitgesloten. Classificatie gebruikt 60.251 geaccepteerde beelden; permutatie gebruikt 13.000 vooraf geselecteerde toetsbeelden. Neem deze verschillende noemers over bij integratie. De al bestaande oriëntatieproeven behouden hun 835-inputreferentie en zijn geen extra hoofdvarianten.
