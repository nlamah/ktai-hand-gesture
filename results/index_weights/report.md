# Gerichte weging wijsvingerhoeken: SVM + Shin + 15 hoeken

Frozen 5 outer folds; 3 inner grouped folds; at most 150 images/class per inner fitting/validation partition. Choose mean inner macro-F1 then accuracy; coordinate order PIP, MCP, DIP, one pass; weights 0.5/1/2. Full outer training and validation; imputation/scaling fitted only on each training partition. Fixed RBF SVM C=10, gamma=scale, balanced. All other angles and Shin unchanged.

Index hypothesis was chosen after inspecting previous results on these same development folds; nested weight selection does not turn them into a fresh independent test. Filename blocks are not known people.

## Conclusie

De beste geteste accuracy is 97,753% bij intern gekozen PIP-gewicht, tegenover
97,748% zonder weging: **slechts drie extra correcte beelden op 60.251**.
De macro-F1 verandert van 0,97524 naar 0,97526. Dat is geen overtuigende
algemene verbetering. Zestien eerder foute beelden worden hersteld, maar
dertien eerdere juiste voorspellingen gaan verloren.

PIP standaard verdubbelen levert netto één extra correct beeld op, maar maakt
O → D juist erger (19 → 23). Het afstemmen van alle drie de wijsvingerhoeken
vermindert O → D wel (19 → 15), maar verliest netto vijf correcte beelden,
onder andere bij N. Dit is geen goede algemene oplossing voor O/D.

Advies: behoud **Shin + 15 hoeken zonder extra weging** als eenvoudige
vervolgkandidaat. Gebruik nieuwe, persoonsgescheiden beelden voor verdere
bevestiging voordat er meer op dezelfde ontwikkelfolds wordt geoptimaliseerd.

| Variant | Accuracy | Macro-F1 | Correct | Hersteld / verloren t.o.v. ongewogen |
|---|---:|---:|---:|---:|
| unweighted | 97.748% | 0.97524 | 58894 | 0 / 0 |
| pip_2 | 97.749% | 0.97524 | 58895 | 22 / 21 |
| tuned_pip | 97.753% | 0.97526 | 58897 | 16 / 13 |
| tuned_index | 97.739% | 0.97507 | 58889 | 21 / 26 |

unweighted: alle hoekgewichten 1. pip_2: vooraf vastgelegde PIP-verhoging naar 2. tuned_pip: alleen PIP intern gekozen. tuned_index: PIP, daarna MCP en DIP intern afgestemd.

## Gewichten per buitenfold

| Fold | Variant | MCP | PIP | DIP |
|---|---|---:|---:|---:|
| 1 | unweighted | 1.0 | 1.0 | 1.0 |
| 1 | pip_2 | 1.0 | 2.0 | 1.0 |
| 1 | tuned_pip | 1.0 | 2.0 | 1.0 |
| 1 | tuned_index | 0.5 | 2.0 | 1.0 |
| 2 | unweighted | 1.0 | 1.0 | 1.0 |
| 2 | pip_2 | 1.0 | 2.0 | 1.0 |
| 2 | tuned_pip | 1.0 | 2.0 | 1.0 |
| 2 | tuned_index | 1.0 | 2.0 | 0.5 |
| 3 | unweighted | 1.0 | 1.0 | 1.0 |
| 3 | pip_2 | 1.0 | 2.0 | 1.0 |
| 3 | tuned_pip | 1.0 | 0.5 | 1.0 |
| 3 | tuned_index | 0.5 | 0.5 | 1.0 |
| 4 | unweighted | 1.0 | 1.0 | 1.0 |
| 4 | pip_2 | 1.0 | 2.0 | 1.0 |
| 4 | tuned_pip | 1.0 | 0.5 | 1.0 |
| 4 | tuned_index | 2.0 | 0.5 | 1.0 |
| 5 | unweighted | 1.0 | 1.0 | 1.0 |
| 5 | pip_2 | 1.0 | 2.0 | 1.0 |
| 5 | tuned_pip | 1.0 | 0.5 | 1.0 |
| 5 | tuned_index | 1.0 | 0.5 | 1.0 |

## Gerichte verwarringen

| Echte letter → voorspelling | Ongewogen | PIP 2 | PIP getuned | Alle wijsvingerhoeken getuned |
|---|---:|---:|---:|---:|
| O → D | 19 | 23 | 16 | 15 |
| D → O | 41 | 39 | 42 | 39 |
| M → N | 102 | 101 | 102 | 103 |
| N → M | 141 | 143 | 143 | 146 |
| R → U | 142 | 137 | 137 | 136 |
| U → R | 44 | 44 | 44 | 44 |

## Recall per letter

| Letter | Beelden | Ongewogen | PIP 2 | PIP getuned | Alle wijsvingerhoeken getuned |
|---|---:|---:|---:|---:|---:|
| A | 2188 | 98.72% | 98.72% | 98.72% | 98.67% |
| B | 2204 | 99.73% | 99.73% | 99.73% | 99.73% |
| C | 1912 | 99.84% | 99.84% | 99.84% | 99.84% |
| D | 2456 | 97.68% | 97.72% | 97.64% | 97.72% |
| E | 2308 | 99.22% | 99.18% | 99.18% | 99.13% |
| F | 2859 | 98.57% | 98.50% | 98.57% | 98.57% |
| G | 2430 | 99.42% | 99.42% | 99.42% | 99.42% |
| H | 2371 | 98.19% | 98.14% | 98.14% | 98.19% |
| I | 2380 | 96.76% | 96.81% | 96.81% | 96.81% |
| J | 2576 | 96.78% | 96.86% | 96.86% | 96.82% |
| K | 2701 | 98.37% | 98.37% | 98.37% | 98.37% |
| L | 2516 | 99.60% | 99.60% | 99.60% | 99.60% |
| M | 1602 | 91.95% | 92.01% | 91.89% | 91.82% |
| N | 1280 | 87.97% | 87.81% | 87.81% | 87.50% |
| O | 2264 | 97.88% | 97.70% | 97.97% | 98.01% |
| P | 2041 | 98.78% | 98.78% | 98.78% | 98.78% |
| Q | 2106 | 97.72% | 97.86% | 97.67% | 97.58% |
| R | 2543 | 93.83% | 94.02% | 94.02% | 94.06% |
| S | 2559 | 98.40% | 98.28% | 98.32% | 98.32% |
| T | 2354 | 98.47% | 98.47% | 98.51% | 98.51% |
| U | 2513 | 96.58% | 96.62% | 96.62% | 96.58% |
| V | 2552 | 98.43% | 98.43% | 98.43% | 98.39% |
| W | 2456 | 98.33% | 98.33% | 98.33% | 98.37% |
| X | 2157 | 97.22% | 97.22% | 97.22% | 97.22% |
| Y | 2580 | 99.38% | 99.42% | 99.42% | 99.38% |
| Z | 2343 | 97.78% | 97.74% | 97.74% | 97.70% |

Gewichten worden na standaardisatie toegepast. Factor 2 betekent factor 4 in het betreffende gekwadrateerde afstandsdeel van de RBF-kernel. Er is niet geselecteerd op alleen O/D en er is geen outer-fold keuze achteraf per letter toegepast. Dit onderzoek betreft bruikbare detecties; de ontbrekende detecties veranderen niet.

```bash
python -m scripts.tune_index_angles
```
