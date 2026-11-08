# Current group paper and RQ3 integration

The complete local Overleaf project is `/Users/tonheijligers/Documents/Github/6aba6ebfe26cf023a6c2efc0`; compile `main.tex` there. The repository's equivalent main is `paper/group_integration/group_report.tex`. The transfer package is `paper/group_integration_ton.zip`.

The current version contains the supplied RQ1/RQ2 prose, tables, bibliography, recovery figure and code appendices. Only the own introduction and RQ3 prose have been shortened. No model or numerical result was changed. Named boundary comments are removed and dotted separators remain.

## Current layout

- Body: pages 1–5.
- References: start on page 5 and continue on page 6.
- Coauthor code/illustrations: appendix A.
- RQ3 definitions, feature tool and practical orientation checks: appendices B–E.

The five-page count concerns the body; whether references are excluded remains a course requirement to check. The existing N47 illustration is now included in appendix A, so its body reference resolves. Long code lines wrap without changing the supplied code files.

## Update Overleaf

1. Use the current local Overleaf project as the working project. Keep any newer coauthor edits.
2. Upload the changed `main.tex`, all `ton_*.tex`, `ton_references.bib` and `code-snippets/n47-example.tex`. Keep the existing three own figures and original group files. The three RQ3 figures were checked against their numerical/photo sources and need no replacement.
3. Do not paste a second RQ3 contribution: the six prose inputs, appendix input and bibliography entry are already present.
4. Compile `main.tex`. Check the feature table, N47 reference, all figures and page count. The generated local `main.pdf` is for review; it is not a source file to upload.
5. Review the diff before committing or uploading. Coauthor prose, numerical table values and both original code-snippet files must remain unchanged.

The package contains a complete equivalent project with `group_report.tex` as main, plus `group_report_paste_ready.tex` as an alternative with the own prose/tables/bibliography embedded. Choose one main. Keep the accompanying group figures and code files; the paste-ready alternative still uses those files.

## If the group has newer main-file edits

Retain that newer main and update only these existing inputs:

| Section | Contribution file |
|---|---|
| Related works | `ton_related_en.tex` |
| Methods | `ton_methods_en.tex` |
| Datasets | `ton_dataset_en.tex` |
| Implementation details | `ton_implementation_en.tex` |
| Results | `ton_results_en.tex` |
| Discussion | `ton_discussion_en.tex` |
| Technical appendix | `ton_appendix_en.tex` |

Keep the four `ton_*_rows_en.tex` table files and `ton_references.bib`. Do not add another research question, bibliography command or copy of the appendix. Filenames and citation/label prefixes avoid collisions; they are not printed author notes.

All active figure paths resolve. N47's original, dilated and overlay views are retained; unused template images and the duplicate recovery figure are removed.
