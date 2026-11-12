# Separate paper project and discussion proposal

The LaTeX paper lives in a separate Overleaf repository. On this computer its checkout is `~/Documents/GitHub/6aba6ebfe26cf023a6c2efc0`; the main document is `main.tex`. The software repository does not contain the former `paper/` working directory. Obtain the current paper source separately for reproduction.

## Discussion version

The separate paper repository contains `review_proposal/` for the 13 November 2026 discussion:

- `changes.pdf`: RQ1/RQ2 suggestions, exact current/proposed wording, abstract/conclusion, compact RQ3 scope and the full-text literature comparison (page five).
- `proposed_paper.pdf`: complete proposed paper, with five body pages, references starting on page six, two original coauthor appendix pages and two RQ3 appendix pages.
- `changes.patch`: complete source comparison against the paper before this proposal.
- `overleaf_proposal.zip`: independent source project; compile `main.tex` after import.
- `README.md`, `scope.json` and `verification.json`: instructions, edit list and preservation/build/test checks.

The proposed sources are also applied locally in the paper checkout, without staging, committing or pushing. This replaces the earlier RQ3-only proposal. RQ1's question and captions match its measured comparison; RQ2's heading, coverage interpretation and figure caption are clarified. Its technical research question is retained. The shared abstract and conclusion summarize the three separate outcomes. RQ1 table bodies, RQ3 numerical cells, original figures, the coauthor bibliography and code appendices are preserved. The own bibliography adds Zhang and Notni (2025), compared in related work and discussion after full-text inspection. No experiment or trained model is changed. The RQ3 appendix is reduced from six pages to two.

## Review before publishing

Read the two PDFs or import the source ZIP as a separate Overleaf project. The comparison works without native Track Changes or additional rights in the shared project.

Overleaf offers [Track Changes](https://docs.overleaf.com/collaborating/track-changes) in projects with the relevant premium feature and permissions. [Git pushes can displace tracked changes and comments](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/git-integration/advanced-git-operations). The supplied PDFs and patch can be reviewed independently of that metadata.

After group approval, inspect the latest `git diff` and commit the intended manuscript changes together with all renamed RQ3 source and figure files. The contribution uses `rq3_*.tex`, `rq3_references.bib` and `figures/rq3_*`; include `main.tex`, `shared_abstract_en.tex` and `shared_conclusion_en.tex` as well. Do not retain duplicate aliases with personal prefixes. Retain any newer coauthor edits. The root `main.pdf` is a local preview; the review PDFs and ZIP can remain local. Do not paste a second copy of the contribution. Compile the latest main and check references and page count before sharing a submission version.
