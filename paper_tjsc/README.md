# SplitSlab — *The Journal of Supercomputing* (Springer) Submission Package

This directory contains the Springer Nature `sn-jnl` (`pdflatex,sn-mathphys-num`) version of the SplitSlab manuscript, rewritten in clear, plain English for **The Journal of Supercomputing (TJSC)**.

## Contents

- `main_tjsc.tex` — Complete Springer Nature `sn-jnl` LaTeX manuscript (25 pages, single-column reviewer-friendly layout, `sn-mathphys-num` bibliography style, step-by-step humanized proofs in Appendix A).
- `main_tjsc.pdf` — Compiled 25-page submission PDF.
- `cover_letter_tjsc.txt` — Scope-aligned cover letter for the Editor-in-Chief of *The Journal of Supercomputing*.
- `sn-jnl.cls` & `sn-mathphys-num.bst` — Official Springer Nature journal template class and bibliography style.
- `references.bib` — 27 BibTeX references.
- `figures/` — 5 vector PDF figures (`fig1_architecture_overview.pdf` through `fig5_fleet_dedup_and_long_horizon.pdf`).
- `splitslab_tjsc_springer.zip` — Self-contained Springer Editorial Manager / Overleaf upload bundle.

## Compilation

With Tectonic:
```bash
tectonic -X compile main_tjsc.tex
```

With standard TeX Live (`pdflatex` + `bibtex`):
```bash
pdflatex main_tjsc.tex
bibtex main_tjsc
pdflatex main_tjsc.tex
pdflatex main_tjsc.tex
```
