# SplitSlab — Elsevier *Future Generation Computer Systems (FGCS)* Submission Package

This directory contains the Elsevier `elsarticle` (`final,5p,times,twocolumn`) version of the SplitSlab manuscript formatted for submission to **Future Generation Computer Systems (FGCS)**.

## Contents

- `main_fgcs.tex` — Complete Elsevier `elsarticle` LaTeX manuscript (12 pages, two-column `5p` layout, `elsarticle-num` bibliography style).
- `main_fgcs.pdf` — Compiled 12-page publication-ready PDF.
- `highlights.txt` — Mandatory Elsevier Highlights (5 bullet points, each strictly $\le 85$ characters including spaces).
- `references.bib` — 27 BibTeX references.
- `figures/` — 5 vector PDF figures (`fig1_architecture.pdf` through `fig5_fleet_and_scaling.pdf`).
- `splitslab_fgcs_overleaf.zip` — Self-contained Overleaf / Editorial Manager upload bundle (`main_fgcs.tex`, `references.bib`, `highlights.txt`, `figures/*.pdf`).

## FGCS Submission Checklist

- [x] `elsarticle` document class with `\journal{Future Generation Computer Systems}`
- [x] Highlights file (`highlights.txt`, 5 bullets, 79–80 chars each, $\le 85$ char Elsevier limit)
- [x] Keywords (`\begin{keyword}...\sep...\end{keyword}`)
- [x] CRediT authorship contribution statement
- [x] Declaration of competing interest
- [x] Data availability statement (GitHub + Hugging Face dataset links)
- [x] Acknowledgments
- [x] `elsarticle-num` numbered bibliography style

## Compilation

With Tectonic:
```bash
tectonic -X compile main_fgcs.tex
```

With standard TeX Live (`pdflatex` + `bibtex`):
```bash
pdflatex main_fgcs.tex
bibtex main_fgcs
pdflatex main_fgcs.tex
pdflatex main_fgcs.tex
```

## Authors

- **Mohammed Kamran Syed** (Corresponding Author, ORCID: `0009-0009-9058-5375`, `mohammedkamran.syed@ieee.org`) — Independent Researcher, Mountain House, CA 95391, USA
- **Anudeep Rentala** (`anudeep.rentala@ieee.org`) — Independent Researcher, Fremont, CA 94538, USA
- **Somesh Rahul** (`someshrahul@ieee.org`) — Independent Researcher, Fremont, CA 94538, USA
- **Raghuram Malpe Pai** (`raghuram.pai@ieee.org`) — Independent Researcher, Sunnyvale, CA 94086, USA
- **Krishna Suman Dara** (`krishna-suman.dara@ieee.org`) — Independent Researcher, Sunnyvale, CA 94086, USA
