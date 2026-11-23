# SplitSlab: IEEE Transactions on Parallel and Distributed Systems Manuscript Pipeline

Contains the two-column **IEEE Transactions on Parallel and Distributed Systems (`IEEEtran`, `10pt,journal,compsoc`)** manuscript, BibTeX bibliography, 5 vector PDF figures, static LaTeX and desk-rejection validator, and one-click Overleaf submission bundle (`splitslab_tpds_overleaf.zip`).

## Quick Commands

### 1. Regenerate All 5 Figures from Verified Hardware JSON Logs
```bash
python3 ../scripts/generate_paper_figures.py
```

### 2. Run Static LaTeX, Citation, Cross-Reference, Abstract, Style & Page-Budget Validator
```bash
python3 validate_latex.py
```

### 3. Compile PDF
* **Overleaf (1-click):** Upload `splitslab_tpds_overleaf.zip` directly as a new project on Overleaf (`pdfLaTeX` compiler, main document `main.tex`).
* **TeX Live / MacTeX:**
  ```bash
  pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex && pdflatex main.tex
  ```

---

## IEEE Transactions on Parallel and Distributed Systems (TPDS) Submission Checklist

### Format & Desk-Rejection Compliance
* **Document Class:** `\documentclass[10pt,journal,compsoc]{IEEEtran}` (`IEEEtran.cls` v1.8b).
* **Page Limit:** Strictly within TPDS's 12-page regular-paper free limit (**12 pages** compiled in `main.pdf`, inclusive of references, Appendix A proofs, and disclosures).
* **Abstract:** Single paragraph, **196 words** (`<= 200` words strictly), zero inline math, zero citations, zero footnotes.
* **Mandatory Disclosures:** Corresponding author ORCID (`0009-0009-9058-5375`), `IEEEkeywords`, Data and Code Availability (GitHub + Hugging Face Hub), and Acknowledgments and Competing Interests.

### Author Roster

| Order | Name | Email | Affiliation | Role / ORCID |
| :---: | :--- | :--- | :--- | :--- |
| 1 | Mohammed Kamran Syed | `mohammedkamran.syed@ieee.org` | IEEE Independent Researcher, Mountain House, CA, USA | Corresponding & Submitting Author; ORCID: `0009-0009-9058-5375` |
| 2 | Anudeep Rentala | `anudeep.rentala@ieee.org` | IEEE Independent Researcher, Fremont, CA, USA | Co-Author |
| 3 | Somesh Rahul | `someshrahul@ieee.org` | IEEE Independent Researcher, Fremont, CA, USA | Co-Author |
| 4 | Raghuram Malpe Pai | `raghuram.pai@ieee.org` | IEEE Independent Researcher, Sunnyvale, CA, USA | Co-Author |
| 5 | Krishna Suman Dara | `krishna-suman.dara@ieee.org` | IEEE Independent Researcher, Sunnyvale, CA, USA | Co-Author |

### Recommended IEEE Taxonomy Keywords
* Disaggregated KV-cache serving, NVMe write amplification, content-addressed deduplication, KV-cache tiering, subspace routing sketches, large language model agents, distributed storage systems.
