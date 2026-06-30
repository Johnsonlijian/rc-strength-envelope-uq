# RC Strength Envelope UQ

Code, derived result tables, and publication figures for the manuscript:

**Machine-learned reinforced-concrete strength predictions fail beyond the training envelope: size and axial-load hazards with a mechanical fallback**

Author: Lijian REN  
ORCID: https://orcid.org/0000-0003-1629-4368  
Contact: renlijian@imut.edu.cn

This repository is the public reproducibility package for the submission-stage manuscript. It is not the journal submission package and does not include the active manuscript, cover letter, peer-review simulation notes, internal logs, or raw third-party datasets.

## What Is Included

- `code/`: analysis, reliability, finite-element, and figure-generation scripts.
- `data/processed/`: derived aggregate result tables and model-output summaries used by the figures and SI.
- `figures/`: final manuscript figures in editable `SVG`, vector `PDF`, and high-resolution `PNG`.
- `DATASETS_AND_LINKS.csv`: source registry for the third-party datasets.
- `REPRODUCIBLE_RUNBOOK.md`: environment, data acquisition, and run order.
- `CITATION.cff`: citation metadata for this reproducibility package.

## What Is Not Included

Raw third-party data are not redistributed. The original records include journal supplements, ORDA/Figshare data, Zenodo data, GitHub-hosted data, ACI/PEER/UW/ACI369 column data, and thesis/report PDFs with their own licenses and access terms. Use `DATASETS_AND_LINKS.csv` to retrieve them from the original sources.

The active manuscript PDFs, cover letter, reviewer-response drafts, internal panel-review reports, logs, and private round files are also excluded.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The included derived tables are sufficient to inspect final aggregate results and audit the archived figures. Full figure regeneration requires restoring the third-party raw data and regenerating the cleaned row-level working tables locally.

```bash
cd code
python generate_SI_tex.py
```

After the source data listed in `DATASETS_AND_LINKS.csv` have been restored locally, the figure scripts can be rerun:

```bash
cd code
python fig_build.py
python a18_mechanism_v2.py
python a21_fig.py
python a27_column_fig.py
```

Those raw and cleaned row-level files are intentionally absent from this repository.

## Public Release Boundary

This release is intended for reproducibility, inspection, and archival DOI minting. It does not grant rights to redistribute any third-party raw data. Code is released under the MIT License. Figures and derived aggregate outputs are released under CC BY 4.0 unless a third-party input license imposes a stricter condition.
