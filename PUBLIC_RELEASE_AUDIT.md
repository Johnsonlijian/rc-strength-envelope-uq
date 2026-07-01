# Public Release Audit

Date: 2026-07-01

## Included

- Analysis and figure-generation code under `code/`.
- Final manuscript-used figures 1--7 in `SVG`, `PDF`, and `PNG`, with legacy figure-name aliases retained only for backward compatibility with earlier release candidates.
- Derived aggregate result tables and JSON summaries under `data/processed/`.
- Dataset source registry, runbook, citation metadata, license files, and release notes.

## Excluded

- Active manuscript PDFs, LaTeX submission files, cover letter, and journal-facing upload materials.
- Raw third-party datasets, downloaded archives, PDFs, spreadsheets, and per-specimen text files.
- Cleaned row-level third-party datasets that may reproduce substantial parts of the original records.
- Locally generated `outputs/` files, including generated SI source, because they are reproduction artifacts rather than public-release source files.
- Superseded reliability scripts and the legacy `reliability_across_size.csv`; the final manuscript route uses `code/a21_steel_reliability.py`, `code/a21_fig.py`, and `data/processed/steel_reliability.csv`.
- Superseded figure/SI generators (`a13_mechanism_fig.py`, `a19_steel_fig2.py`, `a20_steel_fig1.py`, `generate_SI.py`) that could overwrite final manuscript figures or emit stale SI content.
- Internal panel-review reports, prompt/model usage logs, reviewer simulations, and private project logs.
- Build intermediates, cache files, `__pycache__`, LaTeX logs, and QA screenshots.

## Rationale

The release is designed to support reproducibility and DOI archival without redistributing materials whose licenses or submission status are not public-release safe. Users can regenerate the full analysis after retrieving the raw third-party records from their official sources.
