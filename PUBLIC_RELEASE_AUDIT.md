# Public Release Audit

Date: 2026-06-30

## Included

- Analysis and figure-generation code under `code/`.
- Final manuscript-used figures 1, 2, 3, 4, 6, 7, and 8 in `SVG`, `PDF`, and `PNG`.
- Derived aggregate result tables and JSON summaries under `data/processed/`.
- Dataset source registry, runbook, citation metadata, license files, and release notes.

## Excluded

- Active manuscript PDFs, LaTeX submission files, cover letter, and journal-facing upload materials.
- Raw third-party datasets, downloaded archives, PDFs, spreadsheets, and per-specimen text files.
- Cleaned row-level third-party datasets that may reproduce substantial parts of the original records.
- Internal panel-review reports, prompt/model usage logs, reviewer simulations, and private project logs.
- Build intermediates, cache files, `__pycache__`, LaTeX logs, and QA screenshots.

## Rationale

The release is designed to support reproducibility and DOI archival without redistributing materials whose licenses or submission status are not public-release safe. Users can regenerate the full analysis after retrieving the raw third-party records from their official sources.
