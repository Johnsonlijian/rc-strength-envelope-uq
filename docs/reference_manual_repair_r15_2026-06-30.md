# Reference manual repair - r15 - 2026-06-30

This note records the manual repair and direct API verification of the three
references flagged as P0 by the first local Crossref-only audit.

## Repaired entries

- `sheffieldfrp2017`: replaced institutional placeholder author with the
  ORDA/DataCite creator list and retained DOI `10.15131/shef.data.5057527`.
- `megahed2024deepbeam`: replaced the bare GitHub-note entry with the
  Scientific Reports article:
  `Prediction and reliability analysis of shear strength of RC deep beams`,
  DOI `10.1038/s41598-024-64386-w`; retained the GitHub URL only as the
  associated data/code link.
- `bazant2005yu`: corrected the title to the ASCE article
  `Designing Against Size Effect on Shear Strength of Reinforced Concrete
  Beams Without Stirrups: I. Formulation`, narrowed pages to `1877--1885`,
  and retained the official ASCE DOI
  `10.1061/(ASCE)0733-9445(2005)131:12(1877)`.

## Direct verification

- Crossref returned for `10.1038/s41598-024-64386-w`: title
  `Prediction and reliability analysis of shear strength of RC deep beams`;
  journal `Scientific Reports`; year `2024`.
- DataCite returned for `10.15131/shef.data.5057527`: title
  `Shear database of RC FRP beams without shear reinforcement`; publisher
  `The University of Sheffield`; year `2017`.
- Crossref returned for
  `10.1061/(ASCE)0733-9445(2005)131:12(1877)`: title
  `Designing Against Size Effect on Shear Strength of Reinforced Concrete
  Beams Without Stirrups: I. Formulation`; journal
  `Journal of Structural Engineering`; year `2005`; pages `1877-1885`.

## Boundary

The full `imut.cmd references-check` rerun after these repairs timed out on
the live API pass, so `reference_check_r15_2026-06-30.md` remains the earlier
Crossref-first report. The three P0 items from that report have been manually
repaired and directly checked above. Non-Crossref items, standards, books,
NeurIPS/PMLR/arXiv entries, and official code/report pages still require the
usual final human claim-to-source comfort check before journal portal upload.
