# Materials Project XRD Analyzer v4

A Streamlit XRD screening tool that uses **Materials Project XRD JSON files strictly as reference data**.

## Critical v4 fix

v3 could automatically select the first reference containing 002/004/006. That is unsafe because non-MAX hexagonal phases (for example AlN) can also contain 00l reflections. v4 **never silently chooses the target phase**. The user must explicitly select the Materials Project file intended as the layered/MAX reference.

The target selector also only offers MP references whose 002/004/006 are inside the experimental scan range.

## General phase matching

The app no longer reports one reference as the single "best phase." It reports each Materials Project reference separately using:

- **MP reference coverage**: fraction of significant MP reference peak weight represented by detected experimental peaks.
- **Experimental support**: fraction of detected experimental peak prominence explained by that one MP reference.
- matched peak count and mean 2θ error.

These are **not phase fractions or purity**.

## Strict Materials Project rule

Reference peak positions, hkl, d-spacings, and wavelength are read directly from validated Materials Project XRD JSON files. There are no built-in literature peak tables and no fallback wavelength.

## Permanent reference library

Place MP JSON files in `references/`.

Keep the MP-ID in the filename. You may add a readable phase label, for example:

`mp-996162_Ti2AlN_xrd_Cu.json`

The label is for the user only; diffraction calculations still use the JSON contents.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
