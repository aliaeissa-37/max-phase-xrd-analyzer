# MAX Phase XRD Lab Analyzer

A shareable Streamlit website for screening experimental powder XRD patterns for MAX-phase formation.

It was built around the lab workflow discussed for Nb2AlN / Ti2AlN / TiNbAlN synthesis: look for the low-angle (002) reflection, test whether (004) and (006) are consistent with the same c-lattice parameter, then compare the **full experimental pattern** with target and residual/secondary-phase references.

## What it does

- Upload multiple experimental XRD files (`.csv`, `.txt`, `.xy`, `.dat`, simple `.json`).
- Reads two numeric columns: `2theta` and `intensity`; headers are optional and tab-delimited files are accepted.
- Baseline subtraction, smoothing, normalization, and automatic peak detection.
- Editable MAX (002) search target; the included lab presets start at:
  - Nb2AlN: 12.5 degrees 2theta
  - Ti2AlN: 13.5 degrees 2theta
  These are **starting search centers**, not hard-coded phase-identification rules.
- Calculates c from candidate (002), predicts (004)/(006), and checks whether the experimental peaks form a self-consistent 00l series.
- Upload XRD references or CIF files.
- Optional Materials Project lookup using an API key. The app fetches a structure and computes powder XRD with `pymatgen.analysis.diffraction.xrd.XRDCalculator`.
- Compares sample peaks with every loaded reference and reports weighted reference-peak coverage.
- Batch summary CSV export.
- Five provided lab XRD patterns are bundled as optional demo data.

## Scientific limitation

This is a **screening and comparison tool**, not an automatic crystallographic proof. A peak around an expected (002) angle does not by itself confirm a MAX phase. Use the whole pattern, plausible competing phases, sample chemistry, and Rietveld refinement/complementary methods when a quantitative or publication-grade conclusion is needed.

## Run locally

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate            # macOS/Linux
# .venv\\Scripts\\activate         # Windows
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL, normally `http://localhost:8501`.

## Share with the whole lab

### Option A — Streamlit Community Cloud

1. Create a GitHub repository and upload all files in this folder.
2. Go to Streamlit Community Cloud and create an app from the repository.
3. Set the main file to `app.py`.
4. Deploy. Everyone in the lab can use the resulting URL.

For Materials Project access, add a Streamlit secret:

```toml
MP_API_KEY = "your_materials_project_api_key"
```

The app also allows a user to paste an API key for their current session.

### Option B — lab computer/server

Run:

```bash
streamlit run app.py --server.address 0.0.0.0
```

Then expose port 8501 only through your institution's approved network/VPN/reverse proxy.

## Suggested reference library for this project

Add patterns/structures for all phases that are chemically plausible, not just the target. For the current experiments that may include:

- Nb2AlN target MAX
- Ti2AlN structural reference
- Ti/Nb mixed MAX candidate if a validated structure is available
- NbN
- AlN
- Nb
- Al
- TiN where relevant

Do not force a target reference to explain every peak; unexplained peaks are useful information.

## Materials Project behavior

For an `mp-...` material ID, the app uses that exact entry. For an exact formula, it searches available entries and selects a stable entry first, otherwise the lowest-energy-above-hull entry returned. For serious comparison, using the exact material ID is preferable because polymorphs can produce different XRD patterns.

## Included example data

`example_data/` contains the five lab patterns provided during development. They are loaded only when the user checks **Load the 5 bundled lab example datasets**.

## Materials Project JSON compatibility
This build accepts the legacy Materials Project XRD JSON export format with fields such as:
`meta = ["amplitude", "hkl", "two_theta", "d_spacing"]` and a `pattern` array of reflections.
It treats these files as computed stick patterns, so each listed reflection is used directly as a reference peak.
