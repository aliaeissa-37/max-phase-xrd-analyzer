# Materials Project XRD Analyzer v3

This version is designed to be reusable for future samples and phases while using **Materials Project XRD JSON files strictly** as references.

## What changed from the old website

The old MAX routine could choose 002, 004, and 006 independently from broad angle windows. That allowed unrelated peaks to be assembled into a convincing-looking sequence. In particular, the old routine could prefer taller peaks even when they did not produce one consistent c lattice parameter.

v3 fixes that by:

1. Reading the 002/004/006 reference reflections and wavelength directly from the selected Materials Project JSON.
2. Letting a candidate 002 define one c lattice parameter.
3. Predicting 004 and 006 from that same c.
4. Accepting 004/006 only if detected peaks occur near those predicted positions.
5. Grading the result using both c consistency and peak prominence/SNR.
6. Suppressing weak/ambiguous 00l values in the main batch table so a weak mathematical coincidence is not presented as confirmation.
7. Removing the misleading 0–100 “MAX score” from the main result. The app reports strong/moderate/weak/no convincing evidence instead.
8. Keeping general phase matching separate from MAX 00l analysis.

## Strict Materials Project mode

Reference calculations use only uploaded or bundled Materials Project XRD JSON files.

The app rejects a reference if:
- the filename does not contain an `mp-####` Materials Project ID;
- the JSON does not contain the expected MP XRD fields;
- the wavelength is missing.

There are no hard-coded literature peak positions and no fallback wavelength.

## Make your reference library permanent

1. Download each phase's XRD JSON from Materials Project.
2. Keep the MP ID in the filename.
3. Copy the JSON file into the `references/` folder.
4. Commit/push the repo and redeploy Streamlit.

After that, the reference loads automatically on every visit. To support a new phase later, add its MP JSON to `references/`; no Python edits are required.

## Run locally

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

## Deploy on Streamlit Community Cloud

Replace your existing `app.py`, `xrd_core.py`, and `requirements.txt` with the files in this package. Keep/add the `references/` folder. Push to the GitHub repo connected to Streamlit Cloud and reboot the app.

## Important scientific limitation

Reference-peak coverage is not phase fraction, and linked MAX 00l evidence is not proof of purity. Quantitative phase percentages require a method such as Rietveld refinement.
