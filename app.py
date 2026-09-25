import glob
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from xrd_core import (
    analyze_max_00l,
    detect_peaks,
    extract_00l_reference,
    load_experimental_csv,
    load_mp_json_strict,
    match_materials_project_reference,
    reference_c_from_00l,
)

st.set_page_config(page_title="Materials Project XRD Analyzer", layout="wide")
st.title("Materials Project XRD Analyzer")
st.caption("Experimental XRD analysis using Materials Project XRD JSON references only")

st.info(
    "STRICT REFERENCE MODE: this app does not use literature peak tables, COD/ICDD/PDF cards, hard-coded phase positions, "
    "or web lookups. Every reference peak, hkl, d-spacing, and wavelength used in calculations must come from a Materials Project XRD JSON file."
)

with st.expander("What this app can and cannot conclude", expanded=False):
    st.markdown(
        """
**It can:**
- detect peaks in experimental CSV/TXT data;
- compare them to any Materials Project XRD JSON reference you provide;
- rank uploaded MP references by reference-peak coverage;
- inspect matched and unmatched MP peaks;
- for a selected layered/MAX MP reference, test whether 002/004/006 form one linked sequence with a common **c** lattice parameter;
- export tables for your lab notes.

**It cannot:**
- report percent phase purity;
- prove phase identity from one peak;
- replace Rietveld refinement for quantitative phase analysis.

The program therefore reports **evidence / reference matching**, not “% MAX.”
        """
    )


def _load_bundled_references():
    refs = {}
    errors = []
    ref_dir = Path(__file__).resolve().parent / "references"
    for path in sorted(ref_dir.glob("*.json")):
        try:
            refs[path.name] = load_mp_json_strict(str(path), path.name)
        except Exception as e:
            errors.append(f"{path.name}: {e}")
    return refs, errors


bundled_refs, bundled_errors = _load_bundled_references()

left, right = st.columns([1, 1])
with left:
    exp_files = st.file_uploader(
        "1) Upload experimental XRD CSV/TXT/DAT files",
        type=["csv", "txt", "dat"],
        accept_multiple_files=True,
        help="First two numeric columns are interpreted as 2θ and intensity.",
    )
with right:
    uploaded_ref_files = st.file_uploader(
        "2) Add Materials Project XRD JSON references",
        type=["json"],
        accept_multiple_files=True,
        help="Strict mode requires each filename to contain an MP ID (for example mp-996162).",
    )

references = dict(bundled_refs)
ref_errors = list(bundled_errors)
for f in uploaded_ref_files or []:
    try:
        references[f.name] = load_mp_json_strict(f, f.name)
    except Exception as e:
        ref_errors.append(str(e))

if ref_errors:
    with st.expander("Rejected / invalid reference files", expanded=True):
        for e in ref_errors:
            st.error(e)

if references:
    st.success(f"Loaded {len(references)} validated Materials Project reference file(s).")
else:
    st.warning(
        "No Materials Project reference is loaded yet. Add your MP XRD JSON files above, or place them in the app's references/ folder for permanent use."
    )

if not exp_files:
    st.stop()

parsed = {}
exp_errors = []
for f in exp_files:
    try:
        parsed[f.name] = load_experimental_csv(f)
    except Exception as e:
        exp_errors.append(f"{f.name}: {e}")
if exp_errors:
    for e in exp_errors:
        st.error(e)
if not parsed:
    st.stop()

# Global controls for general phase matching.
with st.expander("Reference-matching settings", expanded=False):
    phase_tolerance = st.slider("MP peak match tolerance (° 2θ)", 0.10, 0.60, 0.30, 0.01)
    min_mp_amp = st.slider("Ignore MP reference peaks below this relative intensity (%)", 0.0, 10.0, 1.0, 0.5)
    top_mp_peaks = st.slider("Maximum MP peaks used per phase", 5, 40, 20, 1)
    min_exp_snr = st.slider("Minimum experimental peak SNR for phase matching", 2.0, 8.0, 3.0, 0.5)

# Identify MP references that actually contain 002/004/006.
max_capable = []
for name, ref in references.items():
    oo = extract_00l_reference(ref, (2, 4, 6))
    if all(l in oo for l in (2, 4, 6)):
        max_capable.append(name)

st.divider()
tabs = st.tabs(["Batch summary", "Sample inspector", "MP phase matching", "Reference library", "How to use"])

with tabs[0]:
    st.subheader("Batch summary")
    if not references:
        st.warning("Upload Materials Project references to calculate phase matches or MAX 00l evidence.")
    else:
        selected_max_ref_name = None
        if max_capable:
            selected_max_ref_name = st.selectbox(
                "Materials Project reference for layered/MAX 00l analysis",
                ["Do not run MAX 00l analysis"] + max_capable,
                index=1,
                key="batch_max_ref",
                help="Only MP references containing 002, 004, and 006 are offered here.",
            )
            if selected_max_ref_name == "Do not run MAX 00l analysis":
                selected_max_ref_name = None
        else:
            st.caption("No loaded MP reference contains all of 002/004/006, so MAX 00l analysis is unavailable.")

        rows = []
        detail_cache = {}
        for sample, df in parsed.items():
            # Best general MP reference by weighted coverage.
            phase_matches = []
            for ref_name, ref in references.items():
                m = match_materials_project_reference(
                    df, ref,
                    tolerance_deg=phase_tolerance,
                    min_reference_amplitude=min_mp_amp,
                    top_reference_peaks=top_mp_peaks,
                    min_experimental_snr=min_exp_snr,
                )
                phase_matches.append((ref_name, m))
            phase_matches.sort(key=lambda x: x[1]["coverage"], reverse=True)
            best_name, best_match = phase_matches[0]

            out = {
                "Sample": sample,
                "Best MP reference": best_name,
                "MP reference coverage (%)": round(best_match["coverage"], 1),
                "Matched MP peaks": f"{best_match['matched_count']}/{best_match['reference_count']}",
            }

            if selected_max_ref_name:
                ev = analyze_max_00l(df, references[selected_max_ref_name])
                detail_cache[sample] = ev
                # Suppress exact 00l/c values in the main table if evidence is weak/none.
                report_values = ev.strength in {"strong", "moderate"}
                out.update(
                    {
                        "MAX 00l evidence": ev.status,
                        "002 (°)": round(ev.two_theta_002, 2) if report_values and ev.two_theta_002 is not None else None,
                        "004 (°)": round(ev.two_theta_004, 2) if report_values and ev.two_theta_004 is not None else None,
                        "006 (°)": round(ev.two_theta_006, 2) if report_values and ev.two_theta_006 is not None else None,
                        "c mean (Å)": round(ev.c_mean, 4) if report_values and ev.c_mean is not None else None,
                        "c std (Å)": round(ev.c_std, 4) if report_values and ev.c_std is not None else None,
                    }
                )
            rows.append(out)

        summary = pd.DataFrame(rows)
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.caption(
            "MP reference coverage is a descriptive peak-match metric, not phase percentage. MAX 00l values are shown only for moderate/strong linked evidence."
        )
        st.download_button(
            "Download batch summary CSV",
            summary.to_csv(index=False).encode("utf-8"),
            file_name="materials_project_xrd_batch_summary.csv",
            mime="text/csv",
        )

with tabs[1]:
    st.subheader("Sample inspector")
    sample = st.selectbox("Experimental sample", list(parsed.keys()), key="inspect_sample")
    df = parsed[sample]
    peaks, noise = detect_peaks(df)
    st.caption(f"Detected {len(peaks)} candidate experimental peaks. Robust noise estimate: {noise:.2f} intensity units.")

    ref_options = ["No reference overlay"] + list(references.keys())
    overlay_ref_name = st.selectbox("Overlay one Materials Project reference", ref_options, key="overlay_ref")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.two_theta, y=df.intensity, mode="lines", name="Experimental", line=dict(width=1)))

    if overlay_ref_name != "No reference overlay":
        ref = references[overlay_ref_name]
        r = ref["rows"]
        lo, hi = float(df.two_theta.min()), float(df.two_theta.max())
        r = r[(r.two_theta >= lo) & (r.two_theta <= hi)]
        ymax = float(df.intensity.max())
        scale = 0.22 * ymax / max(float(r.amplitude.max()), 1e-9) if len(r) else 1.0
        for _, rr in r.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[rr.two_theta, rr.two_theta],
                    y=[0, rr.amplitude * scale],
                    mode="lines",
                    line=dict(width=1),
                    showlegend=False,
                    hovertemplate=f"MP {ref['mp_id']}<br>2θ={rr.two_theta:.3f}°<br>hkl={rr.hkl}<br>rel I={rr.amplitude:.2f}<extra></extra>",
                )
            )
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=f"MP sticks: {ref['mp_id']}"))

    fig.update_layout(xaxis_title="2θ (°)", yaxis_title="Intensity (counts)", height=560, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    if references:
        st.markdown("#### Linked MAX / layered 00l check")
        if max_capable:
            max_name = st.selectbox("MP 00l template", max_capable, key="inspect_max_ref")
            ev = analyze_max_00l(df, references[max_name])
            st.write(f"**{ev.status}**")
            if ev.two_theta_002 is not None:
                detail = pd.DataFrame(
                    [
                        ["002", ev.two_theta_002, ev.c_002, ev.snr_002],
                        ["004", ev.two_theta_004, ev.c_004, ev.snr_004],
                        ["006", ev.two_theta_006, ev.c_006, ev.snr_006],
                    ],
                    columns=["Reflection candidate", "Observed 2θ (°)", "c from peak (Å)", "Peak SNR"],
                )
                st.dataframe(detail.round(4), hide_index=True, use_container_width=True)
                st.caption(
                    f"Reference c from {references[max_name]['mp_id']}: {ev.reference_c:.4f} Å. "
                    f"Candidate mean c: {ev.c_mean:.4f} Å; c std: {ev.c_std:.4f} Å; shift vs MP template: {ev.reference_shift_pct:+.2f}%."
                )
                if ev.strength in {"weak", "none"}:
                    st.warning("These candidate positions are shown for diagnosis only. They are not reported as convincing MAX 00l evidence in the batch table.")
        else:
            st.caption("No MP reference with 002/004/006 is loaded.")

with tabs[2]:
    st.subheader("Materials Project phase matching")
    if not references:
        st.warning("Upload Materials Project references first.")
    else:
        sample = st.selectbox("Sample", list(parsed.keys()), key="phase_sample")
        df = parsed[sample]
        phase_rows = []
        phase_details = {}
        for name, ref in references.items():
            m = match_materials_project_reference(
                df, ref,
                tolerance_deg=phase_tolerance,
                min_reference_amplitude=min_mp_amp,
                top_reference_peaks=top_mp_peaks,
                min_experimental_snr=min_exp_snr,
            )
            phase_details[name] = m
            phase_rows.append(
                {
                    "Reference file": name,
                    "MP-ID": ref["mp_id"],
                    "Weighted MP peak coverage (%)": round(m["coverage"], 1),
                    "Matched MP peaks": m["matched_count"],
                    "MP peaks considered": m["reference_count"],
                    "Mean matched Δ2θ (°)": None if np.isnan(m["mean_error_deg"]) else round(m["mean_error_deg"], 3),
                }
            )
        phase_df = pd.DataFrame(phase_rows).sort_values("Weighted MP peak coverage (%)", ascending=False)
        st.dataframe(phase_df, hide_index=True, use_container_width=True)
        st.caption("Coverage answers: 'How much of this MP reference pattern is represented by detected experimental peaks?' It is NOT phase fraction.")

        ref_name = st.selectbox("Inspect one MP reference match", list(phase_df["Reference file"]), key="phase_ref_detail")
        st.dataframe(phase_details[ref_name]["matches"].round(3), hide_index=True, use_container_width=True)
        st.download_button(
            "Download this peak-match table",
            phase_details[ref_name]["matches"].to_csv(index=False).encode("utf-8"),
            file_name=f"{sample}__{references[ref_name]['mp_id']}__matches.csv",
            mime="text/csv",
        )

with tabs[3]:
    st.subheader("Validated Materials Project reference library")
    if not references:
        st.write("No MP references loaded.")
    else:
        lib_rows = []
        for name, ref in references.items():
            oo = extract_00l_reference(ref, (2, 4, 6))
            lib_rows.append(
                {
                    "File": name,
                    "MP-ID": ref["mp_id"],
                    "Wavelength (Å)": ref["wavelength"],
                    "Reference peaks": len(ref["rows"]),
                    "Has 002/004/006": all(l in oo for l in (2, 4, 6)),
                    "Reference c from 00l (Å)": round(reference_c_from_00l(ref), 4) if all(l in oo for l in (2, 4, 6)) else None,
                }
            )
        st.dataframe(pd.DataFrame(lib_rows), hide_index=True, use_container_width=True)
        st.caption("Every row in this library passed strict Materials Project JSON validation.")

with tabs[4]:
    st.subheader("How to use this site for future samples and phases")
    st.markdown(
        """
1. **Download the XRD JSON from Materials Project** for every phase you want to test (MAX phase, NbN, AlN, Nb, Al, TiN, etc.). Keep the **mp-####** ID in the filename.
2. For permanent references, put those JSON files in this app's **`references/`** folder and redeploy. They will load automatically every time.
3. Upload any new experimental XRD CSV/TXT/DAT file. No code change is needed.
4. Use **MP phase matching** to see which uploaded Materials Project patterns are represented in the experiment.
5. If one of your MP references contains **002/004/006**, select it for the linked layered/MAX check.
6. Treat **reference coverage** and **MAX evidence** as identification aids, not quantitative purity. Use Rietveld refinement if you need phase percentages.

**Strict-source guarantee in this version:** the program contains no built-in literature peak positions and no fallback X-ray wavelength. If a reference is not a validated MP XRD JSON, it is rejected.
        """
    )
