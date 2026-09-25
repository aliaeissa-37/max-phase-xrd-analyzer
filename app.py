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

st.set_page_config(page_title="Materials Project XRD Analyzer v4", layout="wide")
st.title("Materials Project XRD Analyzer v4")
st.caption("Experimental XRD analysis using Materials Project XRD JSON references only")

st.info(
    "STRICT MATERIALS PROJECT MODE: every reference peak position, hkl, d-spacing, and wavelength used by this app comes from a validated Materials Project XRD JSON file. "
    "There are no built-in literature peak tables and no fallback X-ray wavelength."
)

with st.expander("What the results mean", expanded=False):
    st.markdown(
        """
- **MP reference coverage** = how much of the significant in-range Materials Project reference pattern has a matching experimental peak.
- **Experimental support** = how much of the detected experimental peak prominence is explained by that one MP reference.
- Neither value is phase percentage or purity.
- **Linked 00l evidence** tests whether 002, 004, and 006 can belong to one layered structure with a common **c** lattice parameter.
- Quantitative phase fractions require Rietveld refinement.
        """
    )


def load_bundled_references():
    refs, errors = {}, []
    ref_dir = Path(__file__).resolve().parent / "references"
    for path in sorted(ref_dir.glob("*.json")):
        try:
            refs[path.name] = load_mp_json_strict(str(path), path.name)
        except Exception as e:
            errors.append(f"{path.name}: {e}")
    return refs, errors


def ref_label(name, ref):
    oo = extract_00l_reference(ref, (2, 4, 6))
    extra = ""
    if 2 in oo:
        extra += f" | 002={oo[2]['two_theta']:.3f}°"
    if all(l in oo for l in (2, 4, 6)):
        c = reference_c_from_00l(ref, (2, 4, 6))
        extra += f" | c≈{c:.4f} Å"
    return f"{name} | {ref['mp_id']}{extra}"


bundled_refs, bundled_errors = load_bundled_references()

left, right = st.columns([1, 1])
with left:
    exp_files = st.file_uploader(
        "1) Upload experimental XRD CSV/TXT/DAT files",
        type=["csv", "txt", "dat"],
        accept_multiple_files=True,
        help="The first two numeric columns are interpreted as 2θ and intensity.",
    )
with right:
    uploaded_ref_files = st.file_uploader(
        "2) Add Materials Project XRD JSON references",
        type=["json"],
        accept_multiple_files=True,
        help="Strict mode requires the filename to contain an MP ID such as mp-996162.",
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
    st.warning("No Materials Project reference is loaded yet.")

if not exp_files:
    st.stop()

parsed, exp_errors = {}, []
for f in exp_files:
    try:
        parsed[f.name] = load_experimental_csv(f)
    except Exception as e:
        exp_errors.append(f"{f.name}: {e}")
for e in exp_errors:
    st.error(e)
if not parsed:
    st.stop()

with st.expander("Reference-matching settings", expanded=False):
    phase_tolerance = st.slider("MP peak match tolerance (° 2θ)", 0.08, 0.50, 0.20, 0.01)
    min_mp_amp = st.slider("Ignore MP reference peaks below this relative intensity (%)", 0.0, 10.0, 1.0, 0.5)
    top_mp_peaks = st.slider("Maximum MP peaks used per phase", 5, 40, 20, 1)
    min_exp_snr = st.slider("Minimum experimental peak SNR for phase matching", 2.0, 10.0, 4.0, 0.5)

# Build 00l-capable reference list, but only offer references whose 002/004/006
# are observable within the experimental scan range. This prevents e.g. AlN from
# being silently used as a MAX template when its 004/006 lie outside the scan.
common_lo = min(float(df.two_theta.min()) for df in parsed.values())
common_hi = max(float(df.two_theta.max()) for df in parsed.values())
max_capable = []
for name, ref in references.items():
    oo = extract_00l_reference(ref, (2, 4, 6))
    if all(l in oo for l in (2, 4, 6)):
        if all(common_lo <= oo[l]["two_theta"] <= common_hi for l in (2, 4, 6)):
            max_capable.append(name)

st.divider()
tabs = st.tabs(["Batch summary", "Sample inspector", "MP phase matching", "Reference library", "How to use"])

with tabs[0]:
    st.subheader("Batch summary")

    target_map = {ref_label(name, references[name]): name for name in max_capable}
    target_choice = st.selectbox(
        "Choose the Materials Project target for linked 002/004/006 analysis",
        ["— Select a target reference —"] + list(target_map.keys()),
        index=0,
        help=(
            "This is intentionally NOT auto-selected. The program cannot infer from an XRD JSON alone which phase you intend to call your MAX target. "
            "For your Ti2AlN comparison, select the MP JSON you downloaded for Ti2AlN."
        ),
    )
    selected_target = None if target_choice.startswith("—") else target_map[target_choice]

    if not max_capable:
        st.warning("No loaded MP reference has 002, 004, and 006 all inside the experimental scan range.")
    elif selected_target is None:
        st.info("Select the intended Materials Project target above before interpreting MAX/layered 00l evidence.")
    else:
        ref = references[selected_target]
        oo = extract_00l_reference(ref, (2, 4, 6))
        st.caption(
            f"Selected target: {selected_target} ({ref['mp_id']}). MP 00l positions: "
            f"002={oo[2]['two_theta']:.3f}°, 004={oo[4]['two_theta']:.3f}°, 006={oo[6]['two_theta']:.3f}°. "
            f"MP wavelength={ref['wavelength']:.5f} Å."
        )

    rows = []
    for sample, df in parsed.items():
        out = {"Sample": sample}
        if selected_target:
            ev = analyze_max_00l(df, references[selected_target])
            report_values = ev.strength in {"strong", "moderate"}
            out.update(
                {
                    "Target MP-ID": references[selected_target]["mp_id"],
                    "00l evidence": ev.status,
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
    if selected_target:
        st.download_button(
            "Download 00l batch summary CSV",
            summary.to_csv(index=False).encode("utf-8"),
            file_name="materials_project_00l_batch_summary.csv",
            mime="text/csv",
        )

    st.markdown("#### Materials Project reference-support matrix")
    st.caption("Each cell is 'MP reference coverage / experimental support'. This is evidence of pattern representation, NOT phase percentage.")
    matrix_rows = []
    for sample, df in parsed.items():
        row = {"Sample": sample}
        for name, ref in references.items():
            m = match_materials_project_reference(
                df,
                ref,
                tolerance_deg=phase_tolerance,
                min_reference_amplitude=min_mp_amp,
                top_reference_peaks=top_mp_peaks,
                min_experimental_snr=min_exp_snr,
            )
            row[f"{ref['mp_id']} ({name})"] = f"{m['coverage']:.0f}% ref / {m['experimental_support']:.0f}% exp ({m['matched_count']}/{m['reference_count']})"
        matrix_rows.append(row)
    matrix = pd.DataFrame(matrix_rows)
    st.dataframe(matrix, use_container_width=True, hide_index=True)

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

    st.markdown("#### Linked layered / MAX 00l check")
    if max_capable:
        inspect_map = {ref_label(name, references[name]): name for name in max_capable}
        choice = st.selectbox("Choose the intended MP target", ["— Select a target reference —"] + list(inspect_map.keys()), index=0, key="inspect_target")
        if not choice.startswith("—"):
            max_name = inspect_map[choice]
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
                    f"Reference c: {ev.reference_c:.4f} Å. Candidate mean c: {ev.c_mean:.4f} Å; "
                    f"c std: {ev.c_std:.4f} Å; shift vs MP target: {ev.reference_shift_pct:+.2f}%."
                )
                if ev.strength in {"weak", "none"}:
                    st.warning("Candidate positions are diagnostic only; they are not reported as convincing 00l evidence in the batch table.")
    else:
        st.caption("No loaded MP reference has 002/004/006 inside this experiment's scan range.")

with tabs[2]:
    st.subheader("Materials Project phase matching")
    if not references:
        st.warning("Upload Materials Project references first.")
    else:
        sample = st.selectbox("Sample", list(parsed.keys()), key="phase_sample")
        df = parsed[sample]
        phase_rows, phase_details = [], {}
        for name, ref in references.items():
            m = match_materials_project_reference(
                df,
                ref,
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
                    "Support": m["support_label"],
                    "MP reference coverage (%)": round(m["coverage"], 1),
                    "Experimental support (%)": round(m["experimental_support"], 1),
                    "Matched MP peaks": f"{m['matched_count']}/{m['reference_count']}",
                    "Matched experimental peaks": m["matched_experimental_count"],
                    "Mean matched Δ2θ (°)": None if np.isnan(m["mean_error_deg"]) else round(m["mean_error_deg"], 3),
                }
            )
        phase_df = pd.DataFrame(phase_rows).sort_values(["MP reference coverage (%)", "Experimental support (%)"], ascending=False)
        st.dataframe(phase_df, hide_index=True, use_container_width=True)
        st.caption(
            "A reference can have 100% MP coverage and still explain only a small part of the experimental pattern. "
            "That means the phase may be present, not that the sample is 100% that phase."
        )
        ref_name = st.selectbox("Inspect one MP reference match", list(phase_df["Reference file"]), key="phase_ref_detail")
        st.dataframe(phase_details[ref_name]["matches"].round(3), hide_index=True, use_container_width=True)

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
                    "002 (°)": round(oo[2]["two_theta"], 3) if 2 in oo else None,
                    "004 (°)": round(oo[4]["two_theta"], 3) if 4 in oo else None,
                    "006 (°)": round(oo[6]["two_theta"], 3) if 6 in oo else None,
                    "c from 00l (Å)": round(reference_c_from_00l(ref), 4) if all(l in oo for l in (2, 4, 6)) else None,
                    "00l usable in current scan": name in max_capable,
                }
            )
        st.dataframe(pd.DataFrame(lib_rows), hide_index=True, use_container_width=True)
        st.caption("The app does not infer chemical formulas from MP XRD JSON files. For clarity, you may rename a file while keeping the MP-ID, e.g. mp-996162_Ti2AlN_xrd_Cu.json.")

with tabs[4]:
    st.subheader("How to use this site")
    st.markdown(
        """
1. Download the XRD JSON from **Materials Project** for every phase you want to test.
2. Keep the **mp-####** ID in the filename. You may add a readable phase label, e.g. `mp-996162_Ti2AlN_xrd_Cu.json`.
3. Put permanent MP JSON references in `references/` on GitHub.
4. Upload your experimental XRD CSV/TXT/DAT files.
5. Use **MP phase matching** to check every reference separately. Do not interpret coverage as phase percentage.
6. For a layered/MAX question, explicitly choose the **intended target MP reference**. The app will never silently choose one for you.
7. The 002 candidate defines a c value; 004 and 006 must appear where that same c predicts them.
8. Use Rietveld refinement if you need quantitative phase fractions.

**Why v4 fixes the previous result:** v3 could automatically pick the first 002/004/006-capable MP file. Your AlN reference also contains 002/004/006, so it could be selected even though it was not your intended MAX target. v4 requires an explicit target and only offers targets whose 002/004/006 are inside the experimental scan range.
        """
    )
