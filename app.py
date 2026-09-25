from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from xrd_core import (
    analyze_max_00l_sequence,
    detect_peaks,
    match_reference_peaks,
    parse_xy_bytes,
    peaks_to_dataframe,
    preprocess_pattern,
    reference_peaks_from_dense_pattern,
)

st.set_page_config(page_title="MAX Phase XRD Lab Analyzer", page_icon="🔬", layout="wide")

APP_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = APP_DIR / "example_data"


@st.cache_data(show_spinner=False)
def parse_file_bytes(data: bytes, filename: str) -> pd.DataFrame:
    return parse_xy_bytes(data, filename)


@st.cache_data(show_spinner=False)
def analyze_pattern_cached(
    data: bytes,
    filename: str,
    prominence_pct: float,
    min_distance_deg: float,
    baseline_window_deg: float,
    smooth_window_deg: float,
):
    raw = parse_xy_bytes(data, filename)
    processed = preprocess_pattern(raw, baseline_window_deg, smooth_window_deg)
    peaks = detect_peaks(processed, prominence_pct, min_distance_deg)
    return raw, processed, peaks


def read_uploaded(uploaded) -> tuple[str, bytes]:
    return uploaded.name, uploaded.getvalue()


def get_demo_files() -> list[tuple[str, bytes]]:
    files = []
    if EXAMPLE_DIR.exists():
        for p in sorted(EXAMPLE_DIR.glob("*.csv")):
            files.append((p.name, p.read_bytes()))
    return files


def add_reference(name: str, df: pd.DataFrame, peaks, source: str, meta: dict[str, Any] | None = None):
    st.session_state.references[name] = {
        "name": name,
        "df": df,
        "peaks": peaks,
        "source": source,
        "meta": meta or {},
    }


def build_reference_from_cif(name: str, data: bytes, wavelength: float):
    try:
        from pymatgen.core import Structure
        from pymatgen.analysis.diffraction.xrd import XRDCalculator
    except ImportError as exc:
        raise RuntimeError("CIF support requires pymatgen. Install dependencies from requirements.txt.") from exc

    text = data.decode("utf-8", errors="replace")
    structure = Structure.from_str(text, fmt="cif")
    calc = XRDCalculator(wavelength=wavelength)
    pattern = calc.get_pattern(structure, two_theta_range=(3, 90))
    df = pd.DataFrame({"two_theta": pattern.x, "intensity": pattern.y})
    peaks = []
    for x, y in zip(pattern.x, pattern.y):
        # calculated patterns are already sticks; prominence isn't meaningful here
        from xrd_core import Peak
        peaks.append(Peak(float(x), float(y), float("nan")))
    hkls = pattern.hkls
    return df, peaks, hkls


def fetch_mp_reference(query: str, api_key: str, wavelength: float):
    try:
        from pymatgen.ext.matproj import MPRester
        from pymatgen.analysis.diffraction.xrd import XRDCalculator
        from xrd_core import Peak
    except ImportError as exc:
        raise RuntimeError("Materials Project support requires pymatgen. Install dependencies from requirements.txt.") from exc

    query = query.strip()
    if not query:
        raise ValueError("Enter a Materials Project ID or exact formula.")

    with MPRester(api_key=api_key) as mpr:
        if query.lower().startswith("mp-"):
            material_id = query
            structure = mpr.get_structure_by_material_id(material_id, conventional_unit_cell=True)
            meta = {"material_id": material_id, "query": query}
        else:
            docs = mpr.summary.search(
                formula=query,
                fields=["material_id", "formula_pretty", "structure", "energy_above_hull", "is_stable"],
            )
            if not docs:
                raise ValueError(f"No Materials Project entries found for formula {query!r}.")
            docs = sorted(docs, key=lambda d: (not bool(getattr(d, "is_stable", False)), float(getattr(d, "energy_above_hull", 999) or 999)))
            doc = docs[0]
            material_id = str(doc.material_id)
            structure = doc.structure
            meta = {
                "material_id": material_id,
                "formula": getattr(doc, "formula_pretty", query),
                "energy_above_hull": getattr(doc, "energy_above_hull", None),
                "is_stable": getattr(doc, "is_stable", None),
                "query": query,
            }

    calc = XRDCalculator(wavelength=wavelength)
    pattern = calc.get_pattern(structure, two_theta_range=(3, 90))
    df = pd.DataFrame({"two_theta": pattern.x, "intensity": pattern.y})
    peaks = [Peak(float(x), float(y), float("nan")) for x, y in zip(pattern.x, pattern.y)]
    meta["hkls"] = pattern.hkls
    return f"{query} [{material_id}]", df, peaks, meta


if "references" not in st.session_state:
    st.session_state.references = {}

st.title("🔬 MAX Phase XRD Lab Analyzer")
st.caption(
    "A lab-facing XRD screening tool for MAX-phase synthesis. It looks for MAX-like (00l) sequences, "
    "compares full patterns with uploaded or computed references, and reports evidence rather than declaring a phase from one peak."
)

with st.sidebar:
    st.header("Instrument & analysis")
    wavelength = st.number_input("X-ray wavelength λ (Å)", min_value=0.1, max_value=3.0, value=1.5406, step=0.0001, format="%.4f")
    st.caption("Default is Cu Kα ≈ 1.5406 Å. Change this if your instrument used another source.")

    preset = st.selectbox(
        "MAX target preset",
        ["Nb₂AlN — lab target", "Ti₂AlN — lab reference", "Custom"],
    )
    preset_default = {"Nb₂AlN — lab target": 12.5, "Ti₂AlN — lab reference": 13.5, "Custom": 12.5}[preset]
    expected_002 = st.number_input("Expected (002) center, 2θ (°)", min_value=3.0, max_value=40.0, value=float(preset_default), step=0.05)
    search_window = st.number_input("(002) search ± (°)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    higher_tol = st.number_input("(004)/(006) tolerance ± (°)", min_value=0.05, max_value=2.0, value=0.45, step=0.05)

    st.divider()
    prominence_pct = st.slider("Peak prominence (% of normalized max)", 0.5, 20.0, 3.0, 0.5)
    min_distance_deg = st.number_input("Minimum peak separation (°)", min_value=0.02, max_value=2.0, value=0.15, step=0.05)
    baseline_window_deg = st.number_input("Baseline window (°)", min_value=0.2, max_value=10.0, value=2.0, step=0.2)
    smooth_window_deg = st.number_input("Smoothing window (°)", min_value=0.02, max_value=1.0, value=0.10, step=0.02)
    ref_match_tol = st.number_input("Reference match tolerance ± (°)", min_value=0.05, max_value=2.0, value=0.35, step=0.05)

    st.divider()
    st.warning("Screening aid only. Confirm phase identity with the full pattern, chemistry, and—when needed—Rietveld refinement or complementary characterization.")

analyze_tab, ref_tab, method_tab = st.tabs(["Analyze samples", "Reference library", "How it works"])

with ref_tab:
    st.subheader("Build a reference library")
    st.write("References can be experimental/computed XRD files, CIF structures, or Materials Project structures. Add likely target and impurity phases, then compare them against each sample.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Upload reference XRD / CIF**")
        ref_uploads = st.file_uploader(
            "Reference files",
            type=["csv", "txt", "xy", "dat", "json", "cif"],
            accept_multiple_files=True,
            key="reference_uploads",
        )
        if st.button("Add uploaded references", use_container_width=True):
            if not ref_uploads:
                st.info("Choose one or more reference files first.")
            else:
                for f in ref_uploads:
                    try:
                        if f.name.lower().endswith(".cif"):
                            df, peaks, hkls = build_reference_from_cif(f.name, f.getvalue(), wavelength)
                            add_reference(f.name, df, peaks, "CIF → pymatgen XRD", {"hkls": hkls})
                        else:
                            df = parse_xy_bytes(f.getvalue(), f.name)
                            peaks = reference_peaks_from_dense_pattern(df)
                            add_reference(f.name, df, peaks, "Uploaded XRD")
                        st.success(f"Added {f.name}")
                    except Exception as e:
                        st.error(f"{f.name}: {e}")

    with c2:
        st.markdown("**Materials Project → computed powder XRD**")
        default_secret = ""
        try:
            default_secret = st.secrets.get("MP_API_KEY", "")
        except Exception:
            pass
        mp_api_key = st.text_input("Materials Project API key", value=default_secret, type="password", help="Stored only for this session unless your deployment provides it as a Streamlit secret.")
        mp_query = st.text_input("MP material ID or exact formula", placeholder="e.g. mp-1234 or Ti2AlN")
        if st.button("Fetch & add MP reference", use_container_width=True):
            if not mp_api_key:
                st.error("Enter an API key or configure MP_API_KEY in Streamlit secrets.")
            else:
                try:
                    with st.spinner("Fetching structure and calculating XRD..."):
                        name, df, peaks, meta = fetch_mp_reference(mp_query, mp_api_key, wavelength)
                        add_reference(name, df, peaks, "Materials Project + pymatgen", meta)
                    st.success(f"Added {name}")
                except Exception as e:
                    st.error(str(e))

    st.divider()
    if st.session_state.references:
        rows = []
        for name, r in st.session_state.references.items():
            rows.append({"Reference": name, "Source": r["source"], "Peaks": len(r["peaks"]), "MP ID": r["meta"].get("material_id", "")})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        remove_name = st.selectbox("Remove reference", [""] + list(st.session_state.references.keys()))
        if remove_name and st.button("Remove selected reference"):
            del st.session_state.references[remove_name]
            st.rerun()
    else:
        st.info("No references added yet. You can still run the MAX (002)/(004)/(006) heuristic without a reference library.")

with analyze_tab:
    st.subheader("Analyze experimental XRD")
    u1, u2 = st.columns([2, 1])
    with u1:
        sample_uploads = st.file_uploader(
            "Upload experimental patterns",
            type=["csv", "txt", "xy", "dat", "json"],
            accept_multiple_files=True,
            key="sample_uploads",
            help="Two numeric columns are expected: 2θ and intensity. Tab-delimited CSV files are accepted.",
        )
    with u2:
        use_demo = st.checkbox("Load the 5 bundled lab example datasets", value=False)

    sample_files: list[tuple[str, bytes]] = []
    if sample_uploads:
        sample_files.extend([read_uploaded(f) for f in sample_uploads])
    if use_demo:
        sample_files.extend(get_demo_files())

    # Deduplicate by name while preserving last occurrence.
    sample_files = list({name: (name, data) for name, data in sample_files}.values())

    if not sample_files:
        st.info("Upload one or more experimental XRD files, or load the bundled lab example data.")
    else:
        sample_results: dict[str, dict[str, Any]] = {}
        summary_rows = []
        for name, data in sample_files:
            try:
                raw, processed, peaks = analyze_pattern_cached(
                    data,
                    name,
                    prominence_pct,
                    min_distance_deg,
                    baseline_window_deg,
                    smooth_window_deg,
                )
                max_result = analyze_max_00l_sequence(
                    peaks,
                    expected_002=expected_002,
                    wavelength=wavelength,
                    search_window_002=search_window,
                    higher_order_tolerance=higher_tol,
                )
                ref_results = {}
                for ref_name, ref in st.session_state.references.items():
                    ref_results[ref_name] = match_reference_peaks(peaks, ref["peaks"], tolerance=ref_match_tol)

                sample_results[name] = {
                    "raw": raw,
                    "processed": processed,
                    "peaks": peaks,
                    "max": max_result,
                    "references": ref_results,
                }
                best_ref = ""
                best_cov = float("nan")
                if ref_results:
                    ranked = [(rn, rr["weighted_coverage_pct"]) for rn, rr in ref_results.items() if np.isfinite(rr["weighted_coverage_pct"])]
                    if ranked:
                        best_ref, best_cov = max(ranked, key=lambda t: t[1])
                summary_rows.append({
                    "Sample": name,
                    "MAX 00l score": round(max_result.score, 1),
                    "MAX 00l evidence": max_result.evidence,
                    "002 (°)": round(max_result.theta_002, 3) if max_result.theta_002 is not None else np.nan,
                    "004 (°)": round(max_result.theta_004, 3) if max_result.theta_004 is not None else np.nan,
                    "006 (°)": round(max_result.theta_006, 3) if max_result.theta_006 is not None else np.nan,
                    "c mean (Å)": round(max_result.c_mean, 4) if max_result.c_mean is not None else np.nan,
                    "c std (Å)": round(max_result.c_std, 4) if max_result.c_std is not None else np.nan,
                    "Best loaded reference": best_ref,
                    "Reference coverage (%)": round(best_cov, 1) if np.isfinite(best_cov) else np.nan,
                })
            except Exception as e:
                st.error(f"Could not analyze {name}: {e}")

        summary_df = pd.DataFrame(summary_rows)
        if len(summary_df):
            st.markdown("### Batch summary")
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download batch summary CSV",
                data=summary_df.to_csv(index=False).encode("utf-8"),
                file_name="xrd_max_batch_summary.csv",
                mime="text/csv",
            )

            st.markdown("### Inspect one sample")
            selected = st.selectbox("Sample", list(sample_results.keys()))
            r = sample_results[selected]
            m = r["max"]

            a, b, c, d = st.columns(4)
            a.metric("MAX 00l score", f"{m.score:.0f}/100")
            b.metric("(002)", f"{m.theta_002:.3f}°" if m.theta_002 is not None else "not found")
            c.metric("Mean c", f"{m.c_mean:.3f} Å" if m.c_mean is not None else "—")
            d.metric("Evidence", m.evidence)

            fig = go.Figure()
            p = r["processed"]
            fig.add_trace(go.Scatter(x=p["two_theta"], y=p["normalized"], mode="lines", name=selected, line={"width": 2}))
            for ref_name, ref in st.session_state.references.items():
                rdf = ref["df"]
                if len(rdf) <= 300:
                    for _, row in rdf.iterrows():
                        inten = 100.0 * float(row["intensity"]) / max(float(rdf["intensity"].max()), 1e-12)
                        fig.add_trace(go.Scatter(
                            x=[float(row["two_theta"]), float(row["two_theta"])],
                            y=[0, inten],
                            mode="lines",
                            name=ref_name,
                            legendgroup=ref_name,
                            showlegend=False,
                            line={"width": 1},
                            opacity=0.45,
                        ))
                    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=ref_name, legendgroup=ref_name))
                else:
                    yy = 100.0 * rdf["intensity"] / max(float(rdf["intensity"].max()), 1e-12)
                    fig.add_trace(go.Scatter(x=rdf["two_theta"], y=yy, mode="lines", name=ref_name, opacity=0.5))

            for label, angle in [("002", m.theta_002), ("004", m.theta_004), ("006", m.theta_006)]:
                if angle is not None:
                    fig.add_vline(x=angle, line_dash="dash", opacity=0.7)
                    fig.add_annotation(x=angle, y=98, text=label, showarrow=False, yanchor="top")

            fig.update_layout(
                xaxis_title="2θ (degrees)",
                yaxis_title="Normalized intensity",
                hovermode="x unified",
                height=540,
                margin=dict(l=30, r=20, t=20, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)

            left, right = st.columns(2)
            with left:
                st.markdown("#### MAX (00l) sequence")
                seq = pd.DataFrame([
                    {"Reflection": "(002)", "Observed 2θ (°)": m.theta_002, "c from peak (Å)": m.c_002},
                    {"Reflection": "(004)", "Observed 2θ (°)": m.theta_004, "c from peak (Å)": m.c_004},
                    {"Reflection": "(006)", "Observed 2θ (°)": m.theta_006, "c from peak (Å)": m.c_006},
                ])
                st.dataframe(seq, use_container_width=True, hide_index=True)
                st.caption("The 00l score is a heuristic. A consistent 002/004/006 sequence is stronger evidence than a single low-angle peak, but it is not by itself a phase identification.")

            with right:
                st.markdown("#### Detected experimental peaks")
                pkdf = peaks_to_dataframe(r["peaks"]).sort_values("intensity", ascending=False)
                st.dataframe(pkdf.head(30), use_container_width=True, hide_index=True)

            if r["references"]:
                st.markdown("#### Reference matching")
                ref_summary = []
                for ref_name, rr in r["references"].items():
                    ref_summary.append({
                        "Reference": ref_name,
                        "Matched peaks": rr["matched"],
                        "Reference peaks considered": rr["reference_peaks"],
                        "Weighted coverage (%)": rr["weighted_coverage_pct"],
                    })
                st.dataframe(pd.DataFrame(ref_summary).sort_values("Weighted coverage (%)", ascending=False), use_container_width=True, hide_index=True)

                chosen_ref = st.selectbox("Show detailed matches", list(r["references"].keys()))
                match_df = pd.DataFrame(r["references"][chosen_ref]["matches"])
                if len(match_df):
                    st.dataframe(match_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No matched peaks within the selected tolerance.")

with method_tab:
    st.subheader("What the analyzer is doing")
    st.markdown(
        """
**1. Preprocesses the pattern.** A rolling low-percentile baseline is subtracted, the signal is lightly smoothed, and intensity is normalized to 100.

**2. Detects experimental peaks.** Peak prominence and minimum separation are adjustable so you can tune the detector to your instrument and sample quality.

**3. Tests the MAX-like (00l) sequence.** Starting from a candidate (002) near the lab's expected position, the app calculates the corresponding **c-lattice parameter** using Bragg's law. It then predicts where (004) and (006) should occur for the same c value and checks whether experimental peaks are actually there.

For a hexagonal MAX structure and a (00l) reflection:

\[
d_{00l}=\frac{c}{l}, \qquad 2d\sin\theta=\lambda
\]

So each of (002), (004), and (006) independently gives an estimate of **c**. If those estimates agree closely, that is more meaningful than seeing only one peak near 12–14°.

**4. Compares the full pattern with references.** Add target phases and likely residual/secondary phases (for example Nb₂AlN, Ti₂AlN, TiNbAlN, NbN, AlN, Nb, Al, TiN). The app reports how much of each reference pattern is represented in the detected experimental peaks.

**5. Keeps the conclusion cautious.** The score says **MAX-like XRD evidence**, not “phase confirmed.” Overlapping peaks, texture, preferred orientation, solid solutions, instrumental offsets, and multiphase samples can all affect a powder pattern.
        """
    )
    st.markdown("### Recommended lab workflow")
    st.markdown(
        "1. Add the target MAX reference and every plausible precursor/secondary phase to the reference library.\n"
        "2. Analyze all samples using identical peak settings.\n"
        "3. Inspect the 002/004/006 sequence and c consistency.\n"
        "4. Compare the **entire pattern**, not one peak.\n"
        "5. For publication-quality phase fractions, move to Rietveld refinement rather than relying on this screening score."
    )
    st.markdown("### Materials Project note")
    st.write("The app retrieves a crystal structure from Materials Project and calculates a powder XRD pattern locally with pymatgen. Materials Project structures are computational references; experimental literature/reference-card patterns should also be used when available.")
