from __future__ import annotations

import io
import json
import math
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter


@dataclass
class Peak:
    pos: float
    intensity: float
    prominence: float
    snr: float
    smooth_pos: float


@dataclass
class OOLEvidence:
    status: str
    strength: str
    two_theta_002: Optional[float]
    two_theta_004: Optional[float]
    two_theta_006: Optional[float]
    c_002: Optional[float]
    c_004: Optional[float]
    c_006: Optional[float]
    c_mean: Optional[float]
    c_std: Optional[float]
    snr_002: Optional[float]
    snr_004: Optional[float]
    snr_006: Optional[float]
    reference_c: Optional[float]
    reference_shift_pct: Optional[float]
    wavelength: float
    weak_candidate: bool
    note: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -------------------------
# Input parsing
# -------------------------

def _read_bytes(source: Any) -> bytes:
    if hasattr(source, "read"):
        raw = source.read()
        try:
            source.seek(0)
        except Exception:
            pass
        if isinstance(raw, str):
            return raw.encode("utf-8")
        return bytes(raw)
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    with open(source, "rb") as f:
        return f.read()


def load_experimental_csv(source: Any) -> pd.DataFrame:
    """Read experimental XRD data from the first two numeric columns.

    Supports comma/tab/semicolon/whitespace-delimited files, with or without headers.
    Returns columns: two_theta, intensity.
    """
    raw = _read_bytes(source)
    text = raw.decode("utf-8-sig", errors="replace")
    df = pd.read_csv(
        io.StringIO(text),
        sep=r"[\t,;\s]+",
        engine="python",
        header=None,
        comment="#",
    )
    if df.shape[1] < 2:
        raise ValueError("Could not find two columns (2θ and intensity).")

    numeric_cols: List[int] = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(10, int(0.5 * len(df))):
            numeric_cols.append(c)
        if len(numeric_cols) == 2:
            break
    if len(numeric_cols) < 2:
        raise ValueError("Could not identify numeric 2θ and intensity columns.")

    out = pd.DataFrame(
        {
            "two_theta": pd.to_numeric(df[numeric_cols[0]], errors="coerce"),
            "intensity": pd.to_numeric(df[numeric_cols[1]], errors="coerce"),
        }
    ).dropna()
    out = out[np.isfinite(out.two_theta) & np.isfinite(out.intensity)].copy()
    out = out.sort_values("two_theta").drop_duplicates("two_theta", keep="last").reset_index(drop=True)
    if len(out) < 20:
        raise ValueError("Too few valid XRD points were found.")
    if not np.all(np.diff(out.two_theta.to_numpy()) > 0):
        raise ValueError("2θ values must increase monotonically after cleaning.")
    return out


def load_mp_json_strict(source: Any, name: str = "reference.json") -> Dict[str, Any]:
    """Parse a Materials Project XRD JSON export in strict mode.

    Strict means:
    - filename must contain an MP id like mp-996162;
    - JSON must contain meta + pattern;
    - meta must contain amplitude, hkl, two_theta, and d_spacing;
    - wavelength.in_angstroms must be present (no fallback wavelength);
    - calculations use only values read from this MP JSON.
    """
    mpid_match = re.search(r"mp-\d+", name or "", flags=re.IGNORECASE)
    if not mpid_match:
        raise ValueError(
            f"{name}: strict MP mode requires the filename to contain a Materials Project ID such as mp-996162."
        )
    mp_id = mpid_match.group(0).lower()

    raw = _read_bytes(source)
    try:
        obj = json.loads(raw.decode("utf-8-sig"))
    except Exception as e:
        raise ValueError(f"{name}: invalid JSON ({e}).") from e

    if not isinstance(obj, dict):
        raise ValueError(f"{name}: top-level JSON must be an object.")
    meta = obj.get("meta")
    pattern = obj.get("pattern")
    wave = obj.get("wavelength")
    if not isinstance(meta, list) or not isinstance(pattern, list) or not pattern:
        raise ValueError(f"{name}: not a recognized Materials Project XRD JSON export (missing meta/pattern).")
    if not isinstance(wave, dict) or "in_angstroms" not in wave:
        raise ValueError(f"{name}: missing Materials Project wavelength.in_angstroms; no fallback is allowed.")

    aliases = {
        "amplitude": {"amplitude", "intensity"},
        "hkl": {"hkl", "hkls"},
        "two_theta": {"two_theta", "2theta", "two theta"},
        "d_spacing": {"d_spacing", "d", "d spacing"},
    }
    normalized = [str(x).strip().lower() for x in meta]
    idx: Dict[str, int] = {}
    for key, names in aliases.items():
        for i, m in enumerate(normalized):
            if m in names:
                idx[key] = i
                break
    missing = [k for k in ("amplitude", "hkl", "two_theta", "d_spacing") if k not in idx]
    if missing:
        raise ValueError(f"{name}: MP JSON is missing required field(s): {', '.join(missing)}.")

    rows = []
    for row in pattern:
        if not isinstance(row, (list, tuple)):
            continue
        try:
            amp = float(row[idx["amplitude"]])
            hkl = row[idx["hkl"]]
            tt = float(row[idx["two_theta"]])
            d = float(row[idx["d_spacing"]])
        except Exception:
            continue
        if np.isfinite(tt) and np.isfinite(amp) and np.isfinite(d):
            rows.append({"amplitude": amp, "hkl": hkl, "two_theta": tt, "d_spacing": d})
    if not rows:
        raise ValueError(f"{name}: no valid XRD pattern rows were found.")

    wavelength = float(wave["in_angstroms"])
    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError(f"{name}: invalid Materials Project wavelength.")

    return {
        "name": name,
        "mp_id": mp_id,
        "wavelength": wavelength,
        "rows": pd.DataFrame(rows).sort_values("two_theta").reset_index(drop=True),
        "raw": obj,
        "source": "Materials Project XRD JSON",
    }


# Backward-compatible alias for older app code.
load_mp_json = load_mp_json_strict


# -------------------------
# Crystallography helpers
# -------------------------

def _00l_l(hkl: Any) -> Optional[int]:
    if not isinstance(hkl, (list, tuple)):
        return None
    try:
        vals = [int(round(float(v))) for v in hkl]
    except Exception:
        return None
    if len(vals) == 4:  # hexagonal h k i l
        if vals[0] == 0 and vals[1] == 0 and vals[2] == 0 and vals[3] != 0:
            return abs(vals[3])
    if len(vals) == 3:
        if vals[0] == 0 and vals[1] == 0 and vals[2] != 0:
            return abs(vals[2])
    return None


def extract_00l_reference(ref: Dict[str, Any], ls: Sequence[int] = (2, 4, 6)) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for _, r in ref["rows"].iterrows():
        l = _00l_l(r.hkl)
        if l in ls:
            candidate = {
                "two_theta": float(r.two_theta),
                "d_spacing": float(r.d_spacing),
                "amplitude": float(r.amplitude),
            }
            # If duplicate hkl entries occur, keep the stronger one.
            if l not in out or candidate["amplitude"] > out[l]["amplitude"]:
                out[l] = candidate
    return out


def c_from_two_theta(two_theta: float, l: int, wavelength: float) -> float:
    theta = math.radians(two_theta / 2.0)
    s = math.sin(theta)
    if s <= 0:
        raise ValueError("Invalid 2θ for Bragg calculation.")
    d = wavelength / (2.0 * s)
    return float(l * d)


def two_theta_from_c(c: float, l: int, wavelength: float) -> Optional[float]:
    arg = l * wavelength / (2.0 * c)
    if not (0 < arg < 1):
        return None
    return float(math.degrees(2.0 * math.asin(arg)))


def reference_c_from_00l(ref: Dict[str, Any], ls: Sequence[int] = (2, 4, 6)) -> Optional[float]:
    oo = extract_00l_reference(ref, ls)
    if not oo:
        return None
    vals = [l * oo[l]["d_spacing"] for l in oo if np.isfinite(oo[l]["d_spacing"])]
    return float(np.median(vals)) if vals else None


# -------------------------
# Peak detection
# -------------------------

def _robust_noise(y: np.ndarray, y_smooth: np.ndarray) -> float:
    residual = y - y_smooth
    med = np.median(residual)
    mad = np.median(np.abs(residual - med))
    return max(float(1.4826 * mad), 1e-9)


def detect_peaks(
    df: pd.DataFrame,
    smooth_width_deg: float = 0.10,
    minimum_distance_deg: float = 0.14,
    minimum_prominence_sigma: float = 1.0,
) -> Tuple[List[Peak], float]:
    x = df.two_theta.to_numpy(dtype=float)
    y = df.intensity.to_numpy(dtype=float)
    step = float(np.median(np.diff(x)))
    if step <= 0:
        raise ValueError("Experimental 2θ spacing must be positive.")

    window = max(5, int(round(smooth_width_deg / step)))
    if window % 2 == 0:
        window += 1
    if window >= len(y):
        window = len(y) - 1 if len(y) % 2 == 0 else len(y)
    if window >= 5:
        y_smooth = savgol_filter(y, window_length=window, polyorder=2, mode="interp")
    else:
        y_smooth = y.copy()

    noise = _robust_noise(y, y_smooth)
    distance_points = max(1, int(round(minimum_distance_deg / step)))
    prominence_floor = max(1.0, minimum_prominence_sigma * noise)
    idxs, props = find_peaks(y_smooth, prominence=prominence_floor, distance=distance_points)

    peaks: List[Peak] = []
    raw_half_window = max(1, int(round(0.06 / step)))
    for idx, prom in zip(idxs, props["prominences"]):
        a = max(0, idx - raw_half_window)
        b = min(len(y), idx + raw_half_window + 1)
        local_y = y[a:b]
        max_val = np.max(local_y)
        ties = np.flatnonzero(local_y == max_val) + a
        raw_idx = int(min(ties, key=lambda j: abs(x[j] - x[idx])))
        peaks.append(
            Peak(
                pos=float(x[raw_idx]),
                intensity=float(y[raw_idx]),
                prominence=float(prom),
                snr=float(prom / noise),
                smooth_pos=float(x[idx]),
            )
        )

    merged: List[Peak] = []
    for p in sorted(peaks, key=lambda z: z.pos):
        if not merged or abs(p.pos - merged[-1].pos) > max(step * 1.5, 0.03):
            merged.append(p)
        elif p.prominence > merged[-1].prominence:
            merged[-1] = p
    return merged, noise


def _choose_peak_near_prediction(
    peaks: Sequence[Peak],
    predicted: float,
    tolerance_deg: float,
    minimum_snr: float,
) -> Tuple[Optional[Peak], Optional[float]]:
    candidates = [p for p in peaks if abs(p.pos - predicted) <= tolerance_deg and p.snr >= minimum_snr]
    if not candidates:
        return None, None

    # Position fidelity dominates; SNR breaks close ties.
    def merit(p: Peak) -> float:
        err = abs(p.pos - predicted)
        pos_term = math.exp(-0.5 * (err / max(tolerance_deg / 3.0, 0.03)) ** 2)
        snr_term = min(p.snr / 10.0, 1.0)
        return 0.88 * pos_term + 0.12 * snr_term

    best = max(candidates, key=merit)
    return best, float(abs(best.pos - predicted))


# -------------------------
# MAX / layered 00l analysis
# -------------------------

def analyze_max_00l(
    df: pd.DataFrame,
    max_reference: Dict[str, Any],
    c_search_fraction: float = 0.10,
    position_tolerance_deg: float = 0.35,
    candidate_snr: float = 3.0,
    report_002_snr: float = 4.0,
    strong_002_snr: float = 6.0,
    strong_secondary_snr: float = 5.0,
) -> OOLEvidence:
    """Assess a linked 002/004/006 sequence using ONLY the selected MP JSON.

    Algorithm:
    1) obtain MP 002/004/006 and wavelength from the uploaded MP JSON;
    2) allow the experimental c lattice parameter to vary around the MP c template;
    3) every candidate 002 defines one c;
    4) that same c predicts 004 and 006;
    5) 004/006 are accepted only near those predicted positions;
    6) classify by c consistency and peak prominence/SNR.

    Weak candidates are deliberately not presented as confirmed reflections in the batch table.
    """
    ref_00l = extract_00l_reference(max_reference, (2, 4, 6))
    if not all(l in ref_00l for l in (2, 4, 6)):
        raise ValueError("Selected MP MAX template must contain 002, 004, and 006 reflections.")

    wavelength = float(max_reference["wavelength"])
    cref = reference_c_from_00l(max_reference, (2, 4, 6))
    if cref is None:
        raise ValueError("Could not calculate c from the selected Materials Project 00l reflections.")

    peaks, _ = detect_peaks(df)
    cmin, cmax = cref * (1 - c_search_fraction), cref * (1 + c_search_fraction)

    candidates = []
    for p002 in peaks:
        if p002.snr < candidate_snr:
            continue
        try:
            c002 = c_from_two_theta(p002.pos, 2, wavelength)
        except Exception:
            continue
        if not (cmin <= c002 <= cmax):
            continue

        pred004 = two_theta_from_c(c002, 4, wavelength)
        pred006 = two_theta_from_c(c002, 6, wavelength)
        if pred004 is None or pred006 is None:
            continue

        p004, e004 = _choose_peak_near_prediction(peaks, pred004, position_tolerance_deg, candidate_snr)
        p006, e006 = _choose_peak_near_prediction(peaks, pred006, position_tolerance_deg, candidate_snr)
        if p004 is None or p006 is None:
            continue

        c004 = c_from_two_theta(p004.pos, 4, wavelength)
        c006 = c_from_two_theta(p006.pos, 6, wavelength)
        cs = np.array([c002, c004, c006], dtype=float)
        cmean = float(np.mean(cs))
        cstd = float(np.std(cs, ddof=0))
        rel_std = cstd / cmean if cmean else np.inf

        # Objective is deliberately crystallography-first. A huge unrelated peak cannot
        # compensate for a poor 00l relationship.
        consistency = math.exp(-0.5 * (cstd / 0.04) ** 2)
        position = math.exp(-0.5 * ((e004 or 0) / 0.15) ** 2) * math.exp(-0.5 * ((e006 or 0) / 0.18) ** 2)
        signal = min(p002.snr, 12) / 12 + min(p004.snr, 12) / 12 + min(p006.snr, 12) / 12
        signal /= 3.0
        objective = 0.62 * consistency + 0.25 * position + 0.13 * signal
        candidates.append(
            dict(
                objective=objective,
                p002=p002, p004=p004, p006=p006,
                c002=c002, c004=c004, c006=c006,
                cmean=cmean, cstd=cstd, rel_std=rel_std,
                e004=e004, e006=e006,
            )
        )

    if not candidates:
        return OOLEvidence(
            status="No convincing linked 00l sequence",
            strength="none",
            two_theta_002=None, two_theta_004=None, two_theta_006=None,
            c_002=None, c_004=None, c_006=None,
            c_mean=None, c_std=None,
            snr_002=None, snr_004=None, snr_006=None,
            reference_c=cref, reference_shift_pct=None, wavelength=wavelength,
            weak_candidate=False,
            note="No linked 002/004/006 candidate satisfied the minimum peak and crystallographic constraints.",
        )

    best = max(candidates, key=lambda z: z["objective"])
    p002, p004, p006 = best["p002"], best["p004"], best["p006"]
    cstd = best["cstd"]
    snrs = (p002.snr, p004.snr, p006.snr)

    # Conservative labels. Weak candidates are retained for inspection but not treated as confirmation.
    if (
        cstd <= 0.03
        and p002.snr >= strong_002_snr
        and p004.snr >= strong_secondary_snr
        and p006.snr >= strong_secondary_snr
    ):
        strength = "strong"
        status = "Strong linked MAX-like 00l evidence"
        weak_candidate = False
    elif (
        cstd <= 0.06
        and p002.snr >= strong_002_snr
        and min(p004.snr, p006.snr) >= candidate_snr
        and sum(s >= strong_secondary_snr for s in snrs) >= 2
    ):
        strength = "moderate"
        status = "Moderate linked 00l evidence (one reflection may be weak)"
        weak_candidate = False
    elif (
        cstd <= 0.08
        and p002.snr >= report_002_snr
        and min(p004.snr, p006.snr) >= candidate_snr
    ):
        strength = "weak"
        status = "Weak / ambiguous linked 00l candidate — not convincing"
        weak_candidate = True
    else:
        strength = "none"
        status = "No convincing linked 00l sequence"
        weak_candidate = True

    shift = 100.0 * (best["cmean"] - cref) / cref

    # For weak/none evidence, keep candidate values internally for detailed inspection,
    # but the app can suppress them in the main summary to avoid overclaiming.
    return OOLEvidence(
        status=status,
        strength=strength,
        two_theta_002=p002.pos,
        two_theta_004=p004.pos,
        two_theta_006=p006.pos,
        c_002=best["c002"],
        c_004=best["c004"],
        c_006=best["c006"],
        c_mean=best["cmean"],
        c_std=best["cstd"],
        snr_002=p002.snr,
        snr_004=p004.snr,
        snr_006=p006.snr,
        reference_c=cref,
        reference_shift_pct=shift,
        wavelength=wavelength,
        weak_candidate=weak_candidate,
        note=(
            "All 00l positions and the X-ray wavelength come from the selected Materials Project JSON. "
            "002 defines c; 004/006 must occur where that same c predicts them. Peak SNR/prominence is then used to grade the evidence."
        ),
    )


# -------------------------
# General Materials Project phase matching
# -------------------------

def match_materials_project_reference(
    df: pd.DataFrame,
    reference: Dict[str, Any],
    tolerance_deg: float = 0.20,
    min_reference_amplitude: float = 1.0,
    top_reference_peaks: int = 20,
    min_experimental_snr: float = 4.0,
) -> Dict[str, Any]:
    """Compare one experimental pattern with one Materials Project XRD JSON.

    The returned metrics are descriptive pattern-support metrics only. They are NOT
    phase fraction, purity, or probability. Reference coverage asks how much of the
    MP reference pattern is represented. Experimental support asks how much of the
    detected experimental peak prominence is explained by that MP reference.
    """
    peaks, _ = detect_peaks(df)
    exp = [p for p in peaks if p.snr >= min_experimental_snr]
    rows = reference["rows"].copy()
    lo, hi = float(df.two_theta.min()), float(df.two_theta.max())
    rows = rows[(rows.two_theta >= lo) & (rows.two_theta <= hi)].copy()
    if rows.empty:
        return {
            "coverage": 0.0,
            "experimental_support": 0.0,
            "matched_count": 0,
            "reference_count": 0,
            "matched_experimental_count": 0,
            "mean_error_deg": np.nan,
            "support_label": "No in-range MP peaks",
            "matches": pd.DataFrame(),
        }

    max_amp = float(rows.amplitude.max()) if len(rows) else 0.0
    amp_floor = max_amp * (min_reference_amplitude / 100.0)
    rows = rows[rows.amplitude >= amp_floor].copy()
    rows = rows.sort_values("amplitude", ascending=False).head(top_reference_peaks)

    records = []
    weights, matched_weights, errors = [], [], []
    matched_exp_ids = set()
    for _, r in rows.iterrows():
        tt = float(r.two_theta)
        amp = float(r.amplitude)
        weight = max(amp, 0.0)
        weights.append(weight)
        if exp:
            j, nearest = min(enumerate(exp), key=lambda jp: abs(jp[1].pos - tt))
            err = abs(nearest.pos - tt)
            matched = err <= tolerance_deg
            exp_tt = nearest.pos if matched else np.nan
            exp_snr = nearest.snr if matched else np.nan
            if matched:
                matched_exp_ids.add(j)
        else:
            err, matched, exp_tt, exp_snr = np.nan, False, np.nan, np.nan
        if matched:
            matched_weights.append(weight)
            errors.append(err)
        else:
            matched_weights.append(0.0)
        records.append(
            {
                "MP 2θ (°)": tt,
                "Experimental 2θ (°)": exp_tt,
                "Δ2θ (°)": err,
                "MP relative intensity": amp,
                "hkl": r.hkl,
                "Experimental peak SNR": exp_snr,
                "Matched": matched,
            }
        )

    denom = sum(weights) or 1.0
    coverage = 100.0 * sum(matched_weights) / denom

    total_exp_prom = sum(max(p.prominence, 0.0) for p in exp) or 1.0
    matched_exp_prom = sum(max(exp[j].prominence, 0.0) for j in matched_exp_ids)
    experimental_support = 100.0 * matched_exp_prom / total_exp_prom

    matched_count = int(sum(bool(r["Matched"]) for r in records))
    ref_count = int(len(records))
    mean_error = float(np.mean(errors)) if errors else np.nan

    # Conservative descriptive label; never interpreted as a phase percentage.
    if matched_count >= 3 and coverage >= 75 and (np.isnan(mean_error) or mean_error <= 0.15):
        support_label = "Strong reference-pattern support"
    elif matched_count >= 2 and coverage >= 50 and (np.isnan(mean_error) or mean_error <= 0.25):
        support_label = "Moderate reference-pattern support"
    elif matched_count >= 1:
        support_label = "Limited / ambiguous reference-pattern support"
    else:
        support_label = "No convincing reference-pattern support"

    return {
        "coverage": float(coverage),
        "experimental_support": float(experimental_support),
        "matched_count": matched_count,
        "reference_count": ref_count,
        "matched_experimental_count": int(len(matched_exp_ids)),
        "mean_error_deg": mean_error,
        "support_label": support_label,
        "matches": pd.DataFrame(records),
    }
