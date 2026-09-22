from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.ndimage import percentile_filter
from scipy.signal import find_peaks, savgol_filter


@dataclass
class Peak:
    two_theta: float
    intensity: float
    prominence: float = float("nan")


@dataclass
class MaxSequenceResult:
    found_002: bool
    theta_002: float | None
    c_002: float | None
    theta_004: float | None
    c_004: float | None
    theta_006: float | None
    c_006: float | None
    c_mean: float | None
    c_std: float | None
    score: float
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _decode_bytes(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    return data.decode("utf-8", errors="replace")


def parse_xy_bytes(data: bytes, filename: str = "pattern.csv") -> pd.DataFrame:
    """Parse a two-column XRD file or a simple Materials Project-style JSON pattern."""
    suffix = Path(filename).suffix.lower()
    text = _decode_bytes(data)

    if suffix == ".json" or text.lstrip().startswith(("{", "[")):
        obj = json.loads(text)

        # Materials Project / pymatgen JSON files are not all exported with
        # exactly the same layout. Search recursively for common XRD schemas.
        x_keys = ("x", "two_theta", "2theta", "twoTheta", "two-theta", "twotheta")
        y_keys = ("y", "intensity", "intensities", "relative_intensity", "relativeIntensity")

        def as_numeric_1d(value):
            if not isinstance(value, (list, tuple)):
                return None
            try:
                arr = np.asarray(value, dtype=float)
            except Exception:
                return None
            if arr.ndim == 1 and arr.size >= 2:
                return arr
            return None

        def find_xy(node):
            # Dict containing paired arrays, e.g. pymatgen DiffractionPattern {x: [...], y: [...]}
            if isinstance(node, dict):
                for xk in x_keys:
                    if xk not in node:
                        continue
                    x = as_numeric_1d(node[xk])
                    if x is None:
                        continue
                    for yk in y_keys:
                        if yk in node:
                            y = as_numeric_1d(node[yk])
                            if y is not None and len(x) == len(y):
                                return x, y

                # A list of point records may live under pattern/data/peaks/etc.
                for value in node.values():
                    found = find_xy(value)
                    if found is not None:
                        return found

            elif isinstance(node, list):
                # [[2theta, intensity], ...]
                if len(node) >= 2 and all(isinstance(r, (list, tuple)) and len(r) >= 2 for r in node):
                    try:
                        arr = np.asarray([[r[0], r[1]] for r in node], dtype=float)
                        if arr.ndim == 2 and arr.shape[1] == 2:
                            return arr[:, 0], arr[:, 1]
                    except Exception:
                        pass

                # [{two_theta: ..., intensity: ...}, ...]
                if len(node) >= 2 and all(isinstance(r, dict) for r in node):
                    xs, ys = [], []
                    for r in node:
                        xv = next((r[k] for k in x_keys if k in r), None)
                        yv = next((r[k] for k in y_keys if k in r), None)
                        if xv is None or yv is None:
                            xs, ys = [], []
                            break
                        try:
                            xs.append(float(xv)); ys.append(float(yv))
                        except Exception:
                            xs, ys = [], []
                            break
                    if len(xs) >= 2:
                        return np.asarray(xs), np.asarray(ys)

                for value in node:
                    found = find_xy(value)
                    if found is not None:
                        return found
            return None

        # Materials Project legacy XRD export schema:
        # {"meta": ["amplitude", "hkl", "two_theta", "d_spacing"],
        #  "pattern": [[100, [1,1,1], 34.9, 2.57], ...]}
        # Use the metadata to locate the intensity and 2theta columns rather than
        # assuming the first two entries are x/y.
        if isinstance(obj, dict) and isinstance(obj.get("meta"), list) and isinstance(obj.get("pattern"), list):
            meta = [str(v).strip().lower() for v in obj["meta"]]
            try:
                x_idx = next(i for i, key in enumerate(meta) if key in {"two_theta", "2theta", "twotheta", "two-theta"})
                y_idx = next(i for i, key in enumerate(meta) if key in {"amplitude", "intensity", "relative_intensity", "relativeintensity"})
            except StopIteration:
                x_idx = y_idx = None
            if x_idx is not None and y_idx is not None:
                xs, ys = [], []
                for row in obj["pattern"]:
                    if not isinstance(row, (list, tuple)) or len(row) <= max(x_idx, y_idx):
                        continue
                    try:
                        xs.append(float(row[x_idx]))
                        ys.append(float(row[y_idx]))
                    except (TypeError, ValueError):
                        continue
                if len(xs) >= 2:
                    return clean_xy(pd.DataFrame({"two_theta": xs, "intensity": ys}))

        found = find_xy(obj)
        if found is not None:
            x, y = found
            return clean_xy(pd.DataFrame({"two_theta": x, "intensity": y}))

        raise ValueError(
            "JSON was read, but no recognizable XRD pattern was found. "
            "Supported formats include Materials Project exports with meta/pattern, "
            "{x:[...], y:[...]}, {two_theta:[...], intensity:[...]}, "
            "a list of point records, or a list of [2theta, intensity] pairs."
        )

    rows: list[tuple[float, float]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";", "//")):
            continue
        # Accept tabs, commas, semicolons, or general whitespace.
        normalized = line.replace(",", " ").replace(";", "\t")
        parts = normalized.split()
        nums: list[float] = []
        for part in parts:
            try:
                nums.append(float(part))
            except ValueError:
                continue
            if len(nums) == 2:
                break
        if len(nums) >= 2:
            rows.append((nums[0], nums[1]))

    if len(rows) < 3:
        raise ValueError("Could not find at least 3 rows with two numeric columns.")
    return clean_xy(pd.DataFrame(rows, columns=["two_theta", "intensity"]))


def clean_xy(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["two_theta", "intensity"]].copy()
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out["two_theta"] = pd.to_numeric(out["two_theta"], errors="coerce")
    out["intensity"] = pd.to_numeric(out["intensity"], errors="coerce")
    out = out.dropna().sort_values("two_theta")
    out = out.groupby("two_theta", as_index=False)["intensity"].mean()
    return out.reset_index(drop=True)


def preprocess_pattern(
    df: pd.DataFrame,
    baseline_window_deg: float = 2.0,
    smooth_window_deg: float = 0.10,
) -> pd.DataFrame:
    x = df["two_theta"].to_numpy(float)
    y = df["intensity"].to_numpy(float)
    if len(x) < 5:
        out = df.copy()
        out["corrected"] = np.maximum(y - np.min(y), 0)
        out["normalized"] = 100 * out["corrected"] / max(np.max(out["corrected"]), 1e-12)
        return out

    dx = float(np.median(np.diff(x))) if len(x) > 1 else 0.02
    dx = max(dx, 1e-6)
    bsize = max(5, int(round(baseline_window_deg / dx)))
    if bsize % 2 == 0:
        bsize += 1
    baseline = percentile_filter(y, percentile=10, size=bsize, mode="nearest")
    corrected = np.clip(y - baseline, 0, None)

    sw = max(5, int(round(smooth_window_deg / dx)))
    if sw % 2 == 0:
        sw += 1
    if sw >= len(corrected):
        sw = len(corrected) - 1 if len(corrected) % 2 == 0 else len(corrected)
    if sw >= 5:
        smoothed = savgol_filter(corrected, window_length=sw, polyorder=min(2, sw - 2), mode="interp")
        smoothed = np.clip(smoothed, 0, None)
    else:
        smoothed = corrected

    norm = 100 * smoothed / max(float(np.max(smoothed)), 1e-12)
    out = df.copy()
    out["baseline"] = baseline
    out["corrected"] = smoothed
    out["normalized"] = norm
    return out


def detect_peaks(
    processed: pd.DataFrame,
    prominence_pct: float = 3.0,
    min_distance_deg: float = 0.15,
    min_two_theta: float | None = None,
    max_two_theta: float | None = None,
) -> list[Peak]:
    df = processed
    if min_two_theta is not None:
        df = df[df["two_theta"] >= min_two_theta]
    if max_two_theta is not None:
        df = df[df["two_theta"] <= max_two_theta]
    if len(df) < 3:
        return []

    x = df["two_theta"].to_numpy(float)
    y = df["normalized"].to_numpy(float)
    dx = max(float(np.median(np.diff(x))), 1e-6)
    distance_pts = max(1, int(round(min_distance_deg / dx)))
    idx, props = find_peaks(y, prominence=prominence_pct, distance=distance_pts)
    proms = props.get("prominences", np.full(len(idx), np.nan))
    peaks = [Peak(float(x[i]), float(y[i]), float(p)) for i, p in zip(idx, proms)]
    return sorted(peaks, key=lambda p: p.two_theta)


def nearest_peak(peaks: Iterable[Peak], target: float, tolerance: float) -> Peak | None:
    candidates = [p for p in peaks if abs(p.two_theta - target) <= tolerance]
    if not candidates:
        return None
    # Favor proximity, then intensity.
    candidates.sort(key=lambda p: (abs(p.two_theta - target), -p.intensity))
    return candidates[0]


def c_from_00l(two_theta: float, l: int, wavelength: float = 1.5406) -> float:
    theta = math.radians(two_theta / 2.0)
    s = math.sin(theta)
    if s <= 0:
        raise ValueError("Invalid diffraction angle")
    return l * wavelength / (2.0 * s)


def two_theta_from_00l(c: float, l: int, wavelength: float = 1.5406) -> float:
    arg = l * wavelength / (2.0 * c)
    if arg <= 0 or arg >= 1:
        return float("nan")
    return math.degrees(2.0 * math.asin(arg))


def analyze_max_00l_sequence(
    peaks: list[Peak],
    expected_002: float,
    wavelength: float = 1.5406,
    search_window_002: float = 1.0,
    higher_order_tolerance: float = 0.45,
) -> MaxSequenceResult:
    p002_candidates = [p for p in peaks if abs(p.two_theta - expected_002) <= search_window_002]
    if not p002_candidates:
        return MaxSequenceResult(False, None, None, None, None, None, None, None, None, 0.0, "Weak / no 00l evidence")

    # Pick a strong peak but penalize large displacement from the expected center.
    def candidate_score(p: Peak) -> float:
        closeness = max(0.0, 1.0 - abs(p.two_theta - expected_002) / max(search_window_002, 1e-9))
        return 0.7 * p.intensity + 30.0 * closeness

    p002 = max(p002_candidates, key=candidate_score)
    c002 = c_from_00l(p002.two_theta, 2, wavelength)
    pred004 = two_theta_from_00l(c002, 4, wavelength)
    pred006 = two_theta_from_00l(c002, 6, wavelength)
    p004 = nearest_peak(peaks, pred004, higher_order_tolerance) if np.isfinite(pred004) else None
    p006 = nearest_peak(peaks, pred006, higher_order_tolerance) if np.isfinite(pred006) else None
    c004 = c_from_00l(p004.two_theta, 4, wavelength) if p004 else None
    c006 = c_from_00l(p006.two_theta, 6, wavelength) if p006 else None

    cvals = [v for v in (c002, c004, c006) if v is not None]
    cmean = float(np.mean(cvals)) if cvals else None
    cstd = float(np.std(cvals, ddof=0)) if len(cvals) >= 2 else None

    score = 40.0  # 002 found
    if p004:
        score += 25.0
    if p006:
        score += 20.0
    if len(cvals) >= 2 and cstd is not None:
        if cstd <= 0.08:
            score += 15.0
        elif cstd <= 0.20:
            score += 10.0
        elif cstd <= 0.40:
            score += 5.0
    score = min(score, 100.0)

    if score >= 80:
        evidence = "Strong MAX-like 00l evidence"
    elif score >= 55:
        evidence = "Partial MAX-like 00l evidence"
    else:
        evidence = "Weak / ambiguous 00l evidence"

    return MaxSequenceResult(
        True,
        p002.two_theta,
        c002,
        p004.two_theta if p004 else None,
        c004,
        p006.two_theta if p006 else None,
        c006,
        cmean,
        cstd,
        score,
        evidence,
    )


def reference_peaks_from_dense_pattern(
    df: pd.DataFrame,
    prominence_pct: float = 2.0,
    min_distance_deg: float = 0.15,
) -> list[Peak]:
    # Computed/reference XRD exports (including Materials Project JSON) are often
    # sparse stick patterns: one row per reflection rather than a dense scan.
    # In that case every listed reflection is already a peak, so do not run a
    # signal-processing peak finder that expects points between peaks.
    clean = clean_xy(df)
    if len(clean) >= 2:
        x = clean["two_theta"].to_numpy(float)
        dx = float(np.median(np.diff(x)))
        if dx > 0.10:
            y = clean["intensity"].to_numpy(float)
            ymax = max(float(np.max(y)), 1e-12)
            yn = 100.0 * y / ymax
            return [Peak(float(xx), float(yy), float("nan")) for xx, yy in zip(x, yn)]

    p = preprocess_pattern(clean)
    return detect_peaks(p, prominence_pct=prominence_pct, min_distance_deg=min_distance_deg)


def match_reference_peaks(
    sample_peaks: list[Peak],
    ref_peaks: list[Peak],
    tolerance: float = 0.35,
    min_ref_intensity: float = 3.0,
) -> dict[str, Any]:
    refs = [p for p in ref_peaks if p.intensity >= min_ref_intensity]
    if not refs:
        return {"matched": 0, "reference_peaks": 0, "weighted_coverage_pct": float("nan"), "matches": []}

    matches = []
    total_weight = sum(max(p.intensity, 0.0) for p in refs)
    matched_weight = 0.0
    for rp in refs:
        sp = nearest_peak(sample_peaks, rp.two_theta, tolerance)
        if sp:
            matched_weight += max(rp.intensity, 0.0)
            matches.append({
                "reference_2theta": rp.two_theta,
                "sample_2theta": sp.two_theta,
                "delta_2theta": sp.two_theta - rp.two_theta,
                "reference_intensity": rp.intensity,
                "sample_intensity": sp.intensity,
            })
    coverage = 100.0 * matched_weight / total_weight if total_weight > 0 else float("nan")
    return {
        "matched": len(matches),
        "reference_peaks": len(refs),
        "weighted_coverage_pct": coverage,
        "matches": matches,
    }


def peaks_to_dataframe(peaks: list[Peak]) -> pd.DataFrame:
    return pd.DataFrame([asdict(p) for p in peaks])
