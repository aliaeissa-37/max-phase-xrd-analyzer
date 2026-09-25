"""Regression check for the five current experimental files.

The test reference below reproduces ONLY the 002/004/006 values from the user's
Materials Project mp-996162 XRD JSON so the linked-peak logic can be checked locally.
Production app calculations never use this fixture; they require real uploaded/bundled MP JSON.
"""
import glob
import os
import pandas as pd
from xrd_core import analyze_max_00l, load_experimental_csv

ref = {
    "name": "mp-996162_test_fixture.json",
    "mp_id": "mp-996162",
    "wavelength": 1.54184,
    "rows": pd.DataFrame([
        {"amplitude": 33.5835, "hkl": [0, 0, 0, 2], "two_theta": 12.7378775, "d_spacing": 6.949597},
        {"amplitude": 8.8353, "hkl": [0, 0, 0, 4], "two_theta": 25.6366473, "d_spacing": 3.4747985},
        {"amplitude": 18.9131, "hkl": [0, 0, 0, 6], "two_theta": 38.8764730, "d_spacing": 2.3165323333},
    ]),
}

for path in sorted(glob.glob("/mnt/data/*.csv")):
    ev = analyze_max_00l(load_experimental_csv(path), ref)
    print(
        os.path.basename(path), "|", ev.status, "|",
        ev.two_theta_002, ev.two_theta_004, ev.two_theta_006,
        "| c=", None if ev.c_mean is None else round(ev.c_mean, 4),
        "std=", None if ev.c_std is None else round(ev.c_std, 4),
        "| SNR=", tuple(None if x is None else round(x, 2) for x in (ev.snr_002, ev.snr_004, ev.snr_006)),
    )
