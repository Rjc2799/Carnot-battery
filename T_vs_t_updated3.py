# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 15:13:27 2025

@author: RuturajChavan
"""

# tanks_custom_profile_legend_box.py
import matplotlib.pyplot as plt
import numpy as np
import csv
from typing import Dict, List, Tuple, Optional

# --- Narrative constants (your latest spec) ---
REFERENCE_TEMP = 293.15
PHASES = [("Preheating", 0.0, 4.0),
          ("Charging", 4.0, 8.0),
          ("Standstill", 8.0, 20.0),
          ("Discharging", 20.0, 24.0),
          ("Standstill", 24.0, 28.0)]
TANKS = ["Hot1", "Mid1", "Cold1", "Hot2", "Mid2", "Cold2"]
EPS = 1e-6  # tiny time offset for jumps

Series = Dict[str, Dict[str, List[float]]]

def init_series() -> Series:
    s = {t: {"time": [], "temp": []} for t in TANKS}
    # Only Hot1 known at t=0
    s["Hot1"]["time"].append(0.0); s["Hot1"]["temp"].append(REFERENCE_TEMP)
    return s

def last_point(s: Series, tank: str):
    if not s[tank]["time"]:
        return None, None
    return s[tank]["time"][-1], s[tank]["temp"][-1]

def ensure_defined(s: Series, tank: str, t: float, v: float):
    lt, lv = last_point(s, tank)
    if lt is None:
        s[tank]["time"].append(t); s[tank]["temp"].append(v)
    elif lt < t:
        s[tank]["time"].append(t); s[tank]["temp"].append(lv)

def jump(s: Series, tank: str, t: float, new_val: float):
    lt, lv = last_point(s, tank)
    if lt is None:
        s[tank]["time"].append(t); s[tank]["temp"].append(new_val); return
    if lt < t:
        s[tank]["time"].append(max(t - EPS, lt)); s[tank]["temp"].append(lv)
    s[tank]["time"].append(t); s[tank]["temp"].append(new_val)

def hold_until(s: Series, tank: str, t_end: float):
    lt, lv = last_point(s, tank)
    if lt is not None and t_end > lt:
        s[tank]["time"].append(t_end); s[tank]["temp"].append(lv)

def ramp_to(s: Series, tank: str, t_end: float, new_val: float):
    lt, lv = last_point(s, tank)
    if lt is None or np.isclose(t_end, lt):
        s[tank]["time"].append(t_end + EPS); s[tank]["temp"].append(new_val)
    else:
        s[tank]["time"].append(t_end); s[tank]["temp"].append(new_val)

def build_from_spec() -> Series:
    s = init_series()

    # Hot1
    hold_until(s, "Hot1", 4.0);  jump(s, "Hot1", 4.0, 361.28)
    hold_until(s, "Hot1", 20.0); jump(s, "Hot1", 20.0, 356.08)
    hold_until(s, "Hot1", 24.0); ramp_to(s, "Hot1", 28.0, 351.90)

    # Mid1
    ensure_defined(s, "Mid1", 4.0, 293.15); hold_until(s, "Mid1", 28.0)

    # Cold1
    ensure_defined(s, "Cold1", 4.0, 293.15)
    hold_until(s, "Cold1", 20.0); jump(s, "Cold1", 20.0, 296.73)
    hold_until(s, "Cold1", 24.0); ramp_to(s, "Cold1", 28.0, 293.99)

    # Hot2
    ensure_defined(s, "Hot2", 4.0, 293.15)
    hold_until(s, "Hot2", 8.0);  jump(s, "Hot2", 8.0, 372.70)
    ramp_to(s, "Hot2", 20.0, 367.42); hold_until(s, "Hot2", 28.0)

    # Mid2
    ensure_defined(s, "Mid2", 4.0, 293.15); jump(s, "Mid2", 4.0, 361.28)
    hold_until(s, "Mid2", 8.0); ramp_to(s, "Mid2", 20.0, 352.24); hold_until(s, "Mid2", 28.0)

    # Cold2  (FIXED: ramp 8→20 to **274.42 K**)
    ensure_defined(s, "Cold2", 4.0, 293.15)   # pre-drop reference
    jump(s, "Cold2", 4.0, 272.70)            # drop at 4 h
    hold_until(s, "Cold2", 8.0)              # hold 4→8
    ramp_to(s, "Cold2", 20.0, 274.42)        # <-- corrected: ramp up to 274.42 by 20 h
    hold_until(s, "Cold2", 28.0)             # hold 20→28

    # sort & dedupe
    for t in TANKS:
        x = np.array(s[t]["time"]); y = np.array(s[t]["temp"])
        order = np.argsort(x); x, y = x[order], y[order]
        keep = [0]
        for i in range(1, len(x)):
            if not (np.isclose(x[i], x[keep[-1]]) and np.isclose(y[i], y[keep[-1]])):
                keep.append(i)
        s[t]["time"] = list(x[keep]); s[t]["temp"] = list(y[keep])
    return s

def export_csv(s: Series, path: str):
    times = sorted(set(t for tank in TANKS for t in s[tank]["time"]))
    def val_at(tank, tt):
        x, y = s[tank]["time"], s[tank]["temp"]
        idx = np.searchsorted(x, tt, side="right") - 1
        return y[idx] if idx >= 0 else np.nan
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["time_h"] + TANKS)
        for tt in times:
            w.writerow([tt] + [val_at(t, tt) for t in TANKS])

def plot_series(s: Series, save_png: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(14, 8))
    for t in TANKS:
        ax.plot(s[t]["time"], s[t]["temp"], label=t)

    # reference
    ax.axhline(REFERENCE_TEMP, linestyle="--", linewidth=1)
    ax.text(0.01, 0.28, "Reference 293.15 K", transform=ax.transAxes, ha="left", va="bottom",fontsize=13)

    # vertical phase lines
    for x in [4, 8, 20, 24]:
        ax.axvline(x, linestyle=":", linewidth=0.8)

    # phase labels in top margin
    total_T = 28.0
    for name, a, b in PHASES:
        fig.text((a + b) / 2 / total_T+0.02, 0.95, name, ha="center", va="top", fontsize=18)

    # # legend INSIDE with a box
    # ax.legend(
    #     title="Tanks",
    #     loc="lower right",   # change to "best" / "upper left" if you prefer
    #     ncol=2,
    #     frameon=True, fancybox=True, framealpha=0.9,
    #     edgecolor="0.3", facecolor="white",
    #     borderpad=0.8, labelspacing=0.6, handlelength=2.0
    # )

    
    ax.set_xlabel("Time [hours]",fontsize=15); ax.set_ylabel("Temperature [K]",fontsize=15)
    ax.set_xlim(0, 28); ax.margins(y=0.1)
    fig.tight_layout()

    if save_png:
        fig.savefig(save_png, dpi=150, bbox_inches="tight")
    try:
        plt.show()
    except Exception:
        pass  # headless case

if __name__ == "__main__":
    series = build_from_spec()
    png_path = "tank_temperatures_plot.png"
    csv_path = "tank_temperatures_timeseries.csv"
    export_csv(series, csv_path)
    plot_series(series, save_png=png_path)
    print(f"Saved plot: {png_path}")
    print(f"Saved data: {csv_path}")
