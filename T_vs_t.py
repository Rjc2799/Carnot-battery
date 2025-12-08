# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 23:40:47 2025

@author: RuturajChavan
"""

"""
Plot tank temperature vs time for the Carnot-battery duty cycle.

Place this file in the same folder as `carnot_battery_2.py` and run:
    python plot_tank_temps_timeseries.py

What it does:
- Imports and runs the main computation functions from carnot_battery_2.py
  (charging_phase, preheating, standstill_after_charging, discharging_phase,
   standstill_after_discharging).
- Builds a piecewise time vector with the following sequence:
    1) Preheat (t = 0 -> preheating_time)
    2) Charging (operation_time)
    3) Standstill after charging (12 h)
    4) Discharging (discharging_time)
    5) Standstill after discharging (4 h)
- Uses the temperatures returned by the functions at the stage boundaries and
  linearly interpolates between stage boundaries to produce smooth curves.
- Shades the plot to show the phases and annotates the transition times.

Notes / assumptions:
- carnot_battery_2's functions return single-valued temperatures at stage
  boundaries rather than time-resolved profiles. This script therefore
  interpolates linearly between those boundary values to create a continuous
  temperature-vs-time plot for visualization.
- The mapping between variable names in carnot_battery_2 and tanks A..F is:
    Tank A -> Temp_A
    Tank B -> Temp_B
    Tank C -> (preheat T_final and mixes to Temp_B during/after charging)
    Tank D -> Temp_D
    Tank E -> Temp_E
    Tank F -> Temp_F
  Where an exact mapping could be ambiguous in places, the script documents
  choices (see comments) and falls back to sensible defaults (surrounding temp).
- The module carnot_battery_2 may depend on REFPROP / CoolProp via `fluid_properties_rp`.
  Run this script in the same environment where carnot_battery_2 runs successfully.

"""

import numpy as np
import matplotlib.pyplot as plt
import sys

# Try to import the user's module
try:
    import carnot_battery_2 as cb
except Exception as e:
    print("Error importing carnot_battery_2:", e)
    print("Make sure you run this script from the repo folder and that dependencies (REFPROP/CoolProp) are available.")
    sys.exit(1)


def safe_get(d, key, default=None):
    return d[key] if (isinstance(d, dict) and key in d) else default


def build_times_and_temps(chg, preheat, stand, disch, stand2):
    # Phase durations (seconds) - taken from carnot_battery_2 module
    t_pre = safe_get(preheat, "preheating_time", 14400)                 # preheat
    t_charge = cb.operation_time                                       # charging
    t_stand1 = 12 * 3600                                               # standstill after charging
    t_discharge = cb.discharging_time                                  # discharging
    t_stand2 = 4 * 3600                                                # final standstill

    # Cumulative times
    t0 = 0
    t1 = t0 + t_pre
    t2 = t1 + t_charge
    t3 = t2 + t_stand1
    t4 = t3 + t_discharge
    t5 = t4 + t_stand2

    times_key = np.array([t0, t1, t2, t3, t4, t5], dtype=float)

    # Temperatures at key times for tanks A-F.
    # We pick values from the dictionaries returned by functions. When exact
    # values are not available, we fall back to T_surrounding.
    T_sur = cb.T_surrounding

    # t0 (initial) - assume all tanks start close to surrounding temp
    T_A_0 = safe_get(chg, "Temp_A", T_sur)
    T_B_0 = T_sur
    T_C_0 = preheat.get("T_C", T_sur) if preheat is not None else T_sur
    T_D_0 = T_sur
    T_E_0 = T_sur
    T_F_0 = T_sur

    # t1 (end preheat) - only tank C has been raised (preheat["T_final"])
    T_A_1 = T_A_0
    T_B_1 = T_B_0
    T_C_1 = safe_get(preheat, "T_final", T_C_0)
    T_D_1 = T_D_0
    T_E_1 = T_E_0
    T_F_1 = T_F_0

    # t2 (end charging) - use the temperatures computed in charging_phase()
    T_A_2 = safe_get(chg, "Temp_A", T_A_1)
    T_B_2 = safe_get(chg, "Temp_B", T_B_1)
    # Tank C after charging: in the code Tank C final after mixing is represented
    # by chg["Temp_B"] / preheat's T_final mixing logic. We set it to Temp_B.
    T_C_2 = T_B_2
    T_D_2 = safe_get(chg, "Temp_D", T_D_1)
    T_E_2 = safe_get(chg, "Temp_E", T_E_1)
    T_F_2 = safe_get(chg, "Temp_F", T_F_1)

    # t3 (after 12h standstill) - use stand results
    T_A_3 = T_sur
    T_B_3 = safe_get(stand, "Temp_at_t_tankB", T_B_2)
    T_C_3 = safe_get(stand, "Temp_at_t_tankC", T_C_2)
    T_D_3 = safe_get(stand, "Temp_at_t_tankD", T_D_2)
    T_E_3 = safe_get(stand, "Temp_at_t_tankE", T_E_2)
    T_F_3 = safe_get(stand, "Temp_at_t_tankF", T_F_2)

    # t4 (end discharging) - use disch results
    # disch["Temp_g"] corresponds to tank B after discharge in the code
    T_A_4 = T_sur
    T_B_4 = safe_get(disch, "Temp_g", T_B_3)
    # disch["Temp_i"] corresponds to tank D after discharge
    T_C_4 = T_B_4  # C often similar to B/D after flow; keep consistent
    T_D_4 = safe_get(disch, "Temp_i", T_D_3)
    T_E_4 = T_sur
    T_F_4 = safe_get(disch, "Temp_k", T_F_3)

    # t5 (after 4h final standstill) - use stand2 results where available.
    # Mapping in the original code: stand2 returns Temp_at_t_tankG (G ~ B),
    # Temp_at_t_tankI (I ~ D), Temp_at_t_tankK (K ~ F).
    T_A_5 = T_sur
    T_B_5 = safe_get(stand2, "Temp_at_t_tankG", T_B_4)
    T_C_5 = T_B_5
    T_D_5 = safe_get(stand2, "Temp_at_t_tankI", T_D_4)
    T_E_5 = T_sur
    T_F_5 = safe_get(stand2, "Temp_at_t_tankK", T_F_4)

    temps_key = {
        "A": np.array([T_A_0, T_A_1, T_A_2, T_A_3, T_A_4, T_A_5]),
        "B": np.array([T_B_0, T_B_1, T_B_2, T_B_3, T_B_4, T_B_5]),
        "C": np.array([T_C_0, T_C_1, T_C_2, T_C_3, T_C_4, T_C_5]),
        "D": np.array([T_D_0, T_D_1, T_D_2, T_D_3, T_D_4, T_D_5]),
        "E": np.array([T_E_0, T_E_1, T_E_2, T_E_3, T_E_4, T_E_5]),
        "F": np.array([T_F_0, T_F_1, T_F_2, T_F_3, T_F_4, T_F_5]),
    }

    # Return the key times, temps per tank, and named phase boundaries
    phase_bounds = {
        "t0": t0, "t_pre_end": t1, "t_charge_end": t2,
        "t_stand1_end": t3, "t_discharge_end": t4, "t_final": t5
    }

    return times_key, temps_key, phase_bounds


def make_interpolated_series(times_key, temps_key, n_points=600):
    t_start, t_end = times_key[0], times_key[-1]
    t_vec = np.linspace(t_start, t_end, n_points)
    series = {}
    for tank, temps_key_vals in temps_key.items():
        # linear interpolation between key times
        series[tank] = np.interp(t_vec, times_key, temps_key_vals)
    return t_vec, series


def plot_series(t_vec, series, phase_bounds, time_unit="hours", outname="tank_temps_timeseries.png"):
    # Convert time axis to hours (default)
    if time_unit == "hours":
        t_plot = t_vec / 3600.0
        xlabel = "Time (hours)"
        conv = 3600.0
    elif time_unit == "minutes":
        t_plot = t_vec / 60.0
        xlabel = "Time (minutes)"
        conv = 60.0
    else:
        t_plot = t_vec
        xlabel = "Time (s)"
        conv = 1.0

    plt.figure(figsize=(11, 6))
    cmap = plt.get_cmap("tab10")
    tanks = sorted(series.keys())
    for i, tank in enumerate(tanks):
        plt.plot(t_plot, series[tank], label=f"Tank {tank}", linewidth=2, color=cmap(i))

    # Phase shading and vertical markers
    t0 = phase_bounds["t0"] / conv
    t1 = phase_bounds["t_pre_end"] / conv
    t2 = phase_bounds["t_charge_end"] / conv
    t3 = phase_bounds["t_stand1_end"] / conv
    t4 = phase_bounds["t_discharge_end"] / conv
    t5 = phase_bounds["t_final"] / conv

    # Colors with alpha for shading
    plt.axvspan(t0, t1, color="#fff3e0", alpha=0.5, label="Preheat (tank C)")
    plt.axvspan(t1, t2, color="#e6f2ff", alpha=0.4, label="Charging (design)")
    plt.axvspan(t2, t3, color="#f0f0f0", alpha=0.4, label="Standstill (12 h)")
    plt.axvspan(t3, t4, color="#e8f8e8", alpha=0.4, label="Discharging (ORC)")
    plt.axvspan(t4, t5, color="#fafafa", alpha=0.4, label="Standstill (4 h)")

    # Vertical lines marking transitions
    for tx in [t0, t1, t2, t3, t4]:
        plt.axvline(tx, color="k", linestyle="--", linewidth=0.7, alpha=0.6)

    plt.xlabel(xlabel)
    plt.ylabel("Temperature (K)")
    plt.title("Storage Tank Temperatures vs Time (piecewise interpolated)")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outname, dpi=300)
    plt.show()
    print(f"Saved figure to {outname}")


def main():
    # Run the computational functions to get stage boundary values
    print("Running charging phase...")
    chg = cb.charging_phase()
    print("Computing standstill after charging...")
    stand = cb.standstill_after_charging(chg)
    print("Computing discharging phase...")
    disch = cb.discharging_phase(chg, stand)
    print("Computing standstill after discharging...")
    stand2 = cb.standstill_after_discharging(chg, stand, disch)

    print("Computing preheating (tank C)...")
    preheat = cb.preheating(
        m_water_1=chg["m_water_1"],
        cp_water=cb.cp_water,
        Temp_B=chg["Temp_B"],
        T_surrounding=cb.T_surrounding,
        m_dot_wf=cb.m_dot_wf,
        comp_out=chg["comp_out"],
        throt_in=chg["throt_in"],
    )

    # Build times and temperature keypoints
    times_key, temps_key, phase_bounds = build_times_and_temps(chg, preheat, stand, disch, stand2)

    # Interpolate to continuous curves
    t_vec, series = make_interpolated_series(times_key, temps_key, n_points=1200)

    # Plot
    plot_series(t_vec, series, phase_bounds, time_unit="hours", outname="tank_temps_timeseries.png")


if __name__ == "__main__":
    main()