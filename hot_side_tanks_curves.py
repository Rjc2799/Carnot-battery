# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 18:11:09 2025

@author: RuturajChavan
"""

import numpy as np
import matplotlib.pyplot as plt
import fluid_properties_rp as flp
from finding_pressures import p_wf_low, p_wf_high

# These are your simulation values from context
m_dot_wf = 0.0071  # kg/s
composition = [0.3, 0.6, 0.1]  # PROPANE;ISOBUTANE;PENTANE
RP = flp.setRPFluid("PROPANE;ISOBUTANE;PENTANE")

# From your simulation results (replace with actual simulation output if needed)
Temp_A = 293.15    # K (Tank A, 20°C)
Temp_B = 361.28    # K (Tank B, ~88°C)
Temp_D = 387.15    # K (Tank D, ~114°C)

def plot_temp_curve_bubble_dew(flp, m_dot_wf, RP, composition, Temp_A, Temp_B, Temp_D, p_wf_high):
    plt.figure(figsize=(10, 7))

    # Bubble and dew lines
    p_vals = np.linspace(1e5, 80.6e5, 150)
    hdot_bubble, T_bubble = [], []
    hdot_dew, T_dew = [], []
    for p in p_vals:
        try:
            bubble = flp.prop_pq(p, 0, RP=RP, composition=composition)
            hdot_bubble.append(bubble[2] * m_dot_wf / 1000)
            T_bubble.append(bubble[0])
            dew = flp.prop_pq(p, 1, RP=RP, composition=composition)
            hdot_dew.append(dew[2] * m_dot_wf / 1000)
            T_dew.append(dew[0])
        except Exception:
            continue

    plt.plot(hdot_bubble, T_bubble, color='black', linestyle='-', linewidth=1.5)
    plt.plot(hdot_dew, T_dew, color='black', linestyle='--', linewidth=1.5)

    # Curve for Temp_A, Temp_B, Temp_D
    h_A = flp.tp(Temp_A, p_wf_high, RP=RP, composition=composition)[2] * m_dot_wf / 1000
    h_B = flp.tp(Temp_B, p_wf_high, RP=RP, composition=composition)[2] * m_dot_wf / 1000
    h_D = flp.tp(Temp_D, p_wf_high, RP=RP, composition=composition)[2] * m_dot_wf / 1000

    T_curve = [Temp_A, Temp_B, Temp_D]
    hdot_curve = [h_A, h_B, h_D]
    plt.plot(hdot_curve, T_curve, marker='o', color='red', linewidth=3)

    # Annotate points
    plt.annotate('Mid1', (h_A, Temp_A), xytext=(10, -15), textcoords='offset points', color='red', fontsize=16)
    plt.annotate('Mid2,Hot1', (h_B, Temp_B), xytext=(10, -15), textcoords='offset points', color='red', fontsize=16)
    plt.annotate('Hot2', (h_D, Temp_D), xytext=(-8, -18), textcoords='offset points', color='red', fontsize=16)


    # Reference temperature line at Temp_A
    x_min = min(hdot_bubble + hdot_dew + hdot_curve) - 0.5
    x_max = max(hdot_bubble + hdot_dew + hdot_curve) + 0.5
    plt.axhline(y=Temp_A, xmin=0, xmax=1, color='blue', linestyle='--', linewidth=2,
                )
    #plt.text(x_max-0.2, Temp_A+2, f'T_ambient = {Temp_A:.2f} K',
             #color='blue', fontsize=12, ha='right', va='bottom', fontweight='bold')
    
    plt.xlabel("Enthalpy Flow Rate, Ḣ (kJ/s)", fontsize=13)
    plt.ylabel("Temperature (K)", fontsize=13)
    plt.title("Carnot Battery: Temp_A, Temp_B, Temp_D with Bubble/Dew Lines", fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_temp_curve_bubble_dew(flp, m_dot_wf, RP, composition, Temp_A, Temp_B, Temp_D, p_wf_high)