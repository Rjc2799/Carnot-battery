# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 13:24:42 2025

@author: RuturajChavan
"""

"""
Standalone plot script: Charging phase only (working-fluid curve + hot & cold storage traces)

Saves a figure 'charging_phase_only.png' showing:
 - saturation dome (bubble & dew)
 - charging (heat-pump) cycle states and connections
 - hot-storage (condensation/subcooling) secondary trace
 - cold-storage (evaporation) secondary trace

Configured to use:
  p_wf_high = 20.27 bar
  p_wf_low  = 2.03  bar

Requirements:
 - Python with NumPy, Matplotlib
 - fluid_properties_rp (same REFPROP wrapper used in the repo)
 - access to the same REFPROP setup / mixture as the main repository

Usage:
  python plot_charging_phase_only.py
"""
import numpy as np
import matplotlib.pyplot as plt
import fluid_properties_rp as flp

# --- User / design inputs ---------------------------------------------------
composition = [0.3, 0.6, 0.1]   # same mixture as repository
RP = flp.setRPFluid("PROPANE;ISOBUTANE;PENTANE")

# Pressures from your request (bar -> Pa)
p_wf_high = 20.27e5   # Pa (20.27 bar)
p_wf_low  = 2.03e5    # Pa (2.03 bar)

# Secondary / plotting parameters (tune if needed)
m_dot_wf = 0.0071     # kg/s (working-fluid mass flow used to convert to Hdot)
T_surrounding = 293.15  # K (ambient)
dT_pinch = 2  # K (approach temp for plotting secondary traces)
cp_water = 4180 # J/kg.K
# ----------------------------------------------------------------------------

# --- Charging-phase thermodynamic states (mirrors charging_phase logic) ---
# compressor inlet: assume slight superheat on low-pressure vapor side
# choose a sensible compressor inlet temperature near ambient minus pinch
T_comp_in = T_surrounding - dT_pinch
comp_in = flp.tp(T_comp_in, p_wf_low, RP=RP, composition=composition)

# isentropic compressor outlet at high pressure
eta_isentropic = 0.75
isentropic_out = flp.sp(comp_in[4], p_wf_high, RP=RP, composition=composition)
h_actual = comp_in[2] + (isentropic_out[2] - comp_in[2]) / eta_isentropic
comp_out = flp.hp(h_actual, p_wf_high, RP=RP, composition=composition)

# condenser / condenser inlet and outlet on high pressure
sat_vap_p_high = flp.prop_pq(p_wf_high, 1, RP=RP, composition=composition)
sat_liq_p_high = flp.prop_pq(p_wf_high, 0, RP=RP, composition=composition)

# throttle (expansion) states: choose a reasonable throt_in at ambient T
throt_in = flp.tp(T_surrounding, p_wf_high, RP=RP, composition=composition)
throt_out = flp.hp(throt_in[2], p_wf_low, RP=RP, composition=composition)

# evaporator (low-pressure vapor)
sat_vap_p_low = flp.prop_pq(p_wf_low, 1, RP=RP, composition=composition)
# ----------------------------------------------------------------------------

# --- Secondary storage temps used for plotting (generic) --------------------
Temp_A = T_surrounding
Temp_B = sat_liq_p_high[0] - dT_pinch   # hot storage target (condense approach)
Temp_hmid = sat_vap_p_high[0] - dT_pinch
Temp_F = throt_out[0] + dT_pinch        # cold-side outlet approach
Temp_E = comp_in[0] + dT_pinch          # cold-side inlet approach


Q_cond_1 = m_dot_wf * (comp_out[2] - sat_liq_p_high[2])
m_dot_water_1 = m_dot_wf * ((sat_vap_p_high[2] - sat_liq_p_high[2]) / (cp_water * (Temp_hmid - Temp_B)))
Temp_D = Temp_B + (Q_cond_1 / (m_dot_water_1 * cp_water))
# ----------------------------------------------------------------------------

# --- Build saturation dome (bubble & dew) ----------------------------------
p_vals = np.linspace(1e5, 80.6e5, 200)
hdot_bubble, T_bubble = [], []
hdot_dew, T_dew = [], []

for p in p_vals:
    try:
        bubble = flp.prop_pq(p, 0, RP=RP, composition=composition)
        dew = flp.prop_pq(p, 1, RP=RP, composition=composition)
        hdot_bubble.append(bubble[2] * m_dot_wf / 1000)  # kJ/s
        T_bubble.append(bubble[0])
        hdot_dew.append(dew[2] * m_dot_wf / 1000)       # kJ/s
        T_dew.append(dew[0])
    except:
        continue
# ----------------------------------------------------------------------------

# --- Cycle points (charging heat pump) -------------------------------------
cycle_points = [comp_in, comp_out, sat_vap_p_high, sat_liq_p_high, throt_in, throt_out, sat_vap_p_low, comp_in]
hdot_cycle = [s[2] * m_dot_wf / 1000 for s in cycle_points]  # kJ/s
T_cycle = [s[0] for s in cycle_points]
# ----------------------------------------------------------------------------

# --- Secondary storage traces for charging (hot & cold storage) ------------
# Hot storage (condensation side) — two segments: condensing and subcooling
# We create smooth linear segments between the key temperatures/enthalpies
hdot_cond_1 = np.linspace(comp_out[2]*m_dot_wf/1000, sat_liq_p_high[2]*m_dot_wf/1000, 80)
T_cond_1 = np.linspace(Temp_D, Temp_B, 80 )   # from compressor-out to hot storage inlet
hdot_cond_2 = np.linspace(sat_liq_p_high[2]*m_dot_wf/1000, throt_in[2]*m_dot_wf/1000, 40)
T_cond_2 = np.linspace( Temp_B, Temp_A, 40 )

hdot_hot = np.concatenate([hdot_cond_1, hdot_cond_2])
T_hot = np.concatenate([T_cond_1, T_cond_2])

# Cold storage (evaporation side) — single sensible path from Temp_F to Temp_E
hdot_evap = np.linspace(throt_out[2]*m_dot_wf/1000, comp_in[2]*m_dot_wf/1000, 80)
T_evap = np.linspace(Temp_F, Temp_E, 80)
# ----------------------------------------------------------------------------

# --- Plotting ---------------------------------------------------------------
plt.figure(figsize=(10, 7))

# Saturation dome
plt.plot(hdot_bubble, T_bubble, color='black', linestyle='-', linewidth=1)
plt.plot(hdot_dew, T_dew, color='black', linestyle='-', linewidth=1)

# Charging cycle (HP) full connections
plt.plot(hdot_cycle[:2], T_cycle[:2], 'g-', linewidth=2)
plt.plot([hdot_cycle[1], hdot_cycle[2]], [T_cycle[1], T_cycle[2]], 'g-', linewidth=2)
# non-ideal curve along high-pressure dome 3→4:
# sample states along the dew/bubble between corresponding enthalpies
h_start = hdot_cycle[2]
h_end   = hdot_cycle[3]
# draw small arc using hp() mapping back to T over Hdot range
hdot_targets = np.linspace(h_start, h_end, 40)
T_curve = []
for hdot in hdot_targets:
    try:
        h_spec = hdot * 1000 / m_dot_wf
        state = flp.hp(h_spec, p_wf_high, RP=RP, composition=composition)
        T_curve.append(state[0])
    except:
        T_curve.append(np.nan)
plt.plot(hdot_targets, T_curve, 'g-', linewidth=2)

plt.plot([hdot_cycle[3], hdot_cycle[4]], [T_cycle[3], T_cycle[4]], 'g-', linewidth=2)
plt.plot([hdot_cycle[4], hdot_cycle[5]], [T_cycle[4], T_cycle[5]], 'g-', linewidth=2)

# Plot hot and cold storage secondary curves
plt.plot(hdot_hot, T_hot, '-', color='red', linewidth=2, label='Hot storage')


plt.plot(hdot_cond_1[0], T_cond_1[0], 'o', color='red')
plt.text(hdot_cond_1[0]-0.1, T_cond_1[0]-8, 'Hot2', color='red', fontsize=15)

plt.plot(hdot_cond_1[-1], T_cond_1[-1], 'o', color='red')
plt.text(hdot_cond_1[-1]+0.11, T_cond_1[-1]-8, 'Mid2,Hot1', color='red', fontsize=15)



# plt.plot(hdot_hot[0], T_hot[0], 'o', color='red')
# plt.text(hdot_hot[0]+0.2, T_hot[0]+0.2, 'Hot2', color='red', fontsize=15)

plt.plot(hdot_hot[-1], T_hot[-1], 'o', color='red')
plt.text(hdot_hot[-1]+0.2, T_hot[-1]+0.2, 'Mid1', color='red', fontsize=15)


plt.plot(hdot_evap, T_evap, '-', color='blue', linewidth=2, label='Cold storage')
plt.plot(hdot_evap[0], T_evap[0], 'o', color='blue')
plt.text(hdot_evap[0]+0.05, T_evap[0]+3.5, 'Cold2', color='blue', fontsize=15)
plt.plot(hdot_evap[-1], T_evap[-1], 'o', color='blue')
plt.text(hdot_evap[0]+2.2, T_evap[0]+25, 'Cold1', color='blue', fontsize=15)

plt.axhline(y=T_surrounding, color='k', linestyle=':', linewidth=1.5, label=f'Ambient ({T_surrounding:.2f} K)')
x_min, x_max = plt.xlim()
plt.text(x_min + 0.02*(x_max - x_min), T_surrounding + 2,
         "Ambient temp", color='k', fontsize=10,
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))


def plot_nonideal_curve_hdot_exact(p, hdot_start, hdot_end, n_points=100, color='r', label=None):
    """Ensures exact endpoint matching for non-ideal processes"""
    # Include explicit start/end points
    hdot_targets = np.linspace(hdot_start, hdot_end, n_points)
    hdot_points, T_points = [], []
    
    for hdot_target in hdot_targets:
        try:
            # Convert back to specific enthalpy for calculation
            h_target = hdot_target * 1000 / m_dot_wf  # Convert back to J/kg
            state = flp.hp(h_target, p, RP=RP, composition=composition)
            hdot_points.append(state[2] * m_dot_wf / 1000)  # Convert back to kJ/s
            T_points.append(state[0])
        except:
            continue
    
    # Force include endpoints
    if hdot_points:
        hdot_points[0] = hdot_start  # Exact start point (State 6)
        hdot_points[-1] = hdot_end   # Exact end point (State 7)
    
    plt.plot(hdot_points, T_points, color=color, linewidth=2, label=label)

# Evaporation process (6→7→1) - Non-ideal
plot_nonideal_curve_hdot_exact(p_wf_low, hdot_cycle[5], hdot_cycle[6], color='g')
plt.plot([hdot_cycle[6], hdot_cycle[7]], [T_cycle[6], T_cycle[7]], 'g-', linewidth=2)  # 7→1 Superheating

# Annotate key cycle points
labels = ['1', '2', '', '3',
          '4', '5', '']
offsets = {
    '1': (20, 3),
    '2': (10, 8),
    '' : (8, 7),
    '3': (-5, 12),
    '4': (-4,13),
    '5': (-4,-16),
   
}

for i, (hdot, T, label) in enumerate(zip(hdot_cycle[:-1], T_cycle[:-1], labels)):
    plt.plot(hdot, T, 'go')
    dx, dy = offsets.get(label, (8, 8))
    plt.annotate(
        label,
        xy=(hdot, T),
        xytext=(dx, dy),
        textcoords='offset points',
        ha='left' if dx>=0 else 'right',
        va='bottom' if dy>=0 else 'top',
        fontsize=18,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
        zorder=10
    )

plt.xlabel('Enthalpy Flow Rate Ḣ (kJ/s)', fontsize=12)
plt.ylabel('Temperature (K)', fontsize=12)
plt.title('Charging phase (Heat Pump) - Working fluid and Storage traces\n'
          f'p_wf_high={p_wf_high/1e5:.2f} bar, p_wf_low={p_wf_low/1e5:.2f} bar', fontsize=13)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left')
plt.ylim(250, 440)

plt.tight_layout()
plt.savefig('charging_phase_only.png', dpi=300)
plt.show()
# ----------------------------------------------------------------------------