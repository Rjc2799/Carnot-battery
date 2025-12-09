"""
Standalone plot script: Discharging phase only (ORC cycle + hot & cold storage traces + black dotted excess-entropy line)

This script draws:
 - saturation dome (bubble & dew lines)
 - ORC (discharging) cycle states and connections (pump, heater/evaporator, turbine, condenser)
 - hot-storage secondary traces (ORC evaporator hot-side)
 - cold-storage secondary traces (ORC condenser/cold-side)
 - black dotted line (visual "excess entropy / enthalpy rejection" path) running from a cold-storage reference to the actual turbine outlet

Notes:
 - Uses the same REFPROP wrapper API as the repository (fluid_properties_rp).
 - Default pressures for the ORC are p_orc_high = 16.0 bar and p_orc_low = 3.5 bar.
 - Default storage temperatures are provided as examples and can be modified at the top of the script.
 - Saves figure as 'discharging_phase_only.png'.
"""
import numpy as np
import matplotlib.pyplot as plt
import fluid_properties_rp as flp

# ------------------------ USER / DESIGN INPUTS -------------------------------
composition = [0.3, 0.6, 0.1]   # PROPANE;ISOBUTANE;PENTANE
RP = flp.setRPFluid("PROPANE;ISOBUTANE;PENTANE")

m_dot_wf = 0.0071     # kg/s (working-fluid mass flow used to convert to Hdot)
p_orc_high = 16.0e5   # Pa (16.0 bar)
p_orc_low  = 3.5e5    # Pa (3.5 bar)
eta_pump = 0.70
eta_turbine = 0.80
T_surrounding = 293.15  # K (ambient)
dT_pinch = 2  # K, plotting approach (used for sketching secondary traces)

# Example storage temperatures (edit these to match your run/results)
# These represent the temperatures used to draw the secondary traces for discharge.
Temp_g = 292.68   # hot-side inlet to ORC evaporator (K)  -- labelled 'Temp g' in plots
Temp_h = 352.24   # hot-side outlet at hot storage (K)    -- 'Temp h'
Temp_i = 356.08   # lower hot-side mixing temperature (K)  -- 'Temp i'
Temp_j = 367.42   # lower hot-side inventory (K)           -- 'Temp j'
Temp_k = 294.10   # cold-side after condenser (K)         -- 'Temp k'
Temp_l = 274.42  # cold sink ambient (K)            -- 'Temp l'
# ---------------------------------------------------------------------------

# ------------------------ COMPUTE ORC STATES --------------------------------
# analogous to the discharging_phase() state calculations
orc_state1 = flp.prop_pq(p_orc_low, 0, RP=RP, composition=composition)  # saturated liquid @ p_low
orc_isentropic_pump = flp.sp(orc_state1[4], p_orc_high, RP=RP, composition=composition)
h_pump_actual = orc_state1[2] + (orc_isentropic_pump[2] - orc_state1[2]) / eta_pump
orc_state2 = flp.hp(h_pump_actual, p_orc_high, RP=RP, composition=composition)

orc_state3 = flp.prop_pq(p_orc_high, 0, RP=RP, composition=composition)  # sat liquid at p_high
orc_state4 = flp.prop_pq(p_orc_high, 1, RP=RP, composition=composition)  # sat vapor at p_high

# Assume working-fluid is saturated vapor at p_orc_high before turbine
orc_state5 = flp.prop_pq(p_orc_high, 1, RP=RP, composition=composition)
ideal_state6 = flp.sp(orc_state5[4], p_orc_low, RP=RP, composition=composition)
h_turb_actual = orc_state5[2] - eta_turbine * (orc_state5[2] - ideal_state6[2])
orc_state6 = flp.hp(h_turb_actual, p_orc_low, RP=RP, composition=composition)

orc_state7 = flp.prop_pq(p_orc_low, 1, RP=RP, composition=composition)  # sat vap @ p_low
orc_state8 = flp.prop_pq(p_orc_low, 0, RP=RP, composition=composition)  # sat liq @ p_low
# ---------------------------------------------------------------------------

# ------------------------ HEAT RATES (for reference) ------------------------
Q_dot_orc_1_evap = m_dot_wf * (orc_state5[2] - orc_state3[2])  # W
Q_dot_orc_2_evap = m_dot_wf * (orc_state3[2] - orc_state2[2])  # W
Q_dot_orc_cond = m_dot_wf * (orc_state7[2] - orc_state8[2])    # W
# ---------------------------------------------------------------------------

# ------------------------ BUILD SATURATION DOME -----------------------------
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
# ---------------------------------------------------------------------------

# ------------------------ ORC Hdot / T arrays -------------------------------
orc_points = [orc_state1, orc_state2, orc_state3, orc_state4,
              orc_state5, orc_state6, orc_state7, orc_state8]
hdot_orc = [s[2] * m_dot_wf / 1000 for s in orc_points]  # kJ/s
T_orc = [s[0] for s in orc_points]
# ---------------------------------------------------------------------------

# ------------------------ STANDSTILL FUNCTION (included, not invoked) -------
def standstill_after_discharging(chg: dict, stand: dict, disch: dict, total_standstill_time_disch: float = 4 * 3600):
    """
    Compute post-discharge standstill temperatures for tanks (returns a dict).
    This function mirrors the behaviour in the original implementation but uses
    safe .get(...) defaults so it can be included here without forcing immediate execution.

    Inputs expected (if available in chg/stand/disch):
      - chg["m_water_1"], chg["m_glycol"], chg["m_water_2"], stand["U_overall"],
        stand["total_surface_area_C_D"], stand["total_surface_area_A_B"], stand["total_surface_area_E_F"],
        disch["m_dot_water_discharge_HG"], disch["m_dot_water_discharge_JI"], disch["m_dot_glycol_discharge_KL"],
        disch["Temp_i"], disch["Temp_g"], disch["Temp_k"]

    Returns:
      dict with keys like "Temp_at_t_tankG", "Temp_at_t_tankI", "Temp_at_t_tankK" etc.
    """
    # default physical properties
    cp_water = 4180.0
    cp_glycol_water = 3390.0

    # safe defaults when keys missing
    U_overall = stand.get("U_overall", 0.968)
    total_surface_area_A_B = stand.get("total_surface_area_A_B", 0.971)
    total_surface_area_C_D = stand.get("total_surface_area_C_D", 4.154)
    total_surface_area_E_F = stand.get("total_surface_area_E_F", 3.601)

    # Mass defaults (if not provided, use placeholders large enough to avoid zero-division)
    m_water_1 = chg.get("m_water_1", 604.2)
    m_water_2 = chg.get("m_water_2", 68.26)
    m_glycol = chg.get("m_glycol", 506.73)

    # discharge flow defaults
    m_dot_water_discharge_HG = disch.get("m_dot_water_discharge_HG", 0.005)
    m_dot_water_discharge_JI = disch.get("m_dot_water_discharge_JI", 0.042)
    m_dot_glycol_discharge_KL = disch.get("m_dot_glycol_discharge_KL", 0.035)

    # initial temperatures (use provided disch temps or fall back to the script-level Temps)
    T_i_tank_G = disch.get("Temp_g", Temp_g)
    T_i_tank_I = disch.get("Temp_i", Temp_i)
    T_i_tank_K = disch.get("Temp_k", Temp_k)

    # effective U multipliers (same approach as original)
    U_overall_G = U_overall * 25
    tau_G = (m_dot_water_discharge_HG * total_standstill_time_disch * cp_water) / (U_overall * total_surface_area_A_B)
    Temp_at_t_tankG = T_surrounding + ((T_i_tank_G - T_surrounding) * np.exp(-(total_standstill_time_disch / tau_G)))
    heat_loss_in_G = m_dot_water_discharge_HG * cp_water * (Temp_at_t_tankG - T_i_tank_G)

    U_overall_I = U_overall * 3
    tau_I = (m_water_1 * cp_water) / (U_overall_I * total_surface_area_C_D)
    Temp_at_t_tankI = T_surrounding + ((T_i_tank_I - T_surrounding) * np.exp(-(total_standstill_time_disch / tau_I)))
    heat_loss_in_I = m_dot_water_discharge_JI * cp_water * (Temp_at_t_tankI - T_i_tank_I)

    U_overall_K = U_overall * 50
    tau_K = (m_glycol * cp_glycol_water) / (U_overall_K * total_surface_area_E_F)
    Temp_at_t_tankK = T_surrounding + ((T_i_tank_K - T_surrounding) * np.exp(-(total_standstill_time_disch / tau_K)))
    heat_loss_in_K = m_dot_glycol_discharge_KL * cp_glycol_water * (Temp_at_t_tankK - T_i_tank_K)

    return {
        "Temp_at_t_tankG": Temp_at_t_tankG,
        "Temp_at_t_tankI": Temp_at_t_tankI,
        "Temp_at_t_tankK": Temp_at_t_tankK,
        "heat_loss_in_G": heat_loss_in_G,
        "heat_loss_in_I": heat_loss_in_I,
        "heat_loss_in_K": heat_loss_in_K,
        "tau_G": tau_G,
        "tau_I": tau_I,
        "tau_K": tau_K
    }
# ---------------------------------------------------------------------------

# ------------------------ SECONDARY TRACE CONSTRUCTION ----------------------
# Hot-side ORC evaporator secondary trace (two segments, mimic code style)
hdot_evap_orc_1 = np.linspace(orc_state2[2]*m_dot_wf/1000, orc_state3[2]*m_dot_wf/1000, 60)
T_evap_orc_1 = np.linspace(Temp_g, Temp_h, 60)   # show hot storage segment from Temp_g to Temp_h

# Use Temp_at_t_tankI variable for the second segment start (default to Temp_i if not computed)
# This allows a future call to standstill_after_discharging(...) to update Temp_at_t_tankI and redraw.
#Temp_at_t_tankI = Temp_i  # default value; calling standstill_after_discharging(...) can overwrite this if desired
Temp_at_t_tankI = 351.9 #K

hdot_evap_orc_2 = np.linspace(orc_state3[2]*m_dot_wf/1000, orc_state5[2]*m_dot_wf/1000, 60)
T_evap_orc_2 = np.linspace(Temp_at_t_tankI, Temp_j, 60)   # Temp j - Temp_at_t_tankI curve as requested

# Cold-side condenser trace
sf_dis_in = flp.tp(Temp_k, p_orc_low, RP=RP, composition=composition)
sf_dis_out = flp.tp(Temp_l, p_orc_low, RP=RP, composition=composition)
hdot_cond_orc = np.linspace(sf_dis_in[2]*m_dot_wf/1000, sf_dis_out[2]*m_dot_wf/1000, 60)
T_cond_orc = np.linspace(Temp_k, Temp_l, 60)
# ---------------------------------------------------------------------------

# ------------------------ BLACK DOTTED "EXCESS ENTROPY" LINE ----------------
# Recreate the dotted line concept: start at cold-storage reference (Temp_k) parallel to cold trace and end at actual turbine outlet
start_hdot = sf_dis_in[2] * m_dot_wf / 1000
start_T = Temp_k
end_hdot = orc_state6[2] * m_dot_wf / 1000
end_T = orc_state6[0]

# compute slope of the temp_k line for a parallel line (avoid division by zero)
if (sf_dis_out[2] - sf_dis_in[2]) != 0:
    temp_k_slope = (Temp_l - Temp_k) / (sf_dis_out[2] - sf_dis_in[2])
else:
    temp_k_slope = 0.0

hdot_range = end_hdot - start_hdot
parallel_end_T = start_T + (temp_k_slope * hdot_range * 1000 / m_dot_wf)

entropy_control_hdot = np.linspace(start_hdot, end_hdot, 50)
entropy_control_T = np.linspace(start_T, parallel_end_T, 50)
# ---------------------------------------------------------------------------

# ------------------------ PLOTTING -----------------------------------------
plt.figure(figsize=(12, 8))

# saturation dome
plt.plot(hdot_bubble, T_bubble, color='black', linewidth=0.8)
plt.plot(hdot_dew, T_dew, color='black', linewidth=0.8)

# ORC cycle (discharging) - thick brown line
plt.plot(hdot_orc[:2], T_orc[:2], color='saddlebrown', linewidth=3)   # 1->2 pump
# 2->3 (liquid heating)
plt.plot([hdot_orc[1], hdot_orc[2]], [T_orc[1], T_orc[2]], color='saddlebrown', linewidth=3)
# 3->4 phase region (plot along dome)
hdot_targets = np.linspace(hdot_orc[2], hdot_orc[3], 40)
T_curve = []
for hdot in hdot_targets:
    try:
        h_spec = hdot * 1000 / m_dot_wf
        state = flp.hp(h_spec, p_orc_high, RP=RP, composition=composition)
        T_curve.append(state[0])
    except:
        T_curve.append(np.nan)
plt.plot(hdot_targets, T_curve, color='saddlebrown', linewidth=3)
# 4->5 sensible heating
plt.plot([hdot_orc[3], hdot_orc[4]], [T_orc[3], T_orc[4]], color='saddlebrown', linewidth=3)
# 5->6 expansion (turbine)
plt.plot([hdot_orc[4], hdot_orc[5]], [T_orc[4], T_orc[5]], color='saddlebrown', linewidth=3)
# 6->7 condensation path (non-ideal)
plt.plot([hdot_orc[5], hdot_orc[6]], [T_orc[5], T_orc[6]], color='saddlebrown', linewidth=3)
# 7->8 and 8->1 completion
plt.plot([hdot_orc[6], hdot_orc[7]], [T_orc[6], T_orc[7]], color='saddlebrown', linewidth=3)
plt.plot([hdot_orc[7], hdot_orc[0]], [T_orc[7], T_orc[0]], color='saddlebrown', linewidth=3)

# plot ORC state markers
state_labels = ['6,7', '', '8', '9', '', '10', '11', '']

offsets = {
    '6,7': (-10, -2),
    '': (10, 8),
    '8' : (8, -5),
    '9': (12, -5),
    '': (-4,13),
    '10': (15, 2),
    '11': (4,-16)
}

for i, (hdot, T, label) in enumerate(zip(hdot_orc[:-1], T_orc[:-1], state_labels)):
    plt.plot(hdot, T, 'o', color='saddlebrown')
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

# secondary traces (hot storage - red; cold storage - blue)
plt.plot(hdot_evap_orc_1, T_evap_orc_1, '--', color='red', linewidth=2, label='ORC Hot storage')
plt.plot(hdot_evap_orc_1[0], T_evap_orc_1[0], 'o', color='red')
plt.text(hdot_evap_orc_1[0] - 0.3, T_evap_orc_1[0] - 2, 'Mid1', color='red', fontsize=15)

plt.plot(hdot_evap_orc_1[-1], T_evap_orc_1[-1], 'o', color='red')
plt.text(hdot_evap_orc_1[-1] - 0.3, T_evap_orc_1[-1] - 2, 'Mid2', color='red', fontsize=15)

plt.plot(hdot_evap_orc_2, T_evap_orc_2, '--', color='red', linewidth=2)
plt.text(hdot_evap_orc_2[0] - 0.2, T_evap_orc_2[0] + 4, 'Hot1', color='red', fontsize=15)
plt.text(hdot_evap_orc_2[-1] - 0.2, T_evap_orc_2[-1] + 4, 'Hot2', color='red', fontsize=15)

plt.plot(hdot_cond_orc, T_cond_orc, '--', color='blue', linewidth=2, label='ORC Cold storage')
plt.plot(hdot_evap_orc_2[0], T_evap_orc_2[0], 'o', color='red')
plt.plot(hdot_evap_orc_2[-1], T_evap_orc_2[-1], 'o', color='red')

# black dotted excess-entropy/enthalpy rejection path
plt.plot(entropy_control_hdot, entropy_control_T, 'k:', linewidth=3, alpha=0.9, label='Excess enthalpy path')

# annotate key temperatures on secondary traces
plt.plot(start_hdot, start_T, 'o', color='blue')
plt.text(start_hdot + 0.02, start_T - 4, 'Cold1', color='blue', fontsize=15)

plt.plot(hdot_cond_orc[-1], T_cond_orc[-1], 'o', color='blue')
plt.text(hdot_cond_orc[-1] + 0.02, T_cond_orc[-1] - 4, 'Cold2', color='blue', fontsize=15)

# appearance
plt.xlabel('Enthalpy Flow Rate Ḣ (kJ/s)', fontsize=12)
plt.ylabel('Temperature (K)', fontsize=12)
plt.title('Discharging Phase (ORC) — Working fluid & Storage traces', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper left')
plt.ylim(250, 420)
plt.xlim(min(hdot_bubble) - 0.5, max(hdot_dew) + 0.5)

plt.tight_layout()
plt.savefig('discharging_phase_only.png', dpi=300)
plt.show()
# ---------------------------------------------------------------------------