# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 17:01:33 2025

@author: RuturajChavan
"""

# -*- coding: utf-8 -*-
"""
Carnot Battery Simulation - Refactored with Functions
"""

import numpy as np
import matplotlib.pyplot as plt
import fluid_properties_rp as flp
import tkinter as tk
from tkinter import scrolledtext


# Set up REFPROP with fluid mixture
composition = [0.3, 0.6, 0.1]  # molar fractions for PROPANE;ISOBUTANE;PENTANE
RP = flp.setRPFluid("PROPANE;ISOBUTANE;PENTANE")
from finding_pressures import p_wf_low, p_wf_high


# Global constants
dT_pinch = 2  #K
dT_pinch_discharge = 10  #K
T_surrounding = 293.15  # K (20°C)
eta_isentropic = 0.75  # 60% isentropic efficiency of compressor
m_dot_wf = 0.0071  # kg/s
U = 0.3               # W/m2/K (typical for insulated tank)


cp_water = 4180             # J/kg·K (water)
cp_glycol_water = 3390      # J/kg·K (35% glycol-water)
cp_wf = 3190                # J/kg.K 


operation_time = 4*3600  # seconds   # 4 hours charging time
discharging_time = 4 * 3600   # 4 Hours discharging time



def charging_phase():
    # --- State calculations (cycle) ---
    
    sat_vap_p_low = flp.prop_pq(p_wf_low, 1, RP=RP, composition=composition)
    T_superheat = T_surrounding - dT_pinch
    comp_in = flp.tp(T_superheat, p_wf_low, RP=RP, composition=composition)
    sf_ch_in = flp.tp(T_surrounding, p_wf_low, RP=RP, composition=composition)

    isentropic_out = flp.sp(comp_in[4], p_wf_high, RP=RP, composition=composition)
    h_actual = comp_in[2] + (isentropic_out[2] - comp_in[2]) / eta_isentropic
    comp_out = flp.hp(h_actual, p_wf_high, RP=RP, composition=composition)
    Delta_s_compressor = m_dot_wf * (comp_out[4] - isentropic_out[4]) / 1000

    sat_vap_p_high = flp.prop_pq(p_wf_high, 1, RP=RP, composition=composition)
    sat_liq_p_high = flp.prop_pq(p_wf_high, 0, RP=RP, composition=composition)
    throt_in = flp.tp(T_surrounding, p_wf_high, RP=RP, composition=composition)
    throt_out = flp.hp(throt_in[2], p_wf_low, RP=RP, composition=composition)
    


    # COP
    m_total = m_dot_wf * operation_time
    Q_cond = m_total * (comp_out[2] - throt_in[2])
    W_comp = m_total * (comp_out[2] - comp_in[2])
    COP = Q_cond / W_comp
    


    # Tank temperatures
    Temp_A = T_surrounding
    Temp_B = sat_liq_p_high[0] - dT_pinch
    Temp_hmid = sat_vap_p_high[0] - dT_pinch
    Temp_F = throt_out[0] + dT_pinch
    Temp_E = comp_in[0] + dT_pinch

    # Heat transfer rates
    Q_cond_1 = m_dot_wf * (comp_out[2] - sat_liq_p_high[2])
    Q_cond_2 = m_dot_wf * (sat_liq_p_high[2] - throt_in[2])
    Q_evap = m_dot_wf * (comp_in[2] - throt_out[2])
    Q_cond_total = m_dot_wf * (comp_out[2] - throt_in[2])
    
    
    Q_cond_total = m_dot_wf * (comp_out[2] - throt_in[2]) #[J/s]
    
    # Heat exchanger entropy generation (condenser and evaporator)
    T_hot_cond = comp_out[0]  # Compressor outlet temp [K]
    T_cold_cond = throt_in[0]  # Condenser outlet temp [K]
    Delta_s_hex_cond = (Q_cond_total * (1/T_cold_cond - 1/T_hot_cond)) / 1000  # kJ/s.K
    
    T_hot_evap = comp_in[0]   # Evaporator inlet temp [K]
    T_cold_evap = throt_out[0]  # Evaporator outlet temp [K]
    Delta_s_hex_evap = (Q_evap * (1/T_hot_evap - 1/T_cold_evap)) / 1000  # kJ/s.K
    
    Delta_s_hex_total = Delta_s_hex_cond + Delta_s_hex_evap  # kJ/K.s    
    
    

    # Mass flow rates (secondary)
    m_dot_water_1 = m_dot_wf * ((sat_vap_p_high[2] - sat_liq_p_high[2]) / (cp_water * (Temp_hmid - Temp_B)))
    m_water_1 = m_dot_water_1 * operation_time
    print(f"m_dot_water_1: {m_dot_water_1:.3f}kg/s")
    m_dot_water_2 = m_dot_wf * ((sat_liq_p_high[2] - throt_in[2]) / (cp_water * (Temp_B - Temp_A)))
    m_water_2 = m_dot_water_2 * operation_time
    print(f"mass of water in tank A and tank B combined: {m_water_2:.3f}kg")
    m_dot_glycol = m_dot_wf * abs((comp_in[2] - throt_out[2]) / (cp_glycol_water * (Temp_F - Temp_E)))
    m_glycol = m_dot_glycol * operation_time
    print(f"m_dot_glycol: {m_dot_glycol:.3f}kg/s")

    # Tank D temp
    Temp_D = Temp_B + (Q_cond_1 / (m_dot_water_1 * cp_water))
    
    
    
    
    
   
    # Temp_D_range = np.linspace(333.15, 415.15, 10)  # Sweep 60°C to 120°C
    # actual_COPs = []
    # carnot_COPs = []
    # delta_Ts = []

    # for Temp_D in Temp_D_range:
    #     # Run charging_phase with override
        

    #     m_total = m_dot_wf * operation_time

    #     Q_cond = m_total * (comp_out[2] - throt_in[2])
    #     W_comp = m_total * (comp_out[2] - comp_in[2])
    #     actual_COP = Q_cond / W_comp if W_comp != 0 else np.nan

    #     delta_T = Temp_D - T_surrounding
    #     carnot_COP = Temp_D / (Temp_D - T_surrounding) if (Temp_D - T_surrounding) != 0 else np.nan

    #     actual_COPs.append(actual_COP)
    #     carnot_COPs.append(carnot_COP)
    #     delta_Ts.append(delta_T)

    # plt.figure(figsize=(8, 8))
    # plt.plot(delta_Ts, actual_COPs, 'o-', label='Actual COP')
    # plt.plot(delta_Ts, carnot_COPs, 's--', label='Carnot COP')
    # plt.xlabel('Temperature Difference (K)')
    # plt.ylabel('COP')
    # plt.title('Actual COP vs Temperature Difference')
    # plt.legend()
    # plt.grid(True)
    # plt.show()






    return locals()


def preheating(m_water_1, cp_water, Temp_B, T_surrounding, m_dot_wf, comp_out, throt_in, preheating_time=14400):
    """
    Handles preheating of tank C:
      - Calculates required electric energy and power for electric heater.
      - Calculates water mass and mixing temperature for heat pump preheating.
    Returns a dictionary with results.
    """
    
    # Using electric heater to heat tank C to temp B (Preheating)
    Q_electric = m_water_1 * cp_water * (Temp_B - T_surrounding)  # Total energy required (Joules)
    P_electric = Q_electric / preheating_time  # Required electric power (Watts)
    print(f"mass of water used when electric heating: {m_water_1:.2f}kg")

    # Using only heat pump to heat up tank C. Closing valve for water to flow from A to B
    T_C = T_surrounding
    Temp_D_test = 371.15 # K  # Manually tested to achieve desired mixing temp 
       
    
    """ from here, heat pump method for preheating tank C """

    # Mass flow rate of water for initial heating
    m_dot_water_initial = m_dot_wf * ((comp_out[2] - throt_in[2]) / (cp_water * (Temp_D_test - T_C)))
    m_water_initial = m_dot_water_initial * preheating_time
    

    # Fractional division (mass_C & mass_D) 
    # mc = Amount of water left back in tank C
    # md = Amount of water went in tank D to get heated to temp_D_test
    Td = Temp_D_test
    Tc = T_surrounding
    T_final_desired = Temp_B
    
    # T_final(mc+md) = mcTc + mdTd --------------- (1)
    # T_final*mc - mc*Tc = mdTd - md*T_final
    # mc(T_final - Tc) = md(Td - T_final)
    # r = mc/md = (Td - T_final)/(T_final - Tc) ---------(2)
    
    # Step 1: Calculate the ratio r = mc/md
    r = (Td - T_final_desired) / (T_final_desired - Tc)
    # mc + md = m_total  ----------(3)
    
    # mc/m_total = Mc ; md/m_total = Md
    # where Mc and Md are % of mass C and mass D respectively wrt m_total
    
    # Mc + Md = 1  ----------- (4)
    
    
    # Now, Td = 371.15K (98°C) Iterated with assumptions
    
    # (4)---->     mc/m_total + md/m_total = 1
    # mc/m_total + mc/(r*m_total) = 1
    # r*mc + mc = r*m_total
    # mc(r+1) = r*m_total
    # mc = (r/r+1)*m_total
    # mc = (1/r+1)*m_total
    
    # Step 2: Calculate fractions Mc and Md
    Mc = r / (r + 1)      # Fraction of m_water_1 in tank C
    Md = 1 / (r + 1)      # Fraction of m_water_1 in tank D
    
    # # Step 3: Calculate actual masses
    # mass_C = Mc * m_water_initial  # Mass of water left in tank C
    # mass_D = Md * m_water_initial  # Mass of water returning from tank D
    
    mass_C = Mc * m_water_1  # Mass of water left in tank C
    mass_D = Md * m_water_1  # Mass of water returning from tank D
    
    
    # Thus, Td = 371.15K , Tc = 293.15K, T_final (desired) = 361.28K
    # r = (371.15 - 361.28) / (361.28 - 293.15)
    # r = 0.144
    
    # Then, mc = (0.144 / 1.144) * m_total = 0.126 * m_total
    # md = (1/1.144) * m_total = 0.874 * m_total
    # Here, m_total = m_water_initial
    
    
    # mass_C = 0.126 * m_water_initial  # Mass of water left in tank C
    # mass_D = 0.874 * m_water_initial  # Mass of water returning from tank D

    T_final = ((mass_C * T_C) + (mass_D * Temp_D_test)) / (mass_C + mass_D)
    print(f"mass of water used when preheating using Heat pump: {m_water_initial:.2f}kg")
    return {
        "Q_electric": Q_electric,
        "P_electric": P_electric,
        "preheating_time": preheating_time,
        "Temp_D_test": Temp_D_test,
        "m_dot_water_initial": m_dot_water_initial,
        "m_water_initial": m_water_initial,
        "mass_C": mass_C,
        "mass_D": mass_D,
        "T_final": T_final,
        "T_C": T_C
    }   
    

def plot_preheating_tankC_mixture(
    m_dot_water_1,
    T_surrounding,
    T_final,
    Temp_D_test,
    preheating_time,
    p_wf_high,
    RP,
    
):
    T_range = np.linspace(T_surrounding, Temp_D_test, 100)
    # Reference enthalpy at starting T, p_wf_high
    h_ref = flp.tp(T_surrounding, p_wf_high, RP=RP, composition=composition)[2]
    h_range = [flp.tp(T, p_wf_high, RP=RP, composition=composition)[2] for T in T_range]
    h_dot = m_dot_water_1 * (np.array(h_range) - h_ref) / 1000  # kW

    plt.figure(figsize=(8, 5))
    plt.plot(h_dot, T_range, 'b-', label='Preheating Path (Working Fluid Mixture)')

    # Mark start point
    plt.plot(h_dot[0], T_range[0], 'go', label=f'Tank Hot1 initial: {T_surrounding:.2f} K')
    plt.text(h_dot[0]+2.0, T_range[0] - 2, f'Tank Hot1 initial\n{T_surrounding:.2f} K', color='green', fontsize=12, ha='right')

    # Mark hot stream temp (Temp_D_test)
    plt.plot(h_dot[-1], T_range[-1], 'ro', label=f'Tank Hot2,chosen: {Temp_D_test:.2f} K')
    plt.text(h_dot[-1], T_range[-1] - 5, f'Tank Hot2,chosen\n{Temp_D_test:.2f} K', color='red', fontsize=12, ha='left')

    # Mark target temperature (T_final)
    h_dot_B = m_dot_water_1 * (
        flp.tp(T_final, p_wf_high, RP=RP, composition=composition)[2] - h_ref
    ) / 1000
    plt.plot(h_dot_B, T_final, 'mo', label=f'Tank Hot1,final (after mixing): {T_final:.2f} K')
    plt.text(h_dot_B, T_final - 9, f'Tank Hot1,final\n{T_final:.2f} K', color='magenta', fontsize=12, ha='left')

    plt.xlabel('Enthalpy Flow Rate $\dot{H}$ (kW)', fontsize=13)
    plt.ylabel('Temperature (K)', fontsize=13)
    plt.title('T–$\dot{H}$ Curve for Preheating Path (Working Fluid Mixture)', fontsize=15)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



# def plot_preheating_tankC(cp_water, m_dot_water_initial, T_surrounding, T_final, Temp_D_test, preheating_time):
#     # Temperatures from ambient to hot stream temp
#     T_range = np.linspace(T_surrounding, Temp_D_test, 100)

#     # Enthalpy flow rate for each T, assuming cp_water const
#     h_dot = m_dot_water_initial * cp_water * (T_range - T_surrounding) / 1000  # kW

#     plt.figure(figsize=(8, 5))
#     plt.plot(h_dot, T_range, 'b-', label='Preheating Path (Tank C)')

#     # Mark start point
#     plt.plot(h_dot[0], T_range[0], 'go', label=f'Start: {T_surrounding:.2f} K')
#     plt.text(h_dot[0]+0.8, T_range[0] - 2, f'Tank C Start\n{T_surrounding:.2f} K', color='green', fontsize=12, ha='right')

#     # Mark hot stream temp (Temp_D_test)
#     plt.plot(h_dot[-1], T_range[-1], 'ro', label=f'Tank D: {Temp_D_test:.2f} K')
#     plt.text(h_dot[-1], T_range[-1] + 4, f'Tank D\n{Temp_D_test:.2f} K', color='red', fontsize=12, ha='left')

#     # Mark target temperature (Temp_B)
#     # Find the enthalpy flow rate at Temp_B
#     h_dot_B = m_dot_water_initial * cp_water * (T_final - T_surrounding) / 1000
#     plt.plot(h_dot_B, T_final, 'mo', label=f'Target (Tank C after mixing): {T_final:.2f} K')
#     plt.text(h_dot_B, T_final -9, f'Target (Tank C)\n{T_final:.2f} K', color='magenta', fontsize=12, ha='left')

#     plt.xlabel('Enthalpy Flow Rate $\dot{H}$ (kW)')
#     plt.ylabel('Temperature (K)')
#     plt.title('T–$\dot{H}$ Curve for Preheating Tank C (Heat Pump)')
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# def plot_charging_phase_with_preheat(
#     cp_water, m_dot_water_initial, T_surrounding, T_final, Temp_D_test,
#     comp_in, comp_out, sat_vap_p_high, sat_liq_p_high, throt_in, throt_out, sat_vap_p_low
# ):
#     plt.figure(figsize=(10,7))

#     # Charging phase (HP cycle)
#     hp_states = [comp_in, comp_out, sat_vap_p_high, sat_liq_p_high, throt_in, throt_out, sat_vap_p_low, comp_in]
#     labels = ['comp_in', 'comp_out', 'sat_vap HP', 'sat_liq HP', 'throt_in', 'throt_out', 'sat_vap LP', 'comp_in']
#     m_dot_wf = 0.0071  # kg/s 
#     h_dot_hp = [s[2] * m_dot_wf / 1000 for s in hp_states] # kW
#     T_hp = [s[0] for s in hp_states]

#     plt.plot(h_dot_hp, T_hp, 'k-', lw=2, label='Charging Phase (HP Cycle)')
#     for h, T, lbl in zip(h_dot_hp, T_hp, labels):
#         plt.plot(h, T, 'ko')
#         plt.text(h, T+3, lbl, fontsize=9, color='black')

    
#     # Preheating curve for Tank C
#     T_range = np.linspace(T_surrounding, Temp_D_test, 100)
#     h_dot_preheat = m_dot_water_initial * cp_water * (T_range - T_surrounding) / 1000
#     plt.plot(h_dot_preheat, T_range, 'b-', label='Preheating Path (Tank C)')

#     # Mark start, hot stream, and target
#     plt.plot(h_dot_preheat[0], T_range[0], 'go', label=f'Start: {T_surrounding:.2f} K')
#     plt.text(h_dot_preheat[0], T_range[0]-7, f'Start\n{T_surrounding:.2f} K', color='green', fontsize=10)
#     plt.plot(h_dot_preheat[-1], T_range[-1], 'ro', label=f'Tank D: {Temp_D_test:.2f} K')
#     plt.text(h_dot_preheat[-1], T_range[-1]+5, f'Tank D\n{Temp_D_test:.2f} K', color='red', fontsize=10)
#     h_dot_B = m_dot_water_initial * cp_water * (T_final - T_surrounding) / 1000
#     plt.plot(h_dot_B, T_final, 'mo', label=f'Target (Tank C): {T_final:.2f} K')
#     plt.text(h_dot_B, T_final+5, f'Target\n{T_final:.2f} K', color='magenta', fontsize=10)

#     plt.xlabel('Enthalpy Flow Rate $\dot{H}$ (kW)')
#     plt.ylabel('Temperature (K)')
#     plt.title('Charging Phase & Preheating Curve for Tank C')
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()





def standstill_after_charging(chg):
    # Area, U values, time constants, standstill temps/losses
    rho_water = 1000 #kg/m^3
    vol_watertank_1 = chg["m_dot_water_1"] * operation_time / rho_water
    vol_watertank_2 = chg["m_dot_water_2"] * operation_time / rho_water
    rho_glycol_water = 1039 # kg/m^3
    vol_glycolwatertank = chg["m_dot_glycol"] * operation_time / rho_glycol_water

    dia_A_B = ((2 * vol_watertank_2) / np.pi) ** (1/3)
    dia_C_D = ((2 * vol_watertank_1) / np.pi) ** (1/3)
    dia_E_F = ((2 * vol_glycolwatertank) / np.pi) ** (1/3)
    Height_A_B, Height_C_D, Height_E_F = 2 * dia_A_B, 2 * dia_C_D, 2 * dia_E_F
 
    side_area_A_B = np.pi * dia_A_B * Height_A_B
    top_bottom_area_A_B = 2 * np.pi * (dia_A_B / 2)** 2 
    total_surface_area_A_B = side_area_A_B + top_bottom_area_A_B
    side_area_C_D = np.pi * dia_C_D * Height_C_D
    top_bottom_area_C_D = 2 * np.pi * (dia_C_D / 2)** 2 
    total_surface_area_C_D = side_area_C_D + top_bottom_area_C_D
    side_area_E_F = np.pi * dia_E_F * Height_E_F
    top_bottom_area_E_F = 2 * np.pi * (dia_E_F / 2)** 2 
    total_surface_area_E_F = side_area_E_F + top_bottom_area_E_F

    ins_thickness = 0.03
    lambda_cond =  0.036
    R_ins = ins_thickness / lambda_cond
    h = 5
    R_conv = 1 / h
    R_total =R_ins + R_conv
    U_overall = 1 / R_total

    tau_B = (chg["m_water_2"] * cp_water) / (U_overall * total_surface_area_A_B)
    tau_D = (chg["m_water_1"] * cp_water) / (U_overall * total_surface_area_C_D)
    tau_F = (chg["m_glycol"] * cp_glycol_water) / (U_overall * total_surface_area_E_F)

    T_infinity = T_surrounding
    T_i_tank_B = chg["Temp_B"]
    T_i_tank_D = chg["Temp_D"]
    T_i_tank_F = chg["Temp_F"]
    T_i_tank_C = chg["Temp_B"]
    T_i_tank_E = chg["Temp_E"]

    total_standstill_time = 12 * 3600
    Temp_at_t_tankB = T_infinity + (T_i_tank_B - T_infinity) * (np.e) ** -(total_standstill_time / tau_B)
    Temp_at_t_tankD = T_infinity + (T_i_tank_D - T_infinity) * (np.e) ** -(total_standstill_time / tau_D)
    Temp_at_t_tankF = T_infinity + (T_i_tank_F - T_infinity) * (np.e) ** -(total_standstill_time / tau_F)
    Temp_at_t_tankC = T_infinity + (T_i_tank_C - T_infinity) * (np.e) ** -(total_standstill_time / tau_D)
    Temp_at_t_tankE = T_infinity + (T_i_tank_E - T_infinity) * (np.e) ** -(total_standstill_time / tau_F)

    heat_loss_in_B = chg["m_water_2"] * cp_water * (Temp_at_t_tankB - T_i_tank_B)
    heat_loss_in_D = chg["m_water_1"] * cp_water * (Temp_at_t_tankD - T_i_tank_D)
    heat_gain_in_F = chg["m_glycol"] * cp_glycol_water * (Temp_at_t_tankF - T_i_tank_F)

    return locals()

def cool_down_time(m, cp, U, A, T_initial, T_threshold, T_ambient):  # Time needed to reheat tank C.
    # m: kg, cp: J/kg/K, U: W/m2/K, A: m2, T_initial/T_final/T_ambient: K
    T_threshold = 350
    t = - (m * cp) / (U * A) * np.log((T_threshold - T_ambient)/(T_initial - T_ambient))

    time_hr = t /3600
    
    print(f"Time to cool from {T_initial:.2f}K to {T_threshold:.2f}K: {time_hr:.2f} hours")

    return t  # seconds


def discharging_phase(chg, stand):
    # --- ORC states ---
    p_orc_low = 3.5e5
    p_orc_high = 16.0e5
    eta_pump = 0.70
    eta_turbine = 0.80

    orc_state1 = flp.prop_pq(p_orc_low, 0, RP=RP, composition=composition) # before pumping state
    orc_isentropic_pump = flp.sp(orc_state1[4], p_orc_high, RP=RP, composition=composition)
    h_pump_actual = orc_state1[2] + (orc_isentropic_pump[2] - orc_state1[2]) / eta_pump
    orc_state2 = flp.hp(h_pump_actual, p_orc_high, RP=RP, composition=composition) # after pumping state
    Delta_s_pump = m_dot_wf * (orc_state2[4] - orc_isentropic_pump[4]) / 1000
    orc_state3 = flp.prop_pq(p_orc_high,0 , RP=RP, composition=composition) # saturated liquid at  p_orc_high
    orc_state4 = flp.prop_pq(p_orc_high,1 , RP=RP, composition=composition) # saturated vapor at p_orc_high
    T_superheat_discharge = chg["Temp_D"] - dT_pinch
    # orc_state5 = flp.tp(T_superheat_discharge, p_orc_high , RP=RP, composition=composition)
    orc_state5 = flp.prop_pq(p_orc_high,1 , RP=RP, composition=composition) # turbine expansion happens after saturated vapor at p_orc_high
    ideal_state6 = flp.sp(orc_state5[4], p_orc_low, RP=RP, composition=composition)
    h_turb_actual = orc_state5[2] - eta_turbine * (orc_state5[2] - ideal_state6[2])
    orc_state6 = flp.hp(h_turb_actual, p_orc_low, RP=RP, composition=composition) # after turbine expansion
    Delta_s_turbine = m_dot_wf * (orc_state6[4] - ideal_state6[4]) / 1000
    orc_state7 = flp.prop_pq(p_orc_low, 1, RP=RP, composition=composition) # saturated vapor at  p_orc_low
    orc_state8 = flp.prop_pq(p_orc_low, 0, RP=RP, composition=composition) # saturated liquid at p_orc_low

    # --- Discharge secondary calculations ---
    Q_dot_orc_1_evap = m_dot_wf * (orc_state5[2] - orc_state3[2])
    Q_dot_orc_2_evap = m_dot_wf * (orc_state3[2] - orc_state2[2])
    Q_dot_orc_cond = m_dot_wf * (orc_state7[2] - orc_state8[2])
    Q_total_JI = Q_dot_orc_1_evap * discharging_time # --(2)
    Q_total_HG = Q_dot_orc_2_evap * discharging_time # --(1)
    Q_total_KL = Q_dot_orc_cond * discharging_time
    
    

    delta_T_JI = Q_total_JI / (chg["m_water_1"] * cp_water)
    # delta_T_HG = Q_total_HG / (chg["m_water_2"] * cp_water)
    delta_T_KL = Q_total_KL / (chg["m_glycol"] * cp_glycol_water)

    Temp_j = stand["Temp_at_t_tankD"]
    Temp_h = stand["Temp_at_t_tankB"]
    Temp_l = stand["Temp_at_t_tankF"]
    Temp_gmid = orc_state2[0] + dT_pinch
    
    m_water_disch_hg = Q_total_HG / (cp_water * (Temp_h - Temp_gmid)) # needed mass for discharge in hg tank
    
    delta_T_HG = Q_total_HG / (m_water_disch_hg * cp_water)

    Temp_i = stand["Temp_at_t_tankD"] - delta_T_JI
    Temp_g = stand["Temp_at_t_tankB"] - delta_T_HG
    Temp_k = stand["Temp_at_t_tankF"] + delta_T_KL

    m_dot_water_discharge_JI = Q_dot_orc_1_evap / (cp_water * (Temp_j - Temp_i))
    m_dot_water_discharge_HG = Q_dot_orc_2_evap / (cp_water * (Temp_h - Temp_gmid))
    #m_dot_water_discharge_HG = Q_dot_orc_2_evap / (cp_water * (Temp_h - Temp_g))
    m_dot_glycol_discharge_KL = Q_dot_orc_cond / (cp_glycol_water * (Temp_k - Temp_l))
    
    print(f"m_dot_water_discharge_HG: {m_dot_water_discharge_HG:.3f}kg/s")
    

    return locals()

def standstill_after_discharging(chg, stand, disch):
    total_standstill_time_disch = 4 * 3600
    U_overall = stand["U_overall"]
    total_surface_area_A_B = stand["total_surface_area_A_B"]
    total_surface_area_C_D = stand["total_surface_area_C_D"]
    total_surface_area_E_F = stand["total_surface_area_E_F"]

    U_overall_G = U_overall * 25
    tau_G = (disch["m_dot_water_discharge_HG"] * total_standstill_time_disch * cp_water) / (U_overall * total_surface_area_A_B)
    T_i_tank_G = disch["Temp_g"]
    T_infinity = T_surrounding
    Temp_at_t_tankG = T_infinity + ((T_i_tank_G - T_infinity) * (np.e) ** -(total_standstill_time_disch / tau_G))
    heat_loss_in_G = disch["m_dot_water_discharge_HG"] * cp_water * (Temp_at_t_tankG - T_i_tank_G)

    U_overall_I = U_overall * 3
    tau_I = (chg["m_water_1"] * cp_water) / (U_overall_I * total_surface_area_C_D)
    T_i_tank_I = disch["Temp_i"]
    Temp_at_t_tankI = T_infinity + (T_i_tank_I - T_infinity) * (np.e) ** -(total_standstill_time_disch / tau_I)
    heat_loss_in_I = disch["m_dot_water_discharge_JI"] * cp_water * (Temp_at_t_tankI - T_i_tank_I)

    U_overall_K = U_overall * 50
    tau_K = (chg["m_glycol"] * cp_glycol_water) / (U_overall_K * total_surface_area_E_F)
    T_i_tank_K = disch["Temp_k"]
    Temp_at_t_tankK = T_infinity + (T_i_tank_K - T_infinity) * (np.e) ** -(total_standstill_time_disch / tau_K)
    heat_loss_in_K = disch["m_dot_glycol_discharge_KL"] * cp_glycol_water * (Temp_at_t_tankK - T_i_tank_K)

    return locals()


def energy_and_performance(chg, stand, disch,preheat):
 
  
    # Calculate total entropy generation and performance impact
    total_entropy_generation = chg["Delta_s_compressor"] + disch["Delta_s_turbine"] + disch["Delta_s_pump"] + chg["Delta_s_hex_total"]  # Total per kg [kJ/s·K]
    # S_gen_rate = m_dot_wf * total_entropy_generation  # Entropy generation rate [kW/K]
    T_dead_state = T_surrounding  # Dead state temperature [K]
    Ex_destroyed = T_dead_state * total_entropy_generation  # Exergy destruction rate [kW]

    # Compressor power (charging phase)
    W_comp_power = m_dot_wf * (chg["comp_out"][2] - chg["comp_in"][2])  # [W]

    W_turb_power = m_dot_wf * (disch["orc_state5"][2] - disch["orc_state6"][2])  # [W]

    # Pump power (discharging phase)
    W_pump_power = m_dot_wf * (disch["orc_state2"][2] - disch["orc_state1"][2])   # [W] 

    # Net power
    W_net_power = abs(W_turb_power - W_comp_power)  # [kW]




    # Exergy efficiency 
    #eta_exergy = W_net_power / (W_net_power + Ex_destroyed) * 100  # [%]

    # Charging phase work (energy stored)
    W_charging = chg["W_comp"] / (1000*3600)  # Convert to kWh
    Q_heat_stored = chg["Q_cond"] / 3600000  # Convert to kWh

    # Discharging phase work (energy recovered)  
    W_turb_power_kW = W_turb_power / 1000  # Convert to kW
    t_hours = discharging_time / 3600      # Convert time to hours
    W_turbine_total = W_turb_power * discharging_time  # J
    E_el_out = (W_turb_power - W_pump_power) * discharging_time   # [J]
    W_pump_total = (W_pump_power) * (discharging_time)     # J
    # W_net_discharge = W_turbine_total - W_pump_total          # kWh
    W_comp_power_kW = W_comp_power / 1000 #[kW]
    W_comp_total = W_comp_power * operation_time # J

    # TRUE Round-trip efficiency
    #eta_roundtrip = (W_net_discharge / W_charging) * 100  # [%]
    
    Q_electric = preheat["Q_electric"]  # Total electrical energy for heating [J]
    energy_elect_heater = W_turbine_total / (Q_electric + W_comp_total + W_pump_total) # Energetic efficiency using electric heater
    
    energy_HP = W_turbine_total / (W_comp_total + W_pump_total) # Energetic efficiency using Heat pump
    


    """
    Comprehensive exergy destruction analysis for electric heater vs heat pump
    """

    
    # =========================
    # ELECTRIC HEATER ANALYSIS  
    # =========================
    
    # # Electric heater energy requirements
    # Q_electric = preheat["Q_electric"]  # Total electrical energy for heating [J]
    # heating_time = preheat["heating_time"]  # 10800 s (3 hours)
    
    # Energy input (electricity has 100% exergy content)
    En_in_electric = Q_electric  # [J] - All electrical energy is energy
    
    # Energy of heat delivered at target temperature
    T_hot_desired = chg["Temp_B"]  # 361.28 K (88°C)
    En_heat_delivered_electric = Q_electric * (1 - T_dead_state / T_hot_desired)
    #En_heat_delivered_electric = Q_electric * (1 - 293.15 / 361.15)
    #En_heat_delivered_electric = Q_electric * 0.1883  # [J]
    
    # Energy destruction in electric heating process
    # En_destroyed_electric_heating = En_in_electric - En_heat_delivered_electric
    # En_destroyed_electric_heating = Q_electric * (1 - 0.1883)
    # En_destroyed_electric_heating = Q_electric * 0.8117  # [J]
    
    preheating_time = preheat["preheating_time"]
    # Electric heating energy destruction rate
    # En_destruction_rate_electric = En_destroyed_electric_heating / preheating_time  # [W]
    
    # =========================
    # HEAT PUMP ANALYSIS
    # =========================
    
    # Heat pump work and heat transfer
    W_comp_total = chg["W_comp"]  # Total compressor work [J]
    Q_cond_total = chg["Q_cond"]  # Total heat delivered [J]
    COP = chg["COP"]  # Coefficient of performance
    
    # Energy inputs to heat pump
    En_in_compressor = W_comp_total  # [J] - Compressor electrical work (100% energy)
    En_in_ambient_heat = 0  # [J] - Ambient heat has zero energy at dead state
    En_in_total_HP = En_in_compressor + En_in_ambient_heat  # [J]
    
    # Energy of heat delivered by heat pump
    En_heat_delivered_HP = Q_cond_total * (1 - T_dead_state / T_hot_desired)
    # En_heat_delivered_HP = Q_cond_total * 0.1883  # [J]
    
    # Energy destruction in heat pump cycle
    # En_destroyed_HP_total = En_in_total_HP - En_heat_delivered_HP
    # En_destroyed_HP_total = W_comp_total - (Q_cond_total * 0.1883)  # [J]
    
    # Heat pump energy destruction rate
    # En_destruction_rate_HP = En_destroyed_HP_total / operation_time  # [W]
    
#     # Component-wise exergy destruction in heat pump
#     # Compressor exergy destruction (already calculated in charging phase)
#     Ex_destroyed_compressor = T_dead_state * chg["Delta_s_compressor"] * operation_time  # [J]
    
#     # Heat exchanger exergy destruction (condenser + evaporator)
#     Ex_destroyed_heat_exchangers = T_dead_state * chg["Delta_s_hex_total"] * operation_time  # [J]
    
#     # Throttling valve exergy destruction
#     # For throttling: Ex_destroyed = h_in - h_out (at constant pressure, ideal gas)
#     # Approximation: Ex_destroyed ≈ m * T_dead_state * Δs_throttling
#     h_before_throttle = chg["throt_in"][2]  # [J/kg]
#     h_after_throttle = chg["throt_out"][2]  # [J/kg]
#     m_total = m_dot_wf * operation_time  # [kg]
    
#     # Throttling entropy generation (isenthalpic expansion)
#     s_before_throttle = chg["throt_in"][4]  # [J/kg⋅K]
#     s_after_throttle = chg["throt_out"][4]  # [J/kg⋅K]
#     Delta_s_throttling = (s_after_throttle - s_before_throttle) / 1000  # [kJ/kg⋅K]
#     Ex_destroyed_throttling = T_dead_state * m_dot_wf * Delta_s_throttling * operation_time  # [J]
    
#     # =========================
#     # DISCHARGING PHASE ANALYSIS
#     # =========================
    
#     # Work output from turbine and pump
#     W_turbine_total = m_dot_wf * (disch["orc_state5"][2] - disch["orc_state6"][2]) * discharging_time  # [J]
#     W_pump_total = m_dot_wf * (disch["orc_state2"][2] - disch["orc_state1"][2]) * discharging_time  # [J]
    W_net_output = W_turbine_total - W_pump_total  # [J]
    
#     # Exergy destruction in discharging components
#     Ex_destroyed_turbine = T_dead_state * disch["Delta_s_turbine"] * discharging_time  # [J]
#     Ex_destroyed_pump = T_dead_state * disch["Delta_s_pump"] * discharging_time  # [J]
    
#     # =========================
#     # OVERALL SYSTEM EXERGY EFFICIENCIES
#     # =========================
    
    # ELECTRIC HEATER SYSTEM
    # Total energy input = Electric heating + Compressor work (for discharge)
    Total_En_in_electric_system = Q_electric + W_comp_total  # [J]
    
    # Useful energy output = Net electrical work from discharge
    Useful_En_out = W_net_output  # [J]
    
    # Electric heater system energy efficiency
    eta_energy_electric_system = (Useful_En_out / Total_En_in_electric_system) * 100  # [%]
    
    # HEAT PUMP SYSTEM  
    # Total energy input = Compressor work (charging) + Energy_hp_normal (Q_electric/COP)
    P_electric = preheat["P_electric"]
    # E_elec_hp = preheat["m_water_initial"] * cp_water * (chg["Temp_B"] - T_surrounding) # Electric work required by Heat pump
    
    E_elec_hp = chg["m_water_1"] * cp_water * (chg["Temp_B"] - T_surrounding) # Electric work required by Heat pump
    En_hp_prep = (E_elec_hp) / COP
    Total_En_in_HP_system = W_comp_total + En_hp_prep   # [J]
    
    # Heat pump system energy efficiency
    eta_energy_HP_system = (Useful_En_out / Total_En_in_HP_system) * 100  # [%]
    
    efficiency_improvement = eta_energy_HP_system / eta_energy_electric_system
    print(f"Compressor work: {W_comp_total/1e6:.3f}MJ")
    print(f"Electric work using electric heater: {Q_electric/1e6:.3f}MJ")
    print(f"Energy required to preheat using HP: {En_hp_prep/1e6:.3f}MJ")
    print(f"Normal energy consumption in heat pump process: {W_comp_total/1e6:.3f}MJ")
    print(f"Ratio of prep/normal: {En_hp_prep / W_comp_total}")
    print(f"Ratio of prep/normal for electric heater: {Q_electric / W_comp_total}")
    print(f"Energy delivered by HP: {En_heat_delivered_HP/1e6:.3f}MJ")
    # Number of cycles (e.g., from 1 to 1000)
    N = np.arange(1, 30)
    
    # Electric heater system
    E_out_electric = Useful_En_out
    E_in_normal_electric = W_comp_total
    E_prep_electric = Q_electric

    # Heat pump system
    E_out_hp = Useful_En_out
    E_in_normal_hp = W_comp_total
    E_prep_hp = En_hp_prep
    
    def round_trip_efficiency(N, E_out, E_in, E_prep):
        return (N * E_out) / ((N * E_in) + E_prep)

    eff_electric = round_trip_efficiency(N, E_out_electric, E_in_normal_electric, E_prep_electric)
    eff_hp = round_trip_efficiency(N, E_out_hp, E_in_normal_hp, E_prep_hp)

    # Per-cycle efficiency limit lines (as %)
    per_cycle_electric = (E_out_electric / (E_in_normal_electric + E_prep_electric)) * 100 if E_in_normal_electric != 0 else 0
    per_cycle_hp = (E_out_hp / (E_in_normal_hp + E_prep_hp)) * 100 if E_in_normal_hp != 0 else 0
    
    
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(N, eff_electric * 100, label='Electric Heater', color='tab:red')
    plt.plot(N, eff_hp * 100, label='Heat Pump', color='tab:blue')
    plt.axhline(y=per_cycle_electric, color='tab:red', linestyle='--', label='Per-cycle: Electric')
    plt.axhline(y=per_cycle_hp, color='tab:blue', linestyle='--', label='Per-cycle: Heat Pump')
    plt.xlabel('Number of Cycles')
    plt.ylabel('Overall Round-trip Efficiency (%)')
    plt.title('E_out / E_in vs Number of Cycles (Including Preheating)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    

    sf_dis_in = flp.tp(stand["Temp_at_t_tankK"], 3.5e5, RP=RP, composition=composition)  # p_orc_low = 3.5e5

    s_actual = disch["orc_state6"][4] / 1000  # kJ/kg·K
    s_ideal_at_same_enthalpy = sf_dis_in[4] / 1000  # kJ/kg·K
    excess_entropy_per_kg = abs(s_actual - s_ideal_at_same_enthalpy)  # kJ/kg·K

    excess_entropy_rate = m_dot_wf * excess_entropy_per_kg  # kW/K
    Q_excess_entropy_process = (m_dot_wf * (disch["orc_state6"][2] - sf_dis_in[2])) / 1000  # kW

    # Avoid divide-by-zero error
    if disch["Q_dot_orc_1_evap"] != 0:
        excess_entropy_process_time = (Q_excess_entropy_process / disch["Q_dot_orc_1_evap"]) * discharging_time  # seconds
    else:
        excess_entropy_process_time = 0

    total_excess_entropy = excess_entropy_rate * excess_entropy_process_time  # kJ/K


    # print(f"s_actual = {s_actual:.6f} kJ/kg·K")
    # print(f"s_ideal_at_same_enthalpy = {s_ideal_at_same_enthalpy:.6f} kJ/kg·K")
    # print(f"excess_entropy_per_kg = {excess_entropy_per_kg:.6f} kJ/kg·K")
    # print(f"excess_entropy_rate = {excess_entropy_rate:.6f} kW/K")
    # print(f"Q_excess_entropy_process = {Q_excess_entropy_process:.6f} kW")
    # print(f"excess_entropy_process_time = {excess_entropy_process_time:.6f} s")
    # print(f"total_excess_entropy = {total_excess_entropy:.6f} kJ/K")
    
    
    
    return locals()






def plot_carnot_battery_cycle(
    flp, m_dot_wf, RP, composition,
    #n All state points and temperatures from charging/discharging
    comp_in, comp_out, sat_vap_p_high, sat_liq_p_high, throt_in, throt_out, sat_vap_p_low,
    COP, Temp_A, Temp_B, Temp_D, Temp_E, Temp_F, Temp_hmid,
    m_dot_water_1, operation_time, vol_watertank_1, m_dot_water_2, vol_watertank_2,
    m_dot_glycol, vol_glycolwatertank, orc_state1, orc_state2, orc_state3, orc_state4,
    orc_state5, orc_state6, orc_state7, orc_state8,
    Temp_at_t_tankB, Temp_at_t_tankD, Temp_at_t_tankF,
    Temp_g, Temp_h, Temp_i, Temp_j, Temp_k, Temp_l,p_orc_high,sf_ch_in,Temp_at_t_tankG,
    Temp_at_t_tankK,p_orc_low,Q_dot_orc_1_evap,Q_cond,Q_evap,total_entropy_generation,Ex_destroyed,
    Temp_at_t_tankI,eta_pump,eta_turbine,dia_A_B,dia_C_D,dia_E_F,U_overall,total_surface_area_A_B,
    P_electric,energy_elect_heater,Temp_D_test,T_final,m_water_2,T_dead_state,
    Delta_s_hex_total,Delta_s_pump, Delta_s_turbine,Delta_s_compressor,heat_loss_in_B,heat_loss_in_D,heat_gain_in_F,
    heat_loss_in_G,heat_loss_in_I,heat_loss_in_K,eta_isentropic,energy_HP,T_hot_desired,En_heat_delivered_electric,
    En_in_compressor,En_in_ambient_heat,En_in_total_HP,
    En_heat_delivered_HP,W_net_output,Total_En_in_electric_system,
    Useful_En_out,eta_energy_electric_system,Total_En_in_HP_system,eta_energy_HP_system,
    
):
    """Plots the full T-Ḣ diagram and all cycle/secondary fluid curves."""
    
    def plot_nonideal_curve_hdot(p, h_start, h_end, n_points=50, color='r', label=None, linewidth=2):
        """Plots non-linear mixture process between two enthalpy flow rates"""
        qualities = np.linspace(0, 1, n_points)
        hdot_points, T_points = [], []
        
        for x in qualities:
            try:
                state = flp.prop_pq(p, x, RP=RP, composition=composition)
                hdot = state[2] * m_dot_wf / 1000  # Convert to kJ/s
                T = state[0]
                if min(h_start, h_end) <= hdot <= max(h_start, h_end):
                    hdot_points.append(hdot)
                    T_points.append(T)
            except:
                continue
        
        if hdot_points:
            plt.plot(hdot_points, T_points, color=color, linewidth=linewidth, label=label)
            
    # ====================== MAIN PLOT ======================
    
    plt.figure(figsize=(14, 8))
    
    # 1. Saturation dome (1 bar to 80.6 bar)
    p_vals = np.linspace(1e5, 80.6e5, 150)
    hdot_bubble, T_bubble = [], []
    hdot_dew, T_dew = [], []
    
    for p in p_vals:
        try:
            # Bubble line
            bubble = flp.prop_pq(p, 0, RP=RP, composition=composition)
            hdot_bubble.append(bubble[2] * m_dot_wf / 1000)  # Convert to kJ/s
            T_bubble.append(bubble[0])
            
            # Dew line
            dew = flp.prop_pq(p, 1, RP=RP, composition=composition)
            hdot_dew.append(dew[2] * m_dot_wf / 1000)  # Convert to kJ/s
            T_dew.append(dew[0])
        except:
            continue
    
    plt.plot(hdot_bubble, T_bubble, color='black', linestyle='-', linewidth=0.8, label='Bubble Line (x=0)')
    plt.plot(hdot_dew, T_dew, color='black', linestyle='-', linewidth=0.8, label='Dew Line (x=1)')
    
    # 2. Thermodynamic cycle - Convert to H_dot
    cycle_points = [
        comp_in, comp_out, sat_vap_p_high, sat_liq_p_high,
        throt_in, throt_out, sat_vap_p_low, comp_in
    ]
    hdot_cycle = [p[2] * m_dot_wf / 1000 for p in cycle_points]  # Convert to kJ/s
    T_cycle = [p[0] for p in cycle_points]
    
    
    
    # Single-phase connections
    plt.plot(hdot_cycle[:2], T_cycle[:2], 'g-', linewidth=2, label='Carnot Battery Cycle')  # 1→2 Compression
    plt.plot([hdot_cycle[1], hdot_cycle[2]], [T_cycle[1], T_cycle[2]], 'g-', linewidth=2)  # 2→3
    plot_nonideal_curve_hdot(p_wf_high, hdot_cycle[2], hdot_cycle[3], color='g')   # 3→4
    plt.plot([hdot_cycle[3], hdot_cycle[4]], [T_cycle[3], T_cycle[4]], 'g-', linewidth=2)  # 4→5
    
    # Throttling process (5→6) - Non-ideal isenthalpic
    plot_nonideal_curve_hdot(p_wf_low, hdot_cycle[4], hdot_cycle[5], color='g')
    plt.plot([hdot_cycle[4], hdot_cycle[5]], [T_cycle[4], T_cycle[5]], 'g-', linewidth=2)  # 5→6 Throttling
    
    # ====================== MODIFIED EVAPORATION PLOTTING ====================== 
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
    
    
    # State annotations
    labels = ['1: Comp In', '2: Comp Out', '3: Sat Vap HP', '4: Sat Liq HP',
              '5: Subcooled', '6: Throttle Out', '7: Sat Vap LP']
    for i, (hdot, T, label) in enumerate(zip(hdot_cycle[:-1], T_cycle[:-1], labels)):
        plt.plot(hdot, T, 'go', markersize=8)
        #plt.annotate(label, (hdot, T), xytext=(7,7), textcoords='offset points',
                    #ha='left', fontsize=9)
        
        if label == '1: Comp In':
            plt.annotate(label, (hdot, T), xytext=(7, -12), textcoords='offset points', ha='left', fontsize=9)
            
        elif label == '6: Throttle Out':
            plt.annotate(label, (hdot, T), xytext=(7, -15), textcoords='offset points', ha='left', fontsize=9)
        
        elif label == '7: Sat Vap LP':
            plt.annotate(label, (hdot, T), xytext=(7, -15), textcoords='offset points', ha='left', fontsize=9)
        else:
            plt.annotate(label, (hdot, T), xytext=(7,7), textcoords='offset points', ha='left', fontsize=9)
        
        
        
    # =================== Plotting Discharging phase ================= #
    # 3. DISCHARGING PHASE (ORC) - Convert to H_dot
    orc_points = [orc_state1, orc_state2, orc_state3, orc_state4, orc_state5, orc_state6,orc_state7, orc_state8]
    hdot_orc = [p[2] * m_dot_wf / 1000 for p in orc_points]  # Convert to kJ/s
    T_orc = [p[0] for p in orc_points]
    
    # Plot ORC cycle 
    plt.plot(hdot_orc[:2], T_orc[:2], 'brown', linewidth=4, label='Discharging Phase (ORC)')  # 1→2 Pump compression
    # Plot heating in first hot tank (2→3) - Liquid heating
    plt.plot(                                                       # 2→3
        [hdot_orc[1], hdot_orc[2]], 
        [T_orc[1], T_orc[2]], 
        'brown', 
        linewidth=4
    )
    
    plot_nonideal_curve_hdot(p_orc_high, hdot_orc[2], hdot_orc[3], color='brown', linewidth=4)      # 3→4 
    plt.plot(                                                       # 4→5
        [hdot_orc[3], hdot_orc[4]], 
        [T_orc[3], T_orc[4]], 
        'brown', 
        linewidth=4
    )
    plt.plot(                                                       # 5→6
        [hdot_orc[4], hdot_orc[5]], 
        [T_orc[4], T_orc[5]], 
        'brown', 
        linewidth=4
    )
    plt.plot(                                                       # 6→7
        [hdot_orc[5], hdot_orc[6]], 
        [T_orc[5], T_orc[6]], 
        'brown', 
        linewidth=4
    )
    plt.plot(                                                       # 7→8
        [hdot_orc[6], hdot_orc[7]], 
        [T_orc[6], T_orc[7]], 
        'brown', 
        linewidth=4
    )
    plt.plot(                                                       # 8→1
        [hdot_orc[7], hdot_orc[0]], 
        [T_orc[7], T_orc[0]], 
        'brown', 
        linewidth=4
    )
    
    hdot_cond_1 = np.linspace(comp_out[2]*m_dot_wf/1000, sat_liq_p_high[2]*m_dot_wf/1000, 50)
    T_cond_1 = np.linspace(Temp_D, Temp_B, 50)    # T_hot_out_1 and T_hot_in
    
    sf_ch_h1 = flp.tp(Temp_A, p_wf_high, RP=RP, composition=composition)
    sf_ch_h2 = flp.tp(Temp_B, p_wf_high, RP=RP, composition=composition)
    # hdot_cond_2 = np.linspace(sat_liq_p_high[2]*m_dot_wf/1000, throt_in[2]*m_dot_wf/1000, 25)
    hdot_cond_2 = np.linspace(sf_ch_h2[2]*m_dot_wf/1000, sf_ch_h1[2]*m_dot_wf/1000, 25)
    T_cond_2 = np.linspace(Temp_B, Temp_A, 25)
    
    
    hdot_hot = np.concatenate([hdot_cond_1, hdot_cond_2])
    T_hot = np.concatenate([T_cond_1, T_cond_2])
    
    
    plt.plot(hdot_hot, T_hot, '-', color='red', label=f'HP-Hot storage(Condensation): {m_dot_water_1*operation_time:.2f} kg/h')
    plt.plot(hdot_hot, T_hot, '-', color='red', label=f'HP-Hot storage(Subcooled): {m_dot_water_2*operation_time:.2f} kg/h')
    plt.plot(hdot_hot[0], T_hot[0], 'o', color='red')
    plt.text(hdot_cond_2[0] + -0.08, T_cond_2[0], 'Temp B', ha='right', va='bottom', fontsize=9, color='black')
    plt.plot(hdot_hot[-1], T_hot[-1], 'o', color='red')
    plt.text(hdot_cond_2[-1], T_cond_2[-1], 'Temp A', ha='right', va='bottom', fontsize=9, color='black')
    plt.text(hdot_cond_1[0] + 0.2, T_cond_1[0], 'Temp D', ha='right', va='bottom', fontsize=9, color='black')
    
    
    
    sf_ch_out = flp.tp(Temp_F, p_wf_low, RP=RP, composition = composition)
    hdot_evap = np.linspace(throt_out[2]*m_dot_wf/1000, sf_ch_in[2]*m_dot_wf/1000, 50)
    # hdot_evap = np.linspace(sf_ch_out[2]*m_dot_wf/1000, sf_ch_in[2]*m_dot_wf/1000, 50)
    T_evap = np.linspace(Temp_F, Temp_E, 50)
    plt.plot(hdot_evap, T_evap, '-', color='darkblue', label=f'HP-Cold storage(Evaporation): {m_dot_glycol*operation_time:.1f} kg/h')
    plt.plot(hdot_evap[0], T_evap[0], 'o', color='darkblue')
    plt.text(hdot_evap[0] + -0.035, T_evap[0] + -2.1, 'Temp F', ha='right', va='bottom', fontsize=9, color='black')
    plt.plot(hdot_evap[-1], T_evap[-1], 'o', color='darkblue')
    plt.text(hdot_evap[-1] + 0.3, T_evap[-1] + -1.5, 'Temp E', ha='right', va='bottom', fontsize=9, color='black')
    
    
    # ================= Plot secondary fluid curves (Discharging phase) ==================== #
    # Convert secondary fluid enthalpies to H_dot as well
    
    
    
    hdot_evap_orc = np.linspace(orc_state2[2]*m_dot_wf/1000, orc_state3[2]*m_dot_wf/1000, 50)
    T_evap_orc = np.linspace(Temp_at_t_tankG, Temp_h, 50) 
    
    plt.plot(hdot_evap_orc, T_evap_orc, '--', color='red', label='ORC Hot storage')
    plt.plot(hdot_evap_orc[0], T_evap_orc[0], 'o', color='red')
    plt.text(hdot_evap_orc[0]-0.05, T_evap_orc[0]-1.6, 'Temp g', ha='right', va='bottom', fontsize=9, color='black')
    plt.plot(hdot_evap_orc[-1], T_evap_orc[-1], 'o', color='red')
    plt.text(hdot_evap_orc[-1], T_evap_orc[-1], 'Temp h', ha='right', va='bottom', fontsize=9, color='black')
    
    
    
    
    hdot_evap_orc_2 = np.linspace(orc_state3[2]*m_dot_wf/1000, orc_state5[2]*m_dot_wf/1000, 50)
    T_evap_orc_2 = np.linspace(Temp_at_t_tankI , Temp_j, 50) 
    
    plt.plot(hdot_evap_orc_2, T_evap_orc_2, '--', color='red')
    plt.plot(hdot_evap_orc_2[0], T_evap_orc_2[0], 'o', color='red')
    plt.text(hdot_evap_orc_2[0] + -0.05, T_evap_orc_2[0], 'Temp i', ha='right', va='bottom', fontsize=9, color='black')
    plt.plot(hdot_evap_orc_2[-1], T_evap_orc_2[-1], 'o', color='red')
    plt.text(hdot_evap_orc_2[-1]+0.02, T_evap_orc_2[-1]-0.8, 'Temp j', ha='left', va='bottom', fontsize=9, color='black')
    
    
    
    
    sf_dis_in = flp.tp(Temp_at_t_tankK, p_orc_low, RP = RP, composition=composition)
    sf_dis_out = flp.tp(Temp_l, p_orc_low, RP = RP, composition=composition)
    # hdot_cond_orc = np.linspace(orc_state7[2]*m_dot_wf/1000, orc_state8[2]*m_dot_wf/1000, 50)
    hdot_cond_orc = np.linspace(sf_dis_in[2]*m_dot_wf/1000, sf_dis_out[2]*m_dot_wf/1000, 50)
    T_cond_orc = np.linspace(Temp_at_t_tankK , Temp_l, 50)
    
    plt.plot(hdot_cond_orc, T_cond_orc, '--', color='darkblue', label='ORC Cold storage')
    plt.plot(hdot_cond_orc[0], T_cond_orc[0], 'o', color='darkblue')
    plt.text(hdot_cond_orc[0] -0.1, T_cond_orc[0]+3.0, 'Temp k', ha='left', va='bottom', fontsize=9, color='black')
    plt.plot(hdot_cond_orc[-1], T_cond_orc[-1], 'o', color='darkblue')
    plt.text(hdot_cond_orc[-1], T_cond_orc[-1]+0.06, 'Temp l', ha='left', va='bottom', fontsize=9, color='black')
    
    
    
    # BLACK DOTTED LINE - Controlled Excess Entropy
    # Shows the entropy control mechanism from ideal turbine expansion to actual
    ideal_turb_hdot = sf_dis_in[2] * m_dot_wf / 1000  # Ideal turbine outlet
    actual_turb_hdot = orc_state6[2] * m_dot_wf / 1000   # Your actual turbine outlet
    ideal_turb_T = sf_dis_in[0]
    actual_turb_T = orc_state6[0]
    
    
    # BLACK DOTTED LINE - Controlled Excess Entropy
    # Shows the entropy control mechanism - starts from temp_k point and runs parallel to temp_k line
    sf_dis_in = flp.tp(Temp_at_t_tankK, p_orc_low, RP=RP, composition=composition)
    sf_dis_out = flp.tp(Temp_l, p_orc_low, RP=RP, composition=composition)
    
    # Start point: where temp_k is located (at the beginning of the temp_k line)
    start_hdot = sf_dis_in[2] * m_dot_wf / 1000  # temp_k starting point
    start_T = Temp_at_t_tankK
    
    # End point: orc_state5
    end_hdot = orc_state6[2] * m_dot_wf / 1000
    end_T = orc_state6[0]
    
    # Calculate the slope of the temp_k line (cold storage line during discharging)
    temp_k_slope = (Temp_l - Temp_at_t_tankK) / (sf_dis_out[2] - sf_dis_in[2])
    
    # Create parallel line: same slope as temp_k line but starting from temp_k point
    hdot_range = end_hdot - start_hdot
    parallel_end_T = start_T + (temp_k_slope * hdot_range * 1000 / m_dot_wf)  # Convert back to J/kg for slope calculation
    



    # Create the parallel line points
    n_points = 50
    entropy_control_hdot_extended = np.linspace(start_hdot, end_hdot, n_points)
    entropy_control_T_extended = np.linspace(start_T, parallel_end_T, n_points)
    
    # Plot the black dotted line parallel to temp_k line
    plt.plot(entropy_control_hdot_extended, entropy_control_T_extended, 'k:', 
             linewidth=3, alpha=0.8, label='Controlled Excess Entropy Path')
    
    # Enhanced textstr with entropy information
    textstr = '\n'.join((
        f'Working Fluid:',
        f'  ṁ = {m_dot_wf*3600:.2f} kg/h',
        f'  Q_cond = {Q_cond/1000:.2f} kW',
        f'  Q_evap = {Q_evap/1000:.2f} kW',
        f'',
        f'Entropy Analysis:',
        f'  Δs_total = {total_entropy_generation:.4f} kJ/kg·K',
        f'  Ex_destroyed = {Ex_destroyed:.2f} kW'))
    
    
    # Plot configuration
    plt.xlabel("Enthalpy Flow Rate, Ḣ (kJ/s)", fontsize=12)
    plt.ylabel("Temperature (K)", fontsize=12)
    plt.title(f"T-Ḣ Diagram: Complete Cycle with Zeotropic Mixture Behavior (COP = {COP:.2f}) \n| Water ΔT = {Temp_D-Temp_B:.0f}K |\n| Water ΔT = {Temp_B - Temp_A:.0f}K "
              f"Glycol ΔT = {Temp_E-Temp_F:.0f}K", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    plt.xlim(min(hdot_bubble)-0.6, max(hdot_dew)+0.6)  # Adjusted for kJ/s scale
    plt.ylim(230, 440)
    plt.tight_layout() 
    
    
    

def show_results_window(args):
    import tkinter as tk
    from tkinter import scrolledtext

    root = tk.Tk()
    root.title("Carnot Battery Simulation Results")
    root.geometry("900x600")
    text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Courier New", 10))
    text_area.pack(expand=True, fill='both', padx=10, pady=10)
    


    results_text = f"""
{'='*80}
{'Carnot Battery Simulation Results'.center(80)}
{'='*80}
{'Working Fluid: PROPANE;ISOBUTANE;PENTANE'.ljust(50)} Composition: {args['composition']}
{'Mass flow rate:'.ljust(50)} {args['m_dot_wf']*3600:.2f} kg/h
{'Efficiency of compressor:'.ljust(50)} {args['eta_isentropic']:.2f}
{'Efficiency of pump:'.ljust(50)} {args['eta_pump']:.2f}
{'Efficiency of turbine:'.ljust(50)} {args['eta_turbine']:.2f}
{'='*80}
{'Diameter of tank A and tank B:'.ljust(50)} {args['dia_A_B']:.2f}m
{'Diameter of tank C and tank D:'.ljust(50)} {args['dia_C_D']:.2f}m
{'Diameter of tank E and tank F:'.ljust(50)} {args['dia_E_F']:.2f}m
{'Overall heat coefficient U:'.ljust(50)} {args['U_overall']:.4f} W/m²K
{'Total surface area:'.ljust(50)} {args['total_surface_area_A_B']:.4f}m²
{'='*80}

{'Calculated COP:'.ljust(50)} {args['COP']:.2f}
{'Mass in tank B:'.ljust(50)} {args['m_water_2']:.5f}kg
{'Water needed in condensation in litres:'.ljust(50)} {args['vol_watertank_1']*1000:.2f}lit
{'Water needed in subcooling in litres:'.ljust(50)} {args['vol_watertank_2']*1000:.2f}lit
{'Water-glycol needed in evaporation:'.ljust(50)} {args['vol_glycolwatertank']*1000:.2f}lit

{'ENERGY AND PERFORMANCE ANALYSIS RESULTS'.center(90)}
{'='*90}
{'Compressor entropy generation:'.ljust(60)} {args['Delta_s_compressor']:.4f} kJ/K·S
{'Turbine entropy generation:'.ljust(60)} {args['Delta_s_turbine']:.4f} kJ/K·S
{'Pump entropy generation:'.ljust(60)} {args['Delta_s_pump']:.4f} kJ/K·S
{'Heat exchanger entropy generation:'.ljust(60)} {args['Delta_s_hex_total']:.4f} kJ/K·S
{'Total entropy generation:'.ljust(60)} {args['total_entropy_generation']:.4f} kJ/K·S
{'Exergy destruction rate:'.ljust(60)} {args['Ex_destroyed']:.2f} kW
{'Dead state temperature:'.ljust(60)} {args['T_dead_state']:.2f} K
{'Excess controlled entropy(black dotted line):'.ljust(60)} {args.get('total_excess_entropy', 0):.3f}kJ/K


{'='*80}
{'Using electric heater'}

{'Required electric heater power for 4h :'.ljust(50)} {args['P_electric']/1000:.2f}kW
{'Energy of heat delivered at target temperature(361.28K) :'.ljust(50)} {args['En_heat_delivered_electric']/1e6:.2f}MJ
{'Energy efficiency electric heater system:'.ljust(50)} {args.get('eta_energy_electric_system',0):.2f}%
{'Mass of water needed using electric heater system:'.ljust(50)} {args.get('m_water_1',0):.2f}kg


{'Using Heat pump'}

{'Temp D needed for preheating:'.ljust(50)} {args['Temp_D_test']:.2f}K
{'Final temperature in tank C after mixing:'.ljust(50)} {args['T_final']:.2f}K
{'Energy of heat delivered by heat pump:'.ljust(50)} {args['En_heat_delivered_HP']/1e6:.2f}MJ
{'Energy efficiency using Heat pump system:'.ljust(50)} {args.get('eta_energy_HP_system',0):.2f}%
{'Mass of water needed using heat pump system:'.ljust(50)} {args.get('m_water_initial',0):.2f}kg

{'Heat pump improvement:'.ljust(50)} {args.get('efficiency_improvement',0):.1f}x better

{'CHARGING PHASE (HEAT PUMP)'.center(80)}
{'='*80}
{'Temperature of water in tank A:'.ljust(50)} {args['Temp_A']:.2f}K
{'Temperature of water in tank B:'.ljust(50)} {args['Temp_B']:.2f}K
{'Temperature of water in tank D:'.ljust(50)} {args['Temp_D']:.2f}K
{'Temperature of water+glycol in tank E:'.ljust(50)} {args['Temp_E']:.2f}K
{'Temperature of water+glycol in tank F:'.ljust(50)} {args['Temp_F']:.2f}K

{'STANDSTILL TIME AFTER CHARGING PHASE (HEAT PUMP)'.center(80)}
{'='*80}
{'Estimated heat loss during storage in tank B:'.ljust(50)} {args['heat_loss_in_B']/1000:.4f} kJ
{'Estimated heat loss during storage in tank D:'.ljust(50)} {args['heat_loss_in_D']/1000:.4f} kJ
{'Estimated heat gain during storage in tank F:'.ljust(50)} {args['heat_gain_in_F']/1000:.4f} kJ
{'Temperature in tank B after 12 hours:'.ljust(50)} {args['Temp_at_t_tankB']:.2f}K
{'Temperature in tank D after 12 hours:'.ljust(50)} {args['Temp_at_t_tankD']:.2f}K
{'Temperature in tank F after 12 hours:'.ljust(50)} {args['Temp_at_t_tankF']:.2f}K

{'DISCHARGING PHASE (ORC CYCLE)'.center(80)}
{'='*80}
{'Temperature of water in tank G:'.ljust(50)} {args['Temp_g']:.2f}K
{'Temperature of water in tank H:'.ljust(50)} {args['Temp_h']:.2f}K
{'Temperature of water in tank I:'.ljust(50)} {args['Temp_i']:.2f}K
{'Temperature of water in tank J:'.ljust(50)} {args['Temp_j']:.2f}K
{'Temperature of water+glycol in tank K:'.ljust(50)} {args['Temp_k']:.2f}K
{'Temperature of water+glycol in tank L:'.ljust(50)} {args['Temp_l']:.2f}K

{'STANDSTILL TIME AFTER DISCHARGING PHASE (ORC CYCLE)'.center(80)}
{'='*80}
{'Estimated heat gain during storage in tank G:'.ljust(50)} {args['heat_loss_in_G']/1000:.4f} kJ
{'Estimated heat loss during storage in tank I:'.ljust(50)} {args['heat_loss_in_I']/1000:.4f} kJ
{'Estimated heat loss during storage in tank K:'.ljust(50)} {args['heat_loss_in_K']/1000:.4f} kJ
{'Temperature in tank G after 4 hours:'.ljust(50)} {args['Temp_at_t_tankG']:.2f}K
{'Temperature in tank I after 4 hours:'.ljust(50)} {args['Temp_at_t_tankI']:.2f}K
{'Temperature in tank K after 4 hours:'.ljust(50)} {args['Temp_at_t_tankK']:.2f}K
"""

    text_area.insert(tk.INSERT, results_text)
    text_area.configure(state='disabled')  # Make it read-only

    close_btn = tk.Button(root, text="Close", command=root.destroy,
                         bg="#4a7abc", fg="white", font=("Arial", 10, "bold"))
    close_btn.pack(pady=10)

    root.mainloop()

plt.show(block=False)  # Non-blocking to allow interaction
# Then show the results window
# show_results_window(args_dict)

def main():
    chg = charging_phase()
    stand = standstill_after_charging(chg)
    disch = discharging_phase(chg, stand)
    stand2 = standstill_after_discharging(chg, stand, disch)

   
    preheat = preheating(
        m_water_1=chg["m_water_1"],
        cp_water=cp_water,
        Temp_B=chg["Temp_B"],
        T_surrounding=T_surrounding,
        m_dot_wf=m_dot_wf,
        comp_out=chg["comp_out"],
        throt_in=chg["throt_in"],
        
    )
    
    # plot_preheating_tankC(
    #     cp_water,
    #     preheat["m_dot_water_initial"],   
    #     T_surrounding,
    #     preheat["T_final"],
    #     preheat["Temp_D_test"],          
    #     preheat["preheating_time"]
    # )
    
    plot_preheating_tankC_mixture(
        chg["m_dot_water_1"],
        T_surrounding,
        preheat["T_final"],
        preheat["Temp_D_test"],
        preheat["preheating_time"],
        p_wf_high,
        RP
    )
    
#     plot_charging_phase_with_preheat(
#     cp_water,
#     preheat["m_dot_water_initial"],
#     T_surrounding,
#     preheat["T_final"],
#     preheat["Temp_D_test"],
#     chg["comp_in"],
#     chg["comp_out"],
#     chg["sat_vap_p_high"],
#     chg["sat_liq_p_high"],
#     chg["throt_in"],
#     chg["throt_out"],
#     chg["sat_vap_p_low"]
# )
    
    T_threshold = 350
    cool_down_time(preheat["m_water_initial"], cp_water, U, stand["total_surface_area_C_D"], preheat["T_final"], T_threshold, T_surrounding)
    
    stand_combined = {**stand, **stand2}
    energy = energy_and_performance(chg, stand_combined, disch,preheat)
    args_dict = {
        # Fluid and basic
        "flp": flp,
        "m_dot_wf": m_dot_wf,
        "RP": RP,
        "composition": composition,
        "eta_isentropic": eta_isentropic,

        # State points from charging
        "comp_in": chg["comp_in"],
        "comp_out": chg["comp_out"],
        "sat_vap_p_high": chg["sat_vap_p_high"],
        "sat_liq_p_high": chg["sat_liq_p_high"],
        "throt_in": chg["throt_in"],
        "throt_out": chg["throt_out"],
        "sat_vap_p_low": chg["sat_vap_p_low"],
        "COP": chg["COP"],
        "Temp_A": chg["Temp_A"],
        "Temp_B": chg["Temp_B"],
        "Temp_D": chg["Temp_D"],
        "Temp_E": chg["Temp_E"],
        "Temp_F": chg["Temp_F"],
        "Temp_hmid": chg["Temp_hmid"],
        "m_dot_water_1": chg["m_dot_water_1"],
        "m_water_1": chg["m_water_1"],
        "operation_time": operation_time,
        "vol_watertank_1": stand["vol_watertank_1"],
        "m_dot_water_2": chg["m_dot_water_2"],
        "vol_watertank_2": stand["vol_watertank_2"],
        "m_dot_glycol": chg["m_dot_glycol"],
        "vol_glycolwatertank": stand["vol_glycolwatertank"],

        # State points from discharging
        "orc_state1": disch["orc_state1"],
        "orc_state2": disch["orc_state2"],
        "orc_state3": disch["orc_state3"],
        "orc_state4": disch["orc_state4"],
        "orc_state5": disch["orc_state5"],
        "orc_state6": disch["orc_state6"],
        "orc_state7": disch["orc_state7"],
        "orc_state8": disch["orc_state8"],

        # Temperatures after standstill
        "Temp_at_t_tankB": stand["Temp_at_t_tankB"],
        "Temp_at_t_tankD": stand["Temp_at_t_tankD"],
        "Temp_at_t_tankF": stand["Temp_at_t_tankF"],
        "Temp_g": disch["Temp_g"],
        "Temp_h": disch["Temp_h"],
        "Temp_i": disch["Temp_i"],
        "Temp_j": disch["Temp_j"],
        "Temp_k": disch["Temp_k"],
        "Temp_l": disch["Temp_l"],

        # Additional parameters
        "p_orc_high": 16.0e5,
        "sf_ch_in": chg["sf_ch_in"],
        "Temp_at_t_tankG": stand2["Temp_at_t_tankG"],
        "Temp_at_t_tankK": stand2["Temp_at_t_tankK"],
        "p_orc_low": 3.5e5,
        "Q_dot_orc_1_evap": disch["Q_dot_orc_1_evap"],
        "Q_cond": chg["Q_cond"],
        "Q_evap": chg["Q_evap"],
        "total_entropy_generation": energy["total_entropy_generation"],
        "Ex_destroyed": energy["Ex_destroyed"],
        "Temp_at_t_tankI": stand2["Temp_at_t_tankI"],
        "eta_pump": 0.6,
        "eta_turbine": 0.75,
        "dia_A_B": stand["dia_A_B"],
        "dia_C_D": stand["dia_C_D"],
        "dia_E_F": stand["dia_E_F"],
        "U_overall": stand["U_overall"],
        "total_surface_area_A_B": stand["total_surface_area_A_B"],
        "P_electric": preheat["P_electric"],
        "energy_elect_heater": energy.get("energy_elect_heater", 0),
        "energy_HP": energy.get("energy_HP", 0),
        "m_water_initial": preheat["m_water_initial"],
        "Temp_D_test": preheat["Temp_D_test"],
        "T_final": preheat["T_final"],
        "m_water_2": chg["m_water_2"],
        
        "T_dead_state": energy["T_dead_state"],
        "Delta_s_hex_total": chg["Delta_s_hex_total"],
        "Delta_s_pump": disch["Delta_s_pump"],
        "Delta_s_turbine": disch["Delta_s_turbine"],
        "Delta_s_compressor": chg["Delta_s_compressor"],
        "heat_loss_in_B": stand["heat_loss_in_B"],
        "heat_loss_in_D": stand["heat_loss_in_D"],
        "heat_gain_in_F": stand["heat_gain_in_F"],
        "heat_loss_in_G": stand2["heat_loss_in_G"],
        "heat_loss_in_I": stand2["heat_loss_in_I"],
        "heat_loss_in_K": stand2["heat_loss_in_K"],
         
    }
    args_dict.update(energy)
    

    plot_carnot_battery_cycle(
    **{k: v for k, v in args_dict.items()
       if k in plot_carnot_battery_cycle.__code__.co_varnames[:plot_carnot_battery_cycle.__code__.co_argcount]}
)
    show_results_window(args_dict)
    

    

if __name__ == "__main__":
    main()
