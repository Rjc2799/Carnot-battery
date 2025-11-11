# -*- coding: utf-8 -*-
"""
Created on Sun May 11 16:18:40 2025

@author: RuturajChavan
"""

'''This code is used to find the pressures of the working fluid at a given pressure ratio and certain other constants.
The obtained pressure values are then used in the heat_pump_cycle.py for further calculations.'''

import fluid_properties_rp as flp
#import numpy as np

# Constants
composition = [0.3, 0.6, 0.1]  # molar fractions for PROPANE;ISOBUTANE;PENTANE
RP = flp.setRPFluid("PROPANE;ISOBUTANE;PENTANE")

T_surrounding = 293.15  # K (20°C)
dT_pinch_superheated = 2  # K below T_surrounding
dT_superheating = 5       # Additional 5 K drop for superheated vapor region

# Final target temperature for superheated vapor
T_target_superheated = T_surrounding - dT_pinch_superheated - dT_superheating

# Direct calculation of saturation pressure using temperature and quality

def find_saturation_pressure_at_temperature(T_target):
    try:
        state = flp.prop_Tq(T_target, 1, RP=RP, composition=composition)
        return state[1]  # Extract pressure directly from state vector
    except Exception as e:
        raise ValueError(f"Could not find saturation pressure for T = {T_target} K. Error: {e}")

p_wf_low = find_saturation_pressure_at_temperature(T_target_superheated)

# Calculate p_wf_high based on pressure ratio
pressure_ratio = 10
p_wf_high = p_wf_low * pressure_ratio

if __name__ == "__main__":

    # Output the results
    print(f"Determined Low Pressure (p_wf_low): {p_wf_low / 100000:.2f} bar at T = {T_target_superheated:.2f} K")
    # Output the result
    print(f"Determined High Pressure (p_wf_high): {p_wf_high / 100000:.2f} bar")