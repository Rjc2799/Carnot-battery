import matplotlib.pyplot as plt

# Reference temperature (used as initial default)
T_REF = 293.15

# Phase definitions: (name, duration_hours, temperatures_at_end_of_phase)
# Any tank not listed in a phase keeps its previous temperature as its end-of-phase value.
phases = [
    ("Preheating", 4, {"Hot1": 361.28}),
    ("Charging", 4, {
        "Mid1": 293.15, "Cold1": 293.15,
        "Mid2": 361.28, "Hot2": 372.70, "Cold2": 272.70
    }),
    ("Standstill", 12, {  # after charging
        "Mid2": 352.24, "Hot2": 367.42, "Cold2": 274.42
    }),
    ("Discharging", 4, {
        "Mid1": 292.68, "Hot1": 356.08, "Cold1": 296.73
    }),
    ("Standstill", 4, {  # after discharging
        "Mid1": 292.70, "Hot1": 351.90, "Cold1": 293.99
    }),
]

tanks = ["Hot1", "Cold1", "Mid1", "Hot2", "Cold2", "Mid2"]
colors = {
    "Hot1": "#d62728",
    "Cold1": "#1f77b4",
    "Mid1": "#2ca02c",
    "Hot2": "#ff7f0e",
    "Cold2": "#17becf",
    "Mid2": "#9467bd",
}

# Build time and temperature profiles with linear ramps per phase
tank_profiles = {tank: [] for tank in tanks}
time_points = []

current_time = 0.0
current_temps = {tank: T_REF for tank in tanks}  # starting temps

time_points.append(current_time)
for tank in tanks:
    tank_profiles[tank].append(current_temps[tank])

for name, duration, overrides in phases:
    phase_start_time = current_time
    phase_end_time = current_time + duration

    # Determine end temps for this phase (carry forward if not overridden)
    end_temps = current_temps.copy()
    end_temps.update(overrides)

    # Append the end-of-phase time and temps (linear interpolation implied by plotting)
    time_points.append(phase_end_time)
    for tank in tanks:
        tank_profiles[tank].append(end_temps[tank])

    # Update for next phase
    current_time = phase_end_time
    current_temps = end_temps

# Plot
plt.figure(figsize=(10, 6))

for tank in tanks:
    plt.plot(time_points, tank_profiles[tank],
             label=tank, color=colors[tank], linewidth=2)

plt.xlabel("Time [hours]")
plt.ylabel("Temperature [K]")
plt.title("Tank Temperatures vs. Time (linear ramps per phase)")
plt.grid(True, alpha=0.3)
plt.legend()

# Shade phases and place labels above curves
phase_start = 0.0
ymax = max(max(vals) for vals in tank_profiles.values())
label_y = ymax + 8  # add headroom for labels
for name, duration, _ in phases:
    phase_end = phase_start + duration
    plt.axvspan(phase_start, phase_end, alpha=0.07, color="gray")
    plt.text((phase_start + phase_end) / 2, label_y,
             name, ha="center", va="bottom", fontsize=11, fontweight="bold")
    phase_start = phase_end

# Extend y-limit so labels do not overlap curves
plt.ylim(None, label_y + 5)

plt.tight_layout()
plt.show()