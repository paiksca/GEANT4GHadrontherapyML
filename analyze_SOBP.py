#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def generate_ideal_profile(depths):
    """
    Generate the ideal normalized SOBP profile over the given depths (in mm).
    New profile specification:
      - For depth d < 24.2 mm: linear ramp from 0.5 at d=0 to 1.0 at d=24.2 mm.
      - For 24.2 mm <= d <= 28.7 mm: flat at 1.0.
      - For 28.7 mm < d <= 30.7 mm: linear drop from 1.0 to 0.
      - For d > 30.7 mm: exactly 0.
    """
    ideal = np.zeros_like(depths)
    ideal = np.where(depths < 25.2, 0.5 + 0.5 * (depths / 24.2), ideal)
    ideal = np.where((depths >= 25.2) & (depths <= 28.7), 1.0, ideal)
    ideal = np.where((depths > 28.7) & (depths <= 30.7), 1.0 - ((depths - 28.7) / 2.0), ideal)
    ideal = np.where(depths > 30.7, 0.0, ideal)
    return ideal

# Path to the Dose.out file from the GEANT4 simulation.
dose_file = "/Users/Children/Downloads/geant4-v11.3.0/examples/advanced/hadrontherapy/build/Dose.out"

# Load the data from Dose.out, skipping the header row.
data = np.loadtxt(dose_file, skiprows=1)

# The first column corresponds to the voxel index along the beam (depth) direction,
# and the fourth column contains the dose in Gy.
slice_indices = data[:, 0]
dose_values = data[:, 3]

# Voxel thickness along the beam direction (in mm) as specified in the macro.
voxel_thickness = 0.1  # mm

# Compute the unique slices and sum the dose values for each slice.
unique_slices = np.unique(slice_indices)
depth_dose = np.array([np.sum(dose_values[slice_indices == idx]) for idx in unique_slices])
depths = unique_slices * voxel_thickness

# Normalize the dose profile to its maximum value.
if np.max(depth_dose) > 0:
    dose_norm = depth_dose / np.max(depth_dose)
else:
    dose_norm = depth_dose

# Generate the ideal SOBP profile using the same depths.
ideal_profile = generate_ideal_profile(depths)

# Compute the overall Mean Squared Error (MSE) between the simulated and ideal profiles.
mse_all = np.mean((dose_norm - ideal_profile) ** 2)

# Compute the MSE specifically for the ideal flat region (25.2 mm to 28.7 mm).
region_mask = (depths >= 25.2) & (depths <= 28.7)
mse_region = np.mean((dose_norm[region_mask] - ideal_profile[region_mask]) ** 2)

# Print the MSE values.
print("Overall MSE:", mse_all)
print("MSE (25.2 mm - 28.7 mm):", mse_region)

# Plot the normalized dose profile and the ideal SOBP profile.
plt.figure(figsize=(10, 6))
plt.plot(depths, dose_norm, marker='o', linestyle='-', label="Simulated Dose Profile")
plt.plot(depths, ideal_profile, linestyle='--', color='red', label="Ideal SOBP Profile")
plt.xlabel("Depth (mm)")
plt.ylabel("Normalized Dose")
plt.title("Comparison of Simulated Dose Profile vs. Ideal SOBP")
plt.axvline(x=25.2, color='gray', linestyle=':', label="24.2 mm")
plt.axvline(x=28.7, color='gray', linestyle=':', label="28.7 mm")
plt.grid(True)
plt.legend()
plt.show()
