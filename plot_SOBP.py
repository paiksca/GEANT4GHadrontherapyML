#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Path to the Dose.out file from GEANT4 simulation
dose_file = "/Users/Children/Downloads/geant4-v11.3.0/examples/advanced/hadrontherapy/build/Dose.out"

# Load the data from Dose.out, skipping the header row.
# Assumes the file is whitespace-separated.
data = np.loadtxt(dose_file, skiprows=1)

# The first three columns are the voxel indices (i, j, k), and the fourth column is Dose(Gy).
# We assume the "i" index (column 0) corresponds to the beam (depth) direction.
slice_indices = data[:, 0]
dose_values = data[:, 3]

# Set the voxel thickness along the beam direction.
# The macro command "/changeDetector/voxelSize .1 40 40 mm" indicates that the voxel size along the beam direction is 0.1 mm.
voxel_thickness = 0.1  # in mm

# Get the unique slice indices along the beam direction.
unique_slices = np.unique(slice_indices)

# For each unique slice, sum up the dose values of all voxels in that slice.
depth_dose = np.array([np.sum(dose_values[slice_indices == idx]) for idx in unique_slices])

# Calculate the physical depth for each slice.
depths = unique_slices * voxel_thickness

# Plot the depth-dose profile, which represents the spread-out Bragg peak.
plt.figure(figsize=(10, 6))
plt.plot(depths, depth_dose, marker='o', linestyle='-')
plt.xlabel("Depth (mm)")
plt.ylabel("Total Dose (Gy)")
plt.title("Spread-Out Bragg Peak")
plt.grid(True)
plt.show()
