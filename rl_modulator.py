#!/usr/bin/env python3
import os
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

# --------------------------
# Helper Functions
# --------------------------

def generate_ideal_profile(depths):
    """
    Generate the ideal normalized SOBP profile over the given depths (in mm).
    Updated Ideal profile specification:
      - For depth d < 25.2 mm: linear ramp from 0.5 to 1.0.
      - For 25.2 mm <= d <= 28.7 mm: flat at 1.0.
      - For 28.7 mm < d <= 30.7 mm: linear drop from 1.0 to 0.
      - For d > 30.7 mm: exactly 0.
    """
    ideal = np.zeros_like(depths)
    # Ramp from 0.5 at d=0 to 1.0 at d=25.2 mm.
    ideal = np.where(depths < 25.2, 0.5 + 0.5 * (depths / 25.2), ideal)
    # Flat region at 1.0 between 25.2 mm and 28.7 mm.
    ideal = np.where((depths >= 25.2) & (depths <= 28.7), 1.0, ideal)
    # Linear drop from 1.0 at 28.7 mm to 0 at 30.7 mm.
    ideal = np.where((depths > 28.7) & (depths <= 30.7), 1.0 - ((depths - 28.7) / 2.0), ideal)
    # Exactly 0 for depths > 30.7 mm.
    ideal = np.where(depths > 30.7, 0.0, ideal)
    return ideal

def parse_dose_profile(dose_file, voxel_thickness=0.1, max_depth=50.0, num_points=500):
    """
    Parse the Dose.out file and generate a normalized depth-dose profile.
    
    Parameters:
      dose_file: Path to Dose.out.
      voxel_thickness: Voxel size along the beam direction (in mm).
      max_depth: Maximum depth to consider (in mm).
      num_points: Number of points for the resampled profile.
      
    Returns:
      depths_resampled: 1D numpy array of depths (mm).
      dose_norm: 1D numpy array of normalized dose values.
    """
    # Load data; skip header row.
    data = np.loadtxt(dose_file, skiprows=1)
    slices = data[:, 0]    # depth index
    doses = data[:, 3]     # Dose(Gy)
    
    unique_slices = np.unique(slices)
    depth_dose = np.array([np.sum(doses[slices == idx]) for idx in unique_slices])
    depths = unique_slices * voxel_thickness
    
    # Interpolate to a fixed grid.
    grid = np.linspace(0, max_depth, num_points)
    interp_func = interp1d(depths, depth_dose, kind='linear', bounds_error=False, fill_value=0)
    dose_profile = interp_func(grid)
    
    # Normalize by maximum dose.
    if np.max(dose_profile) > 0:
        dose_norm = dose_profile / np.max(dose_profile)
    else:
        dose_norm = dose_profile
    return grid, dose_norm

def compute_reward(sim_profile, depths, ideal_profile):
    """
    Compute reward as negative mean squared error between simulated and ideal SOBP profiles.
    """
    mse = np.mean((sim_profile - ideal_profile) ** 2)
    reward = -mse
    return reward

# --------------------------
# Custom Callback for Training Progress
# --------------------------

class TrainingProgressCallback(BaseCallback):
    """
    Custom callback that uses tqdm to display training progress.
    """
    def __init__(self, total_timesteps, verbose=0):
        super(TrainingProgressCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.progress_bar = None

    def _on_training_start(self) -> None:
        self.progress_bar = tqdm(total=self.total_timesteps, desc="Training Progress")

    def _on_step(self) -> bool:
        self.progress_bar.update(1)
        return True

    def _on_training_end(self) -> None:
        self.progress_bar.close()

# --------------------------
# Custom Gymnasium Environment
# --------------------------

class HadronTherapyModulatorEnv(gym.Env):
    """
    Custom Gymnasium environment for optimizing modulator parameters in GEANT4 hadrontherapy.
    
    Action Space (Box with shape (3,)):
      - action[0]: inner_radius, continuous in [1.0, 4.0] (cm)
      - action[1]: outer_radius, continuous in [8.0, 11.0] (cm)
      - action[2]: material indicator, continuous in [0.0, 1.0] (interpreted as 0 if <0.5 else 1)
      
    Observation:
      - A fixed-length normalized dose profile (500 points from 0 to 50 mm)
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(HadronTherapyModulatorEnv, self).__init__()
        # Define a single Box action space of shape (3,)
        self.action_space = spaces.Box(
            low=np.array([0.5, 7.5, 0.0], dtype=np.float32),
            high=np.array([4.5, 11.5, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        # Observation: 500-length normalized dose profile
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(500,), dtype=np.float32)
        
        # Default modulator parameters:
        self.inner_radius = 2.5   # cm
        self.outer_radius = 9.5   # cm
        self.material = 1         # 0 for air, 1 for PMMA (default PMMA)
        
        # Paths:
        self.build_dir = "/Users/Children/Downloads/geant4-v11.3.0/examples/advanced/hadrontherapy/build"
        self.dose_file = os.path.join(self.build_dir, "Dose.out")
        # Update simulation command to use the macro subdirectory:
        self.sim_cmd = "./hadrontherapy ../macro/modulatorMacro.mac"
        
        # Dose profile parameters:
        self.voxel_thickness = 0.1  # mm
        self.max_depth = 50.0       # mm
        self.num_profile_points = 500
        
        # Create depth grid and ideal profile:
        self.depth_grid = np.linspace(0, self.max_depth, self.num_profile_points)
        self.ideal_profile = generate_ideal_profile(self.depth_grid)
        
        # Episode length (n-steps set to 4 simulation steps per episode)
        self.episode_length = 4
        self.current_step = 0

    def _write_macro(self):
        """
        Update only the modulator parameters (material, innerRadius, and outRadius) in the 
        existing modulatorMacro.mac file. This routine will search for the corresponding lines,
        uncomment them if needed, and then update with the current parameters.
        """
        # Determine the new parameter strings.
        material_str = "G4_AIR" if self.material == 0 else "G4_PLEXIGLASS"
        inner_str = "/modulator/innerRadius {:.2f} cm".format(self.inner_radius)
        outer_str = "/modulator/outRadius {:.2f} cm".format(self.outer_radius)
        rmw_str = "/modulator/RMWMat " + material_str

        # Ensure the macro directory exists.
        macro_dir = os.path.join(self.build_dir, "macro")
        os.makedirs(macro_dir, exist_ok=True)
        macro_path = os.path.join(macro_dir, "modulatorMacro.mac")

        # Read the existing macro file.
        with open(macro_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        found_rmw = found_inner = found_outer = False
        for line in lines:
            # Remove leading whitespace.
            stripped = line.lstrip()
            # If the line is commented, remove the '#' for matching.
            uncommented = stripped[1:].lstrip() if stripped.startswith("#") else stripped

            if uncommented.startswith("/modulator/RMWMat"):
                new_lines.append(rmw_str + "\n")
                found_rmw = True
            elif uncommented.startswith("/modulator/innerRadius"):
                new_lines.append(inner_str + "\n")
                found_inner = True
            elif uncommented.startswith("/modulator/outRadius"):
                new_lines.append(outer_str + "\n")
                found_outer = True
            else:
                new_lines.append(line)

        # If any of the three lines were not found, append them.
        if not found_rmw:
            new_lines.append(rmw_str + "\n")
        if not found_inner:
            new_lines.append(inner_str + "\n")
        if not found_outer:
            new_lines.append(outer_str + "\n")

        # Write the updated contents back to the file.
        with open(macro_path, "w") as f:
            f.writelines(new_lines)
        return macro_path

    def _run_simulation(self):
        """
        Execute the GEANT4 simulation.
        """
        self._write_macro()
        process = subprocess.run(self.sim_cmd, shell=True, cwd=self.build_dir,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(process.stdout.decode())
        print(process.stderr.decode())

    def _get_observation(self):
        """
        Parse Dose.out to extract the normalized, resampled SOBP profile.
        """
        grid, dose_norm = parse_dose_profile(self.dose_file,
                                               voxel_thickness=self.voxel_thickness,
                                               max_depth=self.max_depth,
                                               num_points=self.num_profile_points)
        return dose_norm.astype(np.float32)

    def step(self, action):
        # action is a numpy array of shape (3,)
        self.inner_radius = float(action[0])
        self.outer_radius = float(action[1])
        self.material = 0 if action[2] < 0.5 else 1
        
        print("Running simulation with parameters:")
        print("Inner Radius: {:.2f} cm, Outer Radius: {:.2f} cm, Material: {}".format(
            self.inner_radius, self.outer_radius,
            "G4_AIR" if self.material == 0 else "G4_PLEXIGLASS"
        ))
        
        self._run_simulation()
        obs = self._get_observation()
        reward = compute_reward(obs, self.depth_grid, self.ideal_profile)
        print("Computed reward:", reward)
        self.current_step += 1
        
        # Determine termination conditions.
        terminated = (self.current_step >= self.episode_length)
        truncated = False
        info = {"inner_radius": self.inner_radius, "outer_radius": self.outer_radius, "material": self.material}
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Reset parameters to defaults.
        self.inner_radius = 2.5
        self.outer_radius = 9.5
        self.material = 1
        self.current_step = 0
        obs = np.zeros(self.num_profile_points, dtype=np.float32)
        return obs, {}

    def render(self, mode="human"):
        print("Current modulator parameters:")
        print("Inner Radius: {:.2f} cm, Outer Radius: {:.2f} cm, Material: {}".format(
            self.inner_radius, self.outer_radius,
            "G4_AIR" if self.material == 0 else "G4_PLEXIGLASS"
        ))

# --------------------------
# Main Training Script
# --------------------------

def main():
    env = HadronTherapyModulatorEnv()
    # Check environment for Gymnasium compliance.
    check_env(env, warn=True)
    
    total_timesteps = 5  # For prototype testing (each step takes ~130 sec)
    progress_callback = TrainingProgressCallback(total_timesteps=total_timesteps, verbose=1)
    
    # Create PPO model using MLP policy.
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Train the model with the progress callback.
    model.learn(total_timesteps=total_timesteps, callback=progress_callback)
    model.save("ppo_modulator_model")
    
    # Demonstration episode using the trained model.
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
    
    # Plot the simulated SOBP vs. the ideal profile.
    grid, dose_norm = parse_dose_profile(env.dose_file,
                                           voxel_thickness=env.voxel_thickness,
                                           max_depth=env.max_depth,
                                           num_points=env.num_profile_points)
    plt.figure(figsize=(10, 6))
    plt.plot(grid, dose_norm, label="Simulated SOBP")
    plt.plot(env.depth_grid, env.ideal_profile, label="Ideal SOBP", linestyle="--")
    plt.xlabel("Depth (mm)")
    plt.ylabel("Normalized Dose")
    plt.title("Spread-Out Bragg Peak Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
