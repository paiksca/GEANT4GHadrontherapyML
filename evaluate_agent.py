from stable_baselines3 import PPO
from natural_number_game import NaturalNumberGame
import supersuit as ss
from pettingzoo.utils import parallel_to_aec

# Load the trained model
model = PPO.load("ppo_natural_number_game")

# Initialize the environment
env = NaturalNumberGame()
env = parallel_to_aec(env)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')

# Reset the environment
obs = env.reset()

# Run the model for a single episode
done = False
while not done:
    action, _ = model.predict(obs)
    obs, rewards, dones, infos = env.step(action)
    done = dones[0]  # Since we have a single environment

    # Render the environment (optional)
    env.render()

# Close the environment
env.close()
