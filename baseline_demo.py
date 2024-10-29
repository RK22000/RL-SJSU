import gymnasium as gym
import argparse
from stable_baselines3 import DQN, PPO

parser = argparse.ArgumentParser("BaselineDemo")
parser.add_argument("--algo", default='dqn', help="dqn or ppo", type=lambda i: str(i).lower() )

args = parser.parse_args()

if args.algo=='dqn':
  model = DQN.load("baseline_models/dqn-lunarlander/LunarLander-v2.zip")
elif args.algo=='ppo':
  model = PPO.load("baseline_models/ppo-lunarlander/LunarLander-v2.zip")
  

# env = gym.make("LunarLander-v2")

# model = DQN("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10_000, log_interval=4)
# model.save("dqn_cartpole")

# del model # remove to demonstrate saving and loading

# model = DQN.load("dqn_cartpole")

env = gym.make("LunarLander-v2", render_mode='human')

obs, info = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    # env.render()
    if done:
      obs, info = env.reset()
