import gymnasium as gym

from stable_baselines3 import DQN

# env = gym.make("LunarLander-v2")

# model = DQN("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10_000, log_interval=4)
# model.save("dqn_cartpole")

# del model # remove to demonstrate saving and loading

model = DQN.load("rl-baselines3-zoo/rl-trained-agents/dqn/LunarLander-v2_1/LunarLander-v2_")
# model = DQN.load("dqn_cartpole")

env = gym.make("LunarLander-v2", render_mode='human')

obs, info = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    # env.render()
    if done:
      obs, info = env.reset()