from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import logging
import numpy as np

env_logger = logging.getLogger("environmnet_logger")

class Agent(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.runs = [[]]
    @abstractmethod
    def act(self, observation, periphral=None):
        pass
    @abstractmethod
    def record_observation(self, observation_old, action, reward, observation, terminated):
        pass

def run_upto_n_steps(env, agent: Agent, n, continuation=None, runs=[[]]):
    """
    Run n steps in environment or untill termination
    """
    if continuation is not None:
        observation, reward, terminated, truncated, info = continuation
    if continuation is None or terminated or truncated:
        env_logger.info("Resetting")
        observation, info = env.reset()
        terminated = False
        truncated = False
        reward = 0
    step = 0
    # print((observation, reward, terminated, truncated, info))
    while not terminated and not truncated and step < n:
        action = agent.act(observation)
        observation_old = observation
        observation, reward, terminated, truncated, info = env.step(action)
        agent.record_observation(observation_old, action, reward, observation, terminated)
        runs[-1].append(reward)
        step += 1
    if terminated or truncated:
        env_logger.info(f"Finished episode with {len(runs[-1])} steps")
        runs.append([])
    return (observation, reward, terminated, truncated, info), runs

def plot_reward_and_episodes(runs):
    # plt.clf()
    plt.scatter(*zip(*enumerate([sum(i) for i in runs])), s=120, c=[len(i) for i in runs], label='total reward')
    plt.axhline(0,color='k')
    plt.axhline(200,color='k')
    plt.ylabel("Total episode reward")
    plt.colorbar(label='episode length')
    plt.legend()

def plot_one_run(env, agent: Agent, plot_interval=10, pause=0.1, clear_func=lambda:None):
    cont=None
    runs=[[]]
    while len(runs)==1:
        clear_func()
        plt.clf()
        plt.plot(np.cumsum(runs[-1]))
        plt.pause(0.1)
        cont, runs = run_upto_n_steps(env, agent, plot_interval, cont, runs)
    clear_func()
    plt.clf()
    plt.plot(np.cumsum(runs[-2]))
    plt.pause(pause)
    

def run_and_plot(env, agent: Agent, n, cont=None, ):
    """
    Run and plot agent performance
    """
    cont = run_upto_n_steps(env, agent, 10, cont)
    return cont


