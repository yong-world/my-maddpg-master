import keyboard
import torch

from smac.env import StarCraft2Env
import numpy as np
import time
import pickle
from  train_pytorch import mlp_model
from maddpg_pytorch.vae import VAE
from train_pytorch import act_mask_max
env = StarCraft2Env(map_name="3m")
env_info = env.get_env_info()

n_actions = env_info["n_actions"]
n_agents = env_info["n_agents"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_episodes = 100
e = 0
total_reward = []
with open("./MADDPG_SMAC/3mT：0904_0147E：60000/exp1pytorch_model.pt", "rb") as fp:
    all_net = torch.load(fp)
all_net=all_net['trainers']
actors=[]
for agent in all_net:
    actors.append(agent[0])
while e < n_episodes:
    env.reset()
    env_info = env.get_env_info()
    a = env.win_counted
    terminated = False
    episode_reward = 0
    step = 0
    e += 1
    while not terminated:
        step += 1
        obs = env.get_obs()
        state = env.get_state()
        # env.render()  # Uncomment for rendering

        actions = []
        for agent_id in range(n_agents):
            avail_actions = env.get_avail_agent_actions(agent_id)
            action=actors[agent_id](torch.from_numpy(obs[agent_id]).to(device))
            actions.append(action)
        actions = act_mask_max(actors,env)
        reward, terminated, _ = env.step(actions)
        # time.sleep(0.15)
        episode_reward += reward
    total_reward.append(episode_reward)

    print("Total reward in episode {} = {}".format(e, episode_reward))
print('total_reward{}'.format(str(sum(total_reward) / e)))
env.save_replay()
