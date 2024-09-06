import keyboard
import torch
from smac.env import StarCraft2Env
import time
from train_pytorch import MlpModel
from maddpg_pytorch.maddpg_pytorch import ActorEncoder
from maddpg_pytorch.vae import VAE
from train_pytorch import act_mask_max

env = StarCraft2Env(map_name="3m",debug=True)
env_info = env.get_env_info()
n_agents = env_info["n_agents"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_episodes = 100
episode = 0
total_reward = []

with open("./MADDPG_SMAC/6_3m1000/pytorch_model.pt", "rb") as fp:
    all_net = torch.load(fp)
all_agentt = all_net['trainers']
actors = []
for agent in all_agentt:
    actors.append(agent[0])


while episode < max_episodes:
    env.reset()
    env_info = env.get_env_info()
    a = env.win_counted
    terminated = False
    episode_reward = 0
    episode_step = 0
    episode += 1
    while not terminated:
        episode_step += 1
        obs = env.get_obs()
        actions = []
        for agent_id in range(n_agents):
            avail_actions = env.get_avail_agent_actions(agent_id)
            action = actors[agent_id](torch.from_numpy(obs[agent_id]).to(device))
            actions.append(action)
        choice_actions = act_mask_max(actions, env)
        reward, terminated, _ = env.step(choice_actions)
        # time.sleep(0.15)
        episode_reward += reward
    total_reward.append(episode_reward)
    print("Total reward in episode {} = {}".format(episode, episode_reward))
print('total_reward{}'.format(str(sum(total_reward) / episode_step)))
env.save_replay()
