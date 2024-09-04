import keyboard

from smac.env import StarCraft2Env
import numpy as np
import time

env = StarCraft2Env(map_name="3m")
env_info = env.get_env_info()

n_actions = env_info["n_actions"]
n_agents = env_info["n_agents"]

n_episodes = 100
e=0
total_reward = []
while e<n_episodes:
    env.reset()
    env.battles_game
    env_info = env.get_env_info()
    a=env.win_counted
    terminated = False
    episode_reward = 0
    step=0
    e+=1
    while not terminated:
        step+=1
        obs = env.get_obs()
        state = env.get_state()
        # env.render()  # Uncomment for rendering

        actions = []
        for agent_id in range(n_agents):
            avail_actions = env.get_avail_agent_actions(agent_id)
            avail_actions_ind = np.nonzero(avail_actions)[0]
            action = avail_actions_ind[-1]
            actions.append(action)
        reward, terminated, _ = env.step(actions)
        # time.sleep(0.15)
        episode_reward += reward
    total_reward.append(episode_reward)

    print("Total reward in episode {} = {}".format(e, episode_reward))
print('total_reward{}'.format(str(sum(total_reward)/e)))
env.save_replay()
print("11111")

