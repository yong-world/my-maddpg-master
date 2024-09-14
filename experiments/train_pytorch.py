from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os, csv
import sys
import time
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import datetime
import numpy as np
from gym import spaces
from maddpg_pytorch.maddpg_pytorch import MADDPGAgentTrainer, vae_train

from smac.env import StarCraft2Env
from absl import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="3m", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=60, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=3000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=128, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="exp1", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./model_save/TH/", help="directory in which training "
                                                                                 "state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many "
                                                                    "episodes are completed")
    parser.add_argument("--load-dir", type=str, default="./model_save/MADDPG/TH/", help="directory in which"
                                                                                        " training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=1000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where "
                                                                                        "benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot "
                                                                                    "data is saved")
    parser.add_argument("--learning-curves-figure-dir", type=str, default="./learning_curves_figure/",
                        help="learning_curves_figure_directory")
    parser.add_argument("--log-dir", type=str, default="./Logs/", help="log directory")
    parser.add_argument("--log-head", type=str, default="./MADDPG_SMAC/", help="all log head")
    return parser.parse_args()


def act_mask_max(action_n, env):
    indices = []
    avail_agent_actions = [env.get_avail_agent_actions(i) for i in range(env.n_agents)]
    avail_agent_actions_tensor = torch.tensor(avail_agent_actions).to(device)
    for tensor, mask in zip(action_n, avail_agent_actions_tensor):
        masked_tensor = tensor * mask
        # 找到最大值的索引
        max_index = torch.argmax(masked_tensor)
        indices.append(max_index.item())
    return indices


def get_time():
    return str(datetime.datetime.now().strftime('%m%d_%H%M'))


def get_trainers(env, n_agent, obs_space_n, act_space_n, arglist, device):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(env.get_env_info()['n_agents']):
        trainers.append(
            trainer(name="agent_%d" % i, model=model, obs_shape_n=obs_space_n, act_space_n=act_space_n, agent_index=i,
                    args=arglist, device=device, team_num=n_agent))
    return trainers


def make_env(map_name):
    env = StarCraft2Env(map_name=map_name)
    env_info = env.get_env_info()
    action_space = env_info['n_actions']
    obs_space = env_info['obs_shape']
    action_space_n = []
    observation_space_n = []
    for agent in range(env_info['n_agents']):
        d_action_space = spaces.Discrete(action_space)
        b_observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_space,), dtype=np.float32)
        action_space_n.append(d_action_space)
        observation_space_n.append(b_observation_space)
    act_space_acc = action_space * env_info['n_agents']
    obs_space_acc = obs_space * env_info['n_agents']
    obs_n, state = env.reset()
    obs_n = np.array(obs_n)
    obs_n = torch.from_numpy(obs_n).to(device)
    return env, env_info, action_space_n, observation_space_n, act_space_acc, obs_space_acc, obs_n


class mlp_model(nn.Module):
    def __init__(self, input_dim, num_outputs, num_units=64):
        super(mlp_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, num_units)
        self.fc2 = nn.Linear(num_units, num_units)
        self.fc3 = nn.Linear(num_units, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def plot_save(data_input, title, x_label, y_label, png_dir, marker=None):
    plt.plot(range(1, len(data_input) + 1), data_input, linewidth='0.5', marker=marker)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.savefig(png_dir)
    print("figure saved to:", png_dir)
    plt.show()


def read_from_csv(file_name):
    data = []
    with open(file_name, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            converted_row = []
            for item in row:
                # 尝试将字符串转换为整数或浮点数
                if item.isdigit():
                    converted_row.append(int(item))
                else:
                    try:
                        converted_row.append(float(item))
                    except ValueError:
                        converted_row.append(item)  # 保持原始字符串
            data.append(converted_row)
    print(f"数据已从 {file_name} 读取")
    return data


def save_all_data(env, arglist, episode_rewards, final_ep_rewards, log, win_lose, trainers):
    # 创建路径
    exp_dir = arglist.log_head + '{}_{}{}/'.format(arglist.exp_index, arglist.scenario, arglist.num_episodes)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        print("{} created".format(exp_dir))
    # 保存奖励、胜率对象
    with open(exp_dir+'episode_rewards.pkl', 'wb') as fp:
        pickle.dump(episode_rewards, fp)
    with open(exp_dir + 'episode_rewards.pkl', 'wb') as fp:
        pickle.dump(final_ep_rewards, fp)
    with open(exp_dir + 'episode_rewards.pkl', 'wb') as fp:
        pickle.dump(win_lose, fp)

    # 绘图并保存
    mean_reward = np.mean(episode_rewards)
    plot_save(data_input=final_ep_rewards,
              title='1/100TrainStepRewards',
              x_label='episode',
              y_label='reward', png_dir=exp_dir + 'sample_rewards.png', marker='.')

    # 保存训练记录
    log.append('Rewards:{}\t')
    log.append('End iterations')
    with open(exp_dir + 'train_log.txt', 'w+') as fp:
        for log_line in log:
            fp.write(str(log_line) + '\n')
    with open(exp_dir + 'episode_reward.txt', 'w+') as fp:
        for episode_reward in episode_rewards:
            fp.write(str(episode_reward) + '\n')
    # 保存模型
    save_path = exp_dir + "pytorch_model.pt"
    torch.save({'trainers': [[trainers[i].p, trainers[i].target_p, trainers[i].q, trainers[i].target_q, trainers[i].vae]
                             for i in range(trainers[0].n)]}, save_path)

    # 保存自变参数
    arglist.Variable_Parameter[1][1] += 1
    write_to_csv('Variable_Parameter.csv', arglist.Variable_Parameter)


def write_to_csv(file_name, data):
    # data应为列表，每个元素都是一个参数或参数列表（每行的数据）
    with open(file_name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print(f"数据已写入 {file_name}")


def train(arglist):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create environment
    env, env_info, act_space_n, obs_space_n, act_space_acc, obs_space_acc, obs_n = make_env(arglist.scenario)
    # Create agent trainers
    trainers = get_trainers(env, env_info['n_agents'], obs_space_n, act_space_n, arglist, device)
    print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))
    num_adversaries = arglist.num_adversaries
    # Load previous results, if necessary
    if arglist.load_dir == "":
        arglist.load_dir = arglist.save_dir
    if arglist.display or arglist.restore or arglist.benchmark:  # 用torch加载模型
        print('Loading previous state')
    log = []
    episode_rewards = [0.0]
    episode_num = 0
    episode_win_lose = []
    final_ep_rewards = []
    agent_info = [[[]]]
    episode_step = 0
    train_step = 0
    episode1000_start = time.time()

    print('Starting iterations...')
    train_info = ('scenario:{}\tnum-episodes:{}\tbatch-size:{}\tsave-rates:{}\tlr:{}'.format
                  (arglist.scenario, arglist.num_episodes, arglist.batch_size, arglist.save_rate, arglist.lr))
    print(train_info)
    log.append('-----------------------------------------------------------------------------------------')
    log.append('Starting iterations : {}'.format(get_time()))
    log.append(train_info)

    while True:
        action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
        # 获取环境输出的reward, terminated, info_n，rew_n，new_obs_n，action_n
        action_n_index = act_mask_max(action_n, env)
        reward, terminated, info_n = env.step(action_n_index)
        rew_n = [[reward] for _ in range(env.n_agents)]
        new_obs_n = env.get_obs()

        # 每轮里的每步奖励累加
        for i, rew in enumerate(rew_n):
            episode_rewards[-1] += rew[0]
        episode_step += 1

        # 转换成tensor， L*L转L*T,L*N转L*T,因为action本来就是L*T就不用
        rew_n = torch.tensor(rew_n).to(device)
        new_obs_n = np.array(new_obs_n, copy=False)
        new_obs_n = torch.from_numpy(new_obs_n).to(device)
        for i, agent in enumerate(trainers):
            agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i])
        obs_n = new_obs_n

        # 一轮结束后重置环境、代内步和奖励，收集胜利/失败信息
        if terminated:
            obs_n, state = env.reset()
            obs_n = np.array(obs_n)
            obs_n = torch.from_numpy(obs_n).to(device)
            episode_rewards.append(0)
            sys.stdout.write("\repsiode:{},steps:{},episode_step:{}".format(episode_num, train_step, episode_step))
            sys.stdout.flush()
            episode_num += 1
            episode_step = 0
            if info_n == {}:
                episode_win_lose.append(0)
            else:
                if info_n['battle_won']:
                    episode_win_lose.append(1)
                else:
                    episode_win_lose.append(0)
        train_step += 1

        # vae 和p、q训练比例 1:3
        if len(trainers[0].replay_buffer) > trainers[0].max_replay_buffer_len and train_step % 400 == 0:
            vae_train(trainers=trainers)
        else:
            for agent in trainers:
                loss = agent.update(trainers, train_step)

        # 每save_rate(1000)轮保存模型,输出训练信息
        if terminated and (episode_num % arglist.save_rate == 0):
            # 输出最近一千轮信息
            output = "\rsteps: {}, episodes: {}, mean episode reward: {}, won_rate: {} time: {}".format(
                train_step, episode_num, np.mean(episode_rewards[-arglist.save_rate:]),
                np.mean(episode_win_lose[-arglist.save_rate:]), round(time.time() - episode1000_start, 3))
            # 最近一千轮的平均奖励
            final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
            print(output)
            log.append(output)
            episode1000_start = time.time()

        # 保存信息
        if episode_num > arglist.num_episodes:
            print('...Finished total of {} episodes.'.format(len(episode_rewards)))
            save_all_data(env=env, arglist=arglist, episode_rewards=episode_rewards, final_ep_rewards=final_ep_rewards,
                          log=log, win_lose=episode_win_lose, trainers=trainers)
            break


if __name__ == '__main__':
    arglist = parse_args()
    data = read_from_csv('Variable_Parameter.csv')
    setattr(arglist, 'Variable_Parameter', data)
    for para in data[1:]:
        setattr(arglist, para[0], para[1])
    for i in range(1):
        train(arglist)
