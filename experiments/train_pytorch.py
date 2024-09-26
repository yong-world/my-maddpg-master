from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, csv
import sys  # 导入sys模块

import win32con
import win32gui

script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取脚本所在的目录
os.chdir(script_dir)  # 将工作目录切换到脚本所在目录
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
print(sys.path)
import argparse

import sys
import time
import pickle
# import importlib
import logging
import traceback
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import datetime
import numpy as np
from gym import spaces
from maddpg_pytorch.maddpg_pytorch import MADDPGAgentTrainer, vae_train
from smac.env import StarCraft2Env
from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="3m", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=60, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=3000, help="number of episodes")
    parser.add_argument("--max-train-step", type=int, default=3000000, help="maximum train step")
    parser.add_argument("--test-step", type=int, default=10000, help="test step")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-test-episodes", type=int, default=100, help="num test eps")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")

    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="TMMaddpgSmac", help="name of the experiment")
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


def get_logger():
    # logging = importlib.import_module('logging')
    # 创建 Logger 实例
    train_logging = logging.getLogger("mylogger")
    # 设置日志级别
    train_logging.setLevel(logging.DEBUG)
    train_logging.propagate = False

    # 创建控制台 Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # 创建文件处理器
    file_handler = logging.FileHandler(exp_dir + 'experiment.log', mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # 设置输出格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 将 Handler 添加到 Logger
    train_logging.addHandler(console_handler)
    train_logging.addHandler(file_handler)

    # 记录日志
    # train_logging.debug('这是一个调试信息')
    # train_logging.info('这是一个普通信息')
    # train_logging.warning('这是一个警告信息')
    # train_logging.error('这是一个错误信息')
    # train_logging.critical('这是一个严重错误信息')
    return train_logging


def get_time():
    return str(datetime.datetime.now().strftime('%y-%m-%d %H:%M:%S'))


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
    my_logger.debug(f"figure saved to:{png_dir}")
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
    # print(f"数据已从 {file_name} 读取")
    return data


def run_evaluate_episode(env, trainers, tbwriter, csv_writer, episode_num, train_step, interval_time):
    eval_is_win_buffer = []
    eval_reward_buffer = []
    eval_steps_buffer = []
    for i in range(arglist.num_test_episodes):
        episode_reward = 0.0
        episode_step = 0
        terminated = False
        _ = env.reset()
        while not terminated:
            obs_n = env.get_obs()
            new_obs_n = np.array(obs_n, copy=False)
            new_obs_n = torch.from_numpy(new_obs_n).to(device)
            action_n = [agent.action(obs) for agent, obs in zip(trainers, new_obs_n)]
            action_n_index = act_mask_max(action_n, env)
            reward, terminated, info_n = env.step(action_n_index)
            episode_step += 1
            episode_reward += reward

        is_win = env.win_counted
        eval_reward_buffer.append(episode_reward)
        eval_steps_buffer.append(episode_step)
        eval_is_win_buffer.append(is_win)

    tbwriter.add_scalar('eval_win_rate', np.mean(eval_is_win_buffer), train_step)
    tbwriter.add_scalar('eval_reward', np.mean(eval_reward_buffer), train_step)
    tbwriter.add_scalar('eval_steps', np.mean(eval_steps_buffer), train_step)
    output = (f'episode:{episode_num:<8}\ttrain_step:{train_step:<8}\t\tmean_step:{np.mean(eval_steps_buffer):<8.0f}'
              f'\ttime:{interval_time:<8.0f}reward:'
              f'{np.mean(eval_reward_buffer):<8.2f}\twin_rate:{np.mean(eval_is_win_buffer):<8.2f}')
    my_logger.info(output)
    csv_writer.writerow({
        "episode": f'{episode_num:}',
        "train_step": f'{train_step:}',
        "time": f'{interval_time:.0f}',
        "mean_step": f'{np.mean(eval_steps_buffer):.0f}',
        "reward": f'{np.mean(eval_reward_buffer):.2f}',
        "win_rate": f'{np.mean(eval_is_win_buffer):.2f}',
    })
    return


def save_all_data(env, arglist, episode_rewards, final_ep_rewards, win_lose, trainers, exp_dir):
    # 保存奖励、胜率对象
    with open(exp_dir + 'episode_rewards.pkl', 'wb') as fp:
        pickle.dump(episode_rewards, fp)
    with open(exp_dir + 'episode_rewards.pkl', 'wb') as fp:
        pickle.dump(final_ep_rewards, fp)
    with open(exp_dir + 'episode_rewards.pkl', 'wb') as fp:
        pickle.dump(win_lose, fp)
    # 绘图并保存
    mean_reward = np.mean(episode_rewards)
    # plot_save(data_input=final_ep_rewards,
    #           title='Rewards',
    #           x_label='episode',
    #           y_label='reward', png_dir=exp_dir + 'sample_rewards.png', marker='.')

    # 保存模型
    save_path = exp_dir + "pytorch_model.pt"
    torch.save({'trainers': [[trainers[i].p, trainers[i].target_p, trainers[i].q, trainers[i].target_q, trainers[i].vae]
                             for i in range(trainers[0].n)]}, save_path)

    # 保存自变参数
    arglist.Variable_Parameter[1][1] += 1
    write_to_csv('Variable_Parameter.csv', arglist.Variable_Parameter)


class StdToLog:
    def __init__(self, file, std):
        self.file = file  # 打开文件用于写入
        self.std = std  # 保存原始 stdout

    def write(self, message):
        self.file.write(message)  # 写入文件
        self.std.write(message)  # 同时输出到控制台
        self.file.flush()

    def flush(self):
        self.file.flush()  # 刷新文件
        self.std.flush()  # 刷新控制台输出

    def close(self):
        self.file.close()  # 手动关闭文件

    def __enter__(self):
        return self  # 返回当前实例供 with 语句使用

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()  # 在退出时关闭文件


def write_to_csv(file_name, data):
    # data应为列表，每个元素都是一个参数或参数列表（每行的数据）
    with open(file_name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    my_logger.debug(f"数据已写入 {file_name}")


def train():
    # 创建实验保存路径
    fieldnames = ["episode", "train_step", "mean_step", "time", "reward", "win_rate"]
    csv_writer = csv.DictWriter(csv_log_file, fieldnames=fieldnames)
    csv_writer.writeheader()
    tbwriter = SummaryWriter(exp_dir)
    # Create environment
    env, env_info, act_space_n, obs_space_n, act_space_acc, obs_space_acc, obs_n = make_env(arglist.scenario)
    # Create agent trainers
    trainers = get_trainers(env, env_info['n_agents'], obs_space_n, act_space_n, arglist, device)
    # Load previous results, if necessary
    if arglist.load_dir == "":
        arglist.load_dir = arglist.save_dir
    if arglist.display or arglist.restore or arglist.benchmark:  # 用torch加载模型
        my_logger.debug('Loading previous state')
    episode_rewards = [0.0]
    episode_num = 0
    episode_win_lose = []
    final_ep_rewards = []
    agent_info = [[[]]]
    episode_step = 0
    train_step = 0
    episode1000_start = time.time()
    last_test_step = -1e10
    my_logger.info('--------------------------------------------------------'
                   '---------------------------------------------------------------')
    my_logger.info(f'Starting iterations : {get_time()}')
    train_setting = ('exp_name:{}\tscenario:{}\tnum-episodes:{}\tbatch-size:{}\tsave-rates:{}\tlr:{}\tdevice:{}'.format
                     (arglist.exp_name, arglist.scenario, arglist.max_train_step, arglist.batch_size, arglist.save_rate,
                      arglist.lr, device))
    my_logger.info(train_setting)
    train_time = time.time()
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
            # sys.stdout.write("\repisode:{},steps:{},episode_step:{}".format(episode_num, train_step, episode_step))
            # sys.stdout.flush()
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

        if terminated and (train_step - last_test_step > arglist.test_step):
            last_test_step = train_step - (train_step % arglist.test_step)
            run_evaluate_episode(env=env, trainers=trainers, tbwriter=tbwriter, csv_writer=csv_writer,
                                 episode_num=episode_num, train_step=train_step, interval_time=time.time() - train_time)
            train_time = time.time()
        # 保存信息
        if train_step > arglist.max_train_step:
            my_logger.info(f'Finished total of {len(episode_rewards)} episodes at {get_time()}')
            save_all_data(env=env, arglist=arglist, episode_rewards=episode_rewards, final_ep_rewards=final_ep_rewards,
                          win_lose=episode_win_lose, trainers=trainers, exp_dir=exp_dir)
            break


if __name__ == '__main__':
    arglist = parse_args()
    data = read_from_csv('Variable_Parameter.csv')
    setattr(arglist, 'Variable_Parameter', data)
    for para in data[1:]:
        setattr(arglist, para[0], para[1])
    exp_dir = arglist.log_head + '{}_{}{}{}/'.format(arglist.exp_index, arglist.scenario, arglist.max_train_step,
                                                     arglist.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    with open(exp_dir + "train.log", 'a', encoding='utf-8', buffering=1) as std_log:
        # 控制台输出的信息和错误保存到std.log文件
        std_out_log = StdToLog(std_log, sys.stdout)
        std_err_to_log = StdToLog(std_log, sys.stderr)
        sys.stdout = std_out_log
        sys.stderr = std_err_to_log

        for i in range(1):
            arglist = parse_args()
            data = read_from_csv('Variable_Parameter.csv')
            setattr(arglist, 'Variable_Parameter', data)
            for para in data[1:]:
                setattr(arglist, para[0], para[1])
            setattr(arglist, "exp_dir", exp_dir)

            exp_dir = arglist.log_head + '{}_{}{}{}/'.format(arglist.exp_index, arglist.scenario,
                                                             arglist.max_train_step, arglist.exp_name)
            os.makedirs(exp_dir, exist_ok=True)
            my_logger = get_logger()

            with open(arglist.exp_dir + "csv_log.csv", mode="w", newline="", buffering=1) as csv_log_file:
                train()
