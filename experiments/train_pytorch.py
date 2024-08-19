import argparse
import os
import time
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
import numpy as np
from maddpg_pytorch.maddpg_pytorch import MADDPGAgentTrainer
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios


# improve volatile模式，gpu模式

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_world_comm", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=2000, help="number of episodes")
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
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training "
                                                                             "state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many "
                                                                    "episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and "
                                                                 "model are loaded")
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
    return parser.parse_args()


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


def load_state(load_dir, trainers):
    """Load all the variables from the location <load_dir>"""
    checkpoint = torch.load(os.path.join(load_dir, "model.pt"))
    trainers[:] = checkpoint['trainers']
    print(f"Model loaded from {load_dir}")


def make_env(scenario_name, arglist, benchmark=False):
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()

    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def get_trainers(env, num_adversaries, obs_shape_n, arglist,device):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        # action_space是根据动作维度和通信维度得到的box、discrete
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy == 'ddpg'),device=device))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy == 'ddpg'),device=device))
    return trainers


def easy_plot(data_input, title, x_label, y_label, file_name, marker=None):
    plt.plot(range(1, len(data_input) + 1), data_input, linewidth='0.5', marker=marker)
    now_time = datetime.datetime.now()
    time_str = now_time.strftime('%Y%m%d_%H%M%S')
    png_folder_dir = arglist.learning_curves_figure_dir
    png_dir = arglist.learning_curves_figure_dir + file_name + '_' + 'TH' + str(
        arglist.num_episodes) + '_' + arglist.exp_name + '_' + time_str + '.png'
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    if not os.path.exists(png_folder_dir):
        os.makedirs(png_folder_dir)
        print("Folder created")
    plt.savefig(png_dir)
    print("File saved to:", png_dir)
    plt.show()


def train(arglist):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create environment
    env = make_env(arglist.scenario, arglist, arglist.benchmark)

    # Create agent trainers
    obs_shape_n = [env.observation_space[i].shape[0] for i in range(env.n)]
    num_adversaries = min(env.n, arglist.num_adversaries)  # env.n是策略个体数，arglist.num_adversaries默认是0，辅助指定ddpg个数
    trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist,device)
    print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

    # Load previous results, if necessary
    if arglist.load_dir == "":
        arglist.load_dir = arglist.save_dir
    if arglist.display or arglist.restore or arglist.benchmark:  # 用torch加载模型
        print('Loading previous state')
        load_state(arglist.load_dir, trainers)

    episode_rewards = [0.0]
    agent_rewards = [[0.0] for _ in range(env.n)]
    final_ep_rewards = []
    final_ep_ag_rewards = []
    agent_info = [[[]]]
    obs_n = env.reset()
    episode_step = 0
    train_step = 0
    t_start = time.time()

    print('Starting iterations...')
    print('scenario:{}\tnum-episodes:{}\tbatch-size:{}\tsave-rates:{}\tlr:{}'.format(arglist.scenario,
                                                                                     arglist.num_episodes,
                                                                                     arglist.batch_size,
                                                                                     arglist.save_rate, arglist.lr))
    log_file_name = arglist.log_dir + 'TH' + '_' + arglist.exp_name + '_' + 'log.txt'
    with open(log_file_name, 'a') as fp:
        now_time = datetime.datetime.now()
        time_str = now_time.strftime('%Y%m%d  %H:%M:%S')
        fp.write('-----------------------------------------------------------------------------------------\n'
                 + 'Starting iterations : {}\n'.format(time_str))
        fp.write('scenario:{}\t'.format(str(arglist.scenario)))
        fp.write('num-episodes:{}\t'.format(str(arglist.num_episodes)))
        fp.write('batch-size:{}\t'.format(str(arglist.batch_size)))
        fp.write('save-rates:{}\t'.format(str(arglist.save_rate)))
        fp.write('lr:{}\n'.format(str(arglist.lr)))
    while True:
        action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
        # done由于构建环境的时候没有传入done，所以传回都是false，并没有使用
        new_obs_n, rew_n, done_n, info_n = env.step(action_n)
        episode_step += 1
        done = all(done_n)
        terminal = (episode_step >= arglist.max_episode_len)

        for i, agent in enumerate(trainers):
            agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
        obs_n = new_obs_n

        for i, rew in enumerate(rew_n):
            episode_rewards[-1] += rew
            agent_rewards[i][-1] += rew

        if done or terminal:
            obs_n = env.reset()
            episode_step = 0
            episode_rewards.append(0)
            for a in agent_rewards:
                a.append(0)
            agent_info.append([[]])

        train_step += 1

        if arglist.benchmark:
            for i, info in enumerate(info_n):
                agent_info[-1][i].append(info_n['n'])
            if train_step > arglist.benchmark_iters and (done or terminal):
                file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                print('Finished benchmarking, now saving...')
                with open(file_name, 'wb') as fp:
                    pickle.dump(agent_info[:-1], fp)
                break
            continue

        if arglist.display:
            time.sleep(0.1)
            env.render()
            continue

        loss = None
        for agent in trainers:
            agent.preupdate()
        for agent in trainers:
            loss = agent.update(trainers, train_step)
        # 每save_rate(1000)轮保存模型,输出训练信息
        if terminal and (len(episode_rewards) % arglist.save_rate == 0):
            save_path = os.path.join(arglist.save_dir, arglist.exp_name + "pytorch_model.pt")
            torch.save({'trainers': trainers}, save_path)
            # print(f"Model saved to {save_path}")

            if num_adversaries == 0:
                output = "steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                    train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                    round(time.time() - t_start, 3))
                print(output)
                with open(log_file_name, 'a') as fp:
                    fp.write(output + '\n')
            else:
                output = "steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                    train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                    [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time() - t_start, 3))
                print(output)
                with open(log_file_name, 'a') as fp:
                    fp.write(output + '\n')
            t_start = time.time()
            # Keep track of final episode reward 最后一千轮的总奖励和agent奖励
            final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
            for rew in agent_rewards:
                final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

        # EpisodeRewards和Every1000Rewards信息，绘图输出
        if len(episode_rewards) > arglist.num_episodes:
            rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
            with open(rew_file_name, 'wb') as fp:
                pickle.dump(final_ep_rewards, fp)
            agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
            with open(agrew_file_name, 'wb') as fp:
                pickle.dump(final_ep_ag_rewards, fp)
            print('...Finished total of {} episodes.'.format(len(episode_rewards)))
            # 绘图
            easy_plot(data_input=episode_rewards, title='TH_mean_episode_rewards:{}'.format(np.mean(episode_rewards)),
                      x_label='episode', y_label='rewards', file_name='EpisodeRewards')
            easy_plot(data_input=final_ep_rewards, title='TH_Every1000rewards', x_label='episode',
                      y_label='rewards', file_name='Every1000Rewards', marker='.')
            with open(log_file_name, 'a') as fp:
                fp.write('mean rewards:{}\n'.format(str(np.mean(episode_rewards))))
                fp.write(
                    'End iterations\n' + '-----------------------------------------------------------------------------------------\n')
            # while True:
            #     if keyboard.is_pressed('esc'):
            #         break
            #     time.sleep(0.1)
            break


if __name__ == '__main__':
    arglist = parse_args()
    if not os.path.exists(arglist.log_dir):
        os.makedirs(arglist.log_dir)
        print("Folder created")
    train(arglist)
