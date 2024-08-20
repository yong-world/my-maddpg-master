import argparse
import os

import numpy as np
import tensorflow as tf
import time
import pickle
import matplotlib.pyplot as plt
import keyboard
import datetime

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_world_comm", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    # parser.add_argument("--num-episodes", type=int, default=200, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")  # 只有需要
    # 指定为DDPG算法时才需要确定这个
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
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


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world返回的是world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        # 构建世界，初始化随机参数，返回奖励，
        # 返回观察[自身位置 +自身速度 + 地标位置 + 其它可观智能体位置 + 其它智能体的速度 + 自身是否在森林里+通信信息]，
        # good agent的in_forest + other_vel是反过来的且没有comm
        # 返回监测(碰撞数)
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy == 'ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy == 'ddpg')))
    return trainers


def easy_plot(data_input, title, x_label, y_label, file_name, marker=None):
    plt.plot(range(1, len(data_input) + 1), data_input, linewidth='0.5', marker=marker)
    now_time = datetime.datetime.now()
    time_str = now_time.strftime('%Y%m%d_%H%M%S')
    png_folder_dir = arglist.learning_curves_figure_dir
    png_dir = arglist.learning_curves_figure_dir + file_name + '_' + 'TF' + str(
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
    with U.single_threaded_session():  # 单进程运行session
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize 所有变量
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:  # 用tf.nn.train恢复模型变量
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        print('Starting iterations...')
        print('scenario:{}\tnum-episodes:{}\tbatch-size:{}\tsave-rates:{}\tlr:{}\n'.format(arglist.scenario,
                arglist.num_episodes,arglist.batch_size,arglist.save_rate, arglist.lr))
        log_file_name = arglist.log_dir + 'TF' + '_' + arglist.exp_name + '_' + 'log.txt'
        with open(log_file_name, 'a') as fp:
            now_time = datetime.datetime.now()
            time_str = now_time.strftime('%Y%m%d  %H:%M:%S')
            fp.write('-----------------------------------------------------------------------------------------\n'
                +'Starting iterations : {}\n'.format(time_str))
            fp.write('scenario:{}\t'.format(str(arglist.scenario)))
            fp.write('num-episodes:{}\t'.format(str(arglist.num_episodes)))
            fp.write('batch-size:{}\t'.format(str(arglist.batch_size)))
            fp.write('save-rates:{}\t'.format(str(arglist.save_rate)))
            fp.write('lr:{}\n'.format(str(arglist.lr)))
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            # environment step done因为场景给的done_callback是none所以返回的done_n只是[false false ...]
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            # 每过max_episode_len=25步产生一次结束信号
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
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

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies info_n返回值都是，是否发生碰撞
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

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()  # 清空采样索引
            for agent in trainers:  # 大于最大经验池时才更新，且每一百训练步更新一次
                loss = agent.update(trainers, train_step)

            # save model, display training output，保存所有tf变量，输出训练信息，保存最后save_rate轮的总奖励和agent奖励
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):  # 如果一轮结束且是保存轮
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether there are adversaries
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

            # saves final episode reward for plotting training curve later 将最后save_rate轮奖励写入文件
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                # 绘图
                easy_plot(data_input=episode_rewards,
                          title='TF_mean_episode_rewards:{}'.format(np.mean(episode_rewards)),
                          x_label='episode',
                          y_label='rewards', file_name='EpisodeRewards')
                easy_plot(data_input=final_ep_rewards, title='TF_Every1000rewards', x_label='episode',
                          y_label='rewards', file_name='Every1000Rewards', marker='.')
                with open(log_file_name, 'a') as fp:
                    fp.write('mean rewards:{}\n'.format(str(np.mean(episode_rewards))))
                    fp.write(
                        'End iterations\n')
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
